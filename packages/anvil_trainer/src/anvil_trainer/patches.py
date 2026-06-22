"""Monkey-patches applied to lerobot at training time.

``TransformRunner`` owns:
    * The active list of :class:`~anvil_trainer.transforms.Transform` instances.
    * Five monkey-patches on lerobot modules:
        - ``apply_dataset_patches`` — patches ``LeRobotDataset.__getitem__``.
        - ``apply_val_loss_patch`` — patches ``make_dataset`` (split creation),
          captures the preprocessor from ``make_pre_post_processors``, and
          injects EE rel action stats into the returned ``train_dataset``.
        - ``apply_checkpoint_patch`` — patches ``save_checkpoint`` to compute
          test loss and write ``anvil_config.json`` / ``split_info.json`` next
          to each checkpoint.
        - ``apply_val_loss_hook`` — patches ``update_policy`` for periodic val
          loss computation.
        - ``apply_metadata_patches`` — runs ``Transform.patch_metadata`` hooks
          (currently used by ``ExcludeObservationTransform``).

Patches are installed via :meth:`TransformRunner._patch` which tracks the
original attribute so :meth:`restore_all_patches` can put everything back.
For the typical ``train()`` entry point, use the
:func:`patched_lerobot` context manager which guarantees cleanup even when
training raises.
"""
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from anvil_shared.splits import compute_split_episodes, load_split_info, save_split_info

from anvil_trainer.config import TrainingConfig
from anvil_trainer.transforms import (
    DataIntegrityError,
    EERelTransform,
    ExcludeObservationTransform,
    TaskOverrideTransform,
    Transform,
)

log = logging.getLogger(__name__)

# Sentinel used to mark "patch already installed" in the originals list so we
# can keep insertion order + detect re-entrancy without wrapping in tuples.
_PATCHED_MARKER = object()


class TransformRunner:
    """
    Manages and applies dataset transforms.

    Handles:
    - Registration of transforms
    - Metadata patching (before lerobot import)
    - Dataset patching (after lerobot import)
    """

    # Registry of available transforms (add new transforms here).
    # Instantiated fresh per TransformRunner so stateful transforms do not share state across runs.
    TRANSFORMS: list[Transform] = []  # populated in __init__

    def __init__(self, config: TrainingConfig):
        self.config = config
        transforms: list[Transform] = [
            ExcludeObservationTransform(),
            TaskOverrideTransform(),
            EERelTransform(),
        ]
        self.active_transforms = [t for t in transforms if t.is_enabled(config)]
        self._val_dataloader = None   # set by apply_val_loss_patch when make_dataset is called
        self._test_dataloader = None  # set by apply_val_loss_patch when make_dataset is called
        self._split_info: dict = {}   # populated by patched_make_dataset
        self._preprocessor = None     # captured from make_pre_post_processors
        self._val_freq = 0            # set from cfg.log_freq * 5 inside patched_make_dataset
        self._resume_step = 0         # for absolute step tracking in wandb
        # List of (module, attr_name, original_value) — populated by _patch in
        # insertion order so restore_all_patches can revert in reverse.
        self._saved_originals: list[tuple[Any, str, Any]] = []

    # -------------------------------------------------------------------------
    # Patch install / restore infrastructure
    # -------------------------------------------------------------------------

    def _patch(self, module: Any, attr_name: str, new_value: Any) -> None:
        """Install a monkey-patch and remember the original for later restoration.

        Calling this twice with the same ``(module, attr_name)`` from the same
        TransformRunner instance is a no-op (with a debug log) — it will not
        re-wrap and lose the true original.  Use :meth:`restore_all_patches`
        to put things back; the :func:`patched_lerobot` context manager does
        this automatically.
        """
        already_patched = any(
            m is module and n == attr_name for m, n, _ in self._saved_originals
        )
        if already_patched:
            log.debug(
                "[anvil_trainer] Skipping duplicate patch %s.%s",
                getattr(module, "__name__", module), attr_name,
            )
            return
        original = getattr(module, attr_name)
        self._saved_originals.append((module, attr_name, original))
        setattr(module, attr_name, new_value)

    def restore_all_patches(self) -> None:
        """Restore every attribute touched by :meth:`_patch` (LIFO).

        Called by :func:`patched_lerobot` on context exit.  Safe to call more
        than once — the originals list is cleared after restoration.
        """
        while self._saved_originals:
            module, attr_name, original = self._saved_originals.pop()
            try:
                setattr(module, attr_name, original)
            except Exception as e:  # pragma: no cover — extremely defensive
                log.warning(
                    "[anvil_trainer] Failed to restore %s.%s: %s",
                    getattr(module, "__name__", module), attr_name, e,
                )

    def log_config(self) -> None:
        """Log active transforms."""
        if not self.active_transforms:
            log.info("[anvil_trainer] Active transforms: (none - pass-through mode)")
            return

        for transform in self.active_transforms:
            details = self._get_transform_details(transform)
            log.info("[anvil_trainer] Active transform: %s — %s", transform.name, details)

    def _get_transform_details(self, transform: Transform) -> str:
        """Get human-readable details for a transform."""
        if isinstance(transform, ExcludeObservationTransform):
            return f"excluding: {', '.join(self.config.exclude_observation)}"
        elif isinstance(transform, TaskOverrideTransform):
            return f"'{self.config.task_override}'"
        elif isinstance(transform, EERelTransform):
            return "SE(3) relative: delta_xyz + R_state.T @ R_action"
        return "enabled"

    def _compute_ee_rel_stats(self, full_dataset: Any, cfg: Any) -> dict | None:
        """Compute EE relative action AND obs stats for ``ee_rel`` training.

        Both observation.state and action are transformed to SE(3)-relative with the
        SAME anchor (current EE pose), matching UMI.  Stats are computed from the
        actual training distributions so normalization is valid.

        Action target at horizon step k (anchored to state[t]):
            body_delta[k]  = R_state[t].T @ (act_xyz[t+k] - state_xyz[t])
            delta_rot6d[k] = matrices_to_rot6d(R_state[t].T @ R_act[t+k])
            gripper[k]     = action_gripper[t+k]   (absolute, stats restored)

        Obs target at obs step j (anchored to state[t], j = 0..n_obs_steps-1):
            identity step (j == n_obs_steps-1): all zeros + [1,0,0,0,1,0] + abs gripper
            prior steps (j < n_obs_steps-1): obs[t-(n_obs_steps-1-j)] rel to obs[t]

        Returns a dict ``{"action": stats, "observation.state": obs_stats}``,
        or ``None`` on failure.
        """
        if not self.config.is_ee_rel:
            return None

        import numpy as np
        try:
            from anvil_shared.ee_transform import (
                ee_obs_rel_forward,
                ee_rel_forward,
                n_arms_from_dims,
                EE_ACTION_DIM_PER_ARM,
            )

            hf = full_dataset.hf_dataset
            actions_np = np.array(hf["action"], dtype=np.float64)       # (N, 10*n_arms)
            states_np = np.array(hf["observation.state"], dtype=np.float64)  # (N, 8*n_arms)
            episode_idx_np = np.array(hf["episode_index"], dtype=np.int64).ravel()

            if states_np.ndim == 3:
                states_np = states_np[:, -1, :]  # multi-step obs → most recent step

            n_arms = n_arms_from_dims(states_np.shape[-1], actions_np.shape[-1])

            # ------------------------------------------------------------------ #
            # Action stats (relative to current state, per-sample anchor)        #
            # ------------------------------------------------------------------ #
            action_delta_indices = getattr(cfg.policy, "action_delta_indices", None)
            n_steps = len(action_delta_indices) if action_delta_indices else 1
            N = len(actions_np)

            def _ee_rel_action_for_k(k: int) -> np.ndarray:
                if k == 0:
                    act = actions_np
                    sta = states_np
                    mask = np.ones(N, dtype=bool)
                else:
                    act = actions_np[k:]
                    sta = states_np[:-k]
                    mask = episode_idx_np[k:] == episode_idx_np[:-k]
                d = ee_rel_forward(act, sta)
                return d[mask]

            all_deltas = np.concatenate(
                [_ee_rel_action_for_k(k) for k in range(n_steps)], axis=0
            )  # (N_valid_pairs, 10*n_arms)

            orig_action = full_dataset.meta.stats.get("action", {})
            delta_mean = all_deltas.mean(axis=0)
            delta_std = np.where(all_deltas.std(axis=0) < 1e-6, 1e-6, all_deltas.std(axis=0))
            delta_min = all_deltas.min(axis=0)
            delta_max = all_deltas.max(axis=0)

            # Restore gripper stats to absolute range
            orig_arr = lambda key, fallback: np.array(orig_action.get(key, fallback))
            for arm in range(n_arms):
                grip_idx = arm * EE_ACTION_DIM_PER_ARM + 9
                for arr, key in [
                    (delta_mean, "mean"), (delta_std, "std"),
                    (delta_min, "min"), (delta_max, "max"),
                ]:
                    orig_vals = orig_arr(key, arr)
                    if grip_idx < len(orig_vals):
                        arr[grip_idx] = orig_vals[grip_idx]

            action_patched_stats = {
                "mean": delta_mean.tolist(),
                "std": delta_std.tolist(),
                "min": delta_min.tolist(),
                "max": delta_max.tolist(),
                "count": orig_action.get("count", len(all_deltas)),
            }
            full_dataset.meta.stats["action"] = action_patched_stats

            # ------------------------------------------------------------------ #
            # Obs stats (relative to current state, 10-dim rot6d layout)         #
            # ------------------------------------------------------------------ #
            # Identity step: obs[t] relative to obs[t] — always zeros + identity rot6d
            # Prior steps: obs[t-j] relative to obs[t]  (j = 1..n_obs_steps-1)
            # We include both distributions and compute MEAN_STD over all.
            n_obs_steps = getattr(cfg.policy, "n_obs_steps", 2)

            obs_rel_samples = []
            # Identity steps — all N frames (obs relative to itself)
            identity = ee_obs_rel_forward(states_np, states_np)  # zeros+[1,0,0,0,1,0]+grip
            obs_rel_samples.append(identity)

            # Prior steps — obs[t-j] relative to obs[t], episode-bounded
            for j in range(1, n_obs_steps):
                past = states_np[:-j]
                anchor = states_np[j:]
                mask = episode_idx_np[:-j] == episode_idx_np[j:]
                rel = ee_obs_rel_forward(past, anchor)
                obs_rel_samples.append(rel[mask])

            all_obs_rel = np.concatenate(obs_rel_samples, axis=0)  # (N_total, 10*n_arms)

            obs_mean = all_obs_rel.mean(axis=0)
            obs_std = np.where(all_obs_rel.std(axis=0) < 1e-6, 1e-6, all_obs_rel.std(axis=0))
            obs_min = all_obs_rel.min(axis=0)
            obs_max = all_obs_rel.max(axis=0)

            obs_patched_stats = {
                "mean": obs_mean.tolist(),
                "std": obs_std.tolist(),
                "min": obs_min.tolist(),
                "max": obs_max.tolist(),
                "count": len(all_obs_rel),
            }
            full_dataset.meta.stats["observation.state"] = obs_patched_stats

            log.info(
                "[ee_rel_stats] action: %d samples (n_steps=%d); obs: %d samples "
                "(n_obs_steps=%d); %d arm(s)",
                len(all_deltas), n_steps, len(all_obs_rel), n_obs_steps, n_arms,
            )
            return {"action": action_patched_stats, "observation.state": obs_patched_stats}
        except DataIntegrityError:
            raise
        except Exception as e:
            log.warning("[ee_rel_stats] Failed: %s — falling back to absolute stats", e)
            return None

    def apply_metadata_patches(self) -> None:
        """Apply metadata patches before importing lerobot training."""
        for transform in self.active_transforms:
            transform.patch_metadata(self.config, runner=self)

    def apply_dataset_patches(self) -> None:
        """Patch LeRobotDataset.__getitem__ to apply transforms and fix index mapping.

        This patch is always installed (even without active_transforms) because
        EpisodeAwareSampler yields absolute frame indices that must be remapped to
        relative indices for filtered (split) datasets. The mapping is only applied
        to the train dataset instance (flagged via _anvil_uses_abs_sampler).
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # We must capture the original __getitem__ to use it in our patch.
        # LeRobotDataset.__getitem__ in v0.5.1 does not perform index mapping,
        # but EpisodeAwareSampler yields absolute indices. We add the mapping
        # logic here to support filtered datasets (splits).
        original_getitem = LeRobotDataset.__getitem__
        transforms = self.active_transforms
        config = self.config

        def patched_getitem(self, idx):
            # 1. Resolve relative index if the dataset is filtered by episodes.
            # Only the train dataset uses EpisodeAwareSampler (absolute indices).
            # Val/test datasets use DataLoader without a sampler (relative indices
            # 0..N-1) and must NOT be remapped — doing so would corrupt reads when
            # relative indices overlap with the absolute frame index space.
            reader = self._ensure_reader()
            if getattr(self, '_anvil_uses_abs_sampler', False) and reader._absolute_to_relative_idx is not None:
                # Map from absolute HF frame index to relative filtered index
                idx = reader._absolute_to_relative_idx.get(idx, idx)

            # 2. Call original __getitem__ (which calls reader.get_item)
            item = original_getitem(self, idx)

            # 3. Apply transforms (no-op when transforms list is empty)
            for transform in transforms:
                item = transform.apply(item, config)
            return item

        self._patch(LeRobotDataset, "__getitem__", patched_getitem)
        log.info("[anvil_trainer] Patched LeRobotDataset.__getitem__ (%d transform(s))", len(transforms))

    def apply_val_loss_patch(self) -> None:
        """Monkey-patch make_dataset to create train/val/test splits, and capture preprocessor."""
        s = self.config.split_ratio
        total_r = sum(s)
        if total_r <= 0 or (s[1] <= 0 and (len(s) < 3 or s[2] <= 0)):
            return  # no val or test, skip patching

        import lerobot.datasets.factory as factory_mod
        import lerobot.policies.factory as policy_factory_mod
        import lerobot.scripts.lerobot_train as lerobot_train_mod
        import torch
        from lerobot.datasets.factory import make_dataset as original_make_dataset

        val_state = self
        _patched = {"done": False}

        def patched_make_dataset(cfg):
            # Only intercept the first call (main process dataset creation).
            if _patched["done"]:
                return original_make_dataset(cfg)
            _patched["done"] = True

            # Capture val_freq and resume_step from lerobot cfg
            val_state._val_freq = cfg.log_freq * 5 if cfg.log_freq > 0 else 0
            if cfg.resume and hasattr(cfg, "checkpoint_path") and cfg.checkpoint_path:
                try:
                    step_file = Path(cfg.checkpoint_path) / "training_state" / "training_step.json"
                    if step_file.exists():
                        val_state._resume_step = json.loads(step_file.read_text()).get("step", 0)
                except Exception:
                    val_state._resume_step = 0

            # Full dataset to determine total episode count
            full_dataset = original_make_dataset(cfg)
            total_ep = full_dataset.num_episodes

            # Compute relative stats for ee_rel (both action and obs.state).
            # joint_abs / ee_abs use dataset stats as-is.
            if val_state.config.is_ee_rel:
                _patched_ee_stats = val_state._compute_ee_rel_stats(full_dataset, cfg)
            else:
                _patched_ee_stats = None

            # Check if split_info.json already exists in last checkpoint (for resume)
            split_info_path = Path(cfg.output_dir) / "checkpoints" / "last" / "pretrained_model" / "split_info.json"
            loaded_split = load_split_info(split_info_path)
            if loaded_split is not None:
                train_ep = loaded_split.get("train_episodes", [])
                val_ep = loaded_split.get("val_episodes", [])
                test_ep = loaded_split.get("test_episodes", [])
                log.info("[split] Loaded random splits from %s", split_info_path)
            else:
                train_ep = val_ep = test_ep = None

            if train_ep is None:
                # Optional: subsample N episodes before splitting
                import random as _random
                pool = list(range(total_ep))
                max_ep = val_state.config.max_episodes
                if max_ep is not None and max_ep < total_ep:
                    _rng = _random.Random(cfg.seed)
                    pool = sorted(_rng.sample(pool, max_ep))
                    log.info("[split] Subsampled %d / %d episodes (--max-episodes)", len(pool), total_ep)

                # Random three-way split via shared helper (seeded for reproducibility)
                splits = compute_split_episodes(len(pool), s, seed=cfg.seed)
                train_ep = [pool[i] for i in splits["train"]]
                val_ep = [pool[i] for i in splits["val"]]
                test_ep = [pool[i] for i in splits["test"]]

                if len(train_ep) < 1:
                    log.warning("[split] Not enough episodes (%d) for split %s, using all for training", total_ep, s)
                    return full_dataset
                log.info("[split] Generated random splits")

            # Store split info for anvil_config.json (as full lists now)
            val_state._split_info = {
                "split_ratio": list(s),
                "total_episodes": total_ep,
                "train_episodes": train_ep,
                "val_episodes": val_ep,
                "test_episodes": test_ep,
                **({"max_episodes": val_state.config.max_episodes} if val_state.config.max_episodes is not None else {}),
            }

            def _make_dataloader(dataset):
                return torch.utils.data.DataLoader(
                    dataset,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    sampler=None,
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    prefetch_factor=2 if cfg.num_workers > 0 else None,
                )

            # Val dataloader
            if val_ep:
                cfg.dataset.episodes = val_ep
                val_dataset = original_make_dataset(cfg)
                val_state._val_dataloader = _make_dataloader(val_dataset)
                log.info("[split] val=%d ep (randomly selected, %d frames)", len(val_ep), val_dataset.num_frames)

            # Test dataloader
            if test_ep:
                cfg.dataset.episodes = test_ep
                test_dataset = original_make_dataset(cfg)
                val_state._test_dataloader = _make_dataloader(test_dataset)
                log.info("[split] test=%d ep (randomly selected, %d frames)", len(test_ep), test_dataset.num_frames)

            # Train dataset
            cfg.dataset.episodes = train_ep
            train_dataset = original_make_dataset(cfg)
            # Flag this instance so patched_getitem applies absolute→relative mapping.
            # EpisodeAwareSampler (used by ACT and similar policies) yields absolute
            # frame indices; val/test dataloaders use relative indices and must NOT
            # be remapped.
            train_dataset._anvil_uses_abs_sampler = True
            # Inject EE rel stats so lerobot's normalizer uses relative distributions.
            if _patched_ee_stats is not None:
                train_dataset.meta.stats["action"] = _patched_ee_stats["action"]
                train_dataset.meta.stats["observation.state"] = _patched_ee_stats["observation.state"]
                log.info("[ee_rel_stats] Patched train_dataset.meta.stats [action + observation.state]")
            log.info("[split] train=%d ep (randomly selected)", len(train_ep))
            return train_dataset

        self._patch(factory_mod, "make_dataset", patched_make_dataset)
        self._patch(lerobot_train_mod, "make_dataset", patched_make_dataset)
        log.info("[split] Patched make_dataset (split_ratio=%s, random=True)", s)

        # Capture preprocessor when it's created by lerobot
        original_make_processors = policy_factory_mod.make_pre_post_processors

        def capturing_make_processors(*args, **kwargs):
            preprocessor, postprocessor = original_make_processors(*args, **kwargs)
            val_state._preprocessor = preprocessor
            return preprocessor, postprocessor

        self._patch(policy_factory_mod, "make_pre_post_processors", capturing_make_processors)
        self._patch(lerobot_train_mod, "make_pre_post_processors", capturing_make_processors)
        log.info("[split] Patched make_pre_post_processors to capture preprocessor")

    def apply_checkpoint_patch(self) -> None:
        """Monkey-patch lerobot save_checkpoint to:
        1. Compute and log test loss (if test split is active) at save_freq.
        2. Write anvil_config.json (with split info) into each checkpoint's pretrained_model/ directory.
        """
        import time

        import lerobot.scripts.lerobot_train as lerobot_train_mod
        import lerobot.utils.train_utils as train_utils_mod
        import torch
        from lerobot.utils.train_utils import save_checkpoint as original_save_checkpoint

        anvil_cfg_base: dict = {
            "action_type": self.config.action_type,
            "is_ee": self.config.is_ee,
            "is_ee_rel": self.config.is_ee_rel,
        }
        if self.config.task_override:
            anvil_cfg_base["task_description"] = self.config.task_override
        if self.config.note:
            anvil_cfg_base["note"] = self.config.note

        val_state = self

        def patched_save_checkpoint(checkpoint_dir, **kwargs):
            # --- Test loss (computed at save_freq) ---
            if val_state._test_dataloader is not None:
                policy = kwargs.get("policy")
                preprocessor = kwargs.get("preprocessor") or val_state._preprocessor
                step = kwargs.get("step", "?")

                if policy is not None:
                    policy.eval()
                    t0 = time.perf_counter()
                    total_loss = 0.0
                    n_batches = 0

                    # ACTPolicy in evaluation mode has no VAE, but test_loss
                    # needs to calculate the full loss. We set back to train mode
                    # to get the VAE loss if needed.
                    is_act = "ACTPolicy" in str(type(policy))
                    if is_act:
                        policy.train()

                    with torch.no_grad():
                        for batch in val_state._test_dataloader:
                            if preprocessor is not None:
                                batch = preprocessor(batch)
                            else:
                                device = next(policy.parameters()).device
                                batch = {
                                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                                    for k, v in batch.items()
                                }
                            loss, _ = policy.forward(batch)
                            total_loss += loss.item()
                            n_batches += 1

                    if is_act:
                        policy.eval()

                    test_loss = total_loss / max(n_batches, 1)
                    test_s = time.perf_counter() - t0
                    log.info("[eval] test_loss=%.6f @ step %s (%.1fs)", test_loss, step, test_s)

                    try:
                        import wandb as _wandb
                        if _wandb.run is not None:
                            _wandb.log({"eval/test_loss": test_loss}, step=int(step))
                    except Exception:
                        pass

            # --- Original save ---
            original_save_checkpoint(checkpoint_dir, **kwargs)

            # --- Save split_info.json and anvil_config.json ---
            pretrained_dir = checkpoint_dir / "pretrained_model"
            if pretrained_dir.exists():
                # 1. anvil_config.json: only non-split flags
                (pretrained_dir / "anvil_config.json").write_text(json.dumps(anvil_cfg_base, indent=2))

                # 2. split_info.json: all split metadata
                if val_state._split_info:
                    save_split_info(pretrained_dir / "split_info.json", val_state._split_info)

                log.info("[anvil_trainer] Saved configs to %s", pretrained_dir)

        # Patch both the module and the already-imported reference in lerobot_train
        self._patch(train_utils_mod, "save_checkpoint", patched_save_checkpoint)
        self._patch(lerobot_train_mod, "save_checkpoint", patched_save_checkpoint)
        log.info("[anvil_trainer] Patched save_checkpoint for test loss + anvil_config.json")

    def apply_val_loss_hook(self) -> None:
        """Monkey-patch update_policy for periodic val loss computation at val_freq intervals."""
        import time

        import lerobot.scripts.lerobot_train as lerobot_train_mod
        import torch

        original_update_policy = lerobot_train_mod.update_policy
        val_state = self
        _counter = {"n": 0}

        def patched_update_policy(
            train_metrics, policy, batch, optimizer, grad_clip_norm,
            accelerator=None, lr_scheduler=None, lock=None, rabc_weights_provider=None,
        ):
            result = original_update_policy(
                train_metrics, policy, batch, optimizer, grad_clip_norm,
                accelerator=accelerator, lr_scheduler=lr_scheduler,
                lock=lock, rabc_weights_provider=rabc_weights_provider,
            )

            _counter["n"] += 1
            val_freq = val_state._val_freq
            if not val_freq or val_freq <= 0 or val_state._val_dataloader is None:
                return result
            if _counter["n"] % val_freq != 0:
                return result

            abs_step = val_state._resume_step + _counter["n"]
            preprocessor = val_state._preprocessor

            # Unwrap accelerator-wrapped policy for eval
            if accelerator is not None:
                unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
            else:
                unwrapped = policy

            unwrapped.eval()
            t0 = time.perf_counter()
            total_loss = 0.0
            n_batches = 0

            # ACTPolicy in evaluation mode has no VAE, but val_loss
            # needs to calculate the full loss.
            is_act = "ACTPolicy" in str(type(unwrapped))
            if is_act:
                unwrapped.train()

            with torch.no_grad():
                for val_batch in val_state._val_dataloader:
                    if preprocessor is not None:
                        val_batch = preprocessor(val_batch)
                    else:
                        device = next(unwrapped.parameters()).device
                        val_batch = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in val_batch.items()
                        }
                    loss, _ = unwrapped.forward(val_batch)
                    total_loss += loss.item()
                    n_batches += 1

            if is_act:
                unwrapped.eval()

            val_loss = total_loss / max(n_batches, 1)
            val_s = time.perf_counter() - t0
            log.info("[eval] val_loss=%.6f @ step %s (%.1fs)", val_loss, abs_step, val_s)

            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.log({"eval/val_loss": val_loss}, step=abs_step)
            except Exception:
                pass

            return result

        self._patch(lerobot_train_mod, "update_policy", patched_update_policy)
        log.info("[eval] Patched update_policy for periodic val loss (val_freq will be log_freq*5)")


# =============================================================================
# Context manager
# =============================================================================


@contextlib.contextmanager
def patched_lerobot(config: TrainingConfig):
    """Install every anvil-trainer patch for the duration of the ``with`` block.

    Yields the constructed :class:`TransformRunner` so the caller can inspect
    split info, the captured preprocessor, etc.  On exit (normal or via an
    exception), every touched lerobot attribute is restored — so training
    failures no longer leave lerobot's module state permanently mutated, and
    tests run back-to-back without polluting each other.

    Example::

        with patched_lerobot(config) as runner:
            from lerobot.scripts.lerobot_train import train as lerobot_train
            lerobot_train()
    """
    runner = TransformRunner(config)
    runner.log_config()
    runner.apply_metadata_patches()
    # Note: the dataset/val_loss/checkpoint patches need lerobot imported,
    # which apply_metadata_patches typically triggers indirectly via
    # Transform.patch_metadata.  Keep the same install order as train().
    runner.apply_dataset_patches()
    runner.apply_val_loss_patch()
    runner.apply_checkpoint_patch()
    runner.apply_val_loss_hook()
    try:
        yield runner
    finally:
        runner.restore_all_patches()
