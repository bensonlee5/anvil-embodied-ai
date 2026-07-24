"""Monkey-patches applied to lerobot at training time.

``TransformRunner`` owns:
    * The active list of :class:`~anvil_trainer.transforms.Transform` instances.
    * Runtime monkey-patches on lerobot modules:
        - ``apply_dataset_patches`` — patches ``LeRobotDataset.__getitem__`` for transforms.
        - ``apply_priority_sampler_patch`` — optionally replaces uniform train sampling.
        - ``apply_val_loss_patch`` — patches ``make_dataset`` (split creation),
          captures the preprocessor from ``make_pre_post_processors``, and
          injects delta action stats into the returned ``train_dataset``.
        - ``apply_checkpoint_patch`` — patches ``save_checkpoint`` to compute
          test loss and write ``anvil_config.json`` / ``split_info.json`` next
          to each checkpoint.
        - ``apply_val_loss_hook`` — patches ``update_policy`` for periodic val
          loss computation.
        - ``apply_metadata_patches`` — runs ``Transform.patch_metadata`` hooks
          (currently used by ``ExcludeObservationTransform``).
        - ``apply_vla_jepa_input_patch`` — keeps stacked proprioception aligned
          with the current image instead of leaking the final future state.

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
import math
from pathlib import Path
from typing import Any

from anvil_shared.provenance import git_provenance
from anvil_shared.splits import compute_split_episodes, load_split_info, save_split_info

from anvil_trainer.bounded_actions import (
    BoundedActionContract,
    encode_bounded_actions,
    make_processor_steps,
    smooth_horizon,
)
from anvil_trainer.config import TrainingConfig
from anvil_trainer.priority_sampling import PriorityEpisodeAwareSampler, PriorityManifest
from anvil_trainer.rabc_audit import make_audit_verified_sample_weighter
from anvil_trainer.task_space_actions import (
    TaskSpaceActionContract,
    encode_task_space_actions,
    make_task_space_processor_steps,
)
from anvil_trainer.transforms import (
    BoundedRobustnessTransform,
    DataIntegrityError,
    DeltaActionTransform,
    ExcludeObservationTransform,
    TaskOverrideTransform,
    Transform,
)

log = logging.getLogger(__name__)

# Sentinel used to mark "patch already installed" in the originals list so we
# can keep insertion order + detect re-entrancy without wrapping in tuples.
_PATCHED_MARKER = object()


def reconcile_vla_jepa_postprocessor(policy_cfg: Any, postprocessor: Any) -> list[str]:
    """Make inherited VLA-JEPA processor steps match the effective config.

    LeRobot loads serialized processors from ``policy.path`` before applying the
    fine-tuning config. Without reconciliation, a recipe that disables gripper
    snapping can still save the base model's snapping steps. The upstream step
    classes also omit their dimension/threshold from ``get_config``, so enabled
    steps need explicit serialization metadata.

    Returns the class names of disabled steps that were removed.
    """
    from lerobot.policies.vla_jepa.processor_vla_jepa import (
        BinarizeGripperProcessorStep,
        ClipActionsProcessorStep,
        PreSnapGripperProcessorStep,
    )

    reconciled_steps = []
    removed_steps = []
    for step in postprocessor.steps:
        if isinstance(step, ClipActionsProcessorStep):
            if not policy_cfg.clip_normalized_actions:
                removed_steps.append(type(step).__name__)
                continue
        elif isinstance(step, PreSnapGripperProcessorStep):
            if not policy_cfg.pre_snap_gripper_action:
                removed_steps.append(type(step).__name__)
                continue
            step.gripper_dim = policy_cfg.gripper_dim
            step.threshold = policy_cfg.gripper_threshold
            step.get_config = lambda cfg=policy_cfg: {
                "gripper_dim": cfg.gripper_dim,
                "threshold": cfg.gripper_threshold,
            }
        elif isinstance(step, BinarizeGripperProcessorStep):
            if not policy_cfg.binarize_gripper_action:
                removed_steps.append(type(step).__name__)
                continue
            step.gripper_dim = policy_cfg.gripper_dim
            step.threshold = policy_cfg.gripper_threshold
            step.get_config = lambda cfg=policy_cfg: {
                "gripper_dim": cfg.gripper_dim,
                "threshold": cfg.gripper_threshold,
            }
        reconciled_steps.append(step)

    postprocessor.steps = reconciled_steps
    return removed_steps


def _normalize_uint8_camera_images(batch: dict[str, Any], camera_keys: tuple[str, ...]):
    """Match LeRobot's training-loop camera conversion for custom eval hooks."""
    import torch

    for camera_key in camera_keys:
        image = batch.get(camera_key)
        if isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            batch[camera_key] = image.to(dtype=torch.float32) / 255.0
    return batch


class _PerActuatorLossMeter:
    """Aggregate named per-actuator losses into scalar logging metrics."""

    def __init__(self, action_names: tuple[str, ...]):
        if not action_names or any(not isinstance(name, str) or not name for name in action_names):
            raise DataIntegrityError(
                "Per-actuator loss logging requires non-empty action feature names"
            )
        if len(set(action_names)) != len(action_names):
            raise DataIntegrityError(
                f"Per-actuator loss logging requires unique action feature names: {action_names}"
            )
        self.action_names = action_names
        self._weighted_sums = [0.0] * len(action_names)
        self._total_weight = 0.0

    def update(self, loss_dict: dict[str, Any], *, weight: int = 1) -> dict[str, Any]:
        """Consume ``loss_per_dim`` and return only logger-compatible values."""
        cleaned = dict(loss_dict)
        raw_losses = cleaned.pop("loss_per_dim", None)
        if raw_losses is None:
            return cleaned
        if not isinstance(raw_losses, (list, tuple)):
            raise DataIntegrityError(
                "loss_per_dim must be a list or tuple ordered by action_feature_names"
            )
        if len(raw_losses) != len(self.action_names):
            raise DataIntegrityError(
                "loss_per_dim/action feature length mismatch: "
                f"losses={len(raw_losses)}, actions={len(self.action_names)}"
            )
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            raise DataIntegrityError(
                f"Per-actuator loss weight must be a positive integer: {weight}"
            )

        try:
            losses = [float(value) for value in raw_losses]
        except (TypeError, ValueError) as error:
            raise DataIntegrityError(
                f"loss_per_dim contains a non-numeric value: {raw_losses}"
            ) from error
        if not all(math.isfinite(value) for value in losses):
            raise DataIntegrityError(f"loss_per_dim contains non-finite values: {losses}")
        for index, value in enumerate(losses):
            self._weighted_sums[index] += value * weight
        self._total_weight += weight
        return cleaned

    def pop_metrics(self, prefix: str) -> dict[str, float]:
        """Return scalar means keyed by exact actuator name, then reset."""
        if self._total_weight == 0:
            return {}
        metrics = {
            f"{prefix}/{name}": self._weighted_sums[index] / self._total_weight
            for index, name in enumerate(self.action_names)
        }
        self._weighted_sums = [0.0] * len(self.action_names)
        self._total_weight = 0.0
        return metrics


def _action_batch_size(batch: dict[str, Any]) -> int:
    """Return the batch size used to sample-weight holdout metrics."""
    action = batch.get("action")
    shape = getattr(action, "shape", ())
    if not shape or int(shape[0]) <= 0:
        raise DataIntegrityError("Cannot determine a positive batch size from batch['action']")
    return int(shape[0])


def _vla_jepa_current_state(state: Any):
    """Return state at observation time t in LeRobot's [B, 1, D] policy shape."""
    if state.ndim > 2:
        state = state[:, 0, :]
    return state.unsqueeze(1) if state.ndim == 2 else state


def _flatten_config_to_cli_args(data: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten config values without dropping ordered sequence overrides."""
    args: list[str] = []
    for key, value in data.items():
        if key in {"path", "type"}:
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, bool):
            value = str(value).lower()
        if isinstance(value, dict):
            args.extend(_flatten_config_to_cli_args(value, full_key))
        elif isinstance(value, list):
            args.append(f"--{full_key}={json.dumps(value, separators=(',', ':'))}")
        elif value is not None:
            args.append(f"--{full_key}={value}")
    return args


def _mapping_override_keys(cli_overrides: list[str], field_name: str) -> list[str] | None:
    """Return the exact ordered keys requested for a pretrained mapping field."""
    prefix = f"--{field_name}="
    requested: list[str] | None = None
    for override in cli_overrides:
        if not override.startswith(prefix):
            continue
        try:
            value = json.loads(override[len(prefix) :])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for pretrained --{field_name} override: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Pretrained --{field_name} override must be a JSON object")
        requested = list(value)
    return requested


def _apply_exact_pretrained_mapping_overrides(
    policy_cfg: Any,
    cli_overrides: list[str],
) -> dict[str, list[str]]:
    """Make mapping-valued pretrained overrides replace instead of merge.

    Draccus applies dictionary CLI overrides one key at a time.  For a loaded
    checkpoint this retains keys that are absent from the new recipe, which can
    silently add masked cameras to a policy.  A recipe that supplies an entire
    ``input_features`` or ``output_features`` JSON object is an exact contract:
    preserve its order and remove every inherited key not named by the recipe.
    """
    removed: dict[str, list[str]] = {}
    for field_name in ("input_features", "output_features"):
        requested = _mapping_override_keys(cli_overrides, field_name)
        if requested is None:
            continue
        effective = getattr(policy_cfg, field_name, None)
        if not isinstance(effective, dict):
            raise ValueError(
                f"Resolved policy {field_name} is not a mapping: {type(effective).__name__}"
            )
        missing = [key for key in requested if key not in effective]
        if missing:
            raise ValueError(f"Resolved policy {field_name} is missing requested keys: {missing}")
        removed[field_name] = [key for key in effective if key not in requested]
        setattr(policy_cfg, field_name, {key: effective[key] for key in requested})
    return removed


def _remap_molmoact2_processor_overrides(policy_cfg: Any, kwargs: dict[str, Any]):
    """Use MolmoAct2's registered masked-normalizer step names for fine-tuning."""
    if getattr(policy_cfg, "type", None) != "molmoact2":
        return kwargs

    remapped_kwargs = dict(kwargs)
    for kwarg_name, generic_name, molmoact2_name in (
        (
            "preprocessor_overrides",
            "normalizer_processor",
            "molmoact2_masked_normalizer",
        ),
        (
            "postprocessor_overrides",
            "unnormalizer_processor",
            "molmoact2_masked_unnormalizer",
        ),
    ):
        overrides = remapped_kwargs.get(kwarg_name)
        if not isinstance(overrides, dict) or generic_name not in overrides:
            continue
        overrides = dict(overrides)
        overrides[molmoact2_name] = overrides.pop(generic_name)
        remapped_kwargs[kwarg_name] = overrides
    return remapped_kwargs


def _make_pre_post_processors_with_compat(original_make_processors, *args, **kwargs):
    """Load processors while bridging known serialized step-name migrations."""
    policy_cfg = kwargs.get("policy_cfg", args[0] if args else None)
    effective_kwargs = _remap_molmoact2_processor_overrides(policy_cfg, kwargs)
    try:
        return original_make_processors(*args, **effective_kwargs)
    except KeyError as error:
        message = str(error)
        overrides = effective_kwargs.get("preprocessor_overrides")
        is_legacy_delta_mismatch = (
            isinstance(overrides, dict)
            and "relative_actions_processor" in overrides
            and "delta_actions_processor" not in overrides
            and "Override keys" in message
            and "Available step keys" in message
            and "relative_actions_processor" in message
            and "delta_actions_processor" in message
        )
        if not is_legacy_delta_mismatch:
            raise

        remapped_overrides = dict(overrides)
        remapped_overrides["delta_actions_processor"] = remapped_overrides.pop(
            "relative_actions_processor"
        )
        remapped_kwargs = dict(effective_kwargs)
        remapped_kwargs["preprocessor_overrides"] = remapped_overrides
        log.info(
            "[anvil_trainer] Remapped relative_actions_processor override to "
            "legacy delta_actions_processor checkpoint step"
        )
        return original_make_processors(*args, **remapped_kwargs)


class TransformRunner:
    """
    Manages and applies dataset transforms.

    Handles:
    - Registration of transforms
    - Metadata patching (before lerobot import)
    - Dataset patching (after lerobot import)
    """

    # Registry of available transforms (add new transforms here).
    # Instantiated fresh per TransformRunner so stateful transforms (e.g. DeltaActionTransform
    # which caches joint indices) do not share state across runs.
    TRANSFORMS: list[Transform] = []  # populated in __init__

    def __init__(self, config: TrainingConfig):
        self.config = config
        transforms: list[Transform] = [
            ExcludeObservationTransform(),
            TaskOverrideTransform(),
            DeltaActionTransform(),
            BoundedRobustnessTransform(config),
        ]
        self.active_transforms = [t for t in transforms if t.is_enabled(config)]
        self._val_dataloader = None  # set by apply_val_loss_patch when make_dataset is called
        self._test_dataloader = None  # set by apply_val_loss_patch when make_dataset is called
        self._split_info: dict = {}  # populated by patched_make_dataset
        self._preprocessor = None  # captured from make_pre_post_processors
        self._camera_keys: tuple[str, ...] = ()  # captured from dataset metadata
        self._action_feature_names: tuple[str, ...] = ()  # exact dataset action-vector order
        self._train_per_actuator_meter: _PerActuatorLossMeter | None = None
        self._log_freq = 0  # captured from lerobot cfg for window-averaged actuator metrics
        self._val_freq = 0  # set from cfg.log_freq * 5 inside patched_make_dataset
        self._resume_step = 0  # for absolute step tracking in wandb
        self._normalization_contract: dict[str, Any] = {}
        self._bounded_contract = (
            BoundedActionContract.load(config.bounded_action_contract)
            if config.bounded_action_contract
            else None
        )
        self._bounded_statistics: tuple[Any, Any] | None = None
        self._task_space_contract = (
            TaskSpaceActionContract.load(config.task_space_action_contract)
            if config.task_space_action_contract
            else None
        )
        self._task_space_statistics: tuple[Any, Any] | None = None
        self._priority_manifest: PriorityManifest | None = None
        if config.priority_sampling_manifest:
            if not config.dataset_root:
                raise DataIntegrityError(
                    "Priority sampling requires --dataset.root for fingerprint verification"
                )
            self._priority_manifest = PriorityManifest.load(config.priority_sampling_manifest)
            self._priority_manifest.verify_dataset(config.dataset_root)
        # List of (module, attr_name, original_value) — populated by _patch in
        # insertion order so restore_all_patches can revert in reverse.
        self._saved_originals: list[tuple[Any, str, Any]] = []
        # Registry alias names added by apply_processor_compat_aliases — unregistered on restore.
        self._registered_aliases: list[str] = []

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
        already_patched = any(m is module and n == attr_name for m, n, _ in self._saved_originals)
        if already_patched:
            log.debug(
                "[anvil_trainer] Skipping duplicate patch %s.%s",
                getattr(module, "__name__", module),
                attr_name,
            )
            return
        original = getattr(module, attr_name)
        self._saved_originals.append((module, attr_name, original))
        setattr(module, attr_name, new_value)

    def restore_all_patches(self) -> None:
        """Restore every attribute touched by :meth:`_patch` (LIFO).

        Called by :func:`patched_lerobot` on context exit.  Safe to call more
        than once — the originals list is cleared after restoration.
        Also unregisters any processor aliases registered by
        :meth:`apply_processor_compat_aliases`.
        """
        while self._saved_originals:
            module, attr_name, original = self._saved_originals.pop()
            try:
                setattr(module, attr_name, original)
            except Exception as e:  # pragma: no cover — extremely defensive
                log.warning(
                    "[anvil_trainer] Failed to restore %s.%s: %s",
                    getattr(module, "__name__", module),
                    attr_name,
                    e,
                )
        # Unregister processor registry aliases added by apply_processor_compat_aliases.
        if self._registered_aliases:
            try:
                from lerobot.processor.pipeline import ProcessorStepRegistry

                for alias in self._registered_aliases:
                    ProcessorStepRegistry.unregister(alias)
                log.debug(
                    "[anvil_trainer] Unregistered processor aliases: %s",
                    self._registered_aliases,
                )
            except Exception as e:  # pragma: no cover
                log.warning("[anvil_trainer] Failed to unregister processor aliases: %s", e)
            finally:
                self._registered_aliases.clear()

    def apply_config_sequence_patch(self) -> None:
        """Preserve ordered overrides and exact pretrained feature mappings."""
        import lerobot.configs.parser as parser
        from lerobot.configs.train import TrainPipelineConfig

        self._patch(parser, "_flatten_to_cli_args", _flatten_config_to_cli_args)
        original_resolve_pretrained = TrainPipelineConfig._resolve_pretrained_from_cli

        def patched_resolve_pretrained(cfg):
            original_resolve_pretrained(cfg)
            overrides = parser.get_yaml_overrides("policy") + (
                parser.get_cli_overrides("policy") or []
            )
            removed = _apply_exact_pretrained_mapping_overrides(cfg.policy, overrides)
            for field_name, keys in removed.items():
                if keys:
                    log.info(
                        "[anvil_trainer] Exact %s override removed inherited keys: %s",
                        field_name,
                        ", ".join(keys),
                    )

        self._patch(
            TrainPipelineConfig,
            "_resolve_pretrained_from_cli",
            patched_resolve_pretrained,
        )
        log.info(
            "[anvil_trainer] Patched pretrained config overrides to preserve ordered "
            "sequences and replace complete feature mappings"
        )

    def apply_processor_compat_aliases(self) -> None:
        """Register compatibility aliases for renamed action processor registry names.

        LeRobot releases have used both ``relative_actions_processor`` and
        ``delta_actions_processor`` for the same processor class. Keep the
        installed release's canonical name intact, but register the other name
        as an alias so older checkpoints can still deserialize.
        """
        try:
            from lerobot.processor.pipeline import ProcessorStepRegistry
            from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep
        except ImportError as e:
            log.warning(
                "[anvil_trainer] Could not import lerobot processor for compat aliases: %s", e
            )
            return

        canonical_name = getattr(
            RelativeActionsProcessorStep,
            "_registry_name",
            "relative_actions_processor",
        )
        names = ProcessorStepRegistry.list()
        compat_names = {"relative_actions_processor", "delta_actions_processor"}

        try:
            if canonical_name not in names:
                ProcessorStepRegistry.register(canonical_name)(RelativeActionsProcessorStep)
                RelativeActionsProcessorStep._registry_name = canonical_name
                names = ProcessorStepRegistry.list()

            for alias in sorted(compat_names - {canonical_name}):
                if alias in names:
                    continue
                ProcessorStepRegistry.register(alias)(RelativeActionsProcessorStep)
                # register() overwrites _registry_name on the class; restore the
                # installed release's canonical name so newly-saved checkpoints use it.
                RelativeActionsProcessorStep._registry_name = canonical_name
                self._registered_aliases.append(alias)
                names = ProcessorStepRegistry.list()
                log.info(
                    "[anvil_trainer] Registered compat alias '%s' -> %s",
                    alias,
                    canonical_name,
                )
        except Exception as e:  # pragma: no cover
            log.warning("[anvil_trainer] Failed to register processor compat alias: %s", e)

    def log_config(self) -> None:
        """Log active transforms and sampling policy."""
        if not self.active_transforms:
            log.info("[anvil_trainer] Active transforms: (none - pass-through mode)")
        else:
            for transform in self.active_transforms:
                details = self._get_transform_details(transform)
                log.info("[anvil_trainer] Active transform: %s — %s", transform.name, details)
        if self._priority_manifest is not None:
            log.info(
                "[priority-sampling] Validated %s (%s)",
                self._priority_manifest.path,
                self._priority_manifest.sha256,
            )

    def _get_transform_details(self, transform: Transform) -> str:
        """Get human-readable details for a transform."""
        if isinstance(transform, ExcludeObservationTransform):
            if self.config.exclude_observs:
                return f"excluding: {', '.join(self.config.exclude_observs)}"
            return "enabled"
        elif isinstance(transform, TaskOverrideTransform):
            return f"'{self.config.task_override}'"
        elif isinstance(transform, DeltaActionTransform):
            return "action = action - observation.state"
        elif isinstance(transform, BoundedRobustnessTransform):
            return (
                f"camera_dropout={self.config.camera_dropout_probability}, "
                f"state_noise_fraction={self.config.state_noise_std_fraction}"
            )
        return "enabled"

    def _compute_delta_action_stats(self, full_dataset: Any) -> dict | None:
        """Compute delta action stats when DeltaActionTransform is active.

        LeRobot reads ``dataset.meta.stats`` after ``make_dataset()`` returns
        and uses those stats to build the normalizer.  Because
        ``DeltaActionTransform`` runs at ``__getitem__`` time (after stats are
        read), a vanilla setup normalises delta actions with *absolute* action
        statistics — producing a large constant offset in normalised space
        (e.g. −2.66 σ for joint j4) that causes early overfitting.

        This method computes delta action stats from the HF dataset arrays
        (vectorised, fast — ~5 MB for 69K frames) and returns the dict.  It
        also patches ``full_dataset.meta.stats["action"]`` in place so that
        the early-return path (``n_train < 1`` case) inherits the correct
        stats.

        When ``config.delta_stats_n_steps > 1`` the stats are computed over
        multi-step deltas ``action[t+k] - state[t]`` for k = 0 … n_steps,
        sampled only within episode boundaries.  This makes the normalisation
        range reflect the full displacement distribution seen inside an action
        chunk, preventing loss imbalance in ACT-style chunk prediction and
        clip-sample truncation in Diffusion.

        Args:
            full_dataset: LeRobotDataset spanning all episodes.

        Returns:
            Patched action stats dict, or ``None`` when DeltaActionTransform
            is inactive or the computation fails (logged as a warning).
        """
        delta_transform = next(
            (t for t in self.active_transforms if isinstance(t, DeltaActionTransform)),
            None,
        )
        if delta_transform is None:
            return None

        import numpy as np  # numpy is not top-level; keep import local

        try:
            hf = full_dataset.hf_dataset
            actions_np = np.array(hf["action"], dtype=np.float64)
            states_np = np.array(hf["observation.state"], dtype=np.float64)
            episode_idx_np = np.array(hf["episode_index"], dtype=np.int64).ravel()
            # Use most recent obs step if states are stacked (n_obs_steps > 1)
            if states_np.ndim == 3:
                states_np = states_np[:, -1, :]
            d_action = actions_np.shape[-1]
            d_state = states_np.shape[-1]

            # _resolve_exclude_indices triggers _build_mappings which validates names
            exclude_indices = delta_transform._resolve_exclude_indices(self.config)
            action_to_state_map = delta_transform._action_to_state_map
            exclude_set = set(exclude_indices)

            n_steps = max(1, getattr(self.config, "delta_stats_n_steps", 1))

            def _compute_deltas_for_k(k: int) -> np.ndarray:
                """Return delta array for look-ahead k, respecting episode boundaries."""
                if k == 0:
                    act = actions_np
                    sta = states_np
                    mask = np.ones(len(act), dtype=bool)
                else:
                    act = actions_np[k:]
                    sta = states_np[:-k]
                    mask = episode_idx_np[k:] == episode_idx_np[:-k]

                if action_to_state_map is not None:
                    d = act.copy()
                    for a_idx, s_idx in enumerate(action_to_state_map):
                        if a_idx not in exclude_set:
                            d[:, a_idx] = act[:, a_idx] - sta[:, s_idx]
                elif d_action == d_state:
                    d = act - sta
                    # Restore excluded joints to absolute values
                    for idx in exclude_indices:
                        if idx < d_action:
                            d[:, idx] = act[:, idx]
                else:
                    raise DataIntegrityError(
                        f"[delta_stats] action has {d_action} joints but observation.state has "
                        f"{d_state} joints and no info.json is available for name-based mapping."
                    )
                return d[mask]

            all_deltas = np.concatenate([_compute_deltas_for_k(k) for k in range(n_steps)], axis=0)

            orig = full_dataset.meta.stats.get("action", {})
            orig_mean = np.array(orig.get("mean", all_deltas.mean(axis=0)))
            orig_std = np.array(orig.get("std", all_deltas.std(axis=0)))
            orig_min = np.array(orig.get("min", all_deltas.min(axis=0)))
            orig_max = np.array(orig.get("max", all_deltas.max(axis=0)))

            delta_mean = all_deltas.mean(axis=0)
            delta_std = np.where(all_deltas.std(axis=0) < 1e-6, 1e-6, all_deltas.std(axis=0))
            delta_min = all_deltas.min(axis=0)
            delta_max = all_deltas.max(axis=0)

            # Restore excluded joints to their original absolute stats
            for idx in exclude_indices:
                if idx < d_action:
                    delta_mean[idx] = orig_mean[idx]
                    delta_std[idx] = orig_std[idx]
                    delta_min[idx] = orig_min[idx]
                    delta_max[idx] = orig_max[idx]

            patched_stats = {
                "mean": delta_mean.tolist(),
                "std": delta_std.tolist(),
                "min": delta_min.tolist(),
                "max": delta_max.tolist(),
                "count": orig.get("count", len(all_deltas)),
            }
            # Patch full_dataset in place so the early-return path is covered
            full_dataset.meta.stats["action"] = patched_stats

            log.info(
                "[delta_stats] Computed delta action stats over %d samples "
                "(n_steps=%d, %d frames, %d joints, %d kept absolute: %s). "
                "j4 abs_mean=%.3f → delta_mean=%.4f, abs_std=%.3f → delta_std=%.4f",
                len(all_deltas),
                n_steps,
                len(actions_np),
                d_action,
                len(exclude_indices),
                self.config.delta_exclude_joints or [],
                orig_mean[4] if len(orig_mean) > 4 else float("nan"),
                delta_mean[4] if len(delta_mean) > 4 else float("nan"),
                orig_std[4] if len(orig_std) > 4 else float("nan"),
                delta_std[4] if len(delta_std) > 4 else float("nan"),
            )
            return patched_stats
        except DataIntegrityError:
            raise
        except Exception as e:
            log.warning(
                "[delta_stats] Failed to compute delta action stats: %s — "
                "falling back to absolute stats (training may be suboptimal)",
                e,
            )
            return None

    @staticmethod
    def _validate_pi05_dataset_contract(
        policy_cfg: Any,
        full_dataset: Any,
        task_space_contract: TaskSpaceActionContract | None = None,
    ) -> None:
        """Fail before training when Pi0.5 vector or camera contracts diverge."""
        if getattr(policy_cfg, "type", None) != "pi05":
            return

        features = full_dataset.meta.features
        action_feature = features.get("action", {})
        state_feature = features.get("observation.state", {})
        action_names = list(action_feature.get("names") or [])
        state_names = list(state_feature.get("names") or [])
        policy_names = list(getattr(policy_cfg, "action_feature_names", None) or [])

        if not action_names or action_names != state_names:
            raise DataIntegrityError(
                "[pi05_contract] Dataset action/state feature names must be present and identical"
            )
        expected_policy_names = (
            list(task_space_contract.task_action_names)
            if task_space_contract is not None
            else action_names
        )
        if policy_names != expected_policy_names:
            raise DataIntegrityError(
                "[pi05_contract] Policy action_feature_names do not match the effective "
                f"representation: policy={policy_names}, expected={expected_policy_names}"
            )
        if (
            task_space_contract is not None
            and tuple(action_names) != task_space_contract.source_action_names
        ):
            raise DataIntegrityError(
                "[pi05_contract] Dataset joint names do not match the task-space source contract"
            )

        action_shape = tuple(action_feature.get("shape") or ())
        state_shape = tuple(state_feature.get("shape") or ())
        if not action_shape or action_shape != state_shape or len(action_names) != action_shape[0]:
            raise DataIntegrityError(
                "[pi05_contract] Dataset action/state shapes and named dimensions must match: "
                f"action={action_shape}, state={state_shape}, names={len(action_names)}"
            )

        policy_state_shape = tuple(policy_cfg.input_features["observation.state"].shape)
        if policy_state_shape != state_shape:
            raise DataIntegrityError(
                "[pi05_contract] Policy/dataset state shapes differ: "
                f"policy={policy_state_shape}, dataset={state_shape}"
            )

        policy_cameras = list(policy_cfg.image_features)
        dataset_cameras = list(full_dataset.meta.camera_keys)
        if set(policy_cameras) != set(dataset_cameras):
            raise DataIntegrityError(
                "[pi05_contract] Policy/dataset camera keys differ: "
                f"policy={policy_cameras}, dataset={dataset_cameras}"
            )

        for key in dataset_cameras:
            policy_shape = tuple(policy_cfg.input_features[key].shape)
            dataset_shape = tuple(features[key]["shape"])
            if policy_shape != dataset_shape:
                raise DataIntegrityError(
                    f"[pi05_contract] Camera shape mismatch for {key}: "
                    f"policy={policy_shape}, dataset={dataset_shape}"
                )

        policy_action_shape = tuple(policy_cfg.output_features["action"].shape)
        expected_action_shape = (
            (len(task_space_contract.task_action_names),)
            if task_space_contract is not None
            else action_shape
        )
        if policy_action_shape != expected_action_shape:
            raise DataIntegrityError(
                "[pi05_contract] Policy/effective action shapes differ: "
                f"policy={policy_action_shape}, expected={expected_action_shape}"
            )
        log.info(
            "[pi05_contract] Validated %d named action/state dimensions and cameras=%s",
            action_shape[0],
            dataset_cameras,
        )

    def _compute_native_relative_action_stats(
        self,
        full_dataset: Any,
        policy_cfg: Any,
        *,
        num_workers: int,
    ) -> dict | None:
        """Compute chunk-aware stats for LeRobot's native relative processor."""
        if not getattr(policy_cfg, "use_relative_actions", False):
            return None
        if any(isinstance(t, DeltaActionTransform) for t in self.active_transforms):
            raise DataIntegrityError(
                "[relative_stats] Native relative actions and Anvil delta actions cannot both be enabled"
            )

        import numpy as np
        from lerobot.datasets.compute_stats import compute_relative_action_stats

        chunk_size = int(getattr(policy_cfg, "chunk_size", 0))
        if chunk_size < 1:
            raise DataIntegrityError(f"[relative_stats] Invalid policy chunk_size={chunk_size}")
        exclude_joints = list(getattr(policy_cfg, "relative_exclude_joints", None) or [])

        try:
            stats = compute_relative_action_stats(
                hf_dataset=full_dataset.hf_dataset,
                features=full_dataset.meta.features,
                chunk_size=chunk_size,
                exclude_joints=exclude_joints,
                num_workers=max(0, int(num_workers)),
            )
        except Exception as exc:
            raise DataIntegrityError(
                f"[relative_stats] Failed to compute native relative-action stats: {exc}"
            ) from exc

        required = {"mean", "std", "min", "max", "q01", "q99", "count"}
        missing = sorted(required - set(stats))
        if missing:
            raise DataIntegrityError(
                f"[relative_stats] Computed stats are missing required fields: {missing}"
            )
        action_shape = tuple(full_dataset.meta.features["action"]["shape"])
        for name, values in stats.items():
            array = np.asarray(values)
            if name == "count":
                if array.size != 1 or int(array.reshape(-1)[0]) <= 0:
                    raise DataIntegrityError(
                        "[relative_stats] Computed action stats have an invalid sample count"
                    )
                continue
            if array.shape != action_shape:
                raise DataIntegrityError(
                    f"[relative_stats] Computed {name} shape {array.shape} "
                    f"does not match action shape {action_shape}"
                )
            if not np.isfinite(array).all():
                raise DataIntegrityError(
                    f"[relative_stats] Computed action stats contain non-finite {name} values"
                )

        full_dataset.meta.stats["action"] = stats
        count = np.asarray(stats.get("count", 0)).reshape(-1)
        self._normalization_contract.update(
            {
                "action_space": "relative_to_observation_state",
                "chunk_size": chunk_size,
                "exclude_joints": exclude_joints,
                "stats_source": "all_valid_dataset_chunks",
                "stats_sample_count": int(count[0]) if len(count) else 0,
            }
        )
        q01 = np.asarray(stats["q01"])
        q99 = np.asarray(stats["q99"])
        log.info(
            "[relative_stats] Computed native Pi0.5 action stats over %d-step chunks "
            "(%d dimensions, excluded=%s, mean q01=%.4f, mean q99=%.4f)",
            chunk_size,
            len(q01),
            exclude_joints,
            float(q01.mean()),
            float(q99.mean()),
        )
        return stats

    def _fit_bounded_action_statistics(
        self,
        full_dataset: Any,
        policy_cfg: Any,
        train_episodes: list[int],
    ) -> tuple[Any, Any] | None:
        """Fit robust per-actuator/per-horizon statistics on train episodes only."""
        contract = self._bounded_contract
        if contract is None:
            return None
        if getattr(policy_cfg, "type", None) != "pi05":
            raise DataIntegrityError("bounded action representation currently requires Pi0.5")
        if bool(getattr(policy_cfg, "use_relative_actions", False)):
            raise DataIntegrityError(
                "bounded actions replace Pi0.5 native relative actions; set use_relative_actions=false"
            )
        if int(getattr(policy_cfg, "chunk_size", 0)) != contract.chunk_size:
            raise DataIntegrityError("bounded action contract chunk_size does not match the policy")
        policy_names = tuple(getattr(policy_cfg, "action_feature_names", None) or ())
        if policy_names != contract.action_names:
            raise DataIntegrityError(
                "bounded action contract names do not exactly match action_feature_names"
            )
        if sorted(train_episodes) != sorted(contract.training_episode_indices):
            raise DataIntegrityError(
                "resolved training episodes differ from the bounded action contract: "
                f"resolved={sorted(train_episodes)}, contract={sorted(contract.training_episode_indices)}"
            )

        import numpy as np
        import torch

        hf = full_dataset.hf_dataset
        actions = np.asarray(hf["action"], dtype=np.float64)
        states = np.asarray(hf["observation.state"], dtype=np.float64)
        if states.ndim == 3:
            states = states[:, -1, :]
        episode = np.asarray(hf["episode_index"], dtype=np.int64).reshape(-1)
        if actions.shape != states.shape or actions.shape[1] != len(contract.action_names):
            raise DataIntegrityError(
                "bounded action fitting requires matching named action/state vectors"
            )

        train_mask = np.isin(episode, np.asarray(train_episodes, dtype=np.int64))
        dimension = actions.shape[1]
        centers = np.zeros((contract.chunk_size, dimension), dtype=np.float64)
        scales = np.ones((contract.chunk_size, dimension), dtype=np.float64)
        counts: list[int] = []
        clipped = 0
        attempted = 0
        lower = contract.soft_lower
        upper = contract.soft_upper

        for horizon in range(contract.chunk_size):
            starts = np.arange(0, len(actions) - horizon, dtype=np.int64)
            valid = train_mask[starts] & (episode[starts] == episode[starts + horizon])
            starts = starts[valid]
            if len(starts) == 0:
                raise DataIntegrityError(
                    f"bounded action horizon {horizon} has no train-only samples"
                )
            target = actions[starts + horizon]
            reference = states[starts]
            endpoint_tolerance = 1.0e-6
            clipped += int(
                (
                    (target < lower - endpoint_tolerance) | (target > upper + endpoint_tolerance)
                ).sum()
            )
            attempted += int(target.size)
            base = encode_bounded_actions(
                torch.as_tensor(target[:, None, :], dtype=torch.float64),
                torch.as_tensor(reference, dtype=torch.float64),
                lower=torch.as_tensor(lower, dtype=torch.float64),
                upper=torch.as_tensor(upper, dtype=torch.float64),
                arm_indices=contract.arm_indices,
                absolute_indices=contract.absolute_indices,
                center=torch.zeros((1, dimension), dtype=torch.float64),
                scale=torch.ones((1, dimension), dtype=torch.float64),
                clip_value=1.0,
            )[:, 0].numpy()
            arm = list(contract.arm_indices)
            low = np.quantile(base[:, arm], contract.quantile_low, axis=0)
            high = np.quantile(base[:, arm], contract.quantile_high, axis=0)
            centers[horizon, arm] = 0.5 * (low + high)
            scales[horizon, arm] = np.maximum(
                0.5 * (high - low) / contract.clip_value,
                contract.minimum_scale,
            )
            counts.append(len(starts))

        arm = list(contract.arm_indices)
        centers[:, arm] = smooth_horizon(centers[:, arm], contract.smoothing_kernel)
        scales[:, arm] = np.maximum(
            smooth_horizon(scales[:, arm], contract.smoothing_kernel),
            contract.minimum_scale,
        )
        clip_fraction = clipped / max(attempted, 1)
        if clip_fraction > contract.max_training_clip_fraction:
            raise DataIntegrityError(
                "bounded action training targets exceed soft limits too often: "
                f"{clip_fraction:.6f} > {contract.max_training_clip_fraction:.6f}"
            )
        if not np.isfinite(centers).all() or not np.isfinite(scales).all():
            raise DataIntegrityError("bounded action statistics contain non-finite values")

        self._bounded_statistics = (centers, scales)
        self._normalization_contract.update(
            {
                "action_space": "state_relative_soft_limit_fraction",
                "representation_id": contract.representation_id,
                "contract_sha256": contract.sha256,
                "chunk_size": contract.chunk_size,
                "stats_source": "frozen_training_episodes_only",
                "fit_episode_indices": list(contract.training_episode_indices),
                "split_sha256": contract.split_sha256,
                "horizon_sample_counts": counts,
                "quantiles": [contract.quantile_low, contract.quantile_high],
                "minimum_scale": contract.minimum_scale,
                "clip_value": contract.clip_value,
                "training_target_clip_fraction": clip_fraction,
                "soft_lower": lower.tolist(),
                "soft_upper": upper.tolist(),
                "inference_smoothing": {
                    "method": "uniform_cubic_bspline",
                    "kernel": list(contract.inference_smoothing_kernel),
                    "passes": contract.inference_smoothing_passes,
                    "gripper_mode": "absolute_passthrough",
                    "gripper_event_threshold": contract.gripper_event_threshold,
                },
            }
        )
        log.info(
            "[bounded_actions] Fit %dx%d train-only horizon statistics "
            "(episodes=%d, target_clip_fraction=%.6f)",
            contract.chunk_size,
            dimension,
            len(train_episodes),
            clip_fraction,
        )
        return centers, scales

    def _install_bounded_action_processors(
        self,
        policy_cfg: Any,
        preprocessor: Any,
        postprocessor: Any,
    ) -> None:
        """Replace the inherited relative codec and disable generic action normalization."""
        if self._bounded_contract is None:
            return
        if self._bounded_statistics is None:
            raise DataIntegrityError("bounded action statistics were not fit before processors")

        from lerobot.configs.types import FeatureType, NormalizationMode
        from lerobot.processor.normalize_processor import (
            NormalizerProcessorStep,
            UnnormalizerProcessorStep,
        )
        from lerobot.processor.relative_action_processor import (
            AbsoluteActionsProcessorStep,
            RelativeActionsProcessorStep,
        )

        if policy_cfg.normalization_mapping.get(FeatureType.ACTION) != NormalizationMode.IDENTITY:
            raise DataIntegrityError(
                "bounded action processor owns action normalization; set ACTION normalization to IDENTITY"
            )
        center, scale = self._bounded_statistics
        bounded_relative, bounded_absolute = make_processor_steps(
            self._bounded_contract,
            center=center,
            scale=scale,
        )

        relative_indices = [
            index
            for index, step in enumerate(preprocessor.steps)
            if isinstance(step, RelativeActionsProcessorStep)
        ]
        absolute_indices = [
            index
            for index, step in enumerate(postprocessor.steps)
            if isinstance(step, AbsoluteActionsProcessorStep)
        ]
        if len(relative_indices) != 1 or len(absolute_indices) != 1:
            raise DataIntegrityError(
                "bounded actions require exactly one inherited relative/absolute processor pair"
            )
        preprocessor.steps[relative_indices[0]] = bounded_relative
        postprocessor.steps[absolute_indices[0]] = bounded_absolute
        for pipeline in (preprocessor, postprocessor):
            for step in pipeline.steps:
                if isinstance(step, (NormalizerProcessorStep, UnnormalizerProcessorStep)):
                    step.norm_map[FeatureType.ACTION] = NormalizationMode.IDENTITY
        log.info(
            "[bounded_actions] Installed %s (%s)",
            self._bounded_contract.representation_id,
            self._bounded_contract.sha256,
        )

    def _fit_task_space_action_statistics(
        self,
        full_dataset: Any,
        policy_cfg: Any,
        train_episodes: list[int],
    ) -> tuple[Any, Any] | None:
        """Fit robust task-space statistics from the frozen train episodes only."""
        contract = self._task_space_contract
        if contract is None:
            return None
        if getattr(policy_cfg, "type", None) != "pi05":
            raise DataIntegrityError("task-space actions currently require Pi0.5")
        if bool(getattr(policy_cfg, "use_relative_actions", False)):
            raise DataIntegrityError(
                "task-space actions replace Pi0.5 native relative actions; "
                "set use_relative_actions=false"
            )
        if int(getattr(policy_cfg, "chunk_size", 0)) != contract.chunk_size:
            raise DataIntegrityError("task-space contract chunk_size does not match the policy")
        policy_names = tuple(getattr(policy_cfg, "action_feature_names", None) or ())
        if policy_names != contract.task_action_names:
            raise DataIntegrityError(
                "task-space action names do not exactly match action_feature_names"
            )
        if sorted(train_episodes) != sorted(contract.training_episode_indices):
            raise DataIntegrityError(
                "resolved training episodes differ from the task-space action contract: "
                f"resolved={sorted(train_episodes)}, "
                f"contract={sorted(contract.training_episode_indices)}"
            )

        import numpy as np
        import torch

        hf = full_dataset.hf_dataset
        actions = np.asarray(hf["action"], dtype=np.float64)
        states = np.asarray(hf["observation.state"], dtype=np.float64)
        if states.ndim == 3:
            states = states[:, 0, :]
        episodes = np.asarray(hf["episode_index"], dtype=np.int64).reshape(-1)
        if (
            actions.shape != states.shape
            or actions.shape[1] != len(contract.source_action_names)
        ):
            raise DataIntegrityError(
                "task-space fitting requires matching 16-D named joint action/state vectors"
            )

        train_mask = np.isin(episodes, np.asarray(train_episodes, dtype=np.int64))
        task_dimension = len(contract.task_action_names)
        centers = np.zeros((contract.chunk_size, task_dimension), dtype=np.float64)
        scales = np.ones((contract.chunk_size, task_dimension), dtype=np.float64)
        counts: list[int] = []
        raw_horizons: list[np.ndarray] = []
        identity_center = torch.zeros((1, task_dimension), dtype=torch.float64)
        identity_scale = torch.ones_like(identity_center)

        for horizon in range(contract.chunk_size):
            starts = np.arange(0, len(actions) - horizon, dtype=np.int64)
            valid = train_mask[starts] & (episodes[starts] == episodes[starts + horizon])
            starts = starts[valid]
            if len(starts) == 0:
                raise DataIntegrityError(
                    f"task-space action horizon {horizon} has no train-only samples"
                )
            encoded = encode_task_space_actions(
                torch.as_tensor(actions[starts + horizon]).unsqueeze(1),
                torch.as_tensor(states[starts]),
                contract=contract,
                center=identity_center,
                scale=identity_scale,
            )[:, 0].numpy()
            raw_horizons.append(encoded)
            counts.append(len(encoded))
            low = np.quantile(encoded, contract.quantile_low, axis=0)
            high = np.quantile(encoded, contract.quantile_high, axis=0)
            centers[horizon] = 0.5 * (low + high)
            scales[horizon] = np.maximum(
                0.5 * (high - low),
                contract.minimum_scale,
            )
            # Absolute grippers already have a complete physical [-1,1] map.
            centers[horizon, [6, 13]] = 0.0
            scales[horizon, [6, 13]] = 1.0

        clipped = 0
        attempted = 0
        for horizon, raw in enumerate(raw_horizons):
            normalized = (raw - centers[horizon]) / scales[horizon]
            clipped += int(np.count_nonzero(np.abs(normalized) > contract.clip_value))
            attempted += normalized.size
        clip_fraction = clipped / max(attempted, 1)
        if not np.isfinite(centers).all() or not np.isfinite(scales).all():
            raise DataIntegrityError("task-space fitted statistics contain non-finite values")
        self._task_space_statistics = (centers, scales)
        self._normalization_contract.update(
            {
                "action_space": "bimanual_tcp_delta_with_absolute_grippers",
                "representation_id": contract.representation_id,
                "contract_sha256": contract.sha256,
                "chunk_size": contract.chunk_size,
                "stats_source": "frozen_training_episodes_only",
                "fit_episode_indices": list(contract.training_episode_indices),
                "split_sha256": contract.split_sha256,
                "horizon_sample_counts": counts,
                "quantiles": [contract.quantile_low, contract.quantile_high],
                "minimum_scale": contract.minimum_scale,
                "clip_value": contract.clip_value,
                "training_target_clip_fraction": clip_fraction,
                "solver_model_id": contract.model_id,
                "solver_model_sha256": contract.model_sha256,
                "solver_failure_mode": "fail_closed",
            }
        )
        log.info(
            "[task_space_actions] Fit %dx%d train-only horizon statistics "
            "(episodes=%d, target_clip_fraction=%.6f)",
            contract.chunk_size,
            task_dimension,
            len(train_episodes),
            clip_fraction,
        )
        return centers, scales

    def _install_task_space_action_processors(
        self,
        policy_cfg: Any,
        preprocessor: Any,
        postprocessor: Any,
    ) -> None:
        """Replace the inherited joint codec with the task-space/solver pair."""
        if self._task_space_contract is None:
            return
        if self._task_space_statistics is None:
            raise DataIntegrityError("task-space statistics were not fit before processors")

        from lerobot.configs.types import FeatureType, NormalizationMode
        from lerobot.processor.normalize_processor import (
            NormalizerProcessorStep,
            UnnormalizerProcessorStep,
        )
        from lerobot.processor.relative_action_processor import (
            AbsoluteActionsProcessorStep,
            RelativeActionsProcessorStep,
        )

        if policy_cfg.normalization_mapping.get(FeatureType.ACTION) != NormalizationMode.IDENTITY:
            raise DataIntegrityError(
                "task-space processor owns action normalization; "
                "set ACTION normalization to IDENTITY"
            )
        center, scale = self._task_space_statistics
        task_relative, task_absolute = make_task_space_processor_steps(
            self._task_space_contract,
            center=center,
            scale=scale,
        )
        relative_indices = [
            index
            for index, step in enumerate(preprocessor.steps)
            if isinstance(step, RelativeActionsProcessorStep)
        ]
        absolute_indices = [
            index
            for index, step in enumerate(postprocessor.steps)
            if isinstance(step, AbsoluteActionsProcessorStep)
        ]
        if len(relative_indices) != 1 or len(absolute_indices) != 1:
            raise DataIntegrityError(
                "task-space actions require exactly one inherited relative/absolute "
                "processor pair"
            )
        preprocessor.steps[relative_indices[0]] = task_relative
        postprocessor.steps[absolute_indices[0]] = task_absolute
        for pipeline in (preprocessor, postprocessor):
            for step in pipeline.steps:
                if isinstance(step, (NormalizerProcessorStep, UnnormalizerProcessorStep)):
                    step.norm_map[FeatureType.ACTION] = NormalizationMode.IDENTITY
        log.info(
            "[task_space_actions] Installed %s (%s)",
            self._task_space_contract.representation_id,
            self._task_space_contract.sha256,
        )

    def apply_metadata_patches(self) -> None:
        """Apply metadata patches before importing lerobot training."""
        for transform in self.active_transforms:
            transform.patch_metadata(self.config, runner=self)

    def apply_vla_jepa_input_patch(self) -> None:
        """Align VLA-JEPA proprioception with its current-frame visual input.

        World-model training requests observations ``[t, ..., t+7]``. Upstream
        LeRobot 0.6 correctly uses image ``t`` for action conditioning but takes
        state ``t+7`` from the same stacked batch, leaking future information and
        creating a train/inference mismatch. Preserve the future video stack for
        JEPA loss while replacing only the action model's state input with state
        ``t``.
        """
        from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

        original_prepare = VLAJEPAPolicy._prepare_model_inputs

        def patched_prepare(policy, batch, training=True):
            inputs = original_prepare(policy, batch, training=training)
            state = batch.get("observation.state")
            if state is not None:
                inputs["state"] = _vla_jepa_current_state(state).float()
            return inputs

        self._patch(VLAJEPAPolicy, "_prepare_model_inputs", patched_prepare)
        log.info("[vla_jepa] Patched stacked state selection to use observation time t")

    def apply_priority_sampler_patch(self) -> None:
        """Replace LeRobot's uniform train sampler when a manifest is configured."""
        if self._priority_manifest is None:
            return

        import lerobot.datasets.sampler as sampler_mod
        import lerobot.scripts.lerobot_train as lerobot_train_mod

        manifest = self._priority_manifest

        def configured_priority_sampler(*args, **kwargs):
            return PriorityEpisodeAwareSampler(*args, manifest=manifest, **kwargs)

        self._patch(sampler_mod, "EpisodeAwareSampler", configured_priority_sampler)
        self._patch(lerobot_train_mod, "EpisodeAwareSampler", configured_priority_sampler)
        log.info(
            "[priority-sampling] Enabled manifest %s (%s)",
            manifest.path,
            manifest.sha256,
        )

    def apply_rabc_audit_patch(self) -> None:
        """Require an exact progress audit when a recipe opts into provenance locking."""
        import lerobot.utils.sample_weighting as sample_weighting_mod

        original_factory = sample_weighting_mod.make_sample_weighter

        def verified_factory(
            config,
            policy,
            device,
            dataset_root=None,
            dataset_repo_id=None,
        ):
            return make_audit_verified_sample_weighter(
                config,
                policy,
                device,
                dataset_root,
                dataset_repo_id,
                original_factory=original_factory,
            )

        self._patch(sample_weighting_mod, "make_sample_weighter", verified_factory)

    def apply_dataset_patches(self) -> None:
        """Patch LeRobotDataset.__getitem__ to apply Anvil transforms.

        LeRobot 0.6's EpisodeAwareSampler already applies the dataset's
        absolute-to-relative mapping. A second mapping here corrupts split
        datasets whenever a relative index is also an absolute-map key.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        original_getitem = LeRobotDataset.__getitem__
        transforms = self.active_transforms
        config = self.config
        runner = self

        def patched_getitem(dataset, idx):
            # Samplers and ordinary DataLoaders both supply relative indices.
            item = original_getitem(dataset, idx)

            selected = getattr(dataset, "episodes", None)
            train_episodes = set(runner._split_info.get("train_episodes", []))
            if selected is None:
                is_training_dataset = not train_episodes or train_episodes == set(
                    range(getattr(dataset, "num_episodes", 0))
                )
            else:
                is_training_dataset = {int(value) for value in selected} == train_episodes

            # Apply transforms (no-op when transforms list is empty).
            for transform in transforms:
                if transform.training_only and not is_training_dataset:
                    continue
                item = transform.apply(item, config)
            return item

        self._patch(LeRobotDataset, "__getitem__", patched_getitem)
        log.info(
            "[anvil_trainer] Patched LeRobotDataset.__getitem__ (%d transform(s))", len(transforms)
        )

    def apply_val_loss_patch(self) -> None:
        """Patch dataset construction for contracts, stats, splits, and eval."""
        s = self.config.split_ratio
        total_r = sum(s)
        has_holdout = total_r > 0 and (s[1] > 0 or (len(s) >= 3 and s[2] > 0))

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

            # Capture logging frequencies and resume_step from lerobot cfg
            val_state._log_freq = cfg.log_freq
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
            val_state._camera_keys = tuple(full_dataset.meta.camera_keys)
            policy_cfg = cfg.trainable_config
            val_state._validate_pi05_dataset_contract(
                policy_cfg,
                full_dataset,
                val_state._task_space_contract,
            )
            action_feature = full_dataset.meta.features.get("action", {})
            val_state._action_feature_names = (
                val_state._task_space_contract.task_action_names
                if val_state._task_space_contract is not None
                else tuple(action_feature.get("names") or ())
            )

            # Build action stats in the same representation consumed by the
            # processor. Native Pi0.5 relative actions use every valid future
            # action in the configured chunk; Anvil's legacy transform retains
            # its existing delta-stat path.
            _patched_action_stats = None
            if (
                val_state._bounded_contract is None
                and val_state._task_space_contract is None
            ):
                _patched_action_stats = val_state._compute_native_relative_action_stats(
                    full_dataset,
                    policy_cfg,
                    num_workers=cfg.num_workers,
                )
                if _patched_action_stats is None:
                    _patched_action_stats = val_state._compute_delta_action_stats(full_dataset)

            if not has_holdout:
                val_state._split_info = {
                    "split_ratio": list(s),
                    "total_episodes": total_ep,
                    "train_episodes": list(range(total_ep)),
                    "val_episodes": [],
                    "test_episodes": [],
                }
                val_state._fit_bounded_action_statistics(
                    full_dataset,
                    policy_cfg,
                    list(range(total_ep)),
                )
                val_state._fit_task_space_action_statistics(
                    full_dataset,
                    policy_cfg,
                    list(range(total_ep)),
                )
                log.info("[split] Holdout evaluation disabled; using all %d episodes", total_ep)
                return full_dataset

            # Check if split_info.json already exists in last checkpoint (for resume)
            split_info_path = (
                Path(cfg.output_dir)
                / "checkpoints"
                / "last"
                / "pretrained_model"
                / "split_info.json"
            )
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
                    log.info(
                        "[split] Subsampled %d / %d episodes (--max-episodes)", len(pool), total_ep
                    )

                # Random three-way split via shared helper (seeded for reproducibility)
                splits = compute_split_episodes(len(pool), s, seed=cfg.seed)
                train_ep = [pool[i] for i in splits["train"]]
                val_ep = [pool[i] for i in splits["val"]]
                test_ep = [pool[i] for i in splits["test"]]

                if len(train_ep) < 1:
                    log.warning(
                        "[split] Not enough episodes (%d) for split %s, using all for training",
                        total_ep,
                        s,
                    )
                    # full_dataset already has patched action stats if _patched_action_stats is set
                    return full_dataset
                log.info("[split] Generated random splits")

            # Store split info for anvil_config.json (as full lists now)
            val_state._split_info = {
                "split_ratio": list(s),
                "total_episodes": total_ep,
                "train_episodes": train_ep,
                "val_episodes": val_ep,
                "test_episodes": test_ep,
                **(
                    {"max_episodes": val_state.config.max_episodes}
                    if val_state.config.max_episodes is not None
                    else {}
                ),
            }
            val_state._fit_bounded_action_statistics(full_dataset, policy_cfg, train_ep)
            val_state._fit_task_space_action_statistics(full_dataset, policy_cfg, train_ep)

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
                val_dataset.clear_image_transforms()
                val_state._val_dataloader = _make_dataloader(val_dataset)
                log.info(
                    "[split] val=%d ep (randomly selected, %d frames)",
                    len(val_ep),
                    val_dataset.num_frames,
                )

            # Test dataloader
            if test_ep:
                cfg.dataset.episodes = test_ep
                test_dataset = original_make_dataset(cfg)
                test_dataset.clear_image_transforms()
                val_state._test_dataloader = _make_dataloader(test_dataset)
                log.info(
                    "[split] test=%d ep (randomly selected, %d frames)",
                    len(test_ep),
                    test_dataset.num_frames,
                )

            # Train dataset
            cfg.dataset.episodes = train_ep
            train_dataset = original_make_dataset(cfg)
            # The train dataset is a fresh instance that re-reads stats.json,
            # so re-apply the in-memory transformed-action statistics.
            if _patched_action_stats is not None:
                train_dataset.meta.stats["action"] = _patched_action_stats
                log.info(
                    "[anvil_trainer] Patched train_dataset.meta.stats['action'] "
                    "with transformed-action stats"
                )
            log.info("[split] train=%d ep (randomly selected)", len(train_ep))
            return train_dataset

        self._patch(factory_mod, "make_dataset", patched_make_dataset)
        if hasattr(lerobot_train_mod, "make_dataset"):
            self._patch(lerobot_train_mod, "make_dataset", patched_make_dataset)
        if has_holdout and hasattr(lerobot_train_mod, "make_train_eval_datasets"):

            def patched_make_train_eval_datasets(cfg):
                return patched_make_dataset(cfg), None

            self._patch(
                lerobot_train_mod,
                "make_train_eval_datasets",
                patched_make_train_eval_datasets,
            )
            log.info(
                "[split] LeRobot's native dataset eval is disabled while --split-ratio "
                "is active (eval_steps/max_eval_samples are no-ops); anvil-trainer's "
                "val/test loss hooks are used instead"
            )
        log.info(
            "[anvil_trainer] Patched make_dataset (split_ratio=%s, holdout=%s)",
            s,
            has_holdout,
        )

        # Capture preprocessor when it's created by lerobot
        original_make_processors = policy_factory_mod.make_pre_post_processors

        def capturing_make_processors(*args, **kwargs):
            policy_cfg = kwargs.get("policy_cfg", args[0] if args else None)
            preprocessor, postprocessor = _make_pre_post_processors_with_compat(
                original_make_processors, *args, **kwargs
            )
            val_state._install_bounded_action_processors(
                policy_cfg,
                preprocessor,
                postprocessor,
            )
            val_state._install_task_space_action_processors(
                policy_cfg,
                preprocessor,
                postprocessor,
            )
            if getattr(policy_cfg, "type", None) == "vla_jepa":
                removed_steps = reconcile_vla_jepa_postprocessor(policy_cfg, postprocessor)
                if removed_steps:
                    log.info(
                        "[vla_jepa] Removed disabled pretrained postprocessor steps: %s",
                        ", ".join(removed_steps),
                    )
                log.info(
                    "[vla_jepa] Reconciled postprocessor with effective config "
                    "(gripper_dim=%d, pre_snap=%s, binarize=%s)",
                    policy_cfg.gripper_dim,
                    policy_cfg.pre_snap_gripper_action,
                    policy_cfg.binarize_gripper_action,
                )
            val_state._preprocessor = preprocessor
            return preprocessor, postprocessor

        self._patch(policy_factory_mod, "make_pre_post_processors", capturing_make_processors)
        if hasattr(lerobot_train_mod, "make_pre_post_processors"):
            self._patch(lerobot_train_mod, "make_pre_post_processors", capturing_make_processors)
        log.info("[split] Patched make_pre_post_processors to capture preprocessor")

    def apply_checkpoint_patch(self) -> None:
        """Monkey-patch lerobot save_checkpoint to:
        1. Compute and log test loss (if test split is active) at save_freq.
        2. Write anvil_config.json (with split info) into each checkpoint's pretrained_model/ directory.
        """
        import importlib
        import time

        import lerobot.scripts.lerobot_train as lerobot_train_mod
        import torch

        train_utils_mod = None
        with contextlib.suppress(ModuleNotFoundError):
            train_utils_mod = importlib.import_module("lerobot.utils.train_utils")

        original_save_checkpoint = getattr(
            train_utils_mod,
            "save_checkpoint",
            lerobot_train_mod.save_checkpoint,
        )

        anvil_cfg_base: dict = {
            "action_type": self.config.action_type,
            # Backward compat: old inference nodes read use_delta_actions
            "use_delta_actions": self.config.use_delta_actions,
            "delta_sequential": self.config.delta_sequential,
            # This dict is populated after the dataset is loaded.  Keep the
            # shared reference so checkpoint writes capture the resolved
            # normalization representation and statistics source.
            "normalization_contract": self._normalization_contract,
            **git_provenance(),
        }
        if self.config.delta_exclude_joints:
            anvil_cfg_base["delta_exclude_joints"] = self.config.delta_exclude_joints
        if self.config.task_override:
            anvil_cfg_base["task_description"] = self.config.task_override
        if self.config.note:
            anvil_cfg_base["note"] = self.config.note
        if self._priority_manifest is not None:
            anvil_cfg_base["priority_sampling"] = self._priority_manifest.provenance()
        if self._bounded_contract is not None:
            anvil_cfg_base["bounded_action_representation"] = {
                "representation_id": self._bounded_contract.representation_id,
                "contract_sha256": self._bounded_contract.sha256,
                "camera_dropout_probability": self.config.camera_dropout_probability,
                "state_noise_std_fraction": self.config.state_noise_std_fraction,
            }
        if self._task_space_contract is not None:
            anvil_cfg_base["task_space_action_representation"] = {
                "representation_id": self._task_space_contract.representation_id,
                "contract_sha256": self._task_space_contract.sha256,
                "deployment_status": self._task_space_contract.deployment_status,
                "kinematic_model_id": self._task_space_contract.model_id,
                "kinematic_model_sha256": self._task_space_contract.model_sha256,
                "source_action_feature_names": list(
                    self._task_space_contract.source_action_names
                ),
                "task_action_feature_names": list(
                    self._task_space_contract.task_action_names
                ),
                "camera_dropout_probability": self.config.camera_dropout_probability,
                "state_noise_std_fraction": self.config.state_noise_std_fraction,
            }

        val_state = self

        def patched_save_checkpoint(checkpoint_dir, *args, **kwargs):
            # --- Test loss (computed at save_freq) ---
            if val_state._test_dataloader is not None:
                policy = kwargs.get("policy")
                if policy is None and len(args) >= 3:
                    policy = args[2]
                preprocessor = kwargs.get("preprocessor")
                if preprocessor is None and len(args) >= 6:
                    preprocessor = args[5]
                preprocessor = preprocessor or val_state._preprocessor
                step = kwargs.get("step", args[0] if args else "?")

                if policy is not None:
                    policy.eval()
                    t0 = time.perf_counter()
                    total_loss = 0.0
                    total_samples = 0
                    per_actuator_meter = None

                    # ACTPolicy in evaluation mode has no VAE, but test_loss
                    # needs to calculate the full loss. We set back to train mode
                    # to get the VAE loss if needed.
                    is_act = "ACTPolicy" in str(type(policy))
                    if is_act:
                        policy.train()

                    with torch.no_grad():
                        for batch in val_state._test_dataloader:
                            batch = _normalize_uint8_camera_images(batch, val_state._camera_keys)
                            if preprocessor is not None:
                                batch = preprocessor(batch)
                            else:
                                device = next(policy.parameters()).device
                                batch = {
                                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                                    for k, v in batch.items()
                                }
                            loss, loss_dict = policy.forward(batch)
                            batch_size = _action_batch_size(batch)
                            total_loss += loss.item() * batch_size
                            total_samples += batch_size
                            if isinstance(loss_dict, dict) and "loss_per_dim" in loss_dict:
                                if per_actuator_meter is None:
                                    per_actuator_meter = _PerActuatorLossMeter(
                                        val_state._action_feature_names
                                    )
                                per_actuator_meter.update(loss_dict, weight=batch_size)

                    if is_act:
                        policy.eval()

                    test_loss = total_loss / max(total_samples, 1)
                    test_s = time.perf_counter() - t0
                    log.info("[eval] test_loss=%.6f @ step %s (%.1fs)", test_loss, step, test_s)

                    try:
                        import wandb as _wandb

                        if _wandb.run is not None:
                            metrics = {"eval/test_loss": test_loss}
                            if per_actuator_meter is not None:
                                metrics.update(
                                    per_actuator_meter.pop_metrics("eval/test_loss_per_actuator")
                                )
                            _wandb.log(metrics, step=int(step))
                    except Exception:
                        log.exception("[eval] Failed to log test metrics to W&B")

            # --- Original save ---
            original_save_checkpoint(checkpoint_dir, *args, **kwargs)

            # --- Save split_info.json and anvil_config.json ---
            pretrained_dir = checkpoint_dir / "pretrained_model"
            if pretrained_dir.exists():
                # 1. anvil_config.json: only non-split flags
                (pretrained_dir / "anvil_config.json").write_text(
                    json.dumps(anvil_cfg_base, indent=2)
                )

                # 2. split_info.json: all split metadata
                if val_state._split_info:
                    save_split_info(pretrained_dir / "split_info.json", val_state._split_info)

                if val_state._priority_manifest is not None:
                    (pretrained_dir / "priority_sampling_manifest.json").write_bytes(
                        val_state._priority_manifest.path.read_bytes()
                    )
                if val_state._bounded_contract is not None:
                    (pretrained_dir / "bounded_action_contract.json").write_bytes(
                        val_state._bounded_contract.path.read_bytes()
                    )
                if val_state._task_space_contract is not None:
                    (pretrained_dir / "task_space_action_contract.json").write_bytes(
                        val_state._task_space_contract.path.read_bytes()
                    )

                log.info("[anvil_trainer] Saved configs to %s", pretrained_dir)

        # Patch both the source module (when present) and the reference used by lerobot_train.
        if train_utils_mod is not None and hasattr(train_utils_mod, "save_checkpoint"):
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

        def patched_update_policy(*args, **kwargs):
            train_tracker, output_dict = original_update_policy(*args, **kwargs)

            policy = args[1] if len(args) > 1 else kwargs.get("policy")
            accelerator = kwargs.get("accelerator")
            if accelerator is None and len(args) > 5:
                accelerator = args[5]

            _counter["n"] += 1
            abs_step = val_state._resume_step + _counter["n"]
            if isinstance(output_dict, dict) and "loss_per_dim" in output_dict:
                batch = args[2] if len(args) > 2 else kwargs.get("batch")
                if val_state._train_per_actuator_meter is None:
                    val_state._train_per_actuator_meter = _PerActuatorLossMeter(
                        val_state._action_feature_names
                    )
                output_dict = val_state._train_per_actuator_meter.update(
                    output_dict,
                    weight=_action_batch_size(batch),
                )
            if (
                val_state._train_per_actuator_meter is not None
                and val_state._log_freq > 0
                and abs_step % val_state._log_freq == 0
            ):
                output_dict = dict(output_dict or {})
                output_dict.update(
                    val_state._train_per_actuator_meter.pop_metrics("loss_per_actuator")
                )
            result = train_tracker, output_dict

            val_freq = val_state._val_freq
            if not val_freq or val_freq <= 0 or val_state._val_dataloader is None:
                return result
            if _counter["n"] % val_freq != 0:
                return result

            preprocessor = val_state._preprocessor

            # Unwrap accelerator-wrapped policy for eval
            if accelerator is not None:
                unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
            else:
                unwrapped = policy

            unwrapped.eval()
            t0 = time.perf_counter()
            total_loss = 0.0
            total_samples = 0
            per_actuator_meter = None

            # ACTPolicy in evaluation mode has no VAE, but val_loss
            # needs to calculate the full loss.
            is_act = "ACTPolicy" in str(type(unwrapped))
            if is_act:
                unwrapped.train()

            with torch.no_grad():
                for val_batch in val_state._val_dataloader:
                    val_batch = _normalize_uint8_camera_images(val_batch, val_state._camera_keys)
                    if preprocessor is not None:
                        val_batch = preprocessor(val_batch)
                    else:
                        device = next(unwrapped.parameters()).device
                        val_batch = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in val_batch.items()
                        }
                    loss, loss_dict = unwrapped.forward(val_batch)
                    batch_size = _action_batch_size(val_batch)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    if isinstance(loss_dict, dict) and "loss_per_dim" in loss_dict:
                        if per_actuator_meter is None:
                            per_actuator_meter = _PerActuatorLossMeter(
                                val_state._action_feature_names
                            )
                        per_actuator_meter.update(loss_dict, weight=batch_size)

            if is_act:
                unwrapped.eval()

            val_loss = total_loss / max(total_samples, 1)
            val_s = time.perf_counter() - t0
            log.info("[eval] val_loss=%.6f @ step %s (%.1fs)", val_loss, abs_step, val_s)

            try:
                import wandb as _wandb

                if _wandb.run is not None:
                    metrics = {"eval/val_loss": val_loss}
                    if per_actuator_meter is not None:
                        metrics.update(per_actuator_meter.pop_metrics("eval/val_loss_per_actuator"))
                    _wandb.log(metrics, step=abs_step)
            except Exception:
                log.exception("[eval] Failed to log validation metrics to W&B")

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
    runner.apply_config_sequence_patch()
    runner.apply_metadata_patches()
    runner.apply_vla_jepa_input_patch()
    runner.apply_processor_compat_aliases()
    # Note: the dataset/val_loss/checkpoint patches need lerobot imported,
    # which apply_metadata_patches typically triggers indirectly via
    # Transform.patch_metadata.  Keep the same install order as train().
    runner.apply_dataset_patches()
    runner.apply_priority_sampler_patch()
    runner.apply_rabc_audit_patch()
    runner.apply_val_loss_patch()
    runner.apply_checkpoint_patch()
    runner.apply_val_loss_hook()
    try:
        yield runner
    finally:
        runner.restore_all_patches()
