"""Core evaluation logic — replay dataset episodes through a trained policy."""

from __future__ import annotations

import logging
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .substrate import AnchorChunk, EpisodeSubstrate

log = logging.getLogger(__name__)


def _ensure_model_loader_importable() -> None:
    """Add lerobot_control to sys.path for ModelLoader import (zero ROS2 deps)."""
    env_path = os.environ.get("LEROBOT_CONTROL_PATH")
    if env_path:
        target = str(Path(env_path))
    else:
        # Repo-relative: packages/anvil_eval/src/anvil_eval/evaluator.py -> repo root
        repo_root = Path(__file__).resolve().parents[4]
        target = str(repo_root / "ros2" / "src" / "lerobot_control")

    if target not in sys.path:
        sys.path.insert(0, target)


@dataclass
class EpisodeResult:
    """Raw results from evaluating a single episode."""

    episode_idx: int
    split_label: str
    predicted: np.ndarray     # (T, D) absolute actions (after delta restore)
    ground_truth: np.ndarray  # (T, D) absolute ground-truth actions
    joint_names: list[str]
    raw_output: np.ndarray | None = None         # (T, D) model raw output before delta restore
    obs_states: np.ndarray | None = None         # (T, D) observation state at each frame
    raw_ground_truth: np.ndarray | None = None   # (T, D) GT in same space as raw_output
    substrate: EpisodeSubstrate | None = None    # full-chunk capture for horizon analysis


class EpisodeEvaluator:
    """Evaluate a trained policy by replaying dataset episodes."""

    def __init__(
        self,
        model,
        preprocessor,
        postprocessor,
        model_type: str,
        device: str,
        anvil_cfg: dict,
        task_description: str | None,
        joint_names: list[str],
        capture_horizon: bool = True,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.model_type = model_type
        self.device = device
        self.action_type: str = anvil_cfg.get("action_type", "absolute")
        if self.action_type == "absolute" and anvil_cfg.get("use_delta_actions", False):
            self.action_type = "delta_obs_t"
        self.use_delta_actions: bool = self.action_type in ("delta_obs_t", "delta_sequential")
        self.delta_exclude_joints = anvil_cfg.get("delta_exclude_joints", [])
        self.task_description = task_description
        self.joint_names = joint_names
        self._is_vla = model_type in ("pi0", "pi05", "smolvla")
        self._exclude_indices: set[int] | None = None
        self._delta_ref_state: np.ndarray | None = None
        self._prev_gt_action: np.ndarray | None = None  # for delta_sequential GT computation

        # Horizon capture config. The executed trajectory is unchanged; at each inference
        # anchor we additionally grab the full predicted chunk via predict_action_chunk.
        _cfg = getattr(model, "config", None)
        self._native_n_action_steps = int(getattr(_cfg, "n_action_steps", 1) or 1)
        # Temporal-ensemble ACT re-infers every step and does not follow chunk-slicing
        # semantics — skip horizon capture for it (trajectory mode still works).
        _te_act = model_type == "act" and getattr(_cfg, "temporal_ensemble_coeff", None) is not None
        self._capture_horizon = (
            capture_horizon and hasattr(model, "predict_action_chunk") and not _te_act
        )

    def evaluate_episode(
        self,
        dataset,
        frame_indices: list[int],
        episode_idx: int,
        split_label: str,
    ) -> EpisodeResult:
        """Evaluate model predictions for a single episode."""
        _ensure_model_loader_importable()
        from lerobot_control.model_loader import reset_model_state
        from lerobot_control.delta_restore import restore_delta_chunk

        predicted_actions: list[np.ndarray] = []
        ground_truth_actions: list[np.ndarray] = []
        raw_actions: list[np.ndarray] = []
        obs_state_list: list[np.ndarray] = []
        raw_gt_list: list[np.ndarray] = []
        _prev_gt: np.ndarray | None = None  # for delta_sequential GT reference

        reset_model_state(self.model)
        self._delta_ref_state = None   # reset per-episode delta reference state
        self._prev_gt_action = None    # reset sequential GT reference
        # Shadow queue of pre-restored absolute actions (avoids touching model's normalized queue)
        _abs_shadow_queue: deque[np.ndarray] = deque()
        # Captured full predicted chunks for horizon analysis: (pos, raw_chunk, ref_state)
        _horizon_raw: list[tuple[int, np.ndarray, np.ndarray | None]] = []

        exclude_indices = self._resolve_exclude_indices()

        for pos, rel_idx in enumerate(tqdm(frame_indices, desc=f"Episode {episode_idx}", leave=False)):
            item = dataset[rel_idx]

            # Ground truth action (always absolute from dataset)
            gt_action = item["action"].numpy()

            # Build observation dict (observation.* keys only)
            obs = {k: v for k, v in item.items() if k.startswith("observation.")}

            # Observation state for delta restore (raw, before preprocessing)
            obs_state = item["observation.state"].numpy() if "observation.state" in item else None
            _obs_flat = (obs_state[-1] if obs_state is not None and obs_state.ndim > 1 else obs_state)

            # Detect whether a new action chunk is about to be generated.
            # When queue is empty, select_action runs model and fills queue.
            # We capture obs_state as the delta reference for the whole chunk.
            _is_new_chunk = (
                self.use_delta_actions
                and hasattr(self.model, "_queues")
                and len(self.model._queues.get("action", [])) == 0
            )

            # Independent anchor detector (any model, delta or absolute): a fresh inference
            # fires when the action queue is empty. Checked BEFORE select_action fills it.
            _is_anchor = self._capture_horizon and self._action_queue_len() == 0

            # Capture ref state BEFORE inference (queue will be filled after select_action)
            if _is_new_chunk and _obs_flat is not None:
                self._delta_ref_state = _obs_flat.copy()

            # Preprocess + inference
            with torch.inference_mode():
                if self._is_vla:
                    processed = self._preprocess_vla(obs)
                else:
                    if self.preprocessor:
                        processed = self.preprocessor(dict(obs))
                    else:
                        processed = obs
                    processed = self._move_to_device(processed)

                action_raw = self.model.select_action(processed)  # normalized tensor

                # On new chunk: collect all remaining normalized queue items BEFORE postprocessing.
                # The model's queue stores normalized values; we must denormalize the full chunk
                # together so delta restore operates in a consistent physical space.
                if _is_new_chunk and self.use_delta_actions and hasattr(self.model, "_queues"):
                    _rest_norm = [a.detach().clone() for a in self.model._queues.get("action", [])]
                else:
                    _rest_norm = None

            # On an inference anchor, grab the FULL predicted chunk (raw model-output space).
            # Executed path above is untouched — this is purely additional capture.
            if _is_anchor:
                _raw_long = self._capture_full_chunk_raw(processed)
                if _raw_long is not None:
                    _horizon_raw.append(
                        (pos, _raw_long, _obs_flat.copy() if _obs_flat is not None else None)
                    )

            # Postprocess current action (normalized → denormalized delta)
            action = self._tensor_to_np(action_raw)

            raw_actions.append(action)

            # Capture observation state for per-frame diagnostics
            if obs_state is not None:
                obs_state_list.append(_obs_flat.copy())

            # Compute raw ground truth (same space as raw model output)
            if self.use_delta_actions and _obs_flat is not None:
                raw_gt_list.append(self._compute_delta_gt(gt_action, _obs_flat, _prev_gt, exclude_indices))
            else:
                raw_gt_list.append(gt_action)
            _prev_gt = gt_action.copy()

            # Chunk-level delta restore using shadow queue to avoid re-entering normalized space.
            if self.use_delta_actions:
                if _is_new_chunk and self._delta_ref_state is not None:
                    if _rest_norm is not None:
                        # Denormalize the rest of the chunk (each element in physical/delta space)
                        _rest_denorm = [self._tensor_to_np(a) for a in _rest_norm]
                        _chunk = np.stack([action] + _rest_denorm) if _rest_denorm else action[np.newaxis]
                    else:
                        _chunk = action[np.newaxis]
                    _abs = restore_delta_chunk(
                        _chunk, self._delta_ref_state, self.action_type, exclude_indices
                    )
                    # Populate shadow queue with future absolute actions (skip index 0 = current)
                    _abs_shadow_queue = deque(_abs[1:])
                    action = _abs[0]
                elif _abs_shadow_queue:
                    # Non-new-chunk with queue model: pop pre-restored absolute action
                    action = _abs_shadow_queue.popleft()
                elif not hasattr(self.model, "_queues"):
                    # Models without chunk queue (e.g. ACT): per-step restore
                    _ref = self._delta_ref_state if self._delta_ref_state is not None else _obs_flat
                    if _ref is not None:
                        _abs = restore_delta_chunk(
                            action[np.newaxis], _ref, self.action_type, exclude_indices
                        )
                        action = _abs[0]

            predicted_actions.append(action)
            ground_truth_actions.append(gt_action)

        substrate = self._build_substrate(
            episode_idx, split_label, ground_truth_actions, _horizon_raw, exclude_indices
        )

        return EpisodeResult(
            episode_idx=episode_idx,
            split_label=split_label,
            predicted=np.stack(predicted_actions),
            ground_truth=np.stack(ground_truth_actions),
            joint_names=self.joint_names,
            raw_output=np.stack(raw_actions) if raw_actions else None,
            obs_states=np.stack(obs_state_list) if obs_state_list else None,
            raw_ground_truth=np.stack(raw_gt_list) if raw_gt_list else None,
            substrate=substrate,
        )

    def _tensor_to_np(self, a: object) -> np.ndarray:
        """Postprocess (denormalize) a model action and return a flat numpy vector."""
        if self.postprocessor:
            a = self.postprocessor.process_action(a)  # type: ignore[arg-type]
        if isinstance(a, torch.Tensor):
            if a.dim() > 1:
                a = a.squeeze(0)
            return a.detach().cpu().numpy()
        return np.asarray(a).flatten()

    def _action_queue_len(self) -> int | None:
        """Length of the model's pending action queue, or None if it has no queue.

        Diffusion/VLA use ``model._queues['action']``; ACT uses ``model._action_queue``.
        """
        q = getattr(self.model, "_queues", None)
        if isinstance(q, dict) and "action" in q:
            return len(q["action"])
        aq = getattr(self.model, "_action_queue", None)
        if aq is not None:
            return len(aq)
        return None

    def _capture_full_chunk_raw(self, processed) -> np.ndarray | None:
        """Return the full predicted chunk (H, D) in raw model-output space, or None.

        Diffusion caps the returned chunk at ``n_action_steps`` (sliced from
        ``n_obs_steps-1``), so we temporarily raise it to ``horizon - n_obs_steps + 1``
        (the model's hard limit) to expose the full horizon. The first
        ``native n_action_steps`` rows still equal the executed actions. ACT/VLA return
        their full ``chunk_size`` directly and ignore the override.
        """
        cfg = getattr(self.model, "config", None)
        restore_n = None
        if self.model_type == "diffusion" and cfg is not None:
            n_obs = int(getattr(cfg, "n_obs_steps", 1) or 1)
            horizon = getattr(cfg, "horizon", None)
            if horizon is None:
                return None
            restore_n = getattr(cfg, "n_action_steps", None)
            cfg.n_action_steps = int(horizon) - n_obs + 1
        # Queue-based policies (diffusion/VLA) rebuild their batch from the already-populated
        # observation queues, keyed by the queue's own keys (e.g. stacked "observation.images").
        # Pass those keys rather than the raw per-camera batch. ACT has no _queues — pass the
        # real preprocessed batch, which it consumes directly.
        q = getattr(self.model, "_queues", None)
        capture_batch = dict.fromkeys(q) if isinstance(q, dict) and q else processed
        try:
            with torch.inference_mode():
                chunk = self.model.predict_action_chunk(capture_batch)  # (1, H, D) normalized
        except Exception as e:  # pragma: no cover - defensive
            log.warning("[anvil-eval] horizon capture failed: %s", e)
            return None
        finally:
            if restore_n is not None:
                cfg.n_action_steps = restore_n

        if chunk.dim() == 2:
            chunk = chunk.unsqueeze(0)
        rows = [self._tensor_to_np(chunk[:, i, :]) for i in range(chunk.shape[1])]
        return np.stack(rows) if rows else None

    def _build_substrate(
        self,
        episode_idx: int,
        split_label: str,
        ground_truth_actions: list[np.ndarray],
        horizon_raw: list[tuple[int, np.ndarray, np.ndarray | None]],
        exclude_indices: set[int],
    ) -> EpisodeSubstrate:
        """Assemble captured chunks + GT into an EpisodeSubstrate (delta-restored)."""
        if not horizon_raw or not ground_truth_actions:
            return EpisodeSubstrate(episode_idx, split_label, self.joint_names, [])

        _ensure_model_loader_importable()
        from lerobot_control.delta_restore import restore_delta_chunk

        gt_arr = np.stack(ground_truth_actions)
        total = gt_arr.shape[0]
        anchors: list[AnchorChunk] = []
        for pos, raw_long, ref in horizon_raw:
            avail = min(raw_long.shape[0], total - pos)
            if avail <= 0:
                continue
            raw_a = raw_long[:avail]
            if self.use_delta_actions and ref is not None:
                abs_a = restore_delta_chunk(raw_a, ref, self.action_type, exclude_indices)
            else:
                abs_a = raw_a
            anchors.append(
                AnchorChunk(
                    anchor_frame=pos,
                    executed_len=self._native_n_action_steps,
                    abs_pred=np.asarray(abs_a, dtype=np.float64),
                    abs_gt=np.asarray(gt_arr[pos : pos + avail], dtype=np.float64),
                    obs_ref=ref,
                    raw_pred=np.asarray(raw_a, dtype=np.float64),
                )
            )
        return EpisodeSubstrate(episode_idx, split_label, self.joint_names, anchors)

    def _preprocess_vla(self, obs: dict) -> dict:
        """Preprocess observation for VLA models (pi0, pi05, smolvla)."""
        if self.preprocessor:
            batch = dict(obs)
            if self.task_description:
                batch["task"] = [self.task_description]
            processed = self.preprocessor(batch)
            return self._move_to_device(processed)
        return self._move_to_device(obs)

    def _move_to_device(self, data):
        """Recursively move tensors to the configured device."""
        if torch.is_tensor(data):
            return data.to(self.device)
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self._move_to_device(v) for v in data)
        return data

    def _resolve_exclude_indices(self) -> set[int]:
        """Resolve delta_exclude_joints to index set (cached)."""
        if self._exclude_indices is not None:
            return self._exclude_indices

        self._exclude_indices = set()
        for name in self.delta_exclude_joints:
            if name in self.joint_names:
                self._exclude_indices.add(self.joint_names.index(name))
        return self._exclude_indices

    def _compute_delta_gt(
        self,
        gt_action: np.ndarray,
        obs_state: np.ndarray,
        prev_gt: np.ndarray | None = None,
        exclude: set[int] | None = None,
    ) -> np.ndarray:
        """Compute ground-truth in model-output (delta) space.

        delta_obs_t:      raw_gt = gt_action - obs_state
        delta_sequential: raw_gt = gt_action - prev_gt  (falls back to obs_state for t=0)
        Joints in delta_exclude_joints remain as absolute values (matching training).
        """
        if exclude is None:
            exclude = self._resolve_exclude_indices()
        ref = prev_gt if (self.action_type == "delta_sequential" and prev_gt is not None) else obs_state
        n = min(len(gt_action), len(ref))
        delta_gt = gt_action.copy()
        delta_gt[:n] = gt_action[:n] - ref[:n]
        for i in exclude:
            if i < len(delta_gt):
                delta_gt[i] = gt_action[i]
        return delta_gt

def load_model(checkpoint: str, device: str):
    """Load model + processors from checkpoint using ModelLoader.

    Returns (model, preprocessor, postprocessor, model_type).
    """
    _ensure_model_loader_importable()
    from lerobot_control.model_loader import ModelLoader

    loader = ModelLoader(
        model_path=checkpoint,
        device=device,
        logger=None,
        deterministic=True,
        seed=42,
    )
    model, preprocessor, postprocessor = loader.load_with_processors()

    # Detect model type
    model_type = getattr(loader, "model_type", "unknown")

    log.info("[anvil-eval] Loaded model: type=%s, device=%s", model_type, device)
    return model, preprocessor, postprocessor, model_type
