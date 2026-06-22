"""Dataset transforms applied at ``LeRobotDataset.__getitem__`` time.

Each ``Transform`` subclass is enabled by a field on ``TrainingConfig`` and
runs once per loaded sample.  Transforms can also optionally patch lerobot
metadata before training starts — see ``patch_metadata``.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from anvil_trainer.config import TrainingConfig


log = logging.getLogger(__name__)


class DataIntegrityError(ValueError):
    """Raised when dataset features violate expected contracts."""


def _parse_names(info: dict, feat_key: str) -> list[str]:
    """Extract feature names from info.json for the given feature key.

    Handles both flat string lists and grouped dicts with ``motor_names``.
    """
    names = info.get("features", {}).get(feat_key, {}).get("names", [])
    if names and isinstance(names[0], dict):
        names = [n for group in names for n in group.get("motor_names", [])]
    return names


# =============================================================================
# Transform ABC
# =============================================================================


class Transform(ABC):
    """
    Abstract base class for dataset transforms.

    Subclasses implement specific transformations applied to dataset items
    during training. Each transform can optionally patch LeRobot internals
    for metadata filtering.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""

    @abstractmethod
    def is_enabled(self, config: TrainingConfig) -> bool:
        """Check if this transform should be applied."""

    @abstractmethod
    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        """
        Apply transform to a single dataset item.

        Args:
            item: Dataset item from LeRobotDataset.__getitem__
            config: Training configuration

        Returns:
            Transformed item
        """

    def patch_metadata(self, config: TrainingConfig, runner: Any = None) -> None:  # noqa: B027
        """
        Optional: Patch LeRobot metadata/utils before training.

        Override this method if the transform needs to modify how
        LeRobot builds the policy (e.g., filtering input features).

        ``runner`` (when provided) is the owning ``TransformRunner``; use its
        ``_patch(module, attr, new_value)`` method so patches are reverted
        when the :func:`patched_lerobot` context manager exits.
        """


# =============================================================================
# ExcludeObservationTransform
# =============================================================================


class ExcludeObservationTransform(Transform):
    """Exclude observation keys from training via --exclude-observation suffixes.

    Each suffix is prepended with "observation." to form the full dataset key:
      "images.chest"  -> "observation.images.chest"
      "velocity"      -> "observation.velocity"
    """

    @property
    def name(self) -> str:
        return "exclude_observation"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return bool(config.exclude_observation)

    @staticmethod
    def _full_keys(config: TrainingConfig) -> set[str]:
        return {f"observation.{s}" for s in config.exclude_observation}

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        for full_key in self._full_keys(config):
            item.pop(full_key, None)
        return item

    def patch_metadata(self, config: TrainingConfig, runner: Any = None) -> None:
        """Patch dataset_to_policy_features to exclude the specified observation keys."""
        import lerobot.datasets.feature_utils
        import lerobot.policies.factory
        from lerobot.datasets.feature_utils import dataset_to_policy_features

        original_func = dataset_to_policy_features
        excluded = self._full_keys(config)

        def filtered_func(features: dict) -> dict:
            filtered = {}
            for key, value in features.items():
                if key in excluded:
                    log.info("[exclude_observation] Excluding: %s", key)
                    continue
                filtered[key] = value
            return original_func(filtered)

        # Patch both the definition module and the importer (policies/factory.py).
        # Use runner._patch so patches are reverted by patched_lerobot(); fall
        # back to direct assignment for backward compatibility when a transform
        # is used standalone without a runner.
        if runner is not None:
            runner._patch(lerobot.datasets.feature_utils, "dataset_to_policy_features", filtered_func)
            runner._patch(lerobot.policies.factory, "dataset_to_policy_features", filtered_func)
        else:
            lerobot.datasets.feature_utils.dataset_to_policy_features = filtered_func
            lerobot.policies.factory.dataset_to_policy_features = filtered_func


# =============================================================================
# TaskOverrideTransform
# =============================================================================


class TaskOverrideTransform(Transform):
    """Override the task field for all dataset items."""

    @property
    def name(self) -> str:
        return "task_override"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return config.task_override is not None

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        if config.task_override:
            item["task"] = config.task_override
        return item


# =============================================================================
# EERelTransform — SE(3) relative EE actions
# =============================================================================


class EERelTransform(Transform):
    """Convert absolute EE obs and actions to SE(3)-relative representation.

    Both observation.state and action are anchored to the SAME current EE pose
    (last obs step), matching UMI's verified 'relative' mode:
        T_rel = inv(T_anchor) @ T_pose  (full SE(3), translation in body frame)

    obs: 8 dims/arm (quat layout) → 10 dims/arm (rot6d layout), relative to anchor
    action: 10 dims/arm (rot6d layout), unchanged dim, relative to anchor
    """

    def __init__(self):
        self._first_apply: bool = True

    @property
    def name(self) -> str:
        return "ee_rel"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return config.is_ee_rel

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        import torch
        from anvil_shared.ee_transform import ee_obs_rel_forward, ee_rel_forward, n_arms_from_dims

        if "action" not in item or "observation.state" not in item:
            return item

        action = item["action"]                   # (horizon, 10*n_arms) or (10*n_arms,)
        obs_full = item["observation.state"]       # (T, 8*n_arms) or (8*n_arms,)

        # Anchor = most recent obs step (8*n_arms,)
        if obs_full.dim() > 1:
            anchor_tensor = obs_full[-1]
        else:
            anchor_tensor = obs_full

        anchor_np = anchor_tensor.detach().cpu().numpy().astype("float64")
        obs_np = obs_full.detach().cpu().numpy().astype("float64")
        action_np = action.detach().cpu().numpy().astype("float64")

        # Validate action/state dims
        try:
            n_arms = n_arms_from_dims(anchor_np.shape[-1], action_np.shape[-1])
        except ValueError as exc:
            raise DataIntegrityError(str(exc)) from exc

        # Transform obs: (T, 8*n) → (T, 10*n) relative to anchor
        obs_rel_np = ee_obs_rel_forward(obs_np, anchor_np)

        # Transform action: (horizon, 10*n) relative to anchor
        single = action_np.ndim == 1
        if single:
            action_np = action_np[None, :]
        delta_np = ee_rel_forward(action_np, anchor_np)
        if single:
            delta_np = delta_np[0]

        item["observation.state"] = torch.tensor(obs_rel_np, dtype=torch.float32)
        item["action"] = torch.tensor(delta_np, dtype=action.dtype)

        if self._first_apply:
            log.info(
                "[ee_rel] active — %d arm(s), obs (8n abs) → (10n rel), action (abs rot6d) → SE(3) relative",
                n_arms,
            )
            self._first_apply = False

        return item

    def patch_metadata(self, config: TrainingConfig, runner: Any = None) -> None:
        """Patch lerobot's dataset_to_policy_features to report 10-dim obs shape.

        observation.state changes from 8*n_arms (quat layout) to 10*n_arms (rot6d
        relative layout) after this transform. The policy must be initialised with
        the correct input dimension.
        """
        if not config.is_ee_rel:
            return

        import lerobot.datasets.feature_utils as _feat_utils
        import lerobot.policies.factory as _factory
        from lerobot.datasets.feature_utils import dataset_to_policy_features as _original

        def _patched(features: dict) -> dict:
            modified = {}
            for key, feat in features.items():
                if key == "observation.state":
                    shape = feat.get("shape", ())
                    if len(shape) == 1 and shape[0] % 8 == 0:
                        modified[key] = {**feat, "shape": (shape[0] // 8 * 10,)}
                    else:
                        modified[key] = feat
                else:
                    modified[key] = feat
            return _original(modified)

        if runner is not None:
            runner._patch(_feat_utils, "dataset_to_policy_features", _patched)
            runner._patch(_factory, "dataset_to_policy_features", _patched)
        else:
            _feat_utils.dataset_to_policy_features = _patched
            _factory.dataset_to_policy_features = _patched
        log.info("[ee_rel] patched dataset_to_policy_features: obs.state 8n→10n/arm")
