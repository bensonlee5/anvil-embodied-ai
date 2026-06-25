"""EMAModel — Exponential Moving Average of model weights.

Algorithm ported verbatim from UMI (universal_manipulation_interface /
diffusion_policy/model/diffusion/ema_model.py).  No UMI dependency is
required; only ``torch`` is needed.

Typical usage::

    import copy
    from anvil_trainer.ema import EMAModel

    ema = EMAModel(copy.deepcopy(policy), power=0.75, max_value=0.9999)
    ema.averaged_model.to(device)

    # Inside training loop, after optimizer.step():
    ema.step(unwrapped_policy)

    # For eval / checkpointing, use ema.averaged_model directly:
    ema.averaged_model.eval()
    loss = ema.averaged_model.forward(batch)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.nn.modules.batchnorm import _BatchNorm

log = logging.getLogger(__name__)


class EMAModel:
    """Exponential Moving Average of model weights.

    The caller must create ``model`` via ``copy.deepcopy(live_policy)`` before
    passing it in — this class does **not** deepcopy internally.  After
    construction, ``model`` is set to eval mode with ``requires_grad=False``,
    and its parameters are updated in-place by every call to :meth:`step`.

    Decay formula (@ crowsonkb's EMA warmup):

        step  = max(0, optimization_step - update_after_step - 1)
        decay = clamp(1 - (1 + step / inv_gamma) ** -power,
                      min_value, max_value)

    For ``power=0.75`` (UMI production): reaches 0.999 at ~10k steps,
    0.9999 at ~215k steps.  Use ``power=2/3`` for runs beyond 1M steps.

    BatchNorm layers and parameters with ``requires_grad=False`` are
    hard-copied (not averaged) — identical to the UMI implementation.

    Args:
        model:              A deepcopy of the live policy to use as the
                            averaged model.  Modified in-place by step().
        update_after_step:  Skip EMA updates for this many optimizer steps.
        inv_gamma:          Inverse multiplicative factor of EMA warmup.
        power:              Exponential factor.
        min_value:          Minimum EMA decay floor.
        max_value:          Maximum EMA decay ceiling.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ) -> None:
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay: float = 0.0
        self.optimization_step: int = 0

    def get_decay(self, optimization_step: int) -> float:
        """Compute EMA decay for the given optimizer step count."""
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1.0 - (1.0 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model: torch.nn.Module) -> None:
        """Update the EMA weights from the current live model.

        Must be called after every optimizer step (i.e. once per batch).
        Passes the unwrapped live policy — do **not** pass the
        accelerator-wrapped version.
        """
        self.decay = self.get_decay(self.optimization_step)

        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):
            for param, ema_param in zip(
                module.parameters(recurse=False),
                ema_module.parameters(recurse=False),
            ):
                if isinstance(param, dict):
                    raise RuntimeError("Dict parameter not supported by EMAModel.step()")

                if isinstance(module, _BatchNorm) or not param.requires_grad:
                    # Hard-copy for BatchNorm stats and frozen params — matches UMI.
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1.0 - self.decay)

        self.optimization_step += 1

    # ------------------------------------------------------------------
    # Persistence helpers (counter state only — weights live in averaged_model)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of EMA counter state.

        Does **not** include model weights (those are persisted separately via
        ``safetensors`` in ``patched_save_checkpoint``).
        """
        return {
            "optimization_step": self.optimization_step,
            "decay": self.decay,
            "power": self.power,
            "max_value": self.max_value,
            "inv_gamma": self.inv_gamma,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore EMA counter from a dict returned by :meth:`state_dict`."""
        self.optimization_step = int(state["optimization_step"])
        self.decay = float(state.get("decay", 0.0))
        log.info(
            "[ema] Restored EMA counter: optimization_step=%d  decay=%.6f",
            self.optimization_step,
            self.decay,
        )

    @classmethod
    def load_from_dir(cls, training_state_dir: Path, live_model: torch.nn.Module) -> EMAModel | None:
        """Attempt to restore an EMAModel from a checkpoint's training_state dir.

        Returns the restored EMAModel on success, or None if no EMA state is
        found (old checkpoint without EMA support).

        Args:
            training_state_dir: Path to the ``training_state/`` directory of a
                checkpoint (e.g. ``model_zoo/.../checkpoints/last/training_state``).
            live_model:         The current live policy (already loaded by lerobot).
                                Its device is used to place the EMA model.
        """
        import copy

        ema_json = training_state_dir / "ema_state.json"
        raw_path = training_state_dir / "model_raw.safetensors"

        if not ema_json.exists():
            return None

        state = json.loads(ema_json.read_text())
        device = next(live_model.parameters()).device

        # averaged_model seed: deepcopy live_model (which holds EMA weights at resume
        # time, since pretrained_model/ was saved with EMA weights).
        ema_averaged = copy.deepcopy(live_model)

        ema = cls(
            ema_averaged,
            inv_gamma=float(state.get("inv_gamma", 1.0)),
            power=float(state.get("power", 0.75)),
            max_value=float(state.get("max_value", 0.9999)),
        )
        ema.load_state_dict(state)

        # If raw weights were saved, load them back into the live model so the
        # optimizer moment vectors (which track the raw trajectory) stay aligned.
        if raw_path.exists():
            from safetensors.torch import load_file as _st_load

            raw_sd = _st_load(str(raw_path), device=str(device))
            live_model.load_state_dict(raw_sd, strict=False)
            log.info("[ema] Loaded raw weights from %s for resume", raw_path)
        else:
            log.warning(
                "[ema] ema_state.json found but model_raw.safetensors missing at %s. "
                "Live model will start from EMA weights (slight optimizer misalignment).",
                training_state_dir,
            )

        return ema
