"""Tests for EMA / DDPM-IP / DDIM defaults and refactored helpers.

All tests are CPU-only, in-process, no dataset/GPU/subprocess needed.
Covers:
  - EMAModel: get_decay, step, state_dict/load_state_dict, load_from_dir
  - config: _pop_float, from_env_and_args (DDIM/50 injection, opt-out, gating, flags)
  - patches: _compute_mean_loss, _log_wandb, patched_compute_loss (DDPM-IP)
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from anvil_trainer.config import TrainingConfig, _pop_float
from anvil_trainer.ema import EMAModel
from anvil_trainer.patches import TransformRunner, _compute_mean_loss, _log_wandb


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patched_argv(*extra_args: str):
    """Minimal sys.argv for from_env_and_args; restores on exit."""
    saved_argv = sys.argv[:]
    saved_env = os.environ.copy()
    try:
        sys.argv = ["anvil-trainer", "--dataset.root=/x/foo"] + list(extra_args)
        for key in ("LEROBOT_EXCLUDE_OBSERVS", "LEROBOT_TASK_OVERRIDE"):
            os.environ.pop(key, None)
        yield
    finally:
        sys.argv = saved_argv
        os.environ.clear()
        os.environ.update(saved_env)


# ===========================================================================
# EMAModel (ema.py)
# ===========================================================================


class TestEMAGetDecay:
    """Pure math tests — no model needed."""

    def _ema(self, **kwargs) -> EMAModel:
        return EMAModel(nn.Linear(4, 2, bias=False), **kwargs)

    def test_step_zero_returns_zero(self):
        assert self._ema().get_decay(0) == 0.0

    def test_step_one_returns_zero_with_defaults(self):
        # effective step = max(0, 1 - 0 - 1) = 0 → early-return 0.0
        assert self._ema().get_decay(1) == 0.0

    def test_step_two_positive(self):
        ema = self._ema()
        d = ema.get_decay(2)
        # effective step=1 → 1 - (1+1)^-0.75
        expected = 1.0 - (2.0) ** -0.75
        assert abs(d - expected) < 1e-6
        assert 0.0 < d < 1.0

    def test_large_step_approaches_max_value(self):
        ema = self._ema(max_value=0.9999)
        assert ema.get_decay(1_000_000) == pytest.approx(0.9999, abs=1e-4)

    def test_custom_max_value_clamped(self):
        ema = self._ema(max_value=0.5, power=0.75)
        # at large step, raw value > 0.5 → clamped
        assert ema.get_decay(100_000) == pytest.approx(0.5, abs=1e-3)


class TestEMAStep:
    def test_first_step_copies_live_params(self):
        """decay=0 on first step → EMA params become live params exactly."""
        live = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            live.weight.fill_(3.14)

        ema = EMAModel(copy.deepcopy(live))

        # Mutate live AFTER deepcopy so EMA seed differs
        with torch.no_grad():
            live.weight.fill_(2.71)

        ema.step(live)  # optimization_step=0 → decay=0

        assert torch.allclose(ema.averaged_model.weight, live.weight)

    def test_batchnorm_params_hard_copied(self):
        """BN learnable params (weight/bias) are hard-copied regardless of decay."""
        live = nn.Sequential(nn.Linear(4, 4, bias=False), nn.BatchNorm1d(4))
        live_bn = live[1]

        ema = EMAModel(copy.deepcopy(live))
        # Force a step that would normally blend (decay > 0)
        ema.optimization_step = 5
        decay = ema.get_decay(5)
        assert decay > 0.0

        with torch.no_grad():
            live_bn.weight.fill_(7.0)
            live_bn.bias.fill_(3.0)

        ema.step(live)

        ema_bn = ema.averaged_model[1]
        # Hard copy → ema BN weight == live BN weight (not blended)
        assert torch.allclose(ema_bn.weight, live_bn.weight)
        assert torch.allclose(ema_bn.bias, live_bn.bias)

    def test_ema_blend_formula(self):
        """ema_param = old * decay + live * (1-decay) for regular params."""
        ema_seed = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            ema_seed.weight.fill_(0.0)
        ema = EMAModel(ema_seed)

        # Force a specific step so decay > 0
        ema.optimization_step = 5
        decay = ema.get_decay(5)
        assert decay > 0.0

        live = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            live.weight.fill_(1.0)

        ema.step(live)  # old=0, live=1 → expected = 0*decay + 1*(1-decay)

        expected = 1.0 - decay
        assert torch.allclose(ema.averaged_model.weight, torch.full((2, 2), expected), atol=1e-6)

    def test_optimization_step_increments(self):
        live = nn.Linear(2, 2, bias=False)
        ema = EMAModel(copy.deepcopy(live))
        assert ema.optimization_step == 0
        ema.step(live)
        assert ema.optimization_step == 1
        ema.step(live)
        assert ema.optimization_step == 2


class TestEMAStateDict:
    def test_round_trip(self):
        ema = EMAModel(nn.Linear(2, 2, bias=False), power=0.8, max_value=0.9, inv_gamma=2.0)
        ema.optimization_step = 42
        ema.decay = 0.876

        sd = ema.state_dict()
        assert sd["optimization_step"] == 42
        assert abs(sd["decay"] - 0.876) < 1e-9

        ema2 = EMAModel(nn.Linear(2, 2, bias=False))
        ema2.load_state_dict(sd)
        assert ema2.optimization_step == 42
        assert abs(ema2.decay - 0.876) < 1e-9

    def test_state_dict_keys(self):
        sd = EMAModel(nn.Linear(2, 2, bias=False)).state_dict()
        assert set(sd.keys()) == {"optimization_step", "decay", "power", "max_value", "inv_gamma"}

    def test_json_serialisable(self):
        """state_dict() must be JSON-serialisable (used to write ema_state.json)."""
        ema = EMAModel(nn.Linear(2, 2, bias=False))
        ema.optimization_step = 10
        json.dumps(ema.state_dict())  # must not raise


class TestEMALoadFromDir:
    def test_returns_none_when_no_ema_json(self, tmp_path):
        result = EMAModel.load_from_dir(tmp_path, nn.Linear(2, 2, bias=False))
        assert result is None

    def test_restores_counter_and_raw_weights(self, tmp_path):
        """load_from_dir restores optimization_step and loads raw weights into live model."""
        from safetensors.torch import save_file

        live = nn.Linear(4, 2, bias=False)

        # Write all-zeros raw weights
        raw_sd = {k: torch.zeros_like(v) for k, v in live.state_dict().items()}
        save_file(raw_sd, str(tmp_path / "model_raw.safetensors"))

        state = {
            "optimization_step": 500,
            "decay": 0.97,
            "power": 0.75,
            "max_value": 0.9999,
            "inv_gamma": 1.0,
        }
        (tmp_path / "ema_state.json").write_text(json.dumps(state))

        ema = EMAModel.load_from_dir(tmp_path, live)

        assert ema is not None
        assert ema.optimization_step == 500
        assert abs(ema.decay - 0.97) < 1e-9

        # live model should now carry the raw (all-zeros) weights
        for p in live.parameters():
            assert torch.allclose(p, torch.zeros_like(p))

    def test_no_raw_safetensors_still_returns_ema(self, tmp_path):
        """load_from_dir succeeds even if model_raw.safetensors is absent (logs a warning)."""
        state = {
            "optimization_step": 10,
            "decay": 0.5,
            "power": 0.75,
            "max_value": 0.9999,
            "inv_gamma": 1.0,
        }
        (tmp_path / "ema_state.json").write_text(json.dumps(state))

        ema = EMAModel.load_from_dir(tmp_path, nn.Linear(2, 2, bias=False))
        assert ema is not None
        assert ema.optimization_step == 10


# ===========================================================================
# config helpers (config.py)
# ===========================================================================


class TestPopFloat:
    """Unit tests for _pop_float — isolates sys.argv manipulation."""

    def _call(self, argv: list[str], flag: str, default: float) -> float:
        saved = sys.argv[:]
        try:
            sys.argv = argv[:]
            return _pop_float(flag, default)
        finally:
            sys.argv = saved

    def test_present_returns_float(self):
        assert self._call(["prog", "--ema-power=0.8"], "ema-power", 0.75) == pytest.approx(0.8)

    def test_absent_returns_default(self):
        assert self._call(["prog"], "ema-power", 0.75) == pytest.approx(0.75)

    def test_arg_consumed_from_argv(self):
        saved = sys.argv[:]
        try:
            sys.argv = ["prog", "--ddpm-ip-alpha=0.2", "--other=1"]
            _pop_float("ddpm-ip-alpha", 0.1)
            assert not any(a.startswith("--ddpm-ip-alpha=") for a in sys.argv)
            assert "--other=1" in sys.argv
        finally:
            sys.argv = saved

    def test_integer_string_coerced(self):
        assert self._call(["prog", "--ema-inv-gamma=2"], "ema-inv-gamma", 1.0) == pytest.approx(2.0)


class TestFromEnvAndArgsDiffusion:
    """DDIM/50 injection, EMA/DDPM-IP defaults via from_env_and_args."""

    def test_ddim_injected_into_argv_for_diffusion(self):
        """from_env_and_args injects DDIM+50 into sys.argv for diffusion policy."""
        with _patched_argv("--policy.type=diffusion"):
            TrainingConfig.from_env_and_args()
            assert any(a == "--policy.noise_scheduler_type=DDIM" for a in sys.argv)
            assert any(a == "--policy.num_train_timesteps=50" for a in sys.argv)

    def test_ema_defaults(self):
        with _patched_argv("--policy.type=diffusion"):
            cfg = TrainingConfig.from_env_and_args()
        assert cfg.use_ema is True
        assert cfg.ema_power == pytest.approx(0.75)
        assert cfg.ema_max_value == pytest.approx(0.9999)
        assert cfg.ema_inv_gamma == pytest.approx(1.0)

    def test_ddpm_ip_defaults(self):
        with _patched_argv("--policy.type=diffusion"):
            cfg = TrainingConfig.from_env_and_args()
        assert cfg.use_ddpm_ip is True
        assert cfg.ddpm_ip_alpha == pytest.approx(0.1)

    def test_optout_noise_scheduler_type(self):
        """Pre-supplying --policy.noise_scheduler_type=DDPM suppresses DDIM injection."""
        with _patched_argv("--policy.type=diffusion", "--policy.noise_scheduler_type=DDPM"):
            TrainingConfig.from_env_and_args()
            sched_args = [a for a in sys.argv if a.startswith("--policy.noise_scheduler_type=")]
        # Only the user-supplied DDPM remains; DDIM was NOT injected
        assert sched_args == ["--policy.noise_scheduler_type=DDPM"]

    def test_optout_num_train_timesteps(self):
        """Pre-supplying --policy.num_train_timesteps=100 suppresses 50 injection."""
        with _patched_argv("--policy.type=diffusion", "--policy.num_train_timesteps=100"):
            TrainingConfig.from_env_and_args()
            ts_args = [a for a in sys.argv if a.startswith("--policy.num_train_timesteps=")]
        assert ts_args == ["--policy.num_train_timesteps=100"]

    def test_resume_skips_ddim_injection(self):
        """On --resume, DDIM/50 must not be injected (avoids draccus decode errors)."""
        # "some/nonexistent/path" has resume_checkpoint="last" → no FileNotFoundError
        with _patched_argv("--policy.type=diffusion", "--resume=some/nonexistent/path"):
            TrainingConfig.from_env_and_args()
            assert not any(a == "--policy.noise_scheduler_type=DDIM" for a in sys.argv)
            assert not any(a == "--policy.num_train_timesteps=50" for a in sys.argv)

    def test_act_policy_no_ddim_injection(self):
        """ACT policy → DDIM not injected (injection is diffusion-only)."""
        with _patched_argv("--policy.type=act"):
            TrainingConfig.from_env_and_args()
            assert not any(a == "--policy.noise_scheduler_type=DDIM" for a in sys.argv)

    def test_no_ema_flag(self):
        with _patched_argv("--policy.type=diffusion", "--no-ema"):
            cfg = TrainingConfig.from_env_and_args()
        assert cfg.use_ema is False

    def test_ema_power_override(self):
        with _patched_argv("--policy.type=diffusion", "--ema-power=0.9"):
            cfg = TrainingConfig.from_env_and_args()
        assert cfg.ema_power == pytest.approx(0.9)

    def test_ema_max_value_override(self):
        with _patched_argv("--policy.type=diffusion", "--ema-max-value=0.999"):
            cfg = TrainingConfig.from_env_and_args()
        assert cfg.ema_max_value == pytest.approx(0.999)

    def test_ema_inv_gamma_override(self):
        with _patched_argv("--policy.type=diffusion", "--ema-inv-gamma=2.0"):
            cfg = TrainingConfig.from_env_and_args()
        assert cfg.ema_inv_gamma == pytest.approx(2.0)

    def test_no_ddpm_ip_flag(self):
        with _patched_argv("--policy.type=diffusion", "--no-ddpm-ip"):
            cfg = TrainingConfig.from_env_and_args()
        assert cfg.use_ddpm_ip is False

    def test_ddpm_ip_alpha_override(self):
        with _patched_argv("--policy.type=diffusion", "--ddpm-ip-alpha=0.2"):
            cfg = TrainingConfig.from_env_and_args()
        assert cfg.ddpm_ip_alpha == pytest.approx(0.2)


# ===========================================================================
# patches.py helpers
# ===========================================================================


class _FakePolicy(nn.Linear):
    """Minimal policy: one parameter, .forward returns (scalar_loss, {})."""

    def __init__(self, loss: float = 1.0):
        super().__init__(1, 1, bias=False)
        self._loss_val = loss

    def forward(self, batch):  # noqa: D102
        return torch.tensor(self._loss_val), {}


class TestComputeMeanLoss:
    def test_basic_mean(self):
        policy = _FakePolicy(loss=2.0)
        batches = [{"x": torch.zeros(1)} for _ in range(3)]
        result = _compute_mean_loss(policy, batches, preprocessor=None)
        assert result == pytest.approx(2.0)

    def test_mean_of_varying_losses(self):
        losses = iter([1.0, 3.0, 5.0])

        class VaryPolicy(nn.Linear):
            def __init__(self):
                super().__init__(1, 1, bias=False)
            def forward(self, batch):
                return torch.tensor(next(losses)), {}

        result = _compute_mean_loss(VaryPolicy(), [{"x": torch.zeros(1)}] * 3, None)
        assert result == pytest.approx(3.0)  # (1+3+5)/3

    def test_none_policy_returns_none(self):
        assert _compute_mean_loss(None, [{}], preprocessor=None) is None

    def test_none_dataloader_returns_none(self):
        assert _compute_mean_loss(_FakePolicy(), None, preprocessor=None) is None

    def test_preprocessor_called(self):
        calls = []

        def preprocessor(batch):
            calls.append(batch)
            return batch

        _compute_mean_loss(_FakePolicy(0.5), [{"x": torch.zeros(1)}] * 2, preprocessor)
        assert len(calls) == 2

    def test_act_policy_uses_train_mode_during_forward(self):
        """ACTPolicy must be in train mode during forward, eval after."""
        modes: list[bool] = []

        class ACTPolicy(nn.Linear):
            def __init__(self):
                super().__init__(1, 1, bias=False)
            def forward(self, batch):
                modes.append(self.training)
                return torch.tensor(0.0), {}

        policy = ACTPolicy()
        _compute_mean_loss(policy, [{"x": torch.zeros(1)}], preprocessor=None)

        assert modes == [True], f"ACTPolicy should be in train mode during forward, got {modes}"
        assert not policy.training, "ACTPolicy should be back in eval mode after"

    def test_non_act_policy_stays_in_eval(self):
        """Non-ACT policy should stay in eval mode during forward."""
        modes: list[bool] = []

        class NormalPolicy(nn.Linear):
            def __init__(self):
                super().__init__(1, 1, bias=False)
            def forward(self, batch):
                modes.append(self.training)
                return torch.tensor(0.0), {}

        _compute_mean_loss(NormalPolicy(), [{"x": torch.zeros(1)}], preprocessor=None)
        assert modes == [False]


class TestLogWandB:
    def test_no_wandb_is_noop(self):
        """No wandb installed / no active run → must not raise."""
        _log_wandb({"loss": 1.0}, step=100)

    def test_no_active_run_is_noop(self, monkeypatch):
        """wandb.run is None → log must not be called."""

        class FakeWandB:
            run = None

            @staticmethod
            def log(metrics, step):
                raise AssertionError("log must not be called when run is None")

        monkeypatch.setitem(sys.modules, "wandb", FakeWandB())
        _log_wandb({"loss": 1.0}, step=100)

    def test_active_run_calls_log(self, monkeypatch):
        """With an active wandb run, log must be called with correct args."""
        log_calls: list = []

        class FakeWandB:
            run = object()  # non-None

            @staticmethod
            def log(metrics, step):
                log_calls.append((metrics, step))

        monkeypatch.setitem(sys.modules, "wandb", FakeWandB())
        _log_wandb({"eval/loss": 0.5}, step=42)
        assert log_calls == [({"eval/loss": 0.5}, 42)]

    def test_exception_in_wandb_swallowed(self, monkeypatch):
        """Any exception from wandb must be silently swallowed."""

        class BrokenWandB:
            @property
            def run(self):
                raise RuntimeError("wandb broken")

        monkeypatch.setitem(sys.modules, "wandb", BrokenWandB())
        _log_wandb({"x": 1}, step=0)  # must not raise


class TestDDPMIPPatch:
    """Verify patched_compute_loss applies DDPM-IP correctly.

    Invariant: the *noise* passed to add_noise is eps + alpha*extra (perturbed),
    but the *target* for the loss is the original eps (unperturbed).
    """

    @pytest.fixture(autouse=True)
    def _restore_diffusion_model(self):
        """Capture DiffusionModel.compute_loss before each test; restore after."""
        mod = pytest.importorskip("lerobot.policies.diffusion.modeling_diffusion")
        original = mod.DiffusionModel.compute_loss
        yield mod.DiffusionModel
        mod.DiffusionModel.compute_loss = original

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_model_self(B: int, H: int, D: int, N_OBS: int, recorded: dict):
        """SimpleNamespace that fakes the DiffusionModel surface and records noise."""

        class _SchedulerCfg:
            num_train_timesteps = 50

        class _Scheduler:
            config = _SchedulerCfg()

            def add_noise(self, traj, noise, timesteps):
                recorded["noise"] = noise.detach().clone()
                return traj + noise

        return SimpleNamespace(
            config=SimpleNamespace(
                horizon=H,
                n_obs_steps=N_OBS,
                prediction_type="epsilon",
                do_mask_loss_for_padding=False,
            ),
            noise_scheduler=_Scheduler(),
            _prepare_global_conditioning=lambda batch: None,
            unet=lambda noisy, ts, global_cond=None: torch.zeros_like(noisy),
        )

    @staticmethod
    def _make_batch(B: int, H: int, D: int, N_OBS: int) -> dict:
        return {
            "observation.state": torch.zeros(B, N_OBS, 5),
            "action": torch.ones(B, H, D),
            "observation.images": torch.zeros(B, 1, 3, 4, 4),
            "action_is_pad": torch.zeros(B, H, dtype=torch.bool),
        }

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_patch_disabled_when_no_ddpm_ip(self, _restore_diffusion_model):
        """--no-ddpm-ip: DiffusionModel.compute_loss is NOT replaced."""
        DiffusionModel = _restore_diffusion_model
        original_fn = DiffusionModel.compute_loss

        runner = TransformRunner(TrainingConfig(use_ddpm_ip=False))
        runner.apply_ddpm_ip_patch()

        assert DiffusionModel.compute_loss is original_fn

    def test_alpha_zero_no_perturbation(self, _restore_diffusion_model):
        """alpha=0: noise passed to add_noise equals original eps (no perturbation).

        RNG sequence (seeded): eps = randn(...), timesteps = randint(...), extra = randn_like(eps)
        With alpha=0: noise_to_add_noise = eps + 0 * extra = eps.
        """
        DiffusionModel = _restore_diffusion_model
        B, H, D, N_OBS = 2, 4, 3, 2
        recorded: dict = {}

        runner = TransformRunner(TrainingConfig(use_ddpm_ip=True, ddpm_ip_alpha=0.0))
        runner.apply_ddpm_ip_patch()

        model_self = self._make_model_self(B, H, D, N_OBS, recorded)
        batch = self._make_batch(B, H, D, N_OBS)

        torch.manual_seed(99)
        DiffusionModel.compute_loss(model_self, batch)

        # Replicate RNG: only eps is needed (alpha=0 → extra is generated but multiplied by 0)
        torch.manual_seed(99)
        eps_check = torch.randn(B, H, D)

        assert torch.allclose(recorded["noise"], eps_check, atol=1e-6), (
            "With alpha=0, noise passed to add_noise should equal original eps"
        )

        runner.restore_all_patches()

    def test_nonzero_alpha_perturbs_noise(self, _restore_diffusion_model):
        """alpha>0: noise passed to add_noise = eps + alpha*extra.

        RNG sequence (seeded):
          1. eps = randn(trajectory.shape)
          2. timesteps = randint(...) — consume
          3. extra = randn_like(eps)
        Then: noise_to_add_noise = eps + alpha * extra.
        """
        DiffusionModel = _restore_diffusion_model
        B, H, D, N_OBS = 2, 4, 3, 2
        alpha = 0.3
        recorded: dict = {}

        runner = TransformRunner(TrainingConfig(use_ddpm_ip=True, ddpm_ip_alpha=alpha))
        runner.apply_ddpm_ip_patch()

        model_self = self._make_model_self(B, H, D, N_OBS, recorded)
        batch = self._make_batch(B, H, D, N_OBS)

        torch.manual_seed(42)
        DiffusionModel.compute_loss(model_self, batch)

        # Replicate exact RNG sequence with same seed
        torch.manual_seed(42)
        eps_check = torch.randn(B, H, D)
        torch.randint(0, 50, (B,))          # consume timesteps
        extra_check = torch.randn_like(eps_check)
        expected_noise = eps_check + alpha * extra_check

        assert torch.allclose(recorded["noise"], expected_noise, atol=1e-5), (
            f"noise to add_noise should be eps + {alpha}*extra"
        )

        runner.restore_all_patches()

    def test_target_is_original_eps_not_perturbed(self, _restore_diffusion_model):
        """The MSE target is the ORIGINAL eps, not the perturbed eps.

        With unet returning zeros: loss = mse(zeros, eps) = mean(eps^2).
        If target were eps_perturbed, the loss would differ.
        """
        DiffusionModel = _restore_diffusion_model
        B, H, D, N_OBS = 2, 4, 3, 2
        alpha = 0.3
        recorded: dict = {}

        runner = TransformRunner(TrainingConfig(use_ddpm_ip=True, ddpm_ip_alpha=alpha))
        runner.apply_ddpm_ip_patch()

        model_self = self._make_model_self(B, H, D, N_OBS, recorded)
        batch = self._make_batch(B, H, D, N_OBS)

        torch.manual_seed(7)
        loss = DiffusionModel.compute_loss(model_self, batch)

        # loss = mse(zeros, eps) = mean(eps^2)
        torch.manual_seed(7)
        eps_check = torch.randn(B, H, D)
        expected_loss = (eps_check ** 2).mean()

        assert loss.ndim == 0, "loss should be a scalar"
        assert torch.allclose(loss, expected_loss, atol=1e-5), (
            "target should be original eps (not perturbed): loss = mean(eps^2)"
        )

        runner.restore_all_patches()
