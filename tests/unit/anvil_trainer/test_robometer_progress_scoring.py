from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/training/score_openarm2_robometer_progress.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("robometer_scoring", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_anchor_indices_include_exact_episode_end() -> None:
    module = _load_module()
    assert module.anchor_indices(61, 30) == [0, 30, 60]
    assert module.anchor_indices(62, 30) == [0, 30, 60, 61]
    assert module.anchor_indices(1, 30) == [0]


def test_prefix_context_matches_released_four_frame_evaluator() -> None:
    module = _load_module()
    assert module.prefix_context_indices(30) == [0, 10, 20, 30]
    assert module.prefix_context_indices(2) == [0, 1, 1, 2]


def test_interpolation_is_finite_bounded_and_episode_local() -> None:
    module = _load_module()
    values = module.interpolate_progress(5, [0, 2, 4], [-0.1, 0.5, 1.2])
    assert values == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    with pytest.raises(ValueError, match="first and last"):
        module.interpolate_progress(5, [1, 4], [0.0, 1.0])
