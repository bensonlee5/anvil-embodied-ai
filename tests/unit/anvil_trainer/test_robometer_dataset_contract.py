from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/training/materialize_openarm2_robometer_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("robometer_materializer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_checked_in_robometer_contract_is_leakage_safe() -> None:
    module = _load_module()
    priority_path = ROOT / "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v2.json"
    contract_path = (
        ROOT / "configs/training/robometer_manifests/openarm2_shirt_fold_robometer_v1.json"
    )
    priority = json.loads(priority_path.read_text())
    contract = json.loads(contract_path.read_text())

    specs = module.build_trajectory_specs(priority, contract)
    audit = module.validate_specs(specs, contract)

    assert audit["leakage_free"] is True
    assert audit["counts"] == {
        "train": {"episodes": 27, "full_episodes": 27, "stage_clips": 81, "trajectories": 108},
        "validation": {"episodes": 3, "full_episodes": 3, "stage_clips": 9, "trajectories": 12},
        "test": {"episodes": 3, "full_episodes": 3, "stage_clips": 9, "trajectories": 12},
    }
    assert len(specs) == 132
    assert sum(spec.frame_count for spec in specs if spec.kind == "full_episode") == 34850
    assert sum(spec.frame_count for spec in specs if spec.kind == "stage_clip") == 34850
    assert all(spec.data_source.startswith("openarm2_roboreward_") for spec in specs)
    assert all(spec.partial_success == 1.0 for spec in specs if spec.kind == "full_episode")
    assert {spec.partial_success for spec in specs if spec.kind == "stage_clip"} <= {
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    }


def test_every_stage_clip_stays_inside_its_original_episode() -> None:
    module = _load_module()
    priority = json.loads(
        (
            ROOT / "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v2.json"
        ).read_text()
    )
    contract = json.loads(
        (
            ROOT / "configs/training/robometer_manifests/openarm2_shirt_fold_robometer_v1.json"
        ).read_text()
    )
    specs = module.build_trajectory_specs(priority, contract)
    full_by_episode = {spec.episode_index: spec for spec in specs if spec.kind == "full_episode"}
    for spec in specs:
        full = full_by_episode[spec.episode_index]
        assert full.split == spec.split
        assert full.start_frame <= spec.start_frame < spec.end_frame <= full.end_frame
