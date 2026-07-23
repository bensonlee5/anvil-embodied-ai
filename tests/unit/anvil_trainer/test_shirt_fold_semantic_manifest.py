from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "configs/training/semantic_manifests/openarm2_shirt_fold_5stage_v1.json"
STAGE_ORDER = [
    "side_one",
    "recenter_pull",
    "side_two",
    "strip_refinement",
    "bottom_to_top",
]
ABSENT_RECENTER_EPISODES = [2, 5, 10, 20, 22, 27]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def test_five_stage_manifest_is_a_complete_partition() -> None:
    manifest = _manifest()
    assert manifest["schema_version"] == "openarm2.shirt-fold-semantic-segmentation.v1"
    assert manifest["stage_order"] == STAGE_ORDER
    assert manifest["optional_stages"] == ["recenter_pull", "strip_refinement"]
    assert len(manifest["episodes"]) == 33
    assert sum(item["frame_count"] for item in manifest["episodes"]) == 34_850

    for episode in manifest["episodes"]:
        stages = episode["stages"]
        assert [stage["name"] for stage in stages] == STAGE_ORDER
        assert stages[0]["start_frame"] == 0
        assert stages[-1]["end_frame"] == episode["frame_count"]
        assert all(left["end_frame"] == right["start_frame"] for left, right in pairwise(stages))
        assert all(
            stage["present"] == (stage["end_frame"] > stage["start_frame"]) for stage in stages
        )


def test_optional_recenter_is_absent_only_without_a_separated_cycle() -> None:
    manifest = _manifest()
    absent = [
        episode["episode_index"]
        for episode in manifest["episodes"]
        if not episode["stages"][1]["present"]
    ]
    assert absent == ABSENT_RECENTER_EPISODES
    assert all(episode["stages"][3]["present"] for episode in manifest["episodes"])


def test_quality_remains_three_outcomes_not_five_motion_labels() -> None:
    manifest = _manifest()
    assert manifest["outcome_order"] == ["side_one", "side_two", "bottom_to_top"]
    for episode in manifest["episodes"]:
        assert [outcome["name"] for outcome in episode["outcomes"]] == manifest["outcome_order"]
        assert [outcome["observed_after_stage"] for outcome in episode["outcomes"]] == [
            "side_one",
            "strip_refinement",
            "bottom_to_top",
        ]
        assert all(outcome["quality_score"] in {1, 2, 3, 4, 5} for outcome in episode["outcomes"])


def test_manifest_provenance_matches_pinned_inputs() -> None:
    manifest = _manifest()
    provenance = manifest["provenance"]
    source = ROOT / provenance["source_manifest"]
    action_data = ROOT / provenance["action_data"]
    assert provenance["source_manifest_sha256"] == _sha256(source)
    assert provenance["action_data_sha256"] == _sha256(action_data)
    assert manifest["dataset"]["fingerprints"]["data/chunk-000/file-000.parquet"] == _sha256(
        action_data
    )


def test_checked_manifest_matches_deterministic_generator(tmp_path: Path) -> None:
    generated = tmp_path / "semantic-manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/training/build_openarm2_shirt_fold_semantic_manifest.py"),
            "--output",
            str(generated),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(generated.read_text()) == _manifest()
