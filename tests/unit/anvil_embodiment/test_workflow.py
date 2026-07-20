from pathlib import Path

import numpy as np
from anvil_embodiment.artifact import WEIGHTS_NAME
from anvil_embodiment.workflow import (
    _fit_horizon_action_statistics,
    _resample_cached_motion_intensity,
    evaluate_adapter_cache,
    train_residual_adapter,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "configs" / "embodiment_adapters" / "hf_folding_to_anvil_openarm2.json"


def _synthetic_cache(path: Path) -> None:
    count = 15
    steps = 30
    current = np.zeros((count, 16), dtype=np.float32)
    current[:, 3] = 1.1
    current[:, 11] = 1.1
    target = np.repeat(current[:, None, :], steps, axis=1)
    phase = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    target[:, :, 0] += 0.04 * phase
    target[:, :, 8] -= 0.04 * phase
    bridge = target.copy()
    bridge[:, :, 0] += 0.02
    bridge[:, :, 8] -= 0.02
    split = np.asarray(["train"] * 10 + ["val"] * 3 + ["test"] * 2)
    np.savez_compressed(
        path,
        cache_schema_version=np.asarray(3, dtype=np.int64),
        current_state=current,
        bridge_chunk=bridge,
        target_chunk=target,
        bridge_valid=np.ones(count, dtype=np.bool_),
        episode_index=np.arange(count),
        frame_index=np.zeros(count, dtype=np.int64),
        split=split,
    )


def test_train_and_evaluate_synthetic_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache.npz"
    output = tmp_path / "artifact"
    _synthetic_cache(cache)

    provenance = train_residual_adapter(
        manifest=MANIFEST,
        cache=cache,
        output=output,
        device="cpu",
        steps=3,
        batch_size=4,
        eval_every=1,
        seed=7,
    )
    report = evaluate_adapter_cache(
        adapter=output,
        cache=cache,
        device="cpu",
    )

    assert (output / WEIGHTS_NAME).is_file()
    assert (output / "offline_evaluation.json").is_file()
    assert provenance["best_step"] in {1, 2, 3}
    assert provenance["selection_contract"].startswith("lowest_validation_loss")
    assert provenance["selected_quality_gate_pass"] == bool(provenance["gate_passing_evaluations"])
    assert all("quality_gate_pass" in item for item in provenance["history"])
    assert set(report["splits"]) == {"train", "val", "test"}
    assert set(report["splits"]["test"]) == {
        "hold",
        "bridge",
        "adapter",
        "attempted_samples",
    }
    assert report["splits"]["test"]["attempted_samples"] == 2
    assert report["splits"]["test"]["adapter"]["failure_rate"] == 0.0
    assert provenance["horizon_action_statistics"]["stats_source"] == "valid_train_split_only"

    with np.load(cache, allow_pickle=False) as source:
        rejected_arrays = {name: source[name].copy() for name in source.files}
    rejected_arrays["bridge_valid"][-1] = False
    rejected_arrays["bridge_chunk"][-1] = np.nan
    rejected_cache = tmp_path / "rejected-cache.npz"
    np.savez_compressed(rejected_cache, **rejected_arrays)
    rejection_report = evaluate_adapter_cache(
        adapter=output,
        cache=rejected_cache,
        device="cpu",
    )
    adapter_test = rejection_report["splits"]["test"]["adapter"]
    assert adapter_test["attempted_samples"] == 2
    assert adapter_test["valid_samples"] == 1
    assert adapter_test["failure_rate"] == 0.5
    assert adapter_test["failure_adjusted_normalized_joint_mae"] > 0.5


def test_temporal_resampling_uses_only_train_split_and_preserves_raw_bridge() -> None:
    current = np.zeros((3, 16), dtype=np.float32)
    bridge = np.zeros((3, 4, 16), dtype=np.float32)
    target = np.zeros((3, 4, 16), dtype=np.float32)
    bridge[:, :, 0] = np.asarray([1.0, 2.0, 3.0, 4.0])
    target[0, :, 0] = np.asarray([2.0, 4.0, 4.0, 4.0])
    target[1:, :, 0] = 100.0
    arrays = {
        "current_state": current,
        "bridge_chunk": bridge,
        "target_chunk": target,
        "bridge_valid": np.ones(3, dtype=np.bool_),
        "split": np.asarray(["train", "val", "test"]),
    }
    ranges = np.tile(np.asarray([-10.0, 10.0], dtype=np.float32), (16, 1))

    report = _resample_cached_motion_intensity(arrays, ranges)

    assert report["factor"] == 2.0
    assert report["stats_source"] == "train_split_only"
    np.testing.assert_array_equal(arrays["raw_bridge_chunk"], bridge)
    np.testing.assert_array_equal(arrays["bridge_chunk"][0, :, 0], target[0, :, 0])
    np.testing.assert_array_equal(arrays["bridge_chunk"][:, :, 7], 0.0)


def test_horizon_statistics_use_valid_training_rows_only() -> None:
    current = np.zeros((4, 16), dtype=np.float32)
    target = np.zeros((4, 3, 16), dtype=np.float32)
    target[0, :, 0] = [0.1, 0.2, 0.3]
    target[1, :, 0] = [0.2, 0.4, 0.6]
    target[2:, :, 0] = 100.0
    arrays = {
        "current_state": current,
        "target_chunk": target,
        "split": np.asarray(["train", "train", "val", "train"]),
        "bridge_valid": np.asarray([True, True, True, False]),
    }
    ranges = np.tile(np.asarray([-1.0, 1.0], dtype=np.float32), (16, 1))

    scale, report = _fit_horizon_action_statistics(arrays, ranges)

    assert scale.shape == (3, 16)
    assert np.all(np.isfinite(scale))
    assert np.all(scale > 0)
    assert report["train_samples"] == 2
