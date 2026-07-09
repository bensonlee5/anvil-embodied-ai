"""Regression coverage for Anvil training compatibility patches."""

from types import SimpleNamespace

import torch

from anvil_trainer.patches import (
    _normalize_uint8_camera_images,
    _remap_molmoact2_processor_overrides,
)


def test_custom_eval_normalizes_only_uint8_camera_images():
    batch = {
        "observation.images.waist": torch.tensor([0, 255], dtype=torch.uint8),
        "observation.state": torch.tensor([1, 2], dtype=torch.uint8),
    }

    result = _normalize_uint8_camera_images(batch, ("observation.images.waist",))

    assert result is batch
    assert batch["observation.images.waist"].dtype == torch.float32
    assert torch.equal(batch["observation.images.waist"], torch.tensor([0.0, 1.0]))
    assert batch["observation.state"].dtype == torch.uint8


def test_molmoact2_uses_saved_masked_normalizer_step_names():
    generic_stats_override = {"stats": {"action": {"mean": [0.0]}}}
    kwargs = {
        "preprocessor_overrides": {
            "device_processor": {"device": "cuda"},
            "normalizer_processor": generic_stats_override,
        },
        "postprocessor_overrides": {
            "unnormalizer_processor": generic_stats_override,
        },
    }

    result = _remap_molmoact2_processor_overrides(
        SimpleNamespace(type="molmoact2"), kwargs
    )

    assert "normalizer_processor" not in result["preprocessor_overrides"]
    assert (
        result["preprocessor_overrides"]["molmoact2_masked_normalizer"]
        is generic_stats_override
    )
    assert "unnormalizer_processor" not in result["postprocessor_overrides"]
    assert (
        result["postprocessor_overrides"]["molmoact2_masked_unnormalizer"]
        is generic_stats_override
    )
    assert "normalizer_processor" in kwargs["preprocessor_overrides"]
