from pathlib import Path
from types import SimpleNamespace

import pytest
from lerobot.configs import PreTrainedConfig
from lerobot.policies import factory
from lerobot_control.model_loader import ModelLoader, _processor_device_overrides


@pytest.mark.parametrize("requested", ["cpu", "mps", "cuda"])
def test_explicit_runtime_device_overrides_checkpoint_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    requested: str,
) -> None:
    class Config:
        compile_model = True
        device = "checkpoint-auto-selected-device"

    config = Config()

    def fake_from_pretrained(_cls, _path):
        return config

    monkeypatch.setattr(
        PreTrainedConfig,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    loader = ModelLoader(
        model_path=str(tmp_path),
        model_type="pi05",
        device=requested,
    )

    loaded = loader._load_pretrained_config()

    assert loaded is config
    assert loaded.device == requested
    assert loaded.compile_model is False


def test_saved_processors_override_only_runtime_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    device_pipeline = '{"steps":[{"registry_name":"device_processor","config":{}}]}'
    (tmp_path / "policy_preprocessor.json").write_text(device_pipeline)
    (tmp_path / "policy_postprocessor.json").write_text(device_pipeline)
    model = SimpleNamespace(config=object())
    captured = {}

    def fake_make_pre_post_processors(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return object(), object()

    monkeypatch.setattr(factory, "make_pre_post_processors", fake_make_pre_post_processors)
    loader = ModelLoader(
        model_path=str(tmp_path),
        model_type="pi05",
        device="cpu",
    )
    monkeypatch.setattr(loader, "load", lambda: model)

    loaded_model, preprocessor, postprocessor = loader.load_with_processors()

    assert loaded_model is model
    assert preprocessor is not None
    assert postprocessor is not None
    assert captured == {
        "config": model.config,
        "pretrained_path": str(tmp_path),
        "preprocessor_overrides": {"device_processor": {"device": "cpu"}},
        "postprocessor_overrides": {"device_processor": {"device": "cpu"}},
    }


def test_processor_without_device_step_gets_no_override(tmp_path: Path) -> None:
    config = tmp_path / "processor.json"
    config.write_text('{"steps":[{"registry_name":"normalizer_processor"}]}')

    assert _processor_device_overrides(config, "cpu") == {}
