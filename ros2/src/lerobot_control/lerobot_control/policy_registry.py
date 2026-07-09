"""Policy capability registry for Anvil's LeRobot integration."""

from __future__ import annotations

SUPPORTED_POLICIES = {
    "act",
    "diffusion",
    "smolvla",
    "pi0",
    "pi05",
    "molmoact2",
    "groot",
    "multi_task_dit",
    "evo1",
    "fastwam",
    "vla_jepa",
}

LANGUAGE_CONDITIONED_POLICIES = {
    "smolvla",
    "pi0",
    "pi05",
    "molmoact2",
    "groot",
    "multi_task_dit",
    "evo1",
    "fastwam",
    "vla_jepa",
}

RTC_CHUNK_POLICIES = {
    "smolvla",
    "pi0",
    "pi05",
    "molmoact2",
    "groot",
    "evo1",
}

SYNC_CHUNK_POLICIES = {
    "multi_task_dit",
    "fastwam",
    "vla_jepa",
}


def is_supported_policy(model_type: str | None) -> bool:
    return model_type in SUPPORTED_POLICIES


def is_language_conditioned(model_type: str | None) -> bool:
    return model_type in LANGUAGE_CONDITIONED_POLICIES


def uses_rtc_inference(model_type: str | None) -> bool:
    return model_type in RTC_CHUNK_POLICIES


def uses_sync_chunk_inference(model_type: str | None) -> bool:
    return model_type in SYNC_CHUNK_POLICIES
