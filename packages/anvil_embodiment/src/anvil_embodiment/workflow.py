"""Offline cache, supervised training, and evaluation for embodiment adapters."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from anvil_eval.dataset import EvaluationDataset
from anvil_eval.evaluator import load_model
from anvil_shared.embodiment import EmbodimentError
from torch import Tensor

from .artifact import AdapterArtifact, load_adapter_artifact, sha256_file, verify_target_dataset
from .kinematics import torch_forward_kinematics
from .policy import EmbodimentAdaptedPolicy
from .residual import ACTIVE_JOINT_INDICES, AdapterLossWeights, compute_adapter_loss


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _pretrained_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if (candidate / "config.json").is_file():
        return candidate
    nested = candidate / "pretrained_model"
    if (nested / "config.json").is_file():
        return nested
    raise FileNotFoundError(f"no pretrained_model config under {candidate}")


def _register_processor_compat_aliases() -> None:
    """Allow pinned LeRobot checkpoints to load across processor renames."""
    from lerobot.processor.pipeline import ProcessorStepRegistry
    from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

    names = ProcessorStepRegistry.list()
    if "delta_actions_processor" in names:
        return
    canonical_name = getattr(
        RelativeActionsProcessorStep,
        "_registry_name",
        "relative_actions_processor",
    )
    ProcessorStepRegistry.register("delta_actions_processor")(RelativeActionsProcessorStep)
    # Registration annotates the shared class. Keep new artifacts on the
    # installed LeRobot release's canonical registry name.
    RelativeActionsProcessorStep._registry_name = canonical_name


def _load_policy(path: str | Path, device: str) -> tuple[Any, Any, Any]:
    _register_processor_compat_aliases()
    model, preprocessor, postprocessor, _ = load_model(str(_pretrained_dir(path)), device)
    if preprocessor is None or postprocessor is None:
        raise EmbodimentError(f"policy at {path} is missing processor pipelines")
    return model, preprocessor, postprocessor


@torch.no_grad()
def _predict_direct_chunk(
    *,
    model: Any,
    preprocessor: Any,
    postprocessor: Any,
    observation: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    processed = _move_to_device(preprocessor(observation), device)
    raw = model.predict_action_chunk(processed)
    post_input = raw.squeeze(0) if isinstance(raw, Tensor) and raw.ndim == 3 else raw
    absolute = _numpy(postprocessor.process_action(post_input))
    if absolute.ndim == 3 and absolute.shape[0] == 1:
        absolute = absolute[0]
    if absolute.ndim != 2 or absolute.shape[1] != 16:
        raise EmbodimentError(f"policy returned an invalid chunk shape: {absolute.shape}")
    return absolute


def _load_splits(path: str | Path, episode_count: int) -> dict[int, str]:
    raw = json.loads(Path(path).read_text())
    result: dict[int, str] = {}
    for split, key in (
        ("train", "train_episodes"),
        ("val", "val_episodes"),
        ("test", "test_episodes"),
    ):
        for episode in raw.get(key, []):
            episode = int(episode)
            if episode < 0 or episode >= episode_count or episode in result:
                raise EmbodimentError(f"invalid or duplicate split episode: {episode}")
            result[episode] = split
    if len(result) != episode_count:
        missing = sorted(set(range(episode_count)) - set(result))
        raise EmbodimentError(f"split_info does not cover every episode; missing={missing}")
    return result


def _target_chunk(dataset: Any, frame_indices: list[int], offset: int, size: int) -> np.ndarray:
    last = len(frame_indices) - 1
    rows = []
    for horizon in range(size):
        index = frame_indices[min(offset + horizon, last)]
        rows.append(np.asarray(dataset.hf_dataset[index]["action"], dtype=np.float32))
    return np.stack(rows)


def _resample_chunk(chunk: np.ndarray, current: np.ndarray, factor: float) -> np.ndarray:
    """Resample an absolute action chunk in time, preserving its current-state origin."""
    if factor <= 0 or not np.isfinite(factor):
        raise ValueError("temporal resampling factor must be finite and positive")
    steps = chunk.shape[-2]
    source = np.concatenate([current[..., None, :], chunk], axis=-2)
    query = np.minimum((np.arange(steps, dtype=np.float64) + 1.0) * factor, steps)
    lower = np.floor(query).astype(np.int64)
    upper = np.ceil(query).astype(np.int64)
    weight = (query - lower).reshape((1,) * (chunk.ndim - 2) + (steps, 1))
    return source[..., lower, :] * (1.0 - weight) + source[..., upper, :] * weight


def _resample_cached_motion_intensity(
    arrays: dict[str, np.ndarray],
    target_joint_ranges: np.ndarray,
) -> dict[str, Any]:
    """Choose one trajectory-rate factor using valid target rows from train only."""
    train = (arrays["split"] == "train") & arrays["bridge_valid"]
    if not np.any(train):
        raise EmbodimentError("temporal resampling requires valid training samples")
    active = np.asarray(ACTIVE_JOINT_INDICES)
    raw_bridge = arrays["bridge_chunk"].copy()
    current = arrays["current_state"]
    target = arrays["target_chunk"]
    widths = (
        np.asarray(target_joint_ranges, dtype=np.float64)[active, 1]
        - np.asarray(target_joint_ranges, dtype=np.float64)[active, 0]
    )
    candidates = np.linspace(0.25, 2.5, 91, dtype=np.float64)
    target_motion = float(
        np.mean(np.abs(target[train][..., active] - current[train][:, None, :][..., active]))
    )
    if target_motion <= 1e-8:
        raise EmbodimentError("target motion is zero on the training split")
    objectives = []
    motion_values = []
    mae_values = []
    for factor in candidates:
        candidate = _resample_chunk(raw_bridge[train], current[train], float(factor))
        motion = float(
            np.mean(np.abs(candidate[..., active] - current[train][:, None, :][..., active]))
        )
        motion_values.append(motion)
        objectives.append(abs(np.log(max(motion, 1e-8) / target_motion)))
        mae_values.append(
            float(np.mean(np.abs(candidate[..., active] - target[train][..., active]) / widths))
        )
    best_index = int(np.argmin(objectives))
    if best_index in {0, len(candidates) - 1}:
        raise EmbodimentError("temporal resampling optimum is on the search boundary")
    factor = float(candidates[best_index])
    aligned = _resample_chunk(raw_bridge, current, factor)
    lower = np.asarray(target_joint_ranges, dtype=np.float32)[active, 0]
    upper = np.asarray(target_joint_ranges, dtype=np.float32)[active, 1]
    proposed = aligned[..., active]
    clipped = arrays["bridge_valid"][:, None, None] & ((proposed < lower) | (proposed > upper))
    aligned[..., active] = np.clip(proposed, lower, upper)
    # Gripper calibration is deterministic. Do not learn or time-warp it here.
    aligned[..., [7, 15]] = raw_bridge[..., [7, 15]]
    arrays["raw_bridge_chunk"] = raw_bridge
    arrays["bridge_chunk"] = aligned.astype(np.float32)
    return {
        "enabled": True,
        "method": "linear_trajectory_time_resampling_grid_search",
        "stats_source": "train_split_only",
        "factor": factor,
        "candidate_min": float(candidates[0]),
        "candidate_max": float(candidates[-1]),
        "candidate_count": int(len(candidates)),
        "objective": "absolute_log_mean_active_joint_motion_ratio",
        "train_target_motion_rad": target_motion,
        "train_motion_ratio_before": motion_values[int(np.argmin(np.abs(candidates - 1.0)))]
        / target_motion,
        "train_motion_ratio_after": motion_values[best_index] / target_motion,
        "train_normalized_joint_mae_before": mae_values[int(np.argmin(np.abs(candidates - 1.0)))],
        "train_normalized_joint_mae_after": mae_values[best_index],
        "clipped_fraction": float(np.mean(clipped[train])),
        "active_joint_indices": active.tolist(),
        "grippers_resampled": False,
    }


def _fit_horizon_action_statistics(
    arrays: dict[str, np.ndarray], target_joint_ranges: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit smooth per-horizon action statistics from valid training rows only."""
    selected = (arrays["split"] == "train") & arrays["bridge_valid"]
    if not np.any(selected):
        raise EmbodimentError("horizon normalization requires valid training samples")
    delta = arrays["target_chunk"][selected] - arrays["current_state"][selected, None, :]
    steps = delta.shape[1]
    time_axis = np.arange(1, steps + 1, dtype=np.float64)
    mean_design = np.stack([np.ones(steps), time_axis], axis=1)
    std_design = np.stack([np.ones(steps), np.sqrt(time_axis)], axis=1)
    observed_mean = np.mean(delta, axis=0)
    observed_std = np.std(delta, axis=0, ddof=0)
    mean_coefficients = np.linalg.lstsq(mean_design, observed_mean, rcond=None)[0]
    std_coefficients = np.linalg.lstsq(std_design, observed_std, rcond=None)[0]
    fitted_std = std_design @ std_coefficients
    ranges = np.asarray(target_joint_ranges, dtype=np.float64)
    floors = np.maximum((ranges[:, 1] - ranges[:, 0]) * 0.005, 1e-4)
    scale = np.maximum(fitted_std, floors[None, :]).astype(np.float32)
    return scale, {
        "schema_version": 1,
        "stats_source": "valid_train_split_only",
        "mean_model": "intercept_plus_horizon",
        "std_model": "intercept_plus_sqrt_horizon",
        "train_samples": int(np.sum(selected)),
        "mean_coefficients": mean_coefficients.tolist(),
        "std_coefficients": std_coefficients.tolist(),
        "scale_floor": floors.tolist(),
        "scale": scale.tolist(),
    }


def _rejection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, int] = {}
    for row in rows:
        key = "|".join(
            (
                str(row["split"]),
                str(row["episode"]),
                str(row["stage"]),
                str(row["direction"]),
                str(row["reason"]),
            )
        )
        summary[key] = summary.get(key, 0) + 1
    return {
        "key_order": ["split", "episode", "stage", "direction", "reason"],
        "counts": summary,
    }


def cache_policy_predictions(
    *,
    manifest: str | Path,
    base_policy: str | Path,
    dataset_path: str | Path,
    split_info: str | Path,
    output: str | Path,
    task: str,
    device: str = "cuda",
    stride: int = 10,
    seed: int = 42,
    video_backend: str = "pyav",
    baseline_policy: str | Path | None = None,
    resample_motion_intensity: bool = False,
) -> dict[str, Any]:
    """Cache frozen folding predictions after deterministic kinematic bridging."""
    if stride < 1:
        raise ValueError("stride must be positive")
    artifact = load_adapter_artifact(
        manifest,
        base_policy_dir=_pretrained_dir(base_policy),
        device=device,
        require_weights=False,
    )
    verify_target_dataset(artifact.spec, Path(dataset_path), Path(split_info))
    model, preprocessor, postprocessor = _load_policy(base_policy, device)
    adapted = EmbodimentAdaptedPolicy(
        model=model,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        artifact=artifact,
        device=device,
    )
    baseline = _load_policy(baseline_policy, device) if baseline_policy else None
    dataset_wrapper = EvaluationDataset(Path(dataset_path), video_backend=video_backend)
    dataset = dataset_wrapper.dataset
    episode_splits = _load_splits(split_info, dataset_wrapper.total_episodes)
    device_value = torch.device(device)

    current_rows: list[np.ndarray] = []
    bridge_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    baseline_rows: list[np.ndarray] = []
    bridge_valid_rows: list[bool] = []
    baseline_valid_rows: list[bool] = []
    episode_rows: list[int] = []
    frame_rows: list[int] = []
    split_rows: list[str] = []
    rejected: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    attempted_samples = 0
    total_samples = sum(
        (len(dataset_wrapper.get_episode_frames(episode)) + stride - 1) // stride
        for episode in range(dataset_wrapper.total_episodes)
    )

    def log_progress() -> None:
        if attempted_samples != 1 and attempted_samples % 10 and attempted_samples != total_samples:
            return
        elapsed = time.perf_counter() - start_time
        seconds_per_sample = elapsed / max(attempted_samples, 1)
        print(
            json.dumps(
                {
                    "event": "adapter_cache_progress",
                    "attempted": attempted_samples,
                    "accepted": int(sum(bridge_valid_rows)),
                    "rejected": len(rejected),
                    "total": total_samples,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": seconds_per_sample * (total_samples - attempted_samples),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for episode in range(dataset_wrapper.total_episodes):
        frames = dataset_wrapper.get_episode_frames(episode)
        adapted.reset()
        if baseline is not None and callable(getattr(baseline[0], "reset", None)):
            baseline[0].reset()
        for offset in range(0, len(frames), stride):
            attempted_samples += 1
            frame = frames[offset]
            item = dataset[frame]
            observation = {
                key: value for key, value in item.items() if key.startswith("observation.")
            }
            observation["task"] = [task]
            sample_seed = seed + episode * 1_000_003 + int(_numpy(item["frame_index"]))
            random.seed(sample_seed)
            np.random.seed(sample_seed % (2**32))
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
            current_state = _numpy(item["observation.state"]).astype(np.float32)
            target_chunk = _target_chunk(dataset, frames, offset, artifact.spec.residual.chunk_size)
            split = episode_splits[episode]
            bridge_chunk = np.full_like(target_chunk, np.nan, dtype=np.float32)
            bridge_valid = False
            try:
                bridge_chunk = _numpy(adapted.predict_action_chunk(observation))[0]
                bridge_valid = True
            except Exception as exc:
                message = str(exc)
                direction = next(
                    (
                        value
                        for value in ("target_to_reference", "reference_to_target")
                        if value in message
                    ),
                    "unknown",
                )
                rejected.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "split": split,
                        "stage": "bridge",
                        "direction": direction,
                        "reason": type(exc).__name__,
                        "error": message,
                    }
                )

            direct = np.full_like(target_chunk, np.nan, dtype=np.float32)
            baseline_valid = False
            if baseline is not None:
                try:
                    torch.manual_seed(sample_seed)
                    direct = _predict_direct_chunk(
                        model=baseline[0],
                        preprocessor=baseline[1],
                        postprocessor=baseline[2],
                        observation=observation,
                        device=device_value,
                    )
                    if len(direct) < artifact.spec.residual.chunk_size:
                        direct = np.concatenate(
                            [
                                direct,
                                np.repeat(
                                    direct[-1:],
                                    artifact.spec.residual.chunk_size - len(direct),
                                    axis=0,
                                ),
                            ]
                        )
                    direct = direct[: artifact.spec.residual.chunk_size]
                    baseline_valid = True
                except Exception as exc:
                    rejected.append(
                        {
                            "episode": episode,
                            "frame": frame,
                            "split": split,
                            "stage": "baseline",
                            "direction": "direct",
                            "reason": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
            current_rows.append(current_state)
            bridge_rows.append(bridge_chunk.astype(np.float32))
            target_rows.append(target_chunk.astype(np.float32))
            bridge_valid_rows.append(bridge_valid)
            if baseline is not None:
                baseline_rows.append(direct.astype(np.float32))
                baseline_valid_rows.append(baseline_valid)
            episode_rows.append(episode)
            frame_rows.append(int(_numpy(item["frame_index"])))
            split_rows.append(split)
            log_progress()

    if not any(bridge_valid_rows):
        raise EmbodimentError("every cache sample was rejected")
    arrays: dict[str, np.ndarray] = {
        "cache_schema_version": np.asarray(3, dtype=np.int64),
        "current_state": np.stack(current_rows),
        "bridge_chunk": np.stack(bridge_rows),
        "target_chunk": np.stack(target_rows),
        "bridge_valid": np.asarray(bridge_valid_rows, dtype=np.bool_),
        "episode_index": np.asarray(episode_rows, dtype=np.int64),
        "frame_index": np.asarray(frame_rows, dtype=np.int64),
        "split": np.asarray(split_rows),
    }
    if baseline_rows:
        arrays["baseline_chunk"] = np.stack(baseline_rows).astype(np.float32)
        arrays["baseline_valid"] = np.asarray(baseline_valid_rows, dtype=np.bool_)
    temporal_resampling: dict[str, Any] = {"enabled": False}
    if resample_motion_intensity:
        temporal_resampling = _resample_cached_motion_intensity(
            arrays,
            artifact.bridge.target_joint_ranges,
        )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    report = {
        "schema_version": 3,
        "attempted_samples": len(current_rows),
        "valid_bridge_samples": int(sum(bridge_valid_rows)),
        "rejected_bridge_samples": int(len(current_rows) - sum(bridge_valid_rows)),
        "valid_baseline_samples": int(sum(baseline_valid_rows)) if baseline is not None else None,
        "rejected": rejected,
        "rejection_summary": _rejection_summary(rejected),
        "stride": stride,
        "seed": seed,
        "seconds": time.perf_counter() - start_time,
        "base_policy": str(_pretrained_dir(base_policy)),
        "baseline_policy": str(_pretrained_dir(baseline_policy)) if baseline_policy else None,
        "temporal_resampling": temporal_resampling,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report


def _cache_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    required = {
        "cache_schema_version",
        "current_state",
        "bridge_chunk",
        "target_chunk",
        "bridge_valid",
        "episode_index",
        "frame_index",
        "split",
    }
    if not required.issubset(arrays):
        raise EmbodimentError(f"adapter cache is missing {sorted(required - arrays.keys())}")
    if int(arrays["cache_schema_version"]) != 3:
        raise EmbodimentError("adapter cache schema_version must be 3")
    if arrays["bridge_chunk"].shape != arrays["target_chunk"].shape:
        raise EmbodimentError("bridge and target cache shapes differ")
    count = len(arrays["current_state"])
    if any(len(arrays[name]) != count for name in required - {"cache_schema_version"}):
        raise EmbodimentError("adapter cache row counts differ")
    valid = arrays["bridge_valid"].astype(bool)
    if not np.all(np.isfinite(arrays["bridge_chunk"][valid])):
        raise EmbodimentError("valid bridge cache rows must be finite")
    if not np.all(np.isnan(arrays["bridge_chunk"][~valid])):
        raise EmbodimentError("rejected bridge cache rows must be NaN")
    return arrays


def _numeric_metric_leaves(prefix: str, values: dict[str, Any]) -> dict[str, float]:
    """Flatten numeric report leaves for metric backends; retain strings in JSON only."""
    flattened: dict[str, float] = {}
    for name, value in values.items():
        key = f"{prefix}/{name}"
        if isinstance(value, dict):
            flattened.update(_numeric_metric_leaves(key, value))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            flattened[key] = float(value)
    return flattened


def train_residual_adapter(
    *,
    manifest: str | Path,
    cache: str | Path,
    output: str | Path,
    device: str = "cuda",
    steps: int = 5000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    eval_every: int = 100,
    seed: int = 42,
    loss_weights: AdapterLossWeights = AdapterLossWeights(),
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_mode: Literal["online", "offline", "disabled"] = "online",
) -> dict[str, Any]:
    """Train only the bounded target-space residual; the VLA remains frozen."""
    if min(steps, batch_size, eval_every) < 1:
        raise ValueError("steps, batch_size, and eval_every must be positive")
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cache_path = Path(cache)
    cache_hash = sha256_file(cache_path)
    arrays = _cache_arrays(cache_path)
    valid = arrays["bridge_valid"].astype(bool)
    train_indices = np.flatnonzero((arrays["split"] == "train") & valid)
    val_indices = np.flatnonzero((arrays["split"] == "val") & valid)
    if not len(train_indices):
        raise EmbodimentError("cache contains no training samples")
    if not len(val_indices):
        val_indices = train_indices

    artifact = load_adapter_artifact(manifest, device=device, require_weights=False)
    residual = artifact.residual
    residual.train()
    optimizer = torch.optim.AdamW(
        residual.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    device_value = torch.device(device)
    current = torch.as_tensor(arrays["current_state"], dtype=torch.float32)
    bridge = torch.as_tensor(arrays["bridge_chunk"], dtype=torch.float32)
    target = torch.as_tensor(arrays["target_chunk"], dtype=torch.float32)
    target_ranges = torch.as_tensor(
        artifact.bridge.target_joint_ranges, dtype=torch.float32, device=device_value
    )
    horizon_scale_values, horizon_statistics = _fit_horizon_action_statistics(
        arrays, artifact.bridge.target_joint_ranges
    )
    action_scale = torch.as_tensor(horizon_scale_values, dtype=torch.float32, device=device_value)
    correction_bounds = artifact.residual.correction_bounds.to(device_value)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fallback_loss = float("inf")
    fallback_step = 0
    fallback_state: dict[str, Tensor] | None = None
    passing_loss = float("inf")
    passing_step = 0
    passing_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    wandb_run: Any | None = None
    if wandb_project is not None and wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name or f"{artifact.spec.adapter_id}-seed-{seed}",
            job_type="embodiment-adapter",
            mode=wandb_mode,
            tags=["embodiment-adapter", "shirt-fold", "offline-only"],
            config={
                "adapter_id": artifact.spec.adapter_id,
                "deployment_status": artifact.spec.deployment_status,
                "manifest": str(Path(manifest).resolve()),
                "cache": str(cache_path.resolve()),
                "cache_sha256": cache_hash,
                "train_samples": int(len(train_indices)),
                "val_samples": int(len(val_indices)),
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "eval_every": eval_every,
                "seed": seed,
                "loss_weights": asdict(loss_weights),
                "horizon_action_statistics": horizon_statistics,
                "residual_contract": asdict(artifact.spec.residual),
            },
        )
        input_artifact = wandb.Artifact(
            f"{artifact.spec.adapter_id}-inputs-{cache_hash[:8]}",
            type="embodiment-adapter-inputs",
            metadata={
                "cache_sha256": cache_hash,
                "deployment_status": artifact.spec.deployment_status,
            },
        )
        input_artifact.add_file(str(Path(manifest).resolve()), name="manifest.json")
        input_artifact.add_file(str(cache_path.resolve()), name="cache.npz")
        cache_report = cache_path.with_suffix(cache_path.suffix + ".json")
        if cache_report.is_file():
            input_artifact.add_file(str(cache_report.resolve()), name="cache-report.json")
        wandb_run.log_artifact(input_artifact)

    def evaluate(indices: np.ndarray) -> tuple[float, dict[str, float]]:
        residual.eval()
        totals: dict[str, float] = {}
        count = 0
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                state_batch = current[selected].to(device_value)
                bridge_batch = bridge[selected].to(device_value)
                target_batch = target[selected].to(device_value)
                corrected, correction = residual(state_batch, bridge_batch)
                loss, terms = compute_adapter_loss(
                    current_state=state_batch,
                    corrected=corrected,
                    residual=correction,
                    target=target_batch,
                    target_ranges=target_ranges,
                    action_scale=action_scale,
                    target_model=artifact.bridge.target_spec,
                    correction_bounds=correction_bounds,
                    weights=loss_weights,
                )
                size = len(selected)
                count += size
                for name, value in terms.items():
                    totals[name] = totals.get(name, 0.0) + float(value) * size
        residual.train()
        metrics = {name: value / count for name, value in totals.items()}
        return metrics["loss"], metrics

    def evaluate_quality(indices: np.ndarray) -> dict[str, float]:
        residual.eval()
        corrected_rows: list[np.ndarray] = []
        correction_rows: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                corrected, correction = residual(
                    current[selected].to(device_value),
                    bridge[selected].to(device_value),
                )
                corrected_rows.append(_numpy(corrected))
                correction_rows.append(_numpy(correction))
        residual.train()

        corrected_values = np.concatenate(corrected_rows)
        correction_values = np.concatenate(correction_rows)
        selected_current = arrays["current_state"][indices]
        selected_resampled_bridge = arrays["bridge_chunk"][indices]
        selected_bridge = arrays.get("raw_bridge_chunk", arrays["bridge_chunk"])[indices]
        selected_target = arrays["target_chunk"][indices]
        split_name = str(arrays["split"][indices[0]])
        attempted_count = int(np.sum(arrays["split"] == split_name))
        adapter_metrics = _prediction_report(
            corrected_values,
            selected_target,
            selected_current,
            artifact,
            attempted_samples=attempted_count,
        )
        bridge_metrics = _prediction_report(
            selected_bridge,
            selected_target,
            selected_current,
            artifact,
            attempted_samples=attempted_count,
        )
        resampled_bridge_metrics = _prediction_report(
            selected_resampled_bridge,
            selected_target,
            selected_current,
            artifact,
            attempted_samples=attempted_count,
        )
        active = np.asarray(ACTIVE_JOINT_INDICES)
        active_bounds = _numpy(correction_bounds)[active]
        bound_fraction = np.abs(correction_values[..., active]) / np.maximum(
            active_bounds,
            1e-8,
        )
        quality: dict[str, float] = {
            "residual_mean_bound_fraction": float(np.mean(bound_fraction)),
            "residual_max_bound_fraction": float(np.max(bound_fraction)),
            "residual_saturation_fraction": float(np.mean(bound_fraction >= 0.95)),
        }
        for name, value in adapter_metrics.items():
            quality[f"adapter_{name}"] = value
        for name, value in bridge_metrics.items():
            quality[f"bridge_{name}"] = value
        if "raw_bridge_chunk" in arrays:
            for name, value in resampled_bridge_metrics.items():
                quality[f"resampled_bridge_{name}"] = value
        error_metrics = (
            "normalized_joint_mae",
            "failure_adjusted_normalized_joint_mae",
            "joint_rmse_rad",
            "shoulder_mae_rad",
            "tcp_position_mae_m",
            "tcp_orientation_mae_deg",
        )
        for name in error_metrics:
            quality[f"improvement_{name}"] = bridge_metrics[name] - adapter_metrics[name]
        quality["improvement_motion_ratio"] = abs(bridge_metrics["motion_ratio"] - 1.0) - abs(
            adapter_metrics["motion_ratio"] - 1.0
        )
        quality["quality_gate_joint"] = float(quality["improvement_normalized_joint_mae"] > 0.0)
        quality["quality_gate_failure_adjusted_joint"] = float(
            quality["improvement_failure_adjusted_normalized_joint_mae"] > 0.0
        )
        quality["quality_gate_failure_rate"] = float(
            adapter_metrics["failure_rate"] <= bridge_metrics["failure_rate"]
        )
        quality["quality_gate_shoulder"] = float(quality["improvement_shoulder_mae_rad"] > 0.0)
        quality["quality_gate_tcp"] = float(quality["improvement_tcp_position_mae_m"] > 0.0)
        quality["quality_gate_motion"] = float(quality["improvement_motion_ratio"] >= 0.0)
        quality["quality_gate_residual"] = float(quality["residual_saturation_fraction"] <= 0.05)
        quality["quality_gate_pass"] = float(
            quality["quality_gate_joint"]
            and quality["quality_gate_failure_adjusted_joint"]
            and quality["quality_gate_failure_rate"]
            and quality["quality_gate_shoulder"]
            and quality["quality_gate_tcp"]
            and quality["quality_gate_motion"]
            and quality["quality_gate_residual"]
        )
        return quality

    for step in range(1, steps + 1):
        sampled = train_indices[
            torch.randint(len(train_indices), (batch_size,), generator=generator).numpy()
        ]
        state_batch = current[sampled].to(device_value)
        bridge_batch = bridge[sampled].to(device_value)
        target_batch = target[sampled].to(device_value)
        corrected, correction = residual(state_batch, bridge_batch)
        loss, train_terms = compute_adapter_loss(
            current_state=state_batch,
            corrected=corrected,
            residual=correction,
            target=target_batch,
            target_ranges=target_ranges,
            action_scale=action_scale,
            target_model=artifact.bridge.target_spec,
            correction_bounds=correction_bounds,
            weights=loss_weights,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            val_loss, metrics = evaluate(val_indices)
            quality = evaluate_quality(val_indices)
            record = {"step": float(step), **metrics, **quality}
            history.append(record)
            print(
                json.dumps(
                    {
                        "event": "adapter_validation",
                        "step": step,
                        "train_loss": float(train_terms["loss"]),
                        **record,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{f"train/{name}": float(value) for name, value in train_terms.items()},
                        **{f"val/{name}": value for name, value in metrics.items()},
                        **{f"quality/val/{name}": value for name, value in quality.items()},
                        "optimizer/learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )
            state = {
                name: value.detach().cpu().clone() for name, value in residual.state_dict().items()
            }
            if val_loss < fallback_loss:
                fallback_loss = val_loss
                fallback_step = step
                fallback_state = state
            if quality["quality_gate_pass"] and val_loss < passing_loss:
                passing_loss = val_loss
                passing_step = step
                passing_state = {
                    name: value.detach().cpu().clone() for name, value in state.items()
                }

    if fallback_state is None:
        raise RuntimeError("adapter training produced no checkpoint")
    selected_passing_checkpoint = passing_state is not None
    best_loss = passing_loss if selected_passing_checkpoint else fallback_loss
    best_step = passing_step if selected_passing_checkpoint else fallback_step
    best_state = passing_state if selected_passing_checkpoint else fallback_state
    residual.load_state_dict(best_state)
    artifact.residual.eval()
    provenance = {
        "schema_version": 2,
        "cache": str(cache_path.resolve()),
        "cache_sha256": cache_hash,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_step": best_step,
        "best_val_loss": best_loss,
        "selection_contract": (
            "lowest_validation_loss_among_quality_gate_passing_checkpoints_"
            "else_lowest_validation_loss"
        ),
        "gate_passing_evaluations": sum(int(item["quality_gate_pass"]) for item in history),
        "selected_quality_gate_pass": selected_passing_checkpoint,
        "fallback_best_step": fallback_step,
        "fallback_best_val_loss": fallback_loss,
        "loss_weights": asdict(loss_weights),
        "horizon_action_statistics": horizon_statistics,
        "history": history,
        "wandb": (
            {
                "project": wandb_project,
                "entity": wandb_entity,
                "run_id": wandb_run.id,
                "run_name": wandb_run.name,
                "run_path": wandb_run.path,
                "mode": wandb_mode,
            }
            if wandb_run is not None
            else None
        ),
    }
    output_path = Path(output)
    artifact.save(output_path, training_provenance=provenance)
    evaluation_path = output_path / "offline_evaluation.json"
    final_report = evaluate_adapter_cache(
        adapter=output_path,
        cache=cache_path,
        device=device,
        output=evaluation_path,
    )
    if wandb_run is not None:
        final_metrics: dict[str, float] = {}
        for split, split_values in final_report["splits"].items():
            for policy_name, policy_values in split_values.items():
                if isinstance(policy_values, dict):
                    final_metrics.update(
                        _numeric_metric_leaves(f"final/{split}/{policy_name}", policy_values)
                    )
                else:
                    final_metrics[f"final/{split}/{policy_name}"] = float(policy_values)
        wandb_run.log(final_metrics, step=steps)
        best_record = next(item for item in history if int(item["step"]) == best_step)
        wandb_run.summary["best_step"] = best_step
        wandb_run.summary["best_val_loss"] = best_loss
        wandb_run.summary["quality_gate_pass"] = best_record["quality_gate_pass"]
        for name, value in best_record.items():
            if name.startswith(
                ("adapter_", "bridge_", "resampled_bridge_", "improvement_", "residual_")
            ):
                wandb_run.summary[f"best/{name}"] = value

        import wandb

        model_artifact = wandb.Artifact(
            f"{artifact.spec.adapter_id}-model-{wandb_run.id}",
            type="model",
            metadata={
                "cache_sha256": cache_hash,
                "best_step": best_step,
                "best_val_loss": best_loss,
                "deployment_status": artifact.spec.deployment_status,
            },
        )
        model_artifact.add_dir(str(output_path.resolve()))
        wandb_run.log_artifact(model_artifact)
        wandb_run.finish()
    return provenance


def _geodesic_degrees(predicted: Tensor, target: Tensor) -> Tensor:
    relative = predicted.transpose(-1, -2) @ target
    cosine = torch.clamp(
        (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return torch.rad2deg(torch.acos(cosine))


def _prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    artifact: AdapterArtifact,
) -> dict[str, Any]:
    predicted = torch.as_tensor(prediction, dtype=torch.float64)
    expected = torch.as_tensor(target, dtype=torch.float64)
    active = torch.tensor(ACTIVE_JOINT_INDICES)
    ranges = torch.as_tensor(artifact.bridge.target_joint_ranges, dtype=torch.float64)
    widths = (ranges[:, 1] - ranges[:, 0]).clamp_min(1e-8)
    error = predicted - expected
    normalized_mae = torch.mean(torch.abs(error[..., active]) / widths[active])
    shoulder = torch.tensor([0, 1, 8, 9])
    shoulder_mae = torch.mean(torch.abs(error[..., shoulder]))
    current_tensor = torch.as_tensor(current, dtype=torch.float64)[:, None, :]
    predicted_motion = torch.mean(torch.abs(predicted[..., active] - current_tensor[..., active]))
    target_motion = torch.mean(torch.abs(expected[..., active] - current_tensor[..., active]))

    position_errors = []
    orientation_errors = []
    for side_index, side in enumerate(("right", "left")):
        start = side_index * 8
        pred_pos, pred_rot = torch_forward_kinematics(
            artifact.bridge.target_spec, side, predicted[..., start : start + 7]
        )
        target_pos, target_rot = torch_forward_kinematics(
            artifact.bridge.target_spec, side, expected[..., start : start + 7]
        )
        position_errors.append(torch.linalg.vector_norm(pred_pos - target_pos, dim=-1))
        orientation_errors.append(_geodesic_degrees(pred_rot, target_rot))
    per_actuator: dict[str, dict[str, float | str]] = {}
    for index, name in enumerate(artifact.spec.target_vector.names):
        actuator_error = error[..., index]
        predicted_actuator_motion = torch.mean(
            torch.abs(predicted[..., index] - current_tensor[..., index])
        )
        target_actuator_motion = torch.mean(
            torch.abs(expected[..., index] - current_tensor[..., index])
        )
        per_actuator[name] = {
            "kind": "gripper" if index in (7, 15) else "arm_joint",
            "native_unit": "meter" if index in (7, 15) else "radian",
            "mae": float(torch.mean(torch.abs(actuator_error))),
            "rmse": float(torch.sqrt(torch.mean(actuator_error**2))),
            "normalized_mae": float(torch.mean(torch.abs(actuator_error) / widths[index])),
            "final_horizon_mae": float(torch.mean(torch.abs(actuator_error[:, -1]))),
            "mean_commanded_motion": float(predicted_actuator_motion),
            "target_motion": float(target_actuator_motion),
            "motion_ratio": float(
                predicted_actuator_motion / target_actuator_motion.clamp_min(1e-8)
            ),
        }
    return {
        "normalized_joint_mae": float(normalized_mae),
        "joint_rmse_rad": float(torch.sqrt(torch.mean(error[..., active] ** 2))),
        "shoulder_mae_rad": float(shoulder_mae),
        "tcp_position_mae_m": float(torch.stack(position_errors).mean()),
        "tcp_orientation_mae_deg": float(torch.stack(orientation_errors).mean()),
        "mean_commanded_motion_rad": float(predicted_motion),
        "target_motion_rad": float(target_motion),
        "motion_ratio": float(predicted_motion / target_motion.clamp_min(1e-8)),
        "per_actuator": per_actuator,
    }


def _prediction_report(
    prediction: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    artifact: AdapterArtifact,
    *,
    attempted_samples: int,
) -> dict[str, Any]:
    """Report valid-row metrics and a failure-adjusted joint error denominator."""
    valid_samples = len(prediction)
    if attempted_samples < valid_samples or attempted_samples < 1:
        raise ValueError("attempted_samples must cover at least one valid prediction")
    metrics: dict[str, Any] = _prediction_metrics(prediction, target, current, artifact)
    active = np.asarray(ACTIVE_JOINT_INDICES)
    ranges = np.asarray(artifact.bridge.target_joint_ranges, dtype=np.float64)
    widths = ranges[active, 1] - ranges[active, 0]
    sample_error = np.mean(
        np.abs(prediction[..., active] - target[..., active]) / widths, axis=(1, 2)
    )
    failed_samples = attempted_samples - valid_samples
    metrics.update(
        {
            "attempted_samples": attempted_samples,
            "valid_samples": valid_samples,
            "failed_samples": failed_samples,
            "failure_rate": failed_samples / attempted_samples,
            # One full normalized joint range is the explicit penalty for a row
            # that could not produce a bridge command.
            "failure_penalty_normalized_joint_mae": 1.0,
            "failure_adjusted_normalized_joint_mae": float(
                (np.sum(sample_error) + failed_samples) / attempted_samples
            ),
        }
    )
    for actuator in metrics["per_actuator"].values():
        actuator["failure_adjusted_normalized_mae"] = float(
            (actuator["normalized_mae"] * valid_samples + failed_samples) / attempted_samples
        )
    return metrics


def evaluate_adapter_cache(
    *,
    adapter: str | Path,
    cache: str | Path,
    device: str = "cpu",
    output: str | Path | None = None,
) -> dict[str, Any]:
    arrays = _cache_arrays(cache)
    artifact = load_adapter_artifact(adapter, device=device, require_weights=True)
    bridge_valid = arrays["bridge_valid"].astype(bool)
    if not np.any(bridge_valid):
        raise EmbodimentError("adapter cache contains no valid bridge predictions")
    current = torch.as_tensor(
        arrays["current_state"][bridge_valid], dtype=torch.float32, device=device
    )
    bridge = torch.as_tensor(
        arrays["bridge_chunk"][bridge_valid], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        corrected, _ = artifact.residual(current, bridge)
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "hold": (
            np.repeat(arrays["current_state"][:, None, :], arrays["target_chunk"].shape[1], axis=1),
            np.ones(len(arrays["current_state"]), dtype=bool),
        ),
        "bridge": (
            arrays.get("raw_bridge_chunk", arrays["bridge_chunk"])[bridge_valid],
            bridge_valid,
        ),
        "adapter": (corrected.cpu().numpy(), bridge_valid),
    }
    if "raw_bridge_chunk" in arrays:
        predictions["resampled_bridge"] = (arrays["bridge_chunk"][bridge_valid], bridge_valid)
    if "baseline_chunk" in arrays:
        baseline_valid = arrays["baseline_valid"].astype(bool)
        predictions["current_5k"] = (arrays["baseline_chunk"][baseline_valid], baseline_valid)
    report: dict[str, Any] = {
        "schema_version": 3,
        "adapter_id": artifact.spec.adapter_id,
        "cache_sha256": sha256_file(Path(cache)),
        "failure_contract": {
            "valid_row_metrics": "computed only where that prediction path produced a finite chunk",
            "failure_adjusted_normalized_joint_mae": (
                "mean per-sample normalized joint MAE with each failed row assigned penalty 1.0"
            ),
            "per_actuator": (
                "all 16 dimensions in target-vector order; arm joints are radians and grippers "
                "are metres of calibrated aperture. Gripper metrics remain meaningful even when "
                "the residual is configured not to modify them"
            ),
        },
        "splits": {},
    }
    for split in ("train", "val", "test"):
        selected = arrays["split"] == split
        if not np.any(selected):
            continue
        attempted = int(np.sum(selected))
        split_report: dict[str, Any] = {}
        for name, (prediction, prediction_valid) in predictions.items():
            valid_selected = selected & prediction_valid
            if not np.any(valid_selected):
                raise EmbodimentError(f"{name} produced no valid {split} predictions")
            # Prediction arrays for failure-prone paths are compacted to valid rows.
            if len(prediction) == len(prediction_valid):
                split_prediction = prediction[valid_selected]
            else:
                split_prediction = prediction[selected[prediction_valid]]
            split_report[name] = _prediction_report(
                split_prediction,
                arrays["target_chunk"][valid_selected],
                arrays["current_state"][valid_selected],
                artifact,
                attempted_samples=attempted,
            )
        split_report["attempted_samples"] = attempted
        report["splits"][split] = split_report
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def validate_adapter_contract(
    *,
    manifest: str | Path,
    base_policy: str | Path | None = None,
    dataset_path: str | Path | None = None,
    split_info: str | Path | None = None,
    stride: int = 500,
    video_backend: str = "pyav",
) -> dict[str, Any]:
    if stride < 1:
        raise ValueError("stride must be positive")
    artifact = load_adapter_artifact(
        manifest,
        base_policy_dir=_pretrained_dir(base_policy) if base_policy else None,
        require_weights=False,
    )
    report: dict[str, Any] = {
        "adapter_id": artifact.spec.adapter_id,
        "deployment_status": artifact.spec.deployment_status,
        "reference_model_sha256": artifact.spec.reference_model.sha256,
        "target_model_sha256": artifact.spec.target_model.sha256,
        "base_policy_verified": base_policy is not None,
    }
    if dataset_path is None:
        return report
    if split_info is None:
        raise EmbodimentError("dataset validation requires the pinned split_info")
    verify_target_dataset(artifact.spec, Path(dataset_path), Path(split_info))
    dataset_wrapper = EvaluationDataset(Path(dataset_path), video_backend=video_backend)
    if tuple(dataset_wrapper.joint_names) != artifact.spec.target_vector.names:
        raise EmbodimentError("dataset action names do not match target vector contract")
    states = dataset_wrapper.dataset.hf_dataset["observation.state"]
    accepted = 0
    rejected: list[dict[str, Any]] = []
    for index in range(0, len(states), stride):
        try:
            artifact.bridge.target_state_to_policy(np.asarray(states[index]))
            accepted += 1
        except Exception as exc:
            rejected.append({"index": index, "error": str(exc)})
    total = accepted + len(rejected)
    report["dataset"] = {
        "path": str(Path(dataset_path).resolve()),
        "sample_stride": stride,
        "accepted": accepted,
        "rejected": len(rejected),
        "acceptance_rate": accepted / total if total else 0.0,
        "rejections": rejected,
        "target_data_revision": artifact.spec.target_data.revision,
        "trim_manifest_sha256": artifact.spec.target_data.trim_manifest_sha256,
        "split_info_sha256": artifact.spec.target_data.split_info_sha256,
    }
    return report
