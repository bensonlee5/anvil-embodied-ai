#!/usr/bin/env python3
"""Score all trimmed shirt-fold frames with released Robometer inference.

The released evaluator defines a frame-step prediction as four uniformly
spaced prefix frames ending at the frame being scored.  We evaluate those
prefixes at a fixed one-second cadence, then linearly interpolate within each
episode to create the full-frame parquet required by native RA-BC.  The sparse
anchors are retained in the output for auditability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIORITY = ROOT / "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v2.json"
DEFAULT_DATASET = ROOT / "datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-v1"
DEFAULT_TASK = (
    "Fold the T-shirt in three steps: fold one side, fold the other side, "
    "then fold the bottom to the top."
)


def anchor_indices(frame_count: int, stride: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    anchors = list(range(0, frame_count, stride))
    if anchors[-1] != frame_count - 1:
        anchors.append(frame_count - 1)
    return anchors


def prefix_context_indices(anchor: int, context_frames: int = 4) -> list[int]:
    if anchor < 0:
        raise ValueError("anchor must be nonnegative")
    if context_frames <= 0:
        raise ValueError("context_frames must be positive")
    if context_frames == 1:
        return [anchor]
    return [round(i * anchor / (context_frames - 1)) for i in range(context_frames)]


def interpolate_progress(
    frame_count: int, anchors: list[int], predictions: list[float]
) -> list[float]:
    import numpy as np

    if len(anchors) != len(predictions):
        raise ValueError("anchors and predictions must have the same length")
    if anchors[0] != 0 or anchors[-1] != frame_count - 1:
        raise ValueError("anchors must include the first and last frame")
    if any(right <= left for left, right in zip(anchors, anchors[1:])):
        raise ValueError("anchors must be strictly increasing")
    values = np.asarray(predictions, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Robometer anchor predictions must be finite")
    dense = np.interp(
        np.arange(frame_count, dtype=np.float64),
        np.asarray(anchors, dtype=np.float64),
        np.clip(values, 0.0, 1.0),
    )
    return dense.astype("float32").tolist()


def _load_priority(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def score(
    *,
    model_path: str,
    dataset_root: Path,
    priority_path: Path,
    output_path: Path,
    stride: int,
    batch_size: int,
    task: str,
) -> dict:
    import decord
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator

    priority = _load_priority(priority_path)
    video_path = dataset_root / "videos/observation.images.base/chunk-000/file-000.mp4"
    decord.bridge.set_bridge("native")
    reader = decord.VideoReader(str(video_path), num_threads=4)
    expected_frames = int(priority["dataset"]["frames"])
    if len(reader) != expected_frames:
        raise ValueError(f"source video has {len(reader)} frames, expected {expected_frames}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=model_path,
        device=device,
    )
    reward_model.eval()
    collator = setup_batch_collator(
        processor,
        tokenizer,
        exp_config,
        is_eval=True,
    )
    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config
        else False
    )
    num_bins = getattr(loss_config, "progress_discrete_bins", None) or getattr(
        exp_config.model, "progress_discrete_bins", 10
    )

    rows = []
    anchor_records = []
    global_cursor = 0
    with torch.inference_mode():
        for episode in priority["episodes"]:
            episode_index = int(episode["episode_index"])
            frame_count = int(episode["frame_count"])
            anchors = anchor_indices(frame_count, stride)
            samples = []
            for anchor in anchors:
                local_context = prefix_context_indices(anchor)
                global_context = [global_cursor + index for index in local_context]
                frames = reader.get_batch(global_context).asnumpy()
                trajectory = Trajectory(
                    frames=frames,
                    frames_shape=tuple(frames.shape),
                    task=task,
                    id=f"episode-{episode_index:03d}-frame-{anchor:05d}",
                    metadata={
                        "episode_index": episode_index,
                        "frame_index": anchor,
                        "subsequence_length": len(frames),
                    },
                    video_embeddings=None,
                )
                samples.append(ProgressSample(trajectory=trajectory, sample_type="progress"))

            predictions = []
            for start in range(0, len(samples), batch_size):
                batch = collator(samples[start : start + batch_size])
                inputs = batch["progress_inputs"]
                for key, value in inputs.items():
                    if hasattr(value, "to"):
                        inputs[key] = value.to(device)
                result = compute_batch_outputs(
                    reward_model,
                    tokenizer,
                    inputs,
                    sample_type="progress",
                    is_discrete_mode=is_discrete,
                    num_bins=num_bins,
                )
                for values in result.get("progress_pred", []):
                    if not values:
                        raise ValueError("Robometer returned an empty progress prediction")
                    predictions.append(float(values[-1]))
            if len(predictions) != len(anchors):
                raise ValueError(
                    f"episode {episode_index} returned {len(predictions)} anchors, "
                    f"expected {len(anchors)}"
                )

            dense = interpolate_progress(frame_count, anchors, predictions)
            anchor_lookup = dict(zip(anchors, predictions, strict=True))
            for local_frame, progress in enumerate(dense):
                rows.append(
                    {
                        "index": global_cursor + local_frame,
                        "episode_index": episode_index,
                        "frame_index": local_frame,
                        "progress_dense": progress,
                        "is_anchor": local_frame in anchor_lookup,
                        "anchor_progress": anchor_lookup.get(local_frame),
                    }
                )
            anchor_records.append(
                {
                    "episode_index": episode_index,
                    "frame_count": frame_count,
                    "anchor_frames": anchors,
                    "anchor_progress": predictions,
                }
            )
            global_cursor += frame_count

    frame = pd.DataFrame.from_records(rows)
    if len(frame) != expected_frames:
        raise ValueError(f"scored {len(frame)} frames, expected {expected_frames}")
    metadata = {
        b"reward_model_kind": b"released_robometer",
        b"model_path": model_path.encode(),
        b"scoring_method": b"four-frame-prefix-at-fixed-stride-plus-linear-interpolation",
        b"anchor_stride_frames": str(stride).encode(),
        b"priority_manifest_sha256": __import__("hashlib")
        .sha256(priority_path.read_bytes())
        .hexdigest()
        .encode(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(metadata)
    pq.write_table(table, output_path)
    anchors_path = output_path.with_name(f"{output_path.stem}_anchors.json")
    anchors_path.write_text(
        json.dumps(
            {
                "schema_version": "openarm2.robometer-progress-anchors.v1",
                "model_path": model_path,
                "task": task,
                "stride_frames": stride,
                "context_frames": 4,
                "episodes": anchor_records,
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "frames": len(frame),
        "episodes": len(anchor_records),
        "anchors": sum(len(record["anchor_frames"]) for record in anchor_records),
        "progress_path": str(output_path),
        "anchors_path": str(anchors_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--priority-manifest", type=Path, default=DEFAULT_PRIORITY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--task", default=DEFAULT_TASK)
    args = parser.parse_args()
    result = score(
        model_path=args.model_path,
        dataset_root=args.dataset_root.expanduser().resolve(),
        priority_path=args.priority_manifest.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        stride=args.stride,
        batch_size=args.batch_size,
        task=args.task,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
