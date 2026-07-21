#!/usr/bin/env python3
"""Build the leakage-safe Robometer view of the 33 trimmed shirt-fold episodes.

This is an orchestration adapter for the released Robometer repository.  It does
not implement or modify a reward model.  Full episodes supervise temporal
progress; the three annotated stage clips supervise partial progress and
same-task preferences from the blinded 1--5 quality rubric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIORITY = ROOT / "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v2.json"
DEFAULT_CONTRACT = (
    ROOT / "configs/training/robometer_manifests/openarm2_shirt_fold_robometer_v1.json"
)


@dataclass(frozen=True)
class TrajectorySpec:
    id: str
    split: str
    episode_index: int
    kind: str
    stage: str | None
    task: str
    start_frame: int
    end_frame: int
    quality_score: int | None
    partial_success: float
    data_source: str

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def relative_video_path(self) -> str:
        return f"videos/{self.split}/{self.id}.mp4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def build_trajectory_specs(
    priority: dict[str, Any], contract: dict[str, Any]
) -> list[TrajectorySpec]:
    expected_order = contract["stage_order"]
    split_by_episode: dict[int, str] = {}
    for split, episodes in contract["split"].items():
        if split == "seed":
            continue
        for episode_index in episodes:
            index = int(episode_index)
            if index in split_by_episode:
                raise ValueError(f"episode {index} occurs in more than one split")
            split_by_episode[index] = str(split)

    episode_rows = priority["episodes"]
    episode_ids = {int(row["episode_index"]) for row in episode_rows}
    if episode_ids != set(split_by_episode):
        raise ValueError("Robometer split must cover every priority-manifest episode exactly once")

    full_task = contract["tasks"]["full_episode"]
    stage_tasks = contract["tasks"]["stages"]
    specs: list[TrajectorySpec] = []
    global_cursor = 0
    for episode in episode_rows:
        episode_index = int(episode["episode_index"])
        frame_count = int(episode["frame_count"])
        split = split_by_episode[episode_index]
        source = f"openarm2_roboreward_{split}"
        stages = episode["stages"]
        if [stage["name"] for stage in stages] != expected_order:
            raise ValueError(f"episode {episode_index} stage order differs from contract")
        if int(stages[0]["start_frame"]) != 0 or int(stages[-1]["end_frame"]) != frame_count:
            raise ValueError(f"episode {episode_index} stages do not span the trimmed episode")
        for left, right in zip(stages, stages[1:]):
            if int(left["end_frame"]) != int(right["start_frame"]):
                raise ValueError(f"episode {episode_index} stages are not contiguous")

        specs.append(
            TrajectorySpec(
                id=f"episode-{episode_index:03d}-full",
                split=split,
                episode_index=episode_index,
                kind="full_episode",
                stage=None,
                task=full_task,
                start_frame=global_cursor,
                end_frame=global_cursor + frame_count,
                quality_score=None,
                partial_success=1.0,
                data_source=source,
            )
        )
        for stage in stages:
            name = str(stage["name"])
            quality = int(stage["quality_score"])
            if quality not in {1, 2, 3, 4, 5}:
                raise ValueError(
                    f"episode {episode_index} stage {name} has invalid quality {quality}"
                )
            specs.append(
                TrajectorySpec(
                    id=f"episode-{episode_index:03d}-{name}",
                    split=split,
                    episode_index=episode_index,
                    kind="stage_clip",
                    stage=name,
                    task=stage_tasks[name],
                    start_frame=global_cursor + int(stage["start_frame"]),
                    end_frame=global_cursor + int(stage["end_frame"]),
                    quality_score=quality,
                    partial_success=quality / 5.0,
                    data_source=source,
                )
            )
        global_cursor += frame_count

    expected_frames = int(priority["dataset"]["frames"])
    if global_cursor != expected_frames:
        raise ValueError(f"episode frame counts total {global_cursor}, expected {expected_frames}")
    return specs


def validate_specs(specs: list[TrajectorySpec], contract: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    episodes: dict[str, set[int]] = {}
    for split in ("train", "validation", "test"):
        split_specs = [spec for spec in specs if spec.split == split]
        counts[split] = {
            "episodes": len({spec.episode_index for spec in split_specs}),
            "full_episodes": sum(spec.kind == "full_episode" for spec in split_specs),
            "stage_clips": sum(spec.kind == "stage_clip" for spec in split_specs),
            "trajectories": len(split_specs),
        }
        episodes[split] = {spec.episode_index for spec in split_specs}
        expected_episode_count = len(contract["split"][split])
        if counts[split] != {
            "episodes": expected_episode_count,
            "full_episodes": expected_episode_count,
            "stage_clips": expected_episode_count * 3,
            "trajectories": expected_episode_count * 4,
        }:
            raise ValueError(f"unexpected {split} trajectory counts: {counts[split]}")
    if any(
        episodes[a] & episodes[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("episode leakage detected between Robometer splits")
    return {"counts": counts, "leakage_free": True}


def _video_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def _render_clip(
    source: Path, destination: Path, spec: TrajectorySpec, *, fps: int
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _video_frame_count(destination) == spec.frame_count:
        return
    temporary = destination.with_suffix(".tmp.mp4")
    temporary.unlink(missing_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{spec.start_frame / fps:.9f}",
            "-i",
            str(source),
            "-frames:v",
            str(spec.frame_count),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "passthrough",
            str(temporary),
        ],
        check=True,
    )
    actual = _video_frame_count(temporary)
    if actual != spec.frame_count:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"clip {spec.id} has {actual} frames, expected {spec.frame_count}")
    temporary.replace(destination)


def materialize(
    *,
    dataset_root: Path,
    output_root: Path,
    priority_path: Path,
    contract_path: Path,
    hub_repo: str | None,
) -> dict[str, Any]:
    priority = _load_json(priority_path)
    contract = _load_json(contract_path)
    specs = build_trajectory_specs(priority, contract)
    audit = validate_specs(specs, contract)
    api = None
    if hub_repo:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(hub_repo, repo_type="dataset", private=True, exist_ok=True)
    source_video = dataset_root / "videos/observation.images.base/chunk-000/file-000.mp4"
    if not source_video.is_file():
        raise FileNotFoundError(f"missing base-camera video: {source_video}")
    source_frames = _video_frame_count(source_video)
    expected_frames = int(priority["dataset"]["frames"])
    fps = int(priority["dataset"]["fps"])
    if source_frames != expected_frames:
        raise ValueError(f"source video has {source_frames} frames, expected {expected_frames}")

    for spec in specs:
        _render_clip(
            source_video,
            output_root / spec.relative_video_path,
            spec,
            fps=fps,
        )

    from sentence_transformers import SentenceTransformer

    from datasets import Dataset

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tasks = sorted({spec.task for spec in specs})
    vectors = {task: encoder.encode(task).astype("float32").tolist() for task in tasks}
    for split in ("train", "validation", "test"):
        rows = []
        for spec in specs:
            if spec.split != split:
                continue
            rows.append(
                {
                    "id": spec.id,
                    "task": spec.task,
                    "lang_vector": vectors[spec.task],
                    "data_source": spec.data_source,
                    "frames": spec.relative_video_path,
                    "is_robot": True,
                    "quality_label": "successful",
                    "partial_success": spec.partial_success,
                }
            )
        if hub_repo:
            Dataset.from_list(rows).push_to_hub(
                hub_repo,
                config_name=f"openarm2_{split}",
                split="train",
                private=True,
                token=os.environ.get("HF_TOKEN"),
                commit_message=f"Add leakage-safe OpenARM2 Robometer {split} split",
            )

    clip_hashes = {spec.id: _sha256(output_root / spec.relative_video_path) for spec in specs}
    result = {
        "schema_version": "openarm2.robometer-dataset-audit.v1",
        "official_robometer": contract["official_robometer"],
        "source_dataset": contract["source_dataset"],
        "source_video": {
            "path": str(source_video),
            "frames": source_frames,
            "sha256": _sha256(source_video),
        },
        "priority_manifest": {
            "path": str(priority_path),
            "sha256": _sha256(priority_path),
        },
        "split": contract["split"],
        "supervision": contract["supervision"],
        "trajectory_audit": audit,
        "trajectories": [asdict(spec) for spec in specs],
        "clip_sha256": clip_hashes,
    }
    audit_path = output_root / "robometer_dataset_audit_v1.json"
    audit_path.write_text(json.dumps(result, indent=2) + "\n")
    if hub_repo:
        assert api is not None
        api.upload_folder(
            folder_path=output_root / "videos",
            path_in_repo="videos",
            repo_id=hub_repo,
            repo_type="dataset",
            commit_message="Upload exact-frame OpenARM2 Robometer clips",
        )
        result["hub_repo"] = hub_repo
        result["hub_content_revision"] = api.dataset_info(hub_repo).sha
        preprocess_config = {
            "train_datasets": [hub_repo],
            "train_subsets": [["openarm2_train"]],
            "eval_datasets": [hub_repo],
            "eval_subsets": [["openarm2_validation", "openarm2_test"]],
            "max_frames_for_preprocessing": 64,
            "video_frame_sampling": "uniform",
            "num_proc": 1,
            "num_threads": 8,
            "force_reprocess": False,
            "cache_dir": str(output_root.parent.parent / "robometer_processed"),
            "precompute_embeddings": False,
            "embeddings_cache_dir": "embeddings",
            "dinov2_model": "facebook/dinov2-base",
            "sentence_model": "sentence-transformers/all-MiniLM-L12-v2",
            "embedding_batch_size": 32,
        }
        import yaml

        preprocess_path = output_root / "preprocess_openarm2.yaml"
        preprocess_path.write_text(yaml.safe_dump(preprocess_config, sort_keys=False))
        result["preprocess_config"] = str(preprocess_path)
        cutoff_path = output_root / "dataset_success_cutoff.txt"
        cutoff_path.write_text(
            "openarm2_roboreward_train,1.0\n"
            "openarm2_roboreward_validation,1.0\n"
            "openarm2_roboreward_test,1.0\n"
        )
        result["success_cutoff_file"] = str(cutoff_path)
        audit_path.write_text(json.dumps(result, indent=2) + "\n")
        api.upload_file(
            path_or_fileobj=audit_path,
            path_in_repo=audit_path.name,
            repo_id=hub_repo,
            repo_type="dataset",
            commit_message="Add OpenARM2 Robometer dataset audit",
        )
        result["hub_revision"] = api.dataset_info(hub_repo).sha
        audit_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--priority-manifest", type=Path, default=DEFAULT_PRIORITY)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--hub-repo")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the split and supervision contract without rendering videos",
    )
    args = parser.parse_args()

    priority = _load_json(args.priority_manifest)
    contract = _load_json(args.contract)
    specs = build_trajectory_specs(priority, contract)
    if args.check:
        print(json.dumps(validate_specs(specs, contract), indent=2))
        return
    result = materialize(
        dataset_root=args.dataset_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        priority_path=args.priority_manifest.expanduser().resolve(),
        contract_path=args.contract.expanduser().resolve(),
        hub_repo=args.hub_repo,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
