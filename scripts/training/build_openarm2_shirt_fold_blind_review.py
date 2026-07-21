#!/usr/bin/env python3
"""Build anonymized stage-outcome panels for the OpenArm2 blind label pass."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v1.json"
DEFAULT_TRIM_MANIFEST = (
    ROOT / "datasets/shirt-fold/lerobot-hf-phase-aligned/meta/trim_manifest.json"
)
DEFAULT_VIDEO = (
    ROOT / "datasets/shirt-fold/lerobot-pi05-relative/videos/observation.images.base/"
    "chunk-000/file-000.mp4"
)

STAGE_CRITERIA = {
    "side_one": (
        "lateral_placement",
        "sleeve_side_containment",
        "longitudinal_alignment",
        "flatness_body_preservation",
    ),
    "side_two": (
        "second_side_placement",
        "second_sleeve_containment",
        "preservation_bilateral_alignment",
        "flatness_straightness",
    ),
    "bottom_to_top": (
        "fold_placement_coverage",
        "rectangularity_compactness",
        "layer_edge_containment",
        "flatness_stability",
    ),
}
FRAME_OFFSETS_FROM_STAGE_END = (46, 31, 16, 1)


@dataclass(frozen=True)
class ReviewItem:
    token: str
    episode_index: int
    stage: str
    retained_start_frame: int
    retained_end_frame: int
    source_frames: tuple[int, ...]


def _load_items(manifest_path: Path, trim_path: Path, seed: int) -> list[ReviewItem]:
    manifest = json.loads(manifest_path.read_text())
    trims = {
        int(item["episode_index"]): item for item in json.loads(trim_path.read_text())["episodes"]
    }
    items: list[ReviewItem] = []
    for episode in manifest["episodes"]:
        episode_index = int(episode["episode_index"])
        trim = trims[episode_index]
        source_offset = int(trim["source_from_index"]) + int(trim["start"])
        for stage in episode["stages"]:
            start = int(stage["start_frame"])
            end = int(stage["end_frame"])
            local_frames = tuple(max(start, end - delta) for delta in FRAME_OFFSETS_FROM_STAGE_END)
            token = hashlib.sha256(f"{seed}:{stage['name']}:{episode_index}".encode()).hexdigest()[
                :10
            ]
            items.append(
                ReviewItem(
                    token=token,
                    episode_index=episode_index,
                    stage=str(stage["name"]),
                    retained_start_frame=start,
                    retained_end_frame=end,
                    source_frames=tuple(source_offset + frame for frame in local_frames),
                )
            )
    return items


def _extract_frames(video: Path, frames: list[int]) -> dict[int, Image.Image]:
    targets = set(frames)
    decoded: dict[int, Image.Image] = {}
    with av.open(str(video)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index in targets:
                decoded[frame_index] = frame.to_image().convert("RGB")
                if len(decoded) == len(targets):
                    break
    missing = sorted(targets - set(decoded))
    if missing:
        raise RuntimeError(f"video is missing requested frames: {missing}")
    return decoded


def _panel(item: ReviewItem, decoded: dict[int, Image.Image]) -> Image.Image:
    panel = Image.new("RGB", (960, 570), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=20)
    draw.text((12, 5), f"token={item.token}  stage={item.stage}", fill="black", font=font)
    for index, frame in enumerate(item.source_frames):
        image = decoded[frame]
        x = (index % 2) * 480
        y = 30 + (index // 2) * 270
        panel.paste(image, (x, y))
    return panel


def _write_sheet(panels: list[Image.Image], output: Path) -> None:
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 960, rows * 570), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * 960, (index // columns) * 570))
    sheet.save(output)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--trim-manifest", type=Path, default=DEFAULT_TRIM_MANIFEST)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)
    items = _load_items(args.manifest, args.trim_manifest, args.seed)
    all_frames = [frame for item in items for frame in item.source_frames]
    decoded = _extract_frames(args.video, all_frames)
    panels = {item.token: _panel(item, decoded) for item in items}

    for stage in STAGE_CRITERIA:
        stage_items = sorted(
            (item for item in items if item.stage == stage),
            key=lambda item: hashlib.sha256(f"{args.seed}:order:{item.token}".encode()).hexdigest(),
        )
        for page, page_start in enumerate(range(0, len(stage_items), 12), start=1):
            page_items = stage_items[page_start : page_start + 12]
            _write_sheet(
                [panels[item.token] for item in page_items],
                output / f"{stage}-page-{page}.png",
            )

    mapping = {
        "schema_version": "openarm2.blind-stage-review.v1",
        "seed": args.seed,
        "source_manifest": _portable_path(args.manifest),
        "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "trim_manifest": _portable_path(args.trim_manifest),
        "trim_manifest_sha256": hashlib.sha256(args.trim_manifest.read_bytes()).hexdigest(),
        "source_video": _portable_path(args.video),
        "source_video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "review_camera": "base",
        "frame_offsets_from_stage_end": list(FRAME_OFFSETS_FROM_STAGE_END),
        "items": [
            {
                "token": item.token,
                "episode_index": item.episode_index,
                "stage": item.stage,
                "retained_start_frame": item.retained_start_frame,
                "retained_end_frame_exclusive": item.retained_end_frame,
                "source_frames_global": list(item.source_frames),
            }
            for item in items
        ],
    }
    (output / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")

    with (output / "blind_scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "token",
                "stage",
                *[f"criterion_{index + 1}" for index in range(4)],
                "quality_score",
                "visibility_confidence",
                "review_note",
            ]
        )
        for stage in STAGE_CRITERIA:
            stage_items = sorted(
                (item for item in items if item.stage == stage),
                key=lambda item: hashlib.sha256(
                    f"{args.seed}:order:{item.token}".encode()
                ).hexdigest(),
            )
            for item in stage_items:
                writer.writerow([item.token, stage, "", "", "", "", "", "", ""])

    (output / "criteria.json").write_text(json.dumps(STAGE_CRITERIA, indent=2) + "\n")
    print(json.dumps({"output": str(output), "items": len(items)}, indent=2))


if __name__ == "__main__":
    main()
