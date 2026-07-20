"""SARM progress audits must isolate policy holdouts from RA-BC statistics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from anvil_trainer.priority_sampling import PriorityManifest
from anvil_trainer.sarm_annotations import SARMAnnotationContract
from anvil_trainer.sarm_progress import audit_sarm_progress

ROOT = Path(__file__).resolve().parents[3]
PRIORITY = (
    ROOT / "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v1.json"
)
CONTRACT = ROOT / "configs/training/sarm_manifests/openarm2_shirt_fold_sarm_v1.json"


def _ideal_progress_frame(
    manifest: PriorityManifest,
    contract: SARMAnnotationContract,
) -> pd.DataFrame:
    offsets: dict[str, float] = {}
    cumulative = 0.0
    for name in contract.dense_stage_order:
        offsets[name] = cumulative
        cumulative += contract.temporal_proportions[name]

    rows = []
    global_index = 0
    for episode in manifest.episodes:
        for stage in episode.stages:
            length = stage.end_frame - stage.start_frame
            for offset in range(length):
                progress = offsets[stage.name] + contract.temporal_proportions[stage.name] * (
                    offset / max(length - 1, 1)
                )
                rows.append(
                    {
                        "index": global_index + stage.start_frame + offset,
                        "episode_index": episode.episode_index,
                        "frame_index": stage.start_frame + offset,
                        "progress_dense": progress,
                    }
                )
        global_index += episode.frame_count
    return pd.DataFrame(rows)


def test_audit_writes_only_train_episodes_for_native_rabc(tmp_path: Path) -> None:
    manifest = PriorityManifest.load(PRIORITY)
    contract = SARMAnnotationContract.load(CONTRACT, priority_manifest=manifest)
    full_path = tmp_path / "sarm_progress.parquet"
    training_path = tmp_path / "sarm_progress_train.parquet"
    _ideal_progress_frame(manifest, contract).to_parquet(full_path, index=False)

    audit = audit_sarm_progress(
        full_path,
        manifest=manifest,
        contract=contract,
        chunk_size=30,
        training_progress_path=training_path,
    )

    training_frame = pd.read_parquet(training_path)
    assert sorted(training_frame["episode_index"].unique()) == sorted(contract.train_episodes)
    assert not set(training_frame["episode_index"]).intersection(contract.validation_episodes)
    assert not set(training_frame["episode_index"]).intersection(contract.test_episodes)
    assert audit["training_progress"]["frames"] == len(training_frame)
    assert audit["rabc"]["recommended_kappa"] > 0
