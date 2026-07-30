"""Candidate model versioning for AFMLF."""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ModelVersionRecord:
    version_id: str
    created_at: str
    configuration: dict[str, Any]
    training_dataset_summary: dict[str, Any]
    evaluation_summary: dict[str, Any]
    deployment_recommendation: str
    artifact_dir: str
    parent_version: str | None = None
    notes: str = ""


@dataclass
class ModelVersionRegistry:
    versions: list[ModelVersionRecord] = field(default_factory=list)

    def register(self, record: ModelVersionRecord) -> None:
        self.versions.append(record)

    def latest(self) -> ModelVersionRecord | None:
        return self.versions[-1] if self.versions else None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"versions": [asdict(v) for v in self.versions]}
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> ModelVersionRegistry:
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        versions = [ModelVersionRecord(**item) for item in raw.get("versions", [])]
        return cls(versions=versions)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bootstrap_incumbent_version(
    exp_dir: Path,
    config_snapshot: dict[str, Any],
    source_model_path: Path,
    source_stats_path: Path,
) -> ModelVersionRecord:
    version_dir = exp_dir / "versions" / "hybrid_v1"
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model_path, version_dir / "model.keras")
    shutil.copy2(source_stats_path, version_dir / "residual_stats.pkl")
    with source_stats_path.open("rb") as fh:
        stats = pickle.load(fh)
    record = ModelVersionRecord(
        version_id="hybrid_v1",
        created_at=utc_now_iso(),
        configuration=config_snapshot,
        training_dataset_summary={"source": "frozen_baseline_reference"},
        evaluation_summary={"source": "baseline_reference_evaluation_df"},
        deployment_recommendation="incumbent_production_model",
        artifact_dir=str(version_dir),
        parent_version=None,
        notes="Incumbent frozen Hybrid from completed forecasting research.",
    )
    (version_dir / "metadata.json").write_text(
        json.dumps(asdict(record), indent=2) + "\n"
    )
    _ = stats
    return record
