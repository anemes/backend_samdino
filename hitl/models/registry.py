"""Model registry: checkpoint management, metrics history, model selection."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunMetrics:
    """Metrics from a single training run."""

    run_id: str
    iteration: int
    epoch: int
    train_loss: float
    train_mIoU: float
    val_loss: float
    val_mIoU: float
    test_mIoU: Optional[float] = None
    per_class_iou: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    num_train_tiles: int = 0
    num_val_tiles: int = 0
    num_classes: int = 0
    training_time_s: float = 0.0
    learning_rate: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class CheckpointRecord:
    """Metadata for a saved checkpoint."""

    checkpoint_path: str
    run_id: str
    iteration: int
    best_val_mIoU: float
    num_classes: int
    class_names: List[str]
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ModelRegistry:
    """Manages checkpoints and training metrics for a project.

    Directory structure:
        {checkpoint_dir}/{project_id}/
            registry.json          # list of CheckpointRecords
            metrics.json           # list of RunMetrics
            run_{run_id}/
                best.pt            # best val_mIoU checkpoint
                latest.pt          # latest checkpoint
    """

    def __init__(self, checkpoint_dir: str | Path, project_id: str):
        self.root = Path(checkpoint_dir) / project_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.root / "registry.json"
        self._metrics_path = self.root / "metrics.json"

    def _load_json(self, path: Path) -> list:
        if path.exists():
            return json.loads(path.read_text())
        return []

    def _save_json(self, path: Path, data: list) -> None:
        path.write_text(json.dumps(data, indent=2, default=str))

    def save_checkpoint(
        self,
        run_id: str,
        iteration: int,
        checkpoint_path: Path,
        val_mIoU: float,
        num_classes: int,
        class_names: List[str],
        config_snapshot: Optional[dict] = None,
        is_best: bool = False,
    ) -> CheckpointRecord:
        """Register a saved checkpoint."""
        run_dir = self.root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Copy checkpoint to registry
        target_name = "best.pt" if is_best else "latest.pt"
        target = run_dir / target_name
        shutil.copy2(checkpoint_path, target)

        record = CheckpointRecord(
            checkpoint_path=str(target),
            run_id=run_id,
            iteration=iteration,
            best_val_mIoU=val_mIoU,
            num_classes=num_classes,
            class_names=class_names,
            config_snapshot=config_snapshot or {},
        )

        records = self._load_json(self._registry_path)
        records.append(asdict(record))
        self._save_json(self._registry_path, records)

        return record

    def log_metrics(self, metrics: RunMetrics) -> None:
        """Append metrics from a training epoch/run."""
        all_metrics = self._load_json(self._metrics_path)
        all_metrics.append(asdict(metrics))
        self._save_json(self._metrics_path, all_metrics)

    def get_metrics(self, run_id: Optional[str] = None) -> List[dict]:
        """Get all metrics, optionally filtered by run_id."""
        all_metrics = self._load_json(self._metrics_path)
        if run_id:
            return [m for m in all_metrics if m.get("run_id") == run_id]
        return all_metrics

    def list_checkpoints(self) -> List[dict]:
        """List all registered checkpoints."""
        return self._load_json(self._registry_path)

    def get_best_checkpoint(self) -> Optional[dict]:
        """Get the checkpoint with highest val_mIoU across all runs."""
        records = self._load_json(self._registry_path)
        best = [r for r in records if "best" in r.get("checkpoint_path", "")]
        if not best:
            return None
        return max(best, key=lambda r: r.get("best_val_mIoU", 0))

    def get_checkpoint_path(self, run_id: str, which: str = "best") -> Optional[Path]:
        """Get path to a specific checkpoint."""
        p = self.root / f"run_{run_id}" / f"{which}.pt"
        return p if p.exists() else None

    def get_checkpoint_record(self, run_id: str, which: str = "best") -> Optional[dict]:
        """Get the registry record for a specific run and checkpoint type.

        Looks up by run_id (not stored path), so it works even when the data
        directory has moved between environments (local → Docker → ACA).
        Prefers the record whose stored path contains *which* ('best'/'latest').
        Falls back to any record for the run if no type-specific match is found.
        """
        records = self._load_json(self._registry_path)
        fallback = None
        for rec in records:
            if rec.get("run_id") != run_id:
                continue
            if which in rec.get("checkpoint_path", ""):
                return rec
            if fallback is None:
                fallback = rec
        return fallback

    def list_runs(self) -> List[str]:
        """List distinct run_ids that have checkpoints."""
        records = self._load_json(self._registry_path)
        seen = set()
        runs = []
        for r in records:
            rid = r.get("run_id")
            if rid and rid not in seen:
                seen.add(rid)
                runs.append(rid)
        return runs

    def delete_run(self, run_id: str) -> bool:
        """Delete a run's checkpoint directory and registry/metrics entries."""
        run_dir = self.root / f"run_{run_id}"
        if run_dir.exists():
            shutil.rmtree(run_dir)

        records = self._load_json(self._registry_path)
        records = [r for r in records if r.get("run_id") != run_id]
        self._save_json(self._registry_path, records)

        metrics = self._load_json(self._metrics_path)
        metrics = [m for m in metrics if m.get("run_id") != run_id]
        self._save_json(self._metrics_path, metrics)

        if self.get_production_run() == run_id:
            self.clear_production_run()

        return True

    def get_production_run(self) -> Optional[str]:
        """Return the run_id marked as production, or None."""
        prod_path = self.root / "production.json"
        if prod_path.exists():
            data = json.loads(prod_path.read_text())
            return data.get("run_id")
        return None

    def set_production_run(self, run_id: str) -> None:
        """Promote a run to production for this project."""
        run_dir = self.root / f"run_{run_id}"
        if not run_dir.exists():
            raise ValueError(f"Run '{run_id}' not found")
        data = {"run_id": run_id, "promoted_at": datetime.now().isoformat()}
        (self.root / "production.json").write_text(json.dumps(data, indent=2))

    def clear_production_run(self) -> None:
        """Remove production designation."""
        prod_path = self.root / "production.json"
        if prod_path.exists():
            prod_path.unlink()
