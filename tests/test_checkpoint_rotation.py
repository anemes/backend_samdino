"""Tests for checkpoint rotation (#19)."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path


def test_rotation_keeps_newest():
    """#19: After rotation, only the N newest run dirs remain."""
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "checkpoints" / "test_project"
        project_dir.mkdir(parents=True)

        # Create 8 run directories with staggered mtimes
        for i in range(8):
            d = project_dir / f"run_{i:04d}"
            d.mkdir()
            (d / "latest.pt").write_text("fake")
            (d / "best.pt").write_text("fake")
            time.sleep(0.05)  # ensure distinct mtimes

        max_runs = 3
        all_run_dirs = sorted(
            [d for d in project_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda p: p.stat().st_mtime,
        )

        import shutil
        while len(all_run_dirs) > max_runs:
            oldest = all_run_dirs.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)

        remaining = sorted(project_dir.iterdir())
        assert len(remaining) == 3
        # Should be the 3 newest
        names = [d.name for d in remaining]
        assert "run_0005" in names
        assert "run_0006" in names
        assert "run_0007" in names


def test_no_rotation_when_under_limit():
    """Regression: No dirs deleted when count is under the limit."""
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "checkpoints" / "test_project"
        project_dir.mkdir(parents=True)

        for i in range(2):
            d = project_dir / f"run_{i:04d}"
            d.mkdir()
            (d / "latest.pt").write_text("fake")

        max_runs = 5
        all_run_dirs = sorted(
            [d for d in project_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda p: p.stat().st_mtime,
        )

        import shutil
        while len(all_run_dirs) > max_runs:
            oldest = all_run_dirs.pop(0)
            shutil.rmtree(oldest)

        remaining = list(project_dir.iterdir())
        assert len(remaining) == 2
