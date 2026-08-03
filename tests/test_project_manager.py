"""Tests for ProjectManager fixes: path traversal (#18)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hitl.data.project_manager import ProjectManager


def test_path_traversal_blocked():
    """#18: Traversal attempts must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        pm = ProjectManager(tmp)
        try:
            pm.get_project_dir("../../etc")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


def test_normal_project_id_resolves():
    """Regression: normal project IDs resolve correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        pm = ProjectManager(tmp)
        result = pm.get_project_dir("my_project")
        assert result.is_absolute()
        assert str(result).startswith(str(Path(tmp).resolve()))
        assert result.name == "my_project"


def test_resolved_path_returned():
    """#18: get_project_dir returns a resolved (absolute) path."""
    with tempfile.TemporaryDirectory() as tmp:
        pm = ProjectManager(tmp)
        result = pm.get_project_dir("test")
        # Should be resolved — no '..' components
        assert ".." not in str(result)
        assert result == result.resolve()
