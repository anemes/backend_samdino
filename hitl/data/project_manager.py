"""Project lifecycle management.

Each project is a directory under projects_dir containing:
- labels.gpkg (regions + annotations)
- classes.json sidecar (managed by LabelStore)
- project.json (metadata: name, created_at, description, owner, public, members)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Allow only alphanumeric, hyphen, underscore, and dot — no path separators or
# traversal sequences.  Starts with alphanumeric to prevent hidden-file abuse.
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_project_id(project_id: str) -> None:
    """Raise ValueError if project_id is not safe for use as a directory name."""
    if not _PROJECT_ID_RE.match(project_id):
        raise ValueError(
            f"Invalid project_id {project_id!r}. "
            "Must be 1–64 characters, start with alphanumeric, and contain only "
            "letters, digits, hyphens, underscores, or dots."
        )
    if project_id in (".", ".."):
        raise ValueError(f"Reserved project_id: {project_id!r}")

logger = logging.getLogger(__name__)


@dataclass
class ProjectInfo:
    """Metadata about a project."""
    project_id: str
    name: str
    description: str = ""
    created_at: str = ""
    owner: str = "_system"
    public: bool = False
    members: Dict[str, str] = field(default_factory=dict)  # {user_id: "contributor"|"user"}

    def to_dict(self) -> dict:
        return asdict(self)


class ProjectManager:
    """Manages project directories and metadata."""

    def __init__(self, projects_dir: str):
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> List[ProjectInfo]:
        """List all projects."""
        projects = []
        if not self.projects_dir.exists():
            return projects
        for d in sorted(self.projects_dir.iterdir()):
            if d.is_dir():
                info = self._read_project_info(d)
                if info:
                    projects.append(info)
        return projects

    def create_project(
        self,
        project_id: str,
        name: str,
        description: str = "",
        owner: str = "_system",
    ) -> ProjectInfo:
        """Create a new project directory."""
        validate_project_id(project_id)
        project_dir = self.projects_dir / project_id
        if project_dir.exists():
            raise ValueError(f"Project '{project_id}' already exists")

        project_dir.mkdir(parents=True)

        info = ProjectInfo(
            project_id=project_id,
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
            owner=owner,
            public=False,
            members={},
        )
        self._write_project_info(project_dir, info)

        logger.info("Created project '%s' (owner=%s) at %s", project_id, owner, project_dir)
        return info

    def get_project(self, project_id: str) -> Optional[ProjectInfo]:
        """Get project metadata."""
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            return None
        return self._read_project_info(project_dir)

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all its data."""
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            return False
        shutil.rmtree(project_dir)
        logger.info("Deleted project '%s'", project_id)
        return True

    def list_all_project_ids(self) -> List[str]:
        """List all project IDs (directory names under projects_dir)."""
        if not self.projects_dir.exists():
            return []
        return [d.name for d in sorted(self.projects_dir.iterdir()) if d.is_dir()]

    def get_project_dir(self, project_id: str) -> Path:
        """Get the directory path for a project.

        Raises ValueError if project_id would escape the projects root.
        """
        candidate = (self.projects_dir / project_id).resolve()
        root = self.projects_dir.resolve()
        if not str(candidate).startswith(str(root)):
            raise ValueError(f"project_id {project_id!r} escapes the projects directory")
        return self.projects_dir / project_id

    def set_project_info(self, project_id: str, **updates) -> Optional[ProjectInfo]:
        """Partially update project.json fields.  Returns updated info or None."""
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            return None
        info = self._read_project_info(project_dir)
        if info is None:
            return None
        for key, value in updates.items():
            if hasattr(info, key):
                setattr(info, key, value)
        self._write_project_info(project_dir, info)
        return info

    def _read_project_info(self, project_dir: Path) -> Optional[ProjectInfo]:
        meta_path = project_dir / "project.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    data = json.load(f)
                # Use explicit .get() to handle legacy project.json files that
                # predate the owner/public/members fields.
                return ProjectInfo(
                    project_id=data.get("project_id", project_dir.name),
                    name=data.get("name", project_dir.name),
                    description=data.get("description", ""),
                    created_at=data.get("created_at", ""),
                    owner=data.get("owner", "_system"),
                    public=data.get("public", False),
                    members=data.get("members", {}),
                )
            except Exception:
                pass
        # Fallback: directory exists but no metadata
        return ProjectInfo(
            project_id=project_dir.name,
            name=project_dir.name,
        )

    def _write_project_info(self, project_dir: Path, info: ProjectInfo) -> None:
        meta_path = project_dir / "project.json"
        with open(meta_path, "w") as f:
            json.dump(info.to_dict(), f, indent=2)
