"""Project lifecycle management.

Each project is a directory under projects_dir containing:
- labels.gpkg (regions + annotations)
- classes.json sidecar (managed by LabelStore)
- project.json (metadata: name, created_at, description)
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProjectInfo:
    """Metadata about a project."""
    project_id: str
    name: str
    description: str = ""
    created_at: str = ""

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

    def create_project(self, project_id: str, name: str, description: str = "") -> ProjectInfo:
        """Create a new project directory."""
        project_dir = self.projects_dir / project_id
        if project_dir.exists():
            raise ValueError(f"Project '{project_id}' already exists")

        project_dir.mkdir(parents=True)

        info = ProjectInfo(
            project_id=project_id,
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
        )
        self._write_project_info(project_dir, info)

        logger.info("Created project '%s' at %s", project_id, project_dir)
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

    def get_project_dir(self, project_id: str) -> Path:
        """Get the directory path for a project."""
        return self.projects_dir / project_id

    def _read_project_info(self, project_dir: Path) -> Optional[ProjectInfo]:
        meta_path = project_dir / "project.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    data = json.load(f)
                return ProjectInfo(**data)
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
