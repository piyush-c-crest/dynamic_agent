import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import Config


class RunPaths:
    """runs/{batch_id}/{run_id}/ and logs/{batch_id}/{run_id}/"""

    def __init__(
        self,
        run_id: str,
        batch_id: str,
        created_at: datetime,
        relative_path: Optional[str] = None,
    ):
        self.run_id = run_id
        self.batch_id = batch_id
        self.created_at = created_at
        self.relative_path = relative_path or f"{batch_id}/{run_id}"
        self.runs_dir = Config.RUNS_DIR / self.relative_path
        self.logs_dir = Config.LOGS_DIR / self.relative_path

    @property
    def workflow_file(self) -> Path:
        return self.runs_dir / Config.WORKFLOW_FILENAME

    @property
    def log_file(self) -> Path:
        return self.logs_dir / Config.LOG_FILENAME

    @property
    def outputs_dir(self) -> Path:
        return self.runs_dir / Config.OUTPUTS_DIRNAME

    def ensure_dirs(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


def build_run_paths(run_id: str, created_at: Optional[datetime] = None) -> RunPaths:
    created_at = created_at or datetime.now()
    batch_id = created_at.strftime(Config.BATCH_ID_FORMAT)
    return RunPaths(run_id=run_id, batch_id=batch_id, created_at=created_at)


class RunPathStore:
    def __init__(self, index_file: Path = Config.RUN_INDEX_FILE):
        self.index_file = index_file
        self.index_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> dict:
        if not self.index_file.exists():
            return {}
        with open(self.index_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_index(self, index: dict) -> None:
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    def register(self, run_id: str, created_at: Optional[datetime] = None) -> RunPaths:
        paths = build_run_paths(run_id, created_at)
        paths.ensure_dirs()

        index = self._load_index()
        index[run_id] = {
            "batch_id": paths.batch_id,
            "relative_path": paths.relative_path,
            "created_at": paths.created_at.isoformat(),
        }
        self._save_index(index)
        return paths

    def list_index(self) -> Dict[str, Any]:
        return self._load_index()

    def resolve(self, run_id: str) -> Optional[RunPaths]:
        index = self._load_index()
        entry = index.get(run_id)
        if entry:
            created_at = datetime.fromisoformat(entry["created_at"])
            relative_path = entry.get("relative_path")
            batch_id = entry.get("batch_id")
            if not batch_id and relative_path:
                batch_id = Path(relative_path).parts[0]
            elif not batch_id:
                batch_id = "unknown"
            return RunPaths(
                run_id=run_id,
                batch_id=batch_id,
                created_at=created_at,
                relative_path=relative_path,
            )

        for legacy in (
            Config.RUNS_DIR / run_id,
            *Config.RUNS_DIR.glob(f"*/*/{run_id}"),
            *Config.RUNS_DIR.glob(f"*/*/*/{run_id}"),
        ):
            if legacy.exists() and legacy.is_dir():
                batch_id = legacy.parent.name
                rel = str(legacy.relative_to(Config.RUNS_DIR)).replace("\\", "/")
                return RunPaths(
                    run_id=run_id,
                    batch_id=batch_id,
                    created_at=datetime.utcnow(),
                    relative_path=rel,
                )

        return None

    def require(self, run_id: str) -> RunPaths:
        paths = self.resolve(run_id)
        if not paths:
            raise FileNotFoundError(f"No storage path registered for run '{run_id}'.")
        return paths
