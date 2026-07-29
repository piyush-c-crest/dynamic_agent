import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.config import Config
from models.schemas import Goal, RunState, Task, TaskGraph
from storage.run_paths import RunPathStore

# Maps to architecture doc stages 1–8
ARCHITECTURE_STAGES = [
    "goal_intake",       # 1
    "task_graph",        # 2
    "agent_resolution",  # 3
    "execution",         # 4
    "shared_memory",     # 5
    "evaluation",        # 6
    "replanner",         # 7
    "result_assembly",   # 8
]


def _empty_stage() -> dict:
    return {"status": "pending", "started_at": None, "completed_at": None, "details": {}}


class WorkflowStore:
    """Single workflow.json per run — all stages, tasks, memory, and status for UI."""

    def __init__(self):
        self._paths = RunPathStore()

    def _load(self, run_id: str) -> dict:
        paths = self._paths.require(run_id)
        if not paths.workflow_file.exists():
            raise FileNotFoundError(f"Workflow for run {run_id} not found.")
        with open(paths.workflow_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, run_id: str, workflow: dict) -> None:
        paths = self._paths.require(run_id)
        workflow["updated_at"] = datetime.utcnow().isoformat()
        paths.runs_dir.mkdir(parents=True, exist_ok=True)
        with open(paths.workflow_file, "w", encoding="utf-8") as f:
            json.dump(workflow, f, indent=2)

    def create(self, run_id: str, user_prompt: str, storage_path: str, created_at: str) -> dict:
        workflow = {
            "run_id": run_id,
            "storage_path": storage_path,
            "user_prompt": user_prompt,
            "status": "running",
            "created_at": created_at,
            "updated_at": created_at,
            "limits": {
                "max_tasks": Config.MAX_TASKS,
                "max_replans": Config.MAX_REPLANS,
                "max_spawned_agents": Config.MAX_SPAWNED_AGENTS,
                "replans_used": 0,
                "agents_spawned": 0,
            },
            "stages": {name: _empty_stage() for name in ARCHITECTURE_STAGES},
            "tasks": [],
            "memory": {},
            "output_files": [],
            "errors": [],
            "final_response": None,
        }
        self._save(run_id, workflow)
        return workflow

    def begin_stage(self, run_id: str, stage: str) -> None:
        workflow = self._load(run_id)
        entry = workflow["stages"].setdefault(stage, _empty_stage())
        entry["status"] = "running"
        entry["started_at"] = datetime.utcnow().isoformat()
        self._save(run_id, workflow)

    def complete_stage(self, run_id: str, stage: str, **details) -> None:
        workflow = self._load(run_id)
        entry = workflow["stages"].setdefault(stage, _empty_stage())
        entry["status"] = "completed"
        entry["completed_at"] = datetime.utcnow().isoformat()
        entry.setdefault("details", {}).update(details)
        self._save(run_id, workflow)

    def fail_stage(self, run_id: str, stage: str, error: str, **details) -> None:
        workflow = self._load(run_id)
        entry = workflow["stages"].setdefault(stage, _empty_stage())
        entry["status"] = "failed"
        entry["completed_at"] = datetime.utcnow().isoformat()
        entry.setdefault("details", {}).update({"error": error, **details})
        self._save(run_id, workflow)

    def update_stage(self, run_id: str, stage: str, data: dict) -> None:
        workflow = self._load(run_id)
        entry = workflow["stages"].setdefault(stage, _empty_stage())
        entry.update(data)
        self._save(run_id, workflow)

    def record_agent_run(
        self,
        run_id: str,
        task_id: str,
        agent: str,
        status: str,
        output_key: Optional[str] = None,
        error: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
    ) -> None:
        workflow = self._load(run_id)
        entry = workflow["stages"].setdefault("agent_resolution", _empty_stage())
        entry.setdefault("details", {}).setdefault("runs", [])
        entry["details"]["runs"].append(
            {
                "task_id": task_id,
                "agent": agent,
                "status": status,
                "output_key": output_key,
                "error": error,
                "tools_used": tools_used or [],
                "completed_at": datetime.utcnow().isoformat(),
            }
        )
        # Append-only run log; wave status is owned by begin/complete_stage.
        # Never leave a finished stage stuck on "running" after a later complete.
        if entry.get("status") not in ("completed", "failed"):
            entry["status"] = "running"
        self._save(run_id, workflow)

    def set_goal(self, run_id: str, goal: Goal) -> None:
        workflow = self._load(run_id)
        goal_data = goal.model_dump()
        workflow["goal"] = goal_data
        self._save(run_id, workflow)
        self.complete_stage(run_id, "goal_intake", goal=goal_data)

    def set_task_graph(self, run_id: str, graph: TaskGraph) -> None:
        workflow = self._load(run_id)
        workflow["tasks"] = [t.model_dump() for t in graph.tasks]
        self._save(run_id, workflow)
        self.complete_stage(run_id, "task_graph", task_count=len(graph.tasks))

    def load_task_graph(self, run_id: str) -> Optional[TaskGraph]:
        paths = self._paths.resolve(run_id)
        if not paths or not paths.workflow_file.exists():
            return None
        workflow = self._load(run_id)
        tasks = workflow.get("tasks", [])
        return TaskGraph(tasks=[Task(**t) for t in tasks]) if tasks else None

    def update_task(self, run_id: str, task_id: str, **fields) -> None:
        workflow = self._load(run_id)
        for task in workflow["tasks"]:
            if task["id"] == task_id:
                task.update(fields)
                break
        self._save(run_id, workflow)

    def write_memory(self, run_id: str, key: str, value: Any) -> None:
        workflow = self._load(run_id)
        workflow["memory"][key] = value
        entry = workflow["stages"].setdefault("shared_memory", _empty_stage())
        entry["status"] = "completed"
        entry["completed_at"] = datetime.utcnow().isoformat()
        entry["details"] = {"keys": list(workflow["memory"].keys()), "latest_key": key}
        self._save(run_id, workflow)

    def read_memory(self, run_id: str, key: str) -> Optional[Any]:
        paths = self._paths.resolve(run_id)
        if not paths or not paths.workflow_file.exists():
            return None
        workflow = self._load(run_id)
        return workflow.get("memory", {}).get(key)

    def get_all_memory(self, run_id: str) -> Dict[str, Any]:
        paths = self._paths.resolve(run_id)
        if not paths or not paths.workflow_file.exists():
            return {}
        return self._load(run_id).get("memory", {})

    def load_run_state(self, run_id: str) -> Optional[RunState]:
        paths = self._paths.resolve(run_id)
        if not paths or not paths.workflow_file.exists():
            return None
        w = self._load(run_id)
        limits = w.get("limits", {})
        return RunState(
            run_id=run_id,
            storage_path=w.get("storage_path", ""),
            status=w.get("status", "running"),
            replans_used=limits.get("replans_used", 0),
            agents_spawned=limits.get("agents_spawned", 0),
            created_at=w.get("created_at", ""),
            updated_at=w.get("updated_at", ""),
        )

    def save_run_state(self, run_id: str, state: RunState) -> None:
        workflow = self._load(run_id)
        workflow["status"] = state.status
        workflow["limits"]["replans_used"] = state.replans_used
        workflow["limits"]["agents_spawned"] = state.agents_spawned
        self._save(run_id, workflow)

    def increment_replan_counter(self, run_id: str) -> int:
        workflow = self._load(run_id)
        workflow["limits"]["replans_used"] += 1
        count = workflow["limits"]["replans_used"]
        self._save(run_id, workflow)
        return count

    def increment_agent_spawn_counter(self, run_id: str) -> int:
        workflow = self._load(run_id)
        workflow["limits"]["agents_spawned"] += 1
        count = workflow["limits"]["agents_spawned"]
        self._save(run_id, workflow)
        return count

    def set_final_response(self, run_id: str, final_response: str, output_files: List[str]) -> None:
        workflow = self._load(run_id)
        workflow["final_response"] = final_response
        workflow["output_files"] = output_files
        self._save(run_id, workflow)

    def get_workflow(self, run_id: str) -> dict:
        return self._load(run_id)

    def add_error(self, run_id: str, error: str) -> None:
        workflow = self._load(run_id)
        workflow["errors"].append(error)
        self._save(run_id, workflow)
