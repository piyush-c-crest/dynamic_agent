from typing import Any, Dict, Optional

from app.models.schemas import RunState
from app.storage.workflow_store import WorkflowStore


class StateStore:
    def __init__(self):
        self._store = WorkflowStore()

    def load_run_state(self, run_id: str) -> Optional[RunState]:
        return self._store.load_run_state(run_id)

    def save_run_state(self, run_id: str, state: RunState) -> None:
        self._store.save_run_state(run_id, state)

    def increment_replan_counter(self, run_id: str) -> int:
        return self._store.increment_replan_counter(run_id)

    def increment_agent_spawn_counter(self, run_id: str) -> int:
        return self._store.increment_agent_spawn_counter(run_id)
