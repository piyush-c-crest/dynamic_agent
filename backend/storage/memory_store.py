from typing import Any, Dict, Optional

from storage.workflow_store import WorkflowStore


class MemoryStore:
    def __init__(self):
        self._store = WorkflowStore()

    def write_memory(self, run_id: str, key: str, value: Any) -> None:
        self._store.write_memory(run_id, key, value)

    def read_memory(self, run_id: str, key: str) -> Optional[Any]:
        return self._store.read_memory(run_id, key)

    def get_all_memory(self, run_id: str) -> Dict[str, Any]:
        return self._store.get_all_memory(run_id)

    def clear_memory(self, run_id: str) -> None:
        workflow = self._store.get_workflow(run_id)
        workflow["memory"] = {}
        self._store._save(run_id, workflow)
