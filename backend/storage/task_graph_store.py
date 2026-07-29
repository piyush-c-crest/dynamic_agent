from typing import Optional

from models.schemas import TaskGraph
from storage.workflow_store import WorkflowStore


class TaskGraphStore:
    def __init__(self):
        self._store = WorkflowStore()

    def load_task_graph(self, run_id: str) -> Optional[TaskGraph]:
        return self._store.load_task_graph(run_id)

    def save_task_graph(self, run_id: str, graph: TaskGraph) -> None:
        self._store.set_task_graph(run_id, graph)

    def update_task_status(self, run_id: str, task_id: str, status: str, error_message: Optional[str] = None) -> None:
        fields = {"status": status}
        if error_message:
            fields["error_message"] = error_message
        self._store.update_task(run_id, task_id, **fields)
