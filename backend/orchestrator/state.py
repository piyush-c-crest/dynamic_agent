from typing import TypedDict, List, Dict, Any, Optional
from models.schemas import Goal, TaskGraph

class OrchestratorState(TypedDict):
    """The state schema passed between LangGraph nodes."""
    run_id: str
    user_prompt: str
    goal: Optional[Goal]
    task_graph: Optional[TaskGraph]
    current_tasks: List[str]
    evaluated_task_ids: List[str]
    errors: List[str]
    final_response: Optional[str]
