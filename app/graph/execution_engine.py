from typing import List
from app.graph.state import OrchestratorState
from app.graph.agent_resolution import agent_resolution_node
from app.models.schemas import Task, TaskGraph
from app.storage.task_graph_store import TaskGraphStore
from app.storage.workflow_store import WorkflowStore
from app.utils.logging import AuditLogger

def get_ready_tasks(graph: TaskGraph) -> List[Task]:
    """Finds all pending tasks whose dependencies have all completed successfully."""
    ready_tasks = []
    completed_task_ids = {t.id for t in graph.tasks if t.status == "success"}
    
    for task in graph.tasks:
        if task.status != "pending":
            continue
        # Check if all dependencies are satisfied
        if all(dep in completed_task_ids for dep in task.dependencies):
            ready_tasks.append(task)
            
    return ready_tasks

def execution_engine_node(state: OrchestratorState) -> OrchestratorState:
    """Stage 4: Execution Engine (Scheduler).
    Identifies ready tasks, runs them via agent resolution, and manages status transitions.
    """
    run_id = state["run_id"]
    logger = AuditLogger(run_id)
    wf = WorkflowStore()
    wf.begin_stage(run_id, "agent_resolution")
    wf.begin_stage(run_id, "execution")
    logger.log("ExecutionEngine", "stage_started")
    
    task_graph = state["task_graph"]
    tg_store = TaskGraphStore()
    
    # 1. Identify ready tasks
    ready_tasks = get_ready_tasks(task_graph)
    if not ready_tasks:
        logger.log("ExecutionEngine", "no_ready_tasks")
        wf.complete_stage(
            run_id,
            "execution",
            succeeded=[],
            failed=[],
            note="no pending tasks ready",
        )
        wf.complete_stage(run_id, "agent_resolution", note="all agent tasks finished")
        return {**state, "current_tasks": []}
        
    logger.log("ExecutionEngine", "ready_tasks_found", {"tasks": [t.id for t in ready_tasks]})
    
    # Update tasks status to "running" in state and store
    for task in ready_tasks:
        task.status = "running"
        tg_store.update_task_status(run_id, task.id, "running")
        
    state["current_tasks"] = [t.id for t in ready_tasks]
    evaluated_ids = set(state.get("evaluated_task_ids", []))
    for task in ready_tasks:
        evaluated_ids.discard(task.id)
    
    # 2. Invoke Agent Resolution & Tool Execution
    state = agent_resolution_node(state)
    
    # Persist task statuses from agent resolution
    for task_id in state["current_tasks"]:
        task = next(t for t in task_graph.tasks if t.id == task_id)
        if task.status == "failed":
            tg_store.update_task_status(run_id, task_id, "failed", task.error_message)
        elif task.status == "success":
            tg_store.update_task_status(run_id, task_id, "success")
        elif task.status == "running":
            task.status = "failed"
            task.error_message = "Task finished without a resolved status."
            tg_store.update_task_status(run_id, task_id, "failed", task.error_message)
            
    logger.log("ExecutionEngine", "stage_completed", {
        "succeeded": [t.id for t in ready_tasks if t.status == "success"],
        "failed": [t.id for t in ready_tasks if t.status == "failed"],
    })

    succeeded = [t.id for t in ready_tasks if t.status == "success"]
    failed = [t.id for t in ready_tasks if t.status == "failed"]
    wf.complete_stage(run_id, "execution", succeeded=succeeded, failed=failed)
    # Always close this wave — pending downstream tasks must not leave the stage "running".
    wf.complete_stage(
        run_id,
        "agent_resolution",
        wave_succeeded=succeeded,
        wave_failed=failed,
    )

    return {
        **state,
        "task_graph": task_graph,
        "current_tasks": [],
        "evaluated_task_ids": list(evaluated_ids),
    }
