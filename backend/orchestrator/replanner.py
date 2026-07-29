import os
import json
import networkx as nx
from langchain_core.messages import SystemMessage, HumanMessage
from orchestrator.state import OrchestratorState
from orchestrator.task_graph import validate_dag
from models.schemas import Task, TaskGraph, RunState
from storage.state_store import StateStore
from storage.task_graph_store import TaskGraphStore
from storage.registry_store import RegistryStore
from storage.workflow_store import WorkflowStore
from utils.logging import AuditLogger
from config.config import get_chat_model, Config
from utils.llm import invoke_structured
from config.prompts import REPLANNER_SYSTEM_PROMPT

def replanner_node(state: OrchestratorState) -> OrchestratorState:
    """Stage 7: Replanner.
    Alters the remaining task graph DAG upon failures, respecting replan guardrails.
    """
    run_id = state["run_id"]
    logger = AuditLogger(run_id)
    wf = WorkflowStore()
    wf.begin_stage(run_id, "replanner")
    logger.log("Replanner", "stage_started")
    
    state_store = StateStore()
    tg_store = TaskGraphStore()
    
    # 1. Enforce guardrail counter
    replans_used = state_store.increment_replan_counter(run_id)
    max_replans = Config.MAX_REPLANS
    
    if replans_used > max_replans:
        logger.log("Replanner", "max_replans_exceeded", {"replans_used": replans_used})
        wf.fail_stage(run_id, "replanner", "max replans exceeded", replans_used=replans_used)
        run_state = state_store.load_run_state(run_id)
        if run_state:
            run_state.status = "partial_failure"
            state_store.save_run_state(run_id, run_state)
        return {
            **state,
            "errors": state.get("errors", []) + ["Max replans exceeded. Moving to assembly."]
        }
        
    task_graph = state["task_graph"]
    failed_tasks = [t for t in task_graph.tasks if t.status == "failed"]
    if not failed_tasks:
        logger.log("Replanner", "no_failed_tasks_replan_skipped")
        return state
        
    logger.log("Replanner", "replanning_commenced", {"failed": [t.id for t in failed_tasks]})
    
    # 2. Retrieve registry capabilities
    registry = RegistryStore()
    agents = registry.load_agent_registry()
    agent_info_list = [f"- {a.role}: {a.description} (Allowed tools: {a.tools})" for a in agents]
    available_agents_str = "\n".join(agent_info_list)
    
    # 3. Call LLM for Structured Re-planning
    try:
        model = get_chat_model()

        messages = [
            SystemMessage(content=REPLANNER_SYSTEM_PROMPT.format(available_agents=available_agents_str)),
            HumanMessage(
                content=f"Goal description: {state['goal'].description if state['goal'] else 'Dynamic Orchestration'}\n"
                        f"Current Task Graph Status:\n{json.dumps(task_graph.model_dump(), indent=2)}\n\n"
                        f"Please generate a revised TaskGraph that patches the failures and allows the goal to be met."
            )
        ]

        revised_graph = invoke_structured(
            model, TaskGraph, messages, logger=logger, stage="Replanner"
        )
        
        # Validate that successful tasks aren't discarded
        successful_task_ids = {t.id for t in task_graph.tasks if t.status == "success"}
        for task in revised_graph.tasks:
            if task.id in successful_task_ids:
                task.status = "success" # Keep them successful
                
        # Validate DAG validity
        if validate_dag(revised_graph):
            task_graph = revised_graph
            tg_store.save_task_graph(run_id, task_graph)
            wf.complete_stage(
                run_id, "replanner", new_tasks_count=len(task_graph.tasks), replans_used=replans_used
            )
            logger.log("Replanner", "replan_successful", {"new_tasks_count": len(task_graph.tasks)})
        else:
            wf.fail_stage(run_id, "replanner", "DAG validation failed")
            logger.log("Replanner", "replan_failed_validation", {"reason": "DAG validation failed"})

    except Exception as e:
        wf.fail_stage(run_id, "replanner", str(e))
        logger.log("Replanner", "replan_llm_failed", {"error": str(e)})
        raise e
        
    return {
        **state,
        "task_graph": task_graph
    }
