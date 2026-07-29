import networkx as nx
from langchain_core.messages import SystemMessage, HumanMessage
from orchestrator.state import OrchestratorState
from models.schemas import Task, TaskGraph
from storage.task_graph_store import TaskGraphStore
from storage.workflow_store import WorkflowStore
from storage.registry_store import RegistryStore
from utils.logging import AuditLogger
from config.prompts import TASK_GRAPH_SYSTEM_PROMPT
from config.config import get_chat_model, Config
from utils.llm import invoke_structured

def validate_dag(graph: TaskGraph) -> bool:
    """Verifies that the task graph contains no cycles or dangling dependencies."""
    dg = nx.DiGraph()
    task_ids = {t.id for t in graph.tasks}
    
    for task in graph.tasks:
        dg.add_node(task.id)
        for dep in task.dependencies:
            if dep not in task_ids:
                return False
            dg.add_edge(dep, task.id)
            
    return nx.is_directed_acyclic_graph(dg)

def task_graph_node(state: OrchestratorState) -> OrchestratorState:
    """Stage 2: Task Graph Generation.
    Decomposes the Goal into atomic tasks with dependencies, producing a TaskGraph.
    Uses TASK_GRAPH_SYSTEM_PROMPT and selected LLM.
    Validates graph shape deterministically.
    """
    run_id = state["run_id"]
    logger = AuditLogger(run_id)
    WorkflowStore().begin_stage(run_id, "task_graph")
    logger.log("TaskGraph", "stage_started")
    
    goal = state["goal"]
    
    # 1. Fetch available agents from registry
    registry = RegistryStore()
    agents = registry.load_agent_registry()
    agent_info_list = [f"- {agent.role}: {agent.description} (Allowed tools: {agent.tools})" for agent in agents]
    available_agents_str = "\n".join(agent_info_list)
    
    # 2. Formulate prompts
    system_prompt = TASK_GRAPH_SYSTEM_PROMPT.format(available_agents=available_agents_str)
    goal_details = (
        f"Title: {goal.title}\n"
        f"Description: {goal.description}\n"
        f"Deliverables:\n" + "\n".join(f"- {d}" for d in goal.deliverables) + "\n"
        f"Constraints:\n" + "\n".join(f"- {c}" for c in goal.constraints) + "\n"
        f"Assumptions:\n" + "\n".join(f"- {a}" for a in goal.assumptions)
    )
    
    # 3. Call LLM for Structured output
    try:
        model = get_chat_model()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Create a task graph DAG for the following Goal:\n{goal_details}")
        ]
        
        task_graph = invoke_structured(
            model, TaskGraph, messages, logger=logger, stage="TaskGraph"
        )
    except Exception as e:
        logger.log("TaskGraph", "llm_planning_failed", {"error": str(e)})
        raise e
        
    # Check DAG validity and task count guardrail
    is_valid = validate_dag(task_graph)
    if not is_valid:
        logger.log("TaskGraph", "validation_failed", {"reason": "Cycle or dangling dependency detected"})
        return {
            **state,
            "errors": state.get("errors", []) + ["Invalid Task Graph DAG generated."]
        }

    if len(task_graph.tasks) > Config.MAX_TASKS:
        logger.log("TaskGraph", "validation_failed", {"reason": f"Task count exceeds MAX_TASKS ({Config.MAX_TASKS})"})
        return {
            **state,
            "errors": state.get("errors", []) + [f"Task graph has too many tasks (max {Config.MAX_TASKS})."]
        }

    unknown_capabilities = [
        t.required_capability
        for t in task_graph.tasks
        if not registry.find_agent_by_capability(t.required_capability)
    ]
    if unknown_capabilities:
        logger.log("TaskGraph", "validation_failed", {"unknown_capabilities": unknown_capabilities})
        return {
            **state,
            "errors": state.get("errors", []) + [
                f"No registry agent for capabilities: {', '.join(sorted(set(unknown_capabilities)))}"
            ]
        }

    # Normalize capability strings to exact registry role names.
    for task in task_graph.tasks:
        matched = registry.find_agent_by_capability(task.required_capability)
        if matched:
            task.required_capability = matched.role
        
    # Persist the generated task graph
    tg_store = TaskGraphStore()
    tg_store.save_task_graph(run_id, task_graph)
    
    logger.log("TaskGraph", "stage_completed", {"tasks_count": len(task_graph.tasks)})
    
    return {
        **state,
        "task_graph": task_graph
    }
