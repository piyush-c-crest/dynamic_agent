from langgraph.graph import StateGraph, END
from app.config import Config
from app.graph.state import OrchestratorState
from app.graph.goal_intake import goal_intake_node
from app.graph.task_graph import task_graph_node
from app.graph.execution_engine import execution_engine_node
from app.graph.evaluator import evaluator_node
from app.graph.replanner import replanner_node
from app.graph.result_assembly import result_assembly_node
from app.storage.state_store import StateStore

def evaluator_router(state: OrchestratorState) -> str:
    """Conditional router edge following the Evaluator node.
    Decides whether to Replan, Continue Execution, or Assemble Results.
    """
    run_id = state["run_id"]
    task_graph = state["task_graph"]
    
    if not task_graph:
        return "result_assembly"
        
    failed_tasks = [t for t in task_graph.tasks if t.status == "failed"]
    pending_tasks = [t for t in task_graph.tasks if t.status == "pending"]
    
    # 1. If any task failed, check replan counter guardrail
    if failed_tasks:
        state_store = StateStore()
        run_state = state_store.load_run_state(run_id)
        replans_used = run_state.replans_used if run_state else 0
        max_replans = Config.MAX_REPLANS
        
        if replans_used < max_replans:
            return "replanner"
        else:
            # Replan limits reached, route directly to assembly
            return "result_assembly"
            
    # 2. If tasks are still pending, continue executing
    if pending_tasks:
        return "execution_engine"
        
    # 3. If all tasks succeeded, we are done
    return "result_assembly"

def compile_orchestrator():
    """Initializes and compiles the LangGraph StateGraph representing the loop."""
    workflow = StateGraph(OrchestratorState)
    
    # Register stage nodes
    workflow.add_node("goal_intake", goal_intake_node)
    workflow.add_node("task_graph", task_graph_node)
    workflow.add_node("execution_engine", execution_engine_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("result_assembly", result_assembly_node)
    
    # Build graph topology
    workflow.set_entry_point("goal_intake")
    
    workflow.add_edge("goal_intake", "task_graph")
    workflow.add_edge("task_graph", "execution_engine")
    workflow.add_edge("execution_engine", "evaluator")
    
    # Conditional routing edge following evaluator stage
    workflow.add_conditional_edges(
        "evaluator",
        evaluator_router,
        {
            "replanner": "replanner",
            "execution_engine": "execution_engine",
            "result_assembly": "result_assembly"
        }
    )
    
    # Loops back to execution engine after replanning
    workflow.add_edge("replanner", "execution_engine")
    workflow.add_edge("result_assembly", END)
    
    # Compile the workflow graph
    return workflow.compile()
