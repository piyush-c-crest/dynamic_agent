import json
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from app.graph.state import OrchestratorState
from app.storage.memory_store import MemoryStore
from app.storage.run_paths import RunPathStore
from app.storage.state_store import StateStore
from app.storage.workflow_store import WorkflowStore
from app.utils.logging import AuditLogger
from app.utils.llm import log_llm_response
from app.config import get_chat_model
from app.prompts import RESULT_ASSEMBLY_SYSTEM_PROMPT


def _list_output_files(outputs_dir: Path) -> list[str]:
    if not outputs_dir.exists():
        return []
    # Markdown-only deliverables for now (html/docx/csv not supported).
    return sorted(
        f.name for f in outputs_dir.iterdir() if f.is_file() and f.suffix.lower() == ".md"
    )


def result_assembly_node(state: OrchestratorState) -> OrchestratorState:
    run_id = state["run_id"]
    logger = AuditLogger(run_id)
    workflow_store = WorkflowStore()
    workflow_store.begin_stage(run_id, "result_assembly")
    logger.log("ResultAssembly", "stage_started")

    state_store = StateStore()
    mem_store = MemoryStore()
    paths = RunPathStore().require(run_id)
    all_outputs = mem_store.get_all_memory(run_id)

    try:
        model = get_chat_model()
        goal_text = ""
        if state["goal"]:
            goal_text = (
                f"Goal Title: {state['goal'].title}\n"
                f"Deliverables Checklist:\n"
                + "\n".join(f"- {d}" for d in state["goal"].deliverables)
            )

        user_content = (
            f"Original Request: {state['user_prompt']}\n\n"
            f"{goal_text}\n\n"
            f"Subagent Outputs from Memory:\n{json.dumps(all_outputs, indent=2)}\n\n"
            "Please synthesize the final report now."
        )
        messages = [
            SystemMessage(content=RESULT_ASSEMBLY_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        logger.log("ResultAssembly", "llm_request", {"message": user_content})
        response = model.invoke(messages)
        final_text = response.content or ""
        log_llm_response(logger, "ResultAssembly", response, event="llm_response")
    except Exception as e:
        logger.log("ResultAssembly", "stage_failed", {"error": str(e)})
        workflow_store.fail_stage(run_id, "result_assembly", str(e))
        raise

    output_files = _list_output_files(paths.outputs_dir)
    workflow_store.set_final_response(run_id, final_text, output_files)

    # Heal stages left mid-wave when the run still reaches assembly (e.g. pending deps).
    workflow = workflow_store.get_workflow(run_id)
    ar = (workflow.get("stages") or {}).get("agent_resolution") or {}
    if ar.get("status") in (None, "pending", "running"):
        workflow_store.complete_stage(run_id, "agent_resolution", note="finalized at result assembly")

    workflow_store.complete_stage(
        run_id, "result_assembly", final_summary=final_text, output_files=output_files
    )

    run_state = state_store.load_run_state(run_id)
    if run_state:
        failed_tasks = (
            [t for t in state["task_graph"].tasks if t.status == "failed"]
            if state["task_graph"]
            else []
        )
        run_state.status = "partial_failure" if failed_tasks or state.get("errors") else "success"
        state_store.save_run_state(run_id, run_state)

    logger.log(
        "ResultAssembly",
        "stage_completed",
        {"status": run_state.status if run_state else "unknown", "output_files": output_files},
    )

    return {**state, "final_response": final_text}
