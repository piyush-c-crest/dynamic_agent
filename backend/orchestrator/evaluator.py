from langchain_core.messages import SystemMessage, HumanMessage
from orchestrator.state import OrchestratorState
from models.schemas import EvaluationResult
from storage.memory_store import MemoryStore
from storage.task_graph_store import TaskGraphStore
from storage.workflow_store import WorkflowStore
from utils.logging import AuditLogger
from utils.llm import invoke_structured
from config.config import get_chat_model
from config.prompts import EVALUATOR_SYSTEM_PROMPT


def evaluate_task(
    run_id: str,
    task_id: str,
    desc: str,
    output_key: str,
    goal_desc: str,
    logger: AuditLogger,
) -> EvaluationResult:
    """Invokes the LLM to inspect task output quality."""
    mem_store = MemoryStore()
    output = mem_store.read_memory(run_id, output_key)

    if not output:
        return EvaluationResult(
            task_id=task_id,
            passed=False,
            reason="Output is empty or missing from shared memory.",
        )

    output_text = (
        output.get("output")
        if isinstance(output, dict) and isinstance(output.get("output"), str)
        else output
    )

    logger.log(
        "Evaluator",
        "task_evaluation_started",
        {"task_id": task_id, "output_key": output_key, "output_preview": str(output_text)[:500]},
    )

    try:
        model = get_chat_model()
        messages = [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Task ID (use exactly): {task_id}\n"
                    f"Overall Goal: {goal_desc}\n"
                    f"Task Description: {desc}\n"
                    f"Task Output Key: {output_key}\n"
                    f"Task Output Data:\n{output_text}\n\n"
                    "Evaluate whether the output correctly fulfills the task. "
                    "Pass if the substantive content is present in the text "
                    "(a Markdown body counts even if a .md file is also mentioned)."
                )
            ),
        ]

        result = invoke_structured(model, EvaluationResult, messages, logger=logger, stage="Evaluator")
        if result.task_id != task_id:
            result = result.model_copy(update={"task_id": task_id})
        return result
    except Exception as e:
        logger.log("Evaluator", "task_evaluation_error", {"task_id": task_id, "error": str(e)})
        has_content = bool(output.get("output") if isinstance(output, dict) else output)
        return EvaluationResult(
            task_id=task_id,
            passed=has_content,
            reason=(
                f"LLM evaluation unavailable ({e}); "
                f"structural check: {'non-empty output' if has_content else 'empty output'}."
            ),
        )


def evaluator_node(state: OrchestratorState) -> OrchestratorState:
    """Stage 6: Evaluator."""
    run_id = state["run_id"]
    logger = AuditLogger(run_id)
    WorkflowStore().begin_stage(run_id, "evaluation")
    logger.log("Evaluator", "stage_started")

    task_graph = state["task_graph"]
    tg_store = TaskGraphStore()
    goal_desc = state["goal"].description if state["goal"] else "Dynamic agent run"
    evaluated_ids = set(state.get("evaluated_task_ids", []))

    for task in task_graph.tasks:
        if task.status != "success" or not task.output_key or task.id in evaluated_ids:
            continue

        eval_res = evaluate_task(
            run_id, task.id, task.description, task.output_key, goal_desc, logger
        )
        evaluated_ids.add(task.id)

        logger.log(
            "Evaluator",
            "task_evaluated",
            {
                "task_id": task.id,
                "passed": eval_res.passed,
                "reason": eval_res.reason,
            },
        )

        WorkflowStore().update_task(
            run_id,
            task.id,
            evaluation=eval_res.model_dump(),
            status=task.status,
            error_message=task.error_message,
        )

        if not eval_res.passed:
            task.status = "failed"
            task.error_message = f"Evaluation failed: {eval_res.reason}"
            tg_store.update_task_status(run_id, task.id, "failed", task.error_message)

    WorkflowStore().complete_stage(run_id, "evaluation")
    logger.log("Evaluator", "stage_completed")
    return {
        **state,
        "task_graph": task_graph,
        "evaluated_task_ids": list(evaluated_ids),
    }
