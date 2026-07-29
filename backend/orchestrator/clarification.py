"""Early exit when Goal Intake cannot form an actionable goal."""

from models.schemas import Goal
from orchestrator.state import OrchestratorState
from storage.state_store import StateStore
from storage.workflow_store import WorkflowStore
from utils.logging import AuditLogger

DEFAULT_CLARIFICATION = (
    "Your request isn't clear enough to plan work. "
    "Please describe the concrete outcome you want — topic, what to produce "
    "(e.g. a Markdown report), and any constraints."
)


def apply_clarification_heuristic(goal: Goal, prompt: str) -> Goal:
    """Force clarification when deliverables are empty or the model already flagged it."""
    if not goal.needs_clarification and not [d for d in goal.deliverables if str(d).strip()]:
        goal.needs_clarification = True
    if goal.needs_clarification and not (goal.clarification_question or "").strip():
        goal.clarification_question = (
            f'Your request "{prompt.strip()}" is unclear or incomplete. '
            "What concrete outcome should we produce (topic + Markdown deliverable)?"
        )
        if not goal.assumptions:
            goal.assumptions = ["Request is incomplete or unclear"]
    return goal


def goal_needs_clarification(state: OrchestratorState) -> bool:
    goal = state.get("goal")
    return bool(goal and goal.needs_clarification)


def clarification_exit_node(state: OrchestratorState) -> OrchestratorState:
    run_id = state["run_id"]
    goal = state["goal"]
    logger = AuditLogger(run_id)
    workflow_store = WorkflowStore()
    state_store = StateStore()

    question = (goal.clarification_question or "").strip() or DEFAULT_CLARIFICATION
    final_text = (
        "## Clarification needed\n\n"
        f"{question}\n\n"
        "Send a clearer follow-up in Chat describing what you want done."
    )

    logger.log(
        "Clarification",
        "run_paused",
        {"question": question, "title": goal.title if goal else None},
    )

    workflow_store.finalize_needs_clarification(run_id, final_text)

    run_state = state_store.load_run_state(run_id)
    if run_state:
        run_state.status = "needs_clarification"
        state_store.save_run_state(run_id, run_state)

    return {**state, "final_response": final_text}


if __name__ == "__main__":
    # ponytail: assert-based self-check for unclear-goal heuristic
    vague = apply_clarification_heuristic(
        Goal(title="Unknown", description="unclear", deliverables=[]),
        "Hello",
    )
    assert vague.needs_clarification is True
    assert vague.clarification_question and "Hello" in vague.clarification_question

    clear = apply_clarification_heuristic(
        Goal(
            title="Report",
            description="Write a report on X",
            deliverables=["summary.md"],
            needs_clarification=False,
        ),
        "Write a markdown report on X",
    )
    assert clear.needs_clarification is False
    print("clarification self-check ok")
