from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage

from orchestrator.state import OrchestratorState
from models.schemas import Goal, RunState
from storage.run_paths import RunPathStore
from storage.state_store import StateStore
from storage.workflow_store import WorkflowStore
from utils.logging import AuditLogger
from config.prompts import GOAL_INTAKE_SYSTEM_PROMPT
from config.config import get_chat_model
from orchestrator.clarification import apply_clarification_heuristic
from utils.llm import invoke_structured


def goal_intake_node(state: OrchestratorState) -> OrchestratorState:
    run_id = state["run_id"]
    prompt = state["user_prompt"]
    created_at = datetime.now()

    paths = RunPathStore().register(run_id, created_at)
    workflow_store = WorkflowStore()
    logger = AuditLogger(run_id)

    workflow_store.create(
        run_id=run_id,
        user_prompt=prompt,
        storage_path=paths.relative_path,
        created_at=created_at.isoformat(),
    )
    workflow_store.begin_stage(run_id, "goal_intake")
    logger.log("GoalIntake", "stage_started", {"prompt": prompt, "path": paths.relative_path})

    state_store = StateStore()
    state_store.save_run_state(
        run_id,
        RunState(
            run_id=run_id,
            storage_path=paths.relative_path,
            status="running",
            replans_used=0,
            agents_spawned=0,
            created_at=created_at.isoformat(),
            updated_at=created_at.isoformat(),
        ),
    )

    try:
        model = get_chat_model()
        messages = [
            SystemMessage(content=GOAL_INTAKE_SYSTEM_PROMPT),
            HumanMessage(content=f"Decompose the following user request: {prompt}"),
        ]
        goal = invoke_structured(model, Goal, messages, logger=logger, stage="GoalIntake")
    except Exception as e:
        logger.log("GoalIntake", "stage_failed", {"error": str(e)})
        workflow_store.fail_stage(run_id, "goal_intake", str(e))
        raise

    # Heuristic fallback if the model invents work for an empty/vague prompt.
    goal = apply_clarification_heuristic(goal, prompt)

    workflow_store.set_goal(run_id, goal)
    logger.log(
        "GoalIntake",
        "stage_completed",
        {
            "title": goal.title,
            "needs_clarification": goal.needs_clarification,
        },
    )

    return {**state, "goal": goal}
