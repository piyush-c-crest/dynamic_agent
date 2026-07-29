import json
import uuid
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.config import Config, local_llm_summary, uses_request_llm
from orchestrator.orchestrator import compile_orchestrator
from config.llm_runtime import LLMRuntimeConfig, clear_llm_runtime, set_llm_runtime
from models.schemas import RunState
from storage.registry_store import RegistryStore
from storage.run_paths import RunPathStore
from storage.state_store import StateStore
from storage.workflow_store import ARCHITECTURE_STAGES

app = FastAPI(
    title="Dynamic Agent Orchestrator API",
    description="Local R&D FastAPI gateway to run, monitor, and audit the dynamic agent loop.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LLMSettings(BaseModel):
    provider: str  # openai | groq | anthropic (claude)
    model: str
    api_key: str


class ChatRequest(BaseModel):
    prompt: str
    llm_settings: Optional[LLMSettings] = None


class WorkflowLimits(BaseModel):
    max_tasks: int
    max_replans: int
    max_spawned_agents: int


class LocalLLMInfo(BaseModel):
    provider: str
    model: str
    api_key_set: bool


class RegistryAgentSummary(BaseModel):
    role: str
    tools: List[str]


class ConfigResponse(BaseModel):
    orchestrator_mode: str
    llm_source: str  # local | api — backend-only; drives agent credential logic
    limits: WorkflowLimits
    local_llm: LocalLLMInfo
    registry_agents: List[RegistryAgentSummary] = []
    supported_providers: List[str] = ["openai", "groq", "anthropic"]
    dynamic_agents_enabled: bool = False


class ChatResponse(BaseModel):
    run_id: str
    status: str
    message: str
    storage_path: Optional[str] = None


class RunDetailsResponse(BaseModel):
    run_id: str
    state: Optional[RunState] = None
    workflow: Optional[Dict[str, Any]] = None


class HistoryItem(BaseModel):
    run_id: str
    batch_id: Optional[str] = None
    relative_path: str
    created_at: str
    status: Optional[str] = None
    user_prompt: Optional[str] = None
    goal_title: Optional[str] = None
    task_count: int = 0
    updated_at: Optional[str] = None


class HistoryListResponse(BaseModel):
    runs: List[HistoryItem]
    total: int


class HistoryDetailResponse(BaseModel):
    run_id: str
    index: Dict[str, Any]
    workflow: Dict[str, Any]
    stages_order: List[str] = ARCHITECTURE_STAGES


def _load_workflow(run_id: str) -> Optional[Dict[str, Any]]:
    paths = RunPathStore().resolve(run_id)
    if not paths or not paths.workflow_file.exists():
        return None
    with open(paths.workflow_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_workflow_for_api(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Heal display inconsistencies for finished runs (legacy + edge cases)."""
    status = (workflow.get("status") or "").lower()
    stages = workflow.get("stages") or {}
    if status in {"success", "failed", "partial_failure", "completed", "needs_clarification"}:
        ar = stages.get("agent_resolution")
        if isinstance(ar, dict) and ar.get("status") == "running":
            ar = {**ar, "status": "completed" if status != "needs_clarification" else "skipped"}
            stages = {**stages, "agent_resolution": ar}
            workflow = {**workflow, "stages": stages}

    # UI currently surfaces Markdown deliverables only.
    files = workflow.get("output_files") or []
    md_files = [f for f in files if str(f).lower().endswith(".md")]
    if md_files != files:
        workflow = {**workflow, "output_files": md_files}
    return workflow


def _summarize_workflow(workflow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not workflow:
        return {}
    goal = workflow.get("goal") or {}
    return {
        "status": workflow.get("status"),
        "user_prompt": workflow.get("user_prompt"),
        "goal_title": goal.get("title") if isinstance(goal, dict) else None,
        "task_count": len(workflow.get("tasks") or []),
        "updated_at": workflow.get("updated_at"),
    }


async def _run_orchestration(
    run_id: str, prompt: str, llm_settings: Optional[LLMSettings] = None
) -> None:
    if uses_request_llm() and llm_settings:
        set_llm_runtime(
            LLMRuntimeConfig(
                provider=llm_settings.provider,
                model=llm_settings.model,
                api_key=llm_settings.api_key,
            )
        )
    try:
        orchestrator = compile_orchestrator()
        initial_state = {
            "run_id": run_id,
            "user_prompt": prompt,
            "goal": None,
            "task_graph": None,
            "current_tasks": [],
            "evaluated_task_ids": [],
            "errors": [],
            "final_response": None,
        }
        await orchestrator.ainvoke(initial_state)
    except Exception as exc:
        try:
            from storage.workflow_store import WorkflowStore

            wf = WorkflowStore()
            wf.add_error(run_id, str(exc))
            state = wf.load_run_state(run_id)
            if state:
                state.status = "failed"
                wf.save_run_state(run_id, state)
        except Exception:
            pass
    finally:
        clear_llm_runtime()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config", response_model=ConfigResponse)
def get_config():
    """Server config: workflow limits and local (.env) LLM defaults."""
    summary = local_llm_summary()
    agents = RegistryStore().load_agent_registry()
    return ConfigResponse(
        orchestrator_mode=Config.ORCHESTRATOR_MODE,
        llm_source=Config.LLM_SOURCE,
        limits=WorkflowLimits(
            max_tasks=Config.MAX_TASKS,
            max_replans=Config.MAX_REPLANS,
            max_spawned_agents=Config.MAX_SPAWNED_AGENTS,
        ),
        local_llm=LocalLLMInfo(**summary),
        registry_agents=[
            RegistryAgentSummary(role=a.role, tools=a.tools) for a in agents
        ],
        dynamic_agents_enabled=False,
    )


@app.post("/chat", response_model=ChatResponse)
async def start_chat_session(request: ChatRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if uses_request_llm():
        if not request.llm_settings:
            raise HTTPException(
                status_code=400,
                detail="LLM_SOURCE=api on server: provider, model, and api_key are required in llm_settings.",
            )
        if not request.llm_settings.api_key.strip() or not request.llm_settings.model.strip():
            raise HTTPException(status_code=400, detail="llm_settings.model and llm_settings.api_key are required.")

    run_id = str(uuid.uuid4())
    asyncio.create_task(
        _run_orchestration(run_id, request.prompt.strip(), request.llm_settings)
    )

    return ChatResponse(
        run_id=run_id,
        status="starting",
        message="Run started. Poll GET /chat/{run_id} for stages, tasks, and results.",
        storage_path=None,
    )


@app.get("/chat/{run_id}", response_model=RunDetailsResponse)
def get_run_details(run_id: str):
    path_store = RunPathStore()
    paths = path_store.resolve(run_id)
    if not paths:
        raise HTTPException(status_code=404, detail="Session run not found.")

    state_store = StateStore()
    run_state = state_store.load_run_state(run_id)
    workflow = _load_workflow(run_id)

    if not run_state and not workflow:
        raise HTTPException(status_code=404, detail="Session run not found.")

    if workflow:
        workflow = _normalize_workflow_for_api(workflow)

    return RunDetailsResponse(run_id=run_id, state=run_state, workflow=workflow)


@app.get("/history", response_model=HistoryListResponse)
def list_history():
    """List all runs from run_index.json, enriched from each workflow.json."""
    index = RunPathStore().list_index()
    items: List[HistoryItem] = []

    for run_id, meta in index.items():
        summary = _summarize_workflow(_load_workflow(run_id))
        items.append(
            HistoryItem(
                run_id=run_id,
                batch_id=meta.get("batch_id"),
                relative_path=meta.get("relative_path", ""),
                created_at=meta.get("created_at", ""),
                status=summary.get("status"),
                user_prompt=summary.get("user_prompt"),
                goal_title=summary.get("goal_title"),
                task_count=summary.get("task_count") or 0,
                updated_at=summary.get("updated_at"),
            )
        )

    items.sort(key=lambda r: r.created_at or "", reverse=True)
    return HistoryListResponse(runs=items, total=len(items))


@app.get("/history/{run_id}", response_model=HistoryDetailResponse)
def get_history_detail(run_id: str):
    """Full workflow details for a past run (stages, tasks, memory, final result)."""
    index = RunPathStore().list_index()
    meta = index.get(run_id)
    if not meta:
        paths = RunPathStore().resolve(run_id)
        if not paths:
            raise HTTPException(status_code=404, detail="Run not found in history.")
        meta = {
            "batch_id": paths.batch_id,
            "relative_path": paths.relative_path,
            "created_at": paths.created_at.isoformat(),
        }

    workflow = _load_workflow(run_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow file not found for this run.")

    return HistoryDetailResponse(
        run_id=run_id,
        index=meta,
        workflow=_normalize_workflow_for_api(workflow),
        stages_order=ARCHITECTURE_STAGES,
    )
