import asyncio
import uuid
import uvicorn

from app.config import Config
from app.graph.orchestrator import compile_orchestrator
from app.storage.run_paths import RunPathStore

STATIC_TEST_PROMPT = (
    "Research average global smartphone adoption rates by region, analyze which regions are "
    "growing fastest, compute simple percentage comparisons, and write the results to a "
    "markdown report file."
)


async def run_static_prompt():
    """Executes a full run of the dynamic orchestrator using the static test prompt."""
    run_id = str(uuid.uuid4())
    print("[*] Dynamic Agent Orchestrator — LOCAL mode")
    print(f"[*] LLM provider: {Config.ACTIVE_PROVIDER} (LLM_SOURCE={Config.LLM_SOURCE})")
    print(f"[*] Run ID: {run_id}")
    print(f"[*] Prompt: '{STATIC_TEST_PROMPT}'")
    print("-" * 50)

    state = {
        "run_id": run_id,
        "user_prompt": STATIC_TEST_PROMPT,
        "goal": None,
        "task_graph": None,
        "current_tasks": [],
        "evaluated_task_ids": [],
        "errors": [],
        "final_response": None,
    }

    orchestrator = compile_orchestrator()
    result = await orchestrator.ainvoke(state)

    print("-" * 50)
    print("[*] Run Completed.")
    paths = RunPathStore().resolve(run_id)
    if paths:
        print(f"[*] Workflow  : data/runs/{paths.relative_path}/workflow.json")
        print(f"[*] Outputs   : data/runs/{paths.relative_path}/outputs/")
        print(f"[*] Run log   : data/logs/{paths.relative_path}/run.log")
    if result.get("final_response"):
        print(f"\n[Result Assembly Output]:\n{result['final_response']}")
    else:
        print(f"\n[Result Assembly Output]: No response. Errors: {result.get('errors')}")


def start_server():
    """Starts the FastAPI web server."""
    print("[*] Dynamic Agent Orchestrator — SERVER mode")
    print(f"[*] ORCHESTRATOR_MODE=server | API http://{Config.HOST}:{Config.PORT}")
    print(f"[*] LLM_SOURCE={Config.LLM_SOURCE} | default provider: {Config.ACTIVE_PROVIDER}")
    uvicorn.run("app.api:app", host=Config.HOST, port=Config.PORT, reload=True)


if __name__ == "__main__":
    if Config.ORCHESTRATOR_MODE == "server":
        start_server()
    else:
        asyncio.run(run_static_prompt())
