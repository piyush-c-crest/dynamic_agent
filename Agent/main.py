import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from dynamic_langgraph_backend import (
    agent_manager,
    DEFAULT_AGENT_WORKDIR,
    WorkdirSelectionError,
    _workdir_ctx,
)
import document_pipeline as docpipe

app = FastAPI(title="Dynamic Multi-Agent Orchestrator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    goal: str
    thread_id: str | None = None
    doc_id: str | None = None


class ThreadResponse(BaseModel):
    thread_id: str


class WorkdirSelectRequest(BaseModel):
    thread_id: str
    path: str


class SkillCreateRequest(BaseModel):
    name: str
    description: str
    instructions: str = ""
    tool_names: list[str] = []
    triggers: list[str] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/thread", response_model=ThreadResponse)
def new_thread():
    """Create a new conversation thread id (client should reuse it across turns)."""
    return {"thread_id": str(uuid.uuid4())}


@app.get("/tools")
def list_tools():
    return json.loads(agent_manager.get_tool_info())


@app.get("/agents")
def list_agents():
    return json.loads(agent_manager.get_agent_info())


# ---------- skills (Phase 1: registry management -- no discovery or
# LLM-driven selection yet, see dynamic_langgraph_backend.py / skills.py) ----------

@app.get("/skills")
def list_skills():
    """All registered skills with full metadata (name, description,
    instructions, source, tools, triggers, etc.)."""
    return {"skills": json.loads(agent_manager.get_skill_info())}


@app.post("/skills")
def create_skill(req: SkillCreateRequest):
    """Manually register a skill for testing. Registered with
    source="manual", the lowest registration precedence -- Phase 2's
    folder-based discovery (skills/, github_skills/, community_skills/,
    per-workdir project skills) will be able to override a manual skill
    of the same name."""
    ok = agent_manager.add_skill(req.name, req.description, req.instructions, req.tool_names, req.triggers)
    if not ok:
        raise HTTPException(
            status_code=409,
            detail=f"Skill '{req.name}' was not registered (a higher-precedence skill with this name already exists).",
        )
    return {"status": "ok", "name": req.name}


@app.delete("/skills/{name}")
def delete_skill(name: str):
    removed = agent_manager.remove_skill(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found.")
    return {"status": "ok", "name": name}


@app.post("/skills/reindex")
def reindex_skills():
    """Re-scan skills/, github_skills/, community_skills/, and
    project_skills/ on disk for SKILL.md files without restarting the
    process. Does not touch any thread's <workdir>/.skills/ -- that's
    picked up automatically the next time a workdir is selected."""
    results = agent_manager.reindex_skills()
    return {"status": "ok", "results": results, "skills": json.loads(agent_manager.get_skill_info())}


@app.get("/threads")
def list_threads():
    """All threads that have at least one saved message, most recently updated first."""
    return {"threads": agent_manager.list_threads()}


@app.get("/thread/{thread_id}/messages")
def thread_messages(thread_id: str):
    """Full saved message history for one thread, e.g. to load when a recent chat is clicked."""
    return {"thread_id": thread_id, "messages": agent_manager.get_thread_history(thread_id)}


# ---------- working directory (native-file-manager-style folder picker) ----------
# Browsers don't expose real OS filesystem paths from a native picker --
# `<input type=file webkitdirectory>` and the File System Access API's
# showDirectoryPicker() are both sandboxed and never hand back an absolute
# path a server process can open. Since the agent's file tools operate on
# real paths on the server's disk, the only way to select one is to browse
# the server's filesystem from the UI. /fs/browse and /fs/roots back a
# picker modal built to look and behave like a native file manager (address
# bar, up/back navigation, a sidebar of quick-access locations, folders you
# double-click into and single-click to select, files shown for context).

@app.get("/fs/browse")
def browse_workdir(path: str | None = None):
    """List the contents of `path` for the folder-picker UI. Defaults to
    the server's home directory. Both subdirectories and files are
    returned -- files are shown (but not selectable as a working directory)
    so the picker reads like a real file manager rather than a bare folder
    tree. Hidden entries (dotfiles) are omitted."""
    base = Path(path).expanduser().resolve() if path else Path.home().resolve()
    if not base.exists() or not base.is_dir():
        raise HTTPException(status_code=404, detail=f"Not a directory: {base}")

    try:
        raw_entries = [p for p in base.iterdir() if not p.name.startswith(".")]
    except PermissionError:
        raw_entries = []

    # Folders first, then files, each alphabetically -- standard file-manager sort.
    raw_entries.sort(key=lambda p: (not p.is_dir(), p.name.lower()))

    entries = []
    for p in raw_entries:
        try:
            is_dir = p.is_dir()
            entry = {"name": p.name, "path": str(p), "is_dir": is_dir}
            if not is_dir:
                try:
                    entry["size_bytes"] = p.stat().st_size
                except OSError:
                    entry["size_bytes"] = None
            entries.append(entry)
        except OSError:
            continue  # broken symlink or similar -- skip rather than fail the whole listing

    return {
        "path": str(base),
        "parent": str(base.parent) if base != base.parent else None,
        "entries": entries,
    }


@app.get("/fs/roots")
def fs_roots():
    """Quick-access locations for the folder-picker sidebar: home directory,
    the agent's default workspace, and filesystem root(s) -- drive letters
    on Windows, `/` elsewhere."""
    roots = [{"name": "Home", "path": str(Path.home().resolve())}]

    default_ws = str(DEFAULT_AGENT_WORKDIR)
    if default_ws != roots[0]["path"]:
        roots.append({"name": "Agent workspace", "path": default_ws})

    if os.name == "nt":
        import string
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                roots.append({"name": drive, "path": drive})
    else:
        roots.append({"name": "Computer", "path": "/"})

    return {"roots": roots}


@app.get("/workdir")
def get_workdir(thread_id: str):
    """The folder a thread's file tools are currently confined to."""
    path = agent_manager.get_working_directory(thread_id)
    return {"thread_id": thread_id, "workdir": path, "is_default": path == str(DEFAULT_AGENT_WORKDIR)}


@app.post("/workdir")
def select_workdir(req: WorkdirSelectRequest):
    """Point a thread's file tools (read_file/write_file/list_directory/
    view_image/create_artifact) at an existing folder on disk."""
    try:
        result = agent_manager.set_working_directory(req.thread_id, req.path)
    except WorkdirSelectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"thread_id": req.thread_id, "workdir": result["workdir"]}


@app.delete("/workdir")
def clear_workdir(thread_id: str):
    """Reset a thread back to the default agent workspace."""
    agent_manager.clear_working_directory(thread_id)
    return {"thread_id": thread_id, "workdir": agent_manager.get_working_directory(thread_id)}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a file for the agent to see: PDF/DOCX/TXT/CSV/XLSX are
    extracted to plain text; PNG/JPG/GIF/WEBP are stored for vision.
    Either way, returns a doc_id ready to attach to a chat message."""
    data = await file.read()
    try:
        doc_id = docpipe.process_upload(file.filename, data)
    except docpipe.UnsupportedFileType as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:  # e.g. image over the size cap
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process file: {e}")

    doc = docpipe.document_store.get(doc_id)
    return {
        "doc_id": doc_id,
        "filename": doc["filename"],
        "is_image": doc["is_image"],
        "char_count": doc.get("char_count", 0),
        "preview": None if doc["is_image"] else doc["text"][:500],
    }


@app.get("/documents")
def list_documents():
    return {"documents": docpipe.document_store.list()}


@app.get("/generated-docs")
def list_generated_documents():
    """List all files in the generated_documents folder."""
    files = []
    if os.path.isdir(docpipe.GENERATED_DOCS_DIR):
        for fname in sorted(os.listdir(docpipe.GENERATED_DOCS_DIR), reverse=True):
            fpath = os.path.join(docpipe.GENERATED_DOCS_DIR, fname)
            if os.path.isfile(fpath):
                files.append({
                    "filename": fname,
                    "size_bytes": os.path.getsize(fpath),
                    "download_url": f"/generated-docs/{fname}",
                })
    return {"files": files}


@app.post("/chat")
def chat(req: ChatRequest):
    """Run the full graph and return only the final assembled answer.
    If doc_id is set, it's attached to the goal message -- folded into the
    prompt as text for documents, or as real vision content for images."""
    thread_id = req.thread_id or str(uuid.uuid4())
    doc_ids = [req.doc_id] if req.doc_id else None

    result = agent_manager.run(req.goal, thread_id, doc_ids=doc_ids)
    messages = result.get("messages", [])
    final_answer = messages[-1].content if messages else ""
    return {"thread_id": thread_id, "answer": final_answer}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_events(goal: str, thread_id: str, doc_id: str | None = None, workdir: str | None = None):
    # A folder picked before this thread existed rides along on the first
    # message (see index.html's `pendingWorkdir`) -- apply it up front so
    # the rest of the stream, and every later turn on this thread_id, sees it.
    if workdir:
        try:
            agent_manager.set_working_directory(thread_id, workdir)
        except WorkdirSelectionError as e:
            yield _sse("error", {"message": str(e)})
            yield _sse("done", {"answer": ""})
            return

    message_content = docpipe.build_multimodal_message(goal, [doc_id] if doc_id else None)

    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "orchestrator_run",
        "tags": ["orchestrator_run", "fastapi"],
        "metadata": {"goal": goal, "thread_id": thread_id, "doc_id": doc_id},
    }

    yield _sse("thread", {"thread_id": thread_id, "workdir": agent_manager.get_working_directory(thread_id)})

    final_answer = ""
    # Each item Starlette pulls from this generator runs via a separate
    # threadpool call with its own COPY of the request's contextvars.Context
    # (see anyio.to_thread.run_sync / Starlette's iterate_in_threadpool), so
    # a contextvars.Token set before this loop can't be reset later -- it
    # belongs to a different Context by then -- and a single set() before the
    # loop wouldn't even be visible to tool calls made on later resumes.
    # Re-asserting the thread's workdir right before each pull instead puts
    # it in the SAME Context that update's read_file/write_file/
    # list_directory/view_image/create_artifact tool calls actually run in.
    resolved_workdir = agent_manager._thread_workdirs.get(thread_id)
    try:
        stream_iter = agent_manager.chatbot.stream(
            {"messages": [HumanMessage(content=message_content)]},
            config=config,
            stream_mode="updates",
        )
        while True:
            _workdir_ctx.set(resolved_workdir)
            try:
                update = next(stream_iter)
            except StopIteration:
                break

            for node_name, node_output in update.items():

                if node_name == "planner":
                    yield _sse("plan", {"task_plan": node_output.get("task_plan", [])})

                elif node_name == "agent_executor":
                    yield _sse("status", {"message": "agent working on current task"})

                elif node_name == "tools":
                    yield _sse("status", {"message": "executing tool call(s)"})

                elif node_name == "evaluator":
                    verdict = node_output.get("last_verdict", {})
                    yield _sse("evaluation", verdict)

                elif node_name == "assembler":
                    msgs = node_output.get("messages", [])
                    if msgs:
                        final_answer = msgs[-1].content
                        yield _sse("final_answer", {"answer": final_answer})
    except Exception as e:
        print(f"\n❌ Stream error for prompt: {goal[:80]!r}: {e}")
        yield _sse("error", {"message": str(e)})

    print(f"\n✅ Stream finished for prompt: {goal[:80]!r}")
    yield _sse("done", {"answer": final_answer})


@app.get("/chat/stream")
def chat_stream(goal: str, thread_id: str | None = None, doc_id: str | None = None, workdir: str | None = None):
    tid = thread_id or str(uuid.uuid4())
    return StreamingResponse(
        _stream_events(goal, tid, doc_id, workdir),
        media_type="text/event-stream",
    )


# Serve generated documents as downloadable static files
app.mount("/generated-docs", StaticFiles(directory=docpipe.GENERATED_DOCS_DIR), name="generated_docs")

app.mount("/", StaticFiles(directory="static", html=True), name="static")