import json
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from dynamic_langgraph_backend import agent_manager
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


class ThreadResponse(BaseModel):
    thread_id: str


class AskDocumentRequest(BaseModel):
    doc_id: str
    question: str


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


@app.get("/threads")
def list_threads():
    """All threads that have at least one saved message, most recently updated first."""
    return {"threads": agent_manager.list_threads()}


@app.get("/thread/{thread_id}/messages")
def thread_messages(thread_id: str):
    """Full saved message history for one thread, e.g. to load when a recent chat is clicked."""
    return {"thread_id": thread_id, "messages": agent_manager.get_thread_history(thread_id)}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Phase 1: extract a PDF/DOCX/TXT/CSV/XLSX file to plain text and store it in memory."""
    data = await file.read()
    try:
        text = docpipe.extract_text(file.filename, data)
    except docpipe.UnsupportedFileType as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not extract content: {e}")

    doc_id = docpipe.document_store.add(file.filename, text)
    doc = docpipe.document_store.get(doc_id)
    return {
        "doc_id": doc_id,
        "filename": doc["filename"],
        "char_count": doc["char_count"],
        "preview": text[:500],
    }


@app.get("/documents")
def list_documents():
    return {"documents": docpipe.document_store.list()}


@app.post("/documents/ask")
def ask_document(req: AskDocumentRequest):
    """Answer a question grounded only in the uploaded document's extracted text."""
    try:
        return docpipe.ask_document(req.doc_id, req.question)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/chat")
def chat(req: ChatRequest):
    """Run the full graph and return only the final assembled answer."""
    thread_id = req.thread_id or str(uuid.uuid4())
    result = agent_manager.run(req.goal, thread_id)
    messages = result.get("messages", [])
    final_answer = messages[-1].content if messages else ""
    return {"thread_id": thread_id, "answer": final_answer}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_events(goal: str, thread_id: str):
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "orchestrator_run",
        "tags": ["orchestrator_run", "fastapi"],
        "metadata": {"goal": goal, "thread_id": thread_id},
    }

    yield _sse("thread", {"thread_id": thread_id})

    final_answer = ""
    for update in agent_manager.chatbot.stream(
        {"messages": [HumanMessage(content=goal)]},
        config=config,
        stream_mode="updates",
    ):
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

    print(f"\n✅ Stream finished for prompt: {goal[:80]!r}")
    yield _sse("done", {"answer": final_answer})


@app.get("/chat/stream")
def chat_stream(goal: str, thread_id: str | None = None):
    tid = thread_id or str(uuid.uuid4())
    return StreamingResponse(
        _stream_events(goal, tid),
        media_type="text/event-stream",
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")