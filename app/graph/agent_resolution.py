import json
from typing import Any, Dict, Optional

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from app.graph.state import OrchestratorState
from app.models.schemas import Task, AgentDefinition
from app.storage.registry_store import RegistryStore
from app.storage.memory_store import MemoryStore
from app.storage.task_graph_store import TaskGraphStore
from app.storage.workflow_store import WorkflowStore
from app.tools.base import execute_tool_by_name, LANGCHAIN_TOOLS
from app.utils.llm import log_llm_response
from app.utils.logging import AuditLogger
from app.config import get_chat_model, set_active_run_outputs_dir, assert_markdown_path
from app.storage.run_paths import RunPathStore

# Shared rules appended for every registry agent (limited tools + markdown-only).
AGENT_RUNTIME_RULES = """
Runtime rules (must follow):
1. You only have the tools listed for your role. Do not invent tools.
2. Upstream dependency outputs are already in the user message as plain text.
   - Read and use them directly.
   - NEVER paste dependency text into python_repl, file_write, or any tool argument.
3. python_repl (if available): only short calculation snippets (a few lines).
   - Never json.loads / embed research briefs or long documents in code.
4. file_read / file_write: Markdown (.md) filenames only (e.g. analysis.md, report.md).
5. Put your real deliverable in the final assistant reply (structured Markdown).
   file_write is optional for a durable .md copy — keep tool args reasonably sized.
6. If a tool fails, continue and answer from context without that tool.
""".strip()


def lookup_registry(required_capability: str) -> Optional[AgentDefinition]:
    """Queries the agent_registry.json for a matching specialist."""
    return RegistryStore().find_agent_by_capability(required_capability)


def _plain_dependency_text(value: Any) -> str:
    """Unwrap memory payloads to readable text (avoid nested JSON in prompts)."""
    if value is None:
        return ""
    if isinstance(value, dict):
        if "output" in value and isinstance(value["output"], str):
            return value["output"]
        return json.dumps(value, indent=2, ensure_ascii=False)
    return str(value)


def _format_dependency_context(dep_outputs: Dict[str, Any]) -> str:
    if not dep_outputs:
        return ""
    parts = [
        "\n\nUpstream dependency outputs (already available — use directly, do NOT re-parse via tools):\n"
    ]
    for key, val in dep_outputs.items():
        parts.append(f"### {key}\n{_plain_dependency_text(val)}\n")
    return "\n".join(parts)


def _is_tool_call_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "tool_use_failed" in text
        or "failed to call a function" in text
        or "failed_generation" in text
    )


def _maybe_persist_markdown(run_id: str, task: Task, agent: AgentDefinition, final_answer: str) -> None:
    """If the agent produced Markdown but skipped file_write, save a .md artifact."""
    if not final_answer or "file_write" not in (agent.tools or []):
        return
    name = f"{(task.output_key or task.id).replace(' ', '_').lower()}.md"
    try:
        path = assert_markdown_path(name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(final_answer, encoding="utf-8")
    except Exception:
        pass


def execute_ephemeral_agent(run_id: str, task: Task, agent: AgentDefinition) -> str:
    """Executes the task using the resolved agent and its allowed tools."""
    logger = AuditLogger(run_id).for_task(task.id)
    paths = RunPathStore().require(run_id)
    set_active_run_outputs_dir(paths.outputs_dir)

    logger.log("AgentResolution", "executing_task", {"agent": agent.role})

    mem_store = MemoryStore()
    tg_store = TaskGraphStore()
    task_graph = tg_store.load_task_graph(run_id)

    dep_outputs: Dict[str, Any] = {}
    if task_graph:
        for dep_id in task.dependencies:
            dep_task = next((t for t in task_graph.tasks if t.id == dep_id), None)
            if dep_task:
                key = dep_task.output_key or f"result_{dep_id}"
                val = mem_store.read_memory(run_id, key)
                if val:
                    dep_outputs[key] = val

    dependency_context = _format_dependency_context(dep_outputs)
    system_prompt = (agent.system_prompt or "You are a helpful registry agent.").strip()
    system_prompt = f"{system_prompt}\n\n{AGENT_RUNTIME_RULES}"

    try:
        model = get_chat_model()
        allowed_tools = [LANGCHAIN_TOOLS[t] for t in agent.tools if t in LANGCHAIN_TOOLS]
        tool_model = model.bind_tools(allowed_tools) if allowed_tools else model

        user_message = (
            f"Your role: {agent.role}\n"
            f"Task objective: {task.description}\n"
            f"Preferred output key: {task.output_key or f'result_{task.id}'}\n"
            f"{dependency_context}"
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        logger.log(
            "AgentResolution",
            "agent_request",
            {
                "agent": agent.role,
                "tools": agent.tools,
                "system_prompt": system_prompt,
                "user_message": user_message,
            },
        )

        tools_used = []
        final_answer = ""
        for step in range(10):
            try:
                response = tool_model.invoke(messages)
            except Exception as invoke_err:
                if allowed_tools and _is_tool_call_failure(invoke_err):
                    logger.log(
                        "AgentResolution",
                        "tool_use_failed_fallback",
                        {"error": str(invoke_err), "step": step},
                    )
                    # Groq often fails when models stuff huge text into tool args.
                    # Fall back to a plain completion from existing context.
                    fallback = get_chat_model()
                    messages.append(
                        HumanMessage(
                            content=(
                                "Tool calling failed. Do not call any tools. "
                                "Using only the task objective and upstream dependency "
                                "text already in this conversation, produce the complete "
                                "Markdown deliverable now."
                            )
                        )
                    )
                    response = fallback.invoke(messages)
                    log_llm_response(
                        logger,
                        "AgentResolution",
                        response,
                        event="agent_fallback_response",
                        agent=agent.role,
                        step=step,
                    )
                    final_answer = response.content or ""
                    break
                raise

            messages.append(response)
            log_llm_response(
                logger,
                "AgentResolution",
                response,
                event="agent_response",
                agent=agent.role,
                step=step,
                tool_calls=[
                    {"name": tc["name"], "args": tc["args"]}
                    for tc in (response.tool_calls or [])
                ],
            )

            if not response.tool_calls:
                final_answer = response.content or ""
                break

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tools_used.append(tool_name)

                try:
                    tool_output = execute_tool_by_name(tool_name, tool_args)
                except Exception as tool_err:
                    tool_output = f"Error executing tool '{tool_name}': {str(tool_err)}"

                logger.log_tool(
                    "AgentResolution", task.id, agent.role, tool_name, tool_args, tool_output
                )
                messages.append(
                    ToolMessage(
                        content=str(tool_output),
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    )
                )

        if not (final_answer or "").strip():
            raise RuntimeError(
                f"Agent '{agent.role}' finished without a usable text deliverable."
            )

        _maybe_persist_markdown(run_id, task, agent, final_answer)

        output_key = task.output_key or f"result_{task.id}"
        mem_store.write_memory(run_id, output_key, {"output": final_answer, "success": True})

        logger.log(
            "AgentResolution",
            "task_completed",
            {
                "agent": agent.role,
                "output_key": output_key,
                "final_answer": final_answer,
                "tools_used": tools_used,
            },
        )

        WorkflowStore().record_agent_run(
            run_id, task.id, agent.role, "success", output_key=output_key, tools_used=tools_used
        )
        return final_answer

    except Exception as e:
        logger.log("AgentResolution", "task_execution_failed", {"error": str(e)})
        WorkflowStore().record_agent_run(
            run_id, task.id, agent.role, "failed", error=str(e), tools_used=agent.tools
        )
        raise e
    finally:
        set_active_run_outputs_dir(None)


def agent_resolution_node(state: OrchestratorState) -> OrchestratorState:
    """Stage 3 Node for LangGraph.
    Resolves agent roles and runs tasks in state['current_tasks'].
    """
    run_id = state["run_id"]
    logger = AuditLogger(run_id)

    for task_id in state["current_tasks"]:
        task = next((t for t in state["task_graph"].tasks if t.id == task_id), None)
        if not task:
            continue

        logger.log(
            "AgentResolution",
            "resolving_agent",
            {"task_id": task_id, "capability": task.required_capability},
        )

        agent = lookup_registry(task.required_capability)
        if not agent:
            error_msg = (
                f"Agent matching capability '{task.required_capability}' not found in registry. "
                "Available roles: Researcher, Data Analyst, Document Generator."
            )
            logger.log(
                "AgentResolution",
                "agent_resolution_failed",
                {"task_id": task_id, "error": error_msg},
            )
            task.status = "failed"
            task.error_message = error_msg
            WorkflowStore().update_task(
                run_id,
                task.id,
                status=task.status,
                error_message=task.error_message,
            )
            continue

        task.assigned_agent = agent.role
        task.assigned_tools = agent.tools

        try:
            execute_ephemeral_agent(run_id, task, agent)
            task.status = "success"
        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)

        WorkflowStore().update_task(
            run_id,
            task.id,
            status=task.status,
            assigned_agent=task.assigned_agent,
            assigned_tools=task.assigned_tools,
            error_message=task.error_message,
        )

    TaskGraphStore().save_task_graph(run_id, state["task_graph"])
    return state
