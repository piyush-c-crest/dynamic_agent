# dynamic_langgraph_backend.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any, Callable
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, Tool, BaseTool
from dotenv import load_dotenv
import os
import sqlite3
import requests
import json
import re
from datetime import datetime
from langchain_groq import ChatGroq

load_dotenv()

# -------------------
# LangSmith Setup
# -------------------
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Dynamic-LangGraph-Backend"
# Make sure to add LANGCHAIN_API_KEY to your .env file

MAX_RETRIES = 2
MAX_TASKS = 6

# -------------------
# 1. LLM
# -------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# -------------------
# Helpers
# -------------------
def parse_json_safely(text: str, default=None):
    """Extract and parse JSON from an LLM response, tolerating markdown fences."""
    if text is None:
        return default
    cleaned = text.strip()
    try:
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        return json.loads(cleaned.strip())
    except Exception:
        return default


# -------------------
# Orchestrator State
# -------------------
class OrchestratorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    goal: str
    task_plan: list[dict]
    current_task_idx: int
    task_results: dict
    task_messages: list[BaseMessage]
    retry_count: int
    last_verdict: dict


# -------------------
# 2. Dynamic Tool Registry
# -------------------
class DynamicToolRegistry:
    """
    Manages dynamic tool creation and registration.
    Tools can be created from natural language prompts.
    """
    def __init__(self):
        self.tools: dict[str, BaseTool] = {}
        self.tool_code: dict[str, str] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register built-in tools"""
        self.register_tool("search", self._search_tool())
        self.register_tool("calculator", self._calculator_tool())
        self.register_tool("stock_price", self._stock_price_tool())

    def register_tool(self, name: str, tool_obj: BaseTool, code: str = ""):
        """
        Register a tool in the registry.

        Forces tool_obj.name to match the registry key. Without this, a tool
        whose underlying function is named differently from its registry key
        (e.g. get_stock_price registered under "stock_price", or any
        LLM-generated tool whose function name differs from the name typed
        in the UI) will bind to the LLM under its *real* name, so tool_calls
        come back with a name the registry doesn't recognize and
        `_tools_node` reports "tool not found" even though the tool exists.
        """
        tool_obj.name = name
        self.tools[name] = tool_obj
        self.tool_code[name] = code
        print(f"✅ Tool registered: {name}")

    def get_all_tools(self) -> list[BaseTool]:
        """Get all registered tools as a list"""
        return list(self.tools.values())

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a specific tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> dict[str, str]:
        """List all tools with their descriptions"""
        return {name: tool.description for name, tool in self.tools.items()}

    @staticmethod
    def _search_tool() -> BaseTool:
        @tool
        def search(query: str) -> str:
            """Search the web for information"""
            search_tool = DuckDuckGoSearchRun(region="us-en")
            result = search_tool.run(query)
            return result[:500]
        return search

    @staticmethod
    def _calculator_tool() -> BaseTool:
        @tool
        def calculator(first_num: float, second_num: float, operation: str) -> dict:
            """Perform arithmetic: add, sub, mul, div"""
            ops = {
                "add": lambda a, b: a + b,
                "sub": lambda a, b: a - b,
                "mul": lambda a, b: a * b,
                "div": lambda a, b: a / b if b != 0 else "Error: Division by zero"
            }
            result = ops.get(operation, lambda a, b: "Error: Unknown operation")(first_num, second_num)
            return {"result": result, "operation": operation}
        return calculator

    @staticmethod
    def _stock_price_tool() -> BaseTool:
        @tool
        def get_stock_price(symbol: str) -> dict:
            """Fetch stock price for a symbol (e.g., AAPL, TSLA)"""
            api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
            if not api_key:
                return {"error": "ALPHAVANTAGE_API_KEY not set in environment"}
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
            try:
                r = requests.get(url, timeout=5)
                return r.json()
            except Exception as e:
                return {"error": str(e)}
        return get_stock_price

    def create_tool_from_prompt(self, prompt: str, tool_name: str) -> BaseTool:
        """
        Create a new tool from a natural language prompt.
        Code is executed in a restricted-builtins sandbox: no file/network-escape
        primitives (open, os, subprocess, eval, __import__, etc.) are exposed.
        """
        print(f"🔨 Creating tool '{tool_name}' from prompt...")

        creation_prompt = f"""
        Create a Python tool function for the following requirement:
        {prompt}

        Requirements:
        1. Use the @tool decorator from langchain_core.tools
        2. Include a clear docstring
        3. Handle errors gracefully
        4. Return JSON-serializable data
        5. Keep it focused and simple
        6. Do not use file I/O, os, subprocess, eval, exec, or import statements

        Example format:

        @tool
        def {tool_name}(input_data: str) -> dict:
            Description of what the tool does.
            code implementation here

        Return ONLY the function code, starting with @tool and ending with the return statement.
        No imports needed (they're already available: requests, json, datetime, re).
        """

        response = llm.invoke(creation_prompt)
        tool_code = response.content
        print(f"📝 Generated code for tool '{tool_name}':\n{tool_code}")

        try:
            if "```python" in tool_code:
                tool_code = tool_code.split("```python")[1].split("```")[0]
            elif "```" in tool_code:
                tool_code = tool_code.split("```")[1].split("```")[0]

            # --- Sandboxed execution ---
            # Only a minimal, safe set of builtins is exposed. This blocks the
            # obvious escape hatches (open, os, subprocess, eval, __import__)
            # that would otherwise be reachable from LLM-generated code.
            safe_builtins = {
                "len": len, "str": str, "int": int, "float": float, "bool": bool,
                "list": list, "dict": dict, "tuple": tuple, "set": set,
                "range": range, "enumerate": enumerate, "zip": zip,
                "min": min, "max": max, "sum": sum, "sorted": sorted, "round": round,
                "abs": abs, "isinstance": isinstance, "print": print,
                "Exception": Exception, "ValueError": ValueError,
                "TypeError": TypeError, "KeyError": KeyError, "True": True,
                "False": False, "None": None,
            }
            safe_globals = {
                "__builtins__": safe_builtins,
                "tool": tool,
                "requests": requests,
                "json": json,
                "datetime": datetime,
                "re": re,
            }
            local_vars = {}
            exec(tool_code, safe_globals, local_vars)

            for var_name, var_obj in local_vars.items():
                if isinstance(var_obj, BaseTool):
                    self.register_tool(tool_name, var_obj, tool_code)
                    return var_obj

            raise ValueError("No tool created from prompt")

        except Exception as e:
            print(f"❌ Error creating tool: {str(e)}")
            raise


# -------------------
# 3. Dynamic Agent Factory
# -------------------
class DynamicAgentFactory:
    """
    Creates and caches specialized sub-agents at runtime.
    Each agent is a distinct (system_prompt, tool_subset, llm) triple built for
    a specific role the planner identified — this is what makes agent creation
    dynamic rather than just tool creation on a single fixed agent.
    """
    def __init__(self, tool_registry: DynamicToolRegistry):
        self.tool_registry = tool_registry
        self.agents: dict[str, dict] = {}

    def get_or_create(self, role: str, task_description: str) -> dict:
        if role in self.agents:
            return self.refresh_tools(role, task_description)
        return self.create_agent(role, task_description)

    def refresh_tools(self, role: str, task_description: str) -> dict:
        """
        Re-check an existing agent's tool set against the *current* tool
        registry and the *current* task.

        Without this, an agent's tools are decided once from whatever the
        first task it ever handled needed, and never revisited — so (a) a
        role reused later for a task needing a different tool stays stuck
        without it, and (b) a tool added from the sidebar after the role
        already exists is invisible to it forever. This only ever expands
        the tool set (never removes tools), and skips the rebind entirely
        if nothing changed.
        """
        agent_conf = self.agents[role]
        available_tools = self.tool_registry.list_tools()

        selection_prompt = f"""
The "{role}" agent currently has these tools: {agent_conf['tool_names']}
It now needs to handle this task: "{task_description}"

Available tools (name: description): {json.dumps(available_tools)}

Return ONLY JSON in this exact shape:
{{"tools": ["tool_name1", "tool_name2"]}}

List ALL tools this agent should have going forward: its current tools plus
any additional ones this new task genuinely needs. Do not drop a currently
held tool unless it is clearly irrelevant.
"""
        response = llm.invoke(selection_prompt)
        result = parse_json_safely(response.content, default=None)
        if result is None:
            print(f"⚠️ Tool refresh JSON parse failed for role '{role}'; keeping existing tools.")
            return agent_conf

        desired_names = set(result.get("tools", [])) | set(agent_conf["tool_names"])
        if desired_names == set(agent_conf["tool_names"]):
            return agent_conf

        selected_tools = [
            self.tool_registry.get_tool(name)
            for name in desired_names
            if self.tool_registry.get_tool(name) is not None
        ]
        agent_conf["llm"] = llm.bind_tools(selected_tools) if selected_tools else llm
        agent_conf["tool_names"] = [t.name for t in selected_tools]
        print(f"🔄 Agent '{role}' tools refreshed: {agent_conf['tool_names']}")
        return agent_conf

    def create_agent(self, role: str, task_description: str) -> dict:
        print(f"🧬 Creating new agent for role: '{role}'")
        available_tools = self.tool_registry.list_tools()

        selection_prompt = f"""
You are configuring a specialized AI agent for the role "{role}".
It will handle tasks like: "{task_description}"

Available tools (name: description): {json.dumps(available_tools)}

Return ONLY JSON in this exact shape:
{{
  "system_prompt": "a system prompt defining this agent's persona, scope and behavior",
  "tools": ["tool_name1", "tool_name2"]
}}

Only select tools genuinely relevant to this role. It is fine to select zero tools.
"""
        response = llm.invoke(selection_prompt)
        config = parse_json_safely(response.content, default=None)
        if config is None:
            print(f"⚠️ Tool selection JSON parse failed for role '{role}'. Raw response: {response.content[:300]!r}")
            config = {
                "system_prompt": f"You are a focused, helpful '{role}' agent. Be concise and accurate.",
                "tools": [],
            }

        selected_tools = [
            self.tool_registry.get_tool(name)
            for name in config.get("tools", [])
            if self.tool_registry.get_tool(name) is not None
        ]

        agent_llm = llm.bind_tools(selected_tools) if selected_tools else llm

        agent_conf = {
            "role": role,
            "system_prompt": config.get("system_prompt", f"You are a helpful '{role}' agent."),
            "tool_names": [t.name for t in selected_tools],
            "llm": agent_llm,
        }
        self.agents[role] = agent_conf
        print(f"✅ Agent '{role}' created with tools: {agent_conf['tool_names']}")
        return agent_conf

    def list_agents(self) -> dict:
        return {
            role: {"system_prompt": conf["system_prompt"], "tools": conf["tool_names"]}
            for role, conf in self.agents.items()
        }


# -------------------
# 4. Dynamic Agent Manager (Orchestrator)
# -------------------
class DynamicAgentManager:
    """
    Owns the full orchestration graph:
    planner -> agent_executor <-> tools -> evaluator -> (retry | next task) -> assembler
    """
    def __init__(self):
        self.tool_registry = DynamicToolRegistry()
        self.agent_factory = DynamicAgentFactory(self.tool_registry)
        self.behavior_style = "standard"
        self.extra_instruction = ""
        self.temperature = 0.0
        self.chatbot = self._build_graph()

    # ---------- dynamic configuration ----------
    def set_behavior_style(self, style: str):
        self.behavior_style = style or "standard"

    def set_extra_instruction(self, instruction: str):
        self.extra_instruction = instruction or ""

    def set_temperature(self, temperature: float):
        self.temperature = temperature
        try:
            llm.temperature = temperature
        except Exception:
            pass

    # ---------- graph nodes ----------
    def _planner_node(self, state: OrchestratorState):
        goal = state["messages"][-1].content
        print(f"🧭 Planning for goal: {goal}")

        style_note = f"\nDesired response style: {self.behavior_style}." if self.behavior_style else ""
        extra_note = f"\nAdditional instruction: {self.extra_instruction}" if self.extra_instruction else ""

        planning_prompt = f"""
You are a task planning system for a multi-agent orchestrator.
Break the goal below into a short sequence of atomic, actionable tasks.
For each task, assign an "agent_role": a short label for the kind of
specialist needed (e.g. "researcher", "calculator", "analyst", "writer").
Reuse the same role across tasks when it genuinely fits the same specialty.

Goal: "{goal}"{style_note}{extra_note}

Return ONLY a JSON array, no prose, like:
[
  {{"id": "T1", "description": "...", "agent_role": "researcher"}},
  {{"id": "T2", "description": "...", "agent_role": "writer"}}
]

Use between 1 and {MAX_TASKS} tasks. Keep each task atomic.
"""
        response = llm.invoke(planning_prompt)
        plan = parse_json_safely(response.content, default=None)
        if not plan or not isinstance(plan, list):
            plan = [{"id": "T1", "description": goal, "agent_role": "general"}]
        plan = plan[:MAX_TASKS]

        print(f"📋 Task plan: {json.dumps(plan, indent=2)}")

        return {
            "goal": goal,
            "task_plan": plan,
            "current_task_idx": 0,
            "task_results": {},
            "task_messages": [],
            "retry_count": 0,
            "last_verdict": {},
        }

    def _agent_executor_node(self, state: OrchestratorState):
        idx = state["current_task_idx"]
        task = state["task_plan"][idx]
        role = task["agent_role"]
        agent = self.agent_factory.get_or_create(role, task["description"])

        task_messages = state.get("task_messages") or []
        if not task_messages:
            context = self._build_context(state.get("task_results", {}), task)
            init_prompt = f"Task: {task['description']}"
            if context:
                init_prompt += f"\n\nRelevant context from earlier tasks:\n{context}"
            task_messages = [
                SystemMessage(content=agent["system_prompt"]),
                HumanMessage(content=init_prompt),
            ]

        print(f"🤖 [{role}] executing task {task['id']}: {task['description']}")
        response = agent["llm"].invoke(task_messages)
        task_messages = task_messages + [response]

        return {"task_messages": task_messages}

    def _tools_node(self, state: OrchestratorState):
        """Custom tool executor operating on task_messages (not the main transcript)."""
        last = state["task_messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        tool_messages = []
        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call.get("args", {})
            tool_obj = self.tool_registry.get_tool(tool_name)
            try:
                if tool_obj is None:
                    result = f"Error: tool '{tool_name}' not found"
                else:
                    result = tool_obj.invoke(tool_args)
            except Exception as e:
                result = f"Error executing tool '{tool_name}': {e}"
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return {"task_messages": state["task_messages"] + tool_messages}

    def _evaluator_node(self, state: OrchestratorState):
        idx = state["current_task_idx"]
        task = state["task_plan"][idx]
        task_messages = state["task_messages"]
        final_content = task_messages[-1].content or ""

        eval_prompt = f"""
Task: {task['description']}
Agent output: {final_content}

Does this output satisfactorily complete the task? Respond with ONLY JSON:
{{"status": "PASS" or "RETRY", "reason": "short reason", "feedback": "what to fix if RETRY"}}
"""
        eval_response = llm.invoke(eval_prompt)
        verdict = parse_json_safely(
            eval_response.content,
            default={"status": "PASS", "reason": "auto-accepted (unparseable verdict)", "feedback": ""},
        )
        print(f"🧪 Evaluation for {task['id']}: {verdict}")

        retry_count = state.get("retry_count", 0)
        if verdict.get("status") == "RETRY" and retry_count < MAX_RETRIES:
            feedback_msg = HumanMessage(
                content=f"Evaluator feedback: {verdict.get('feedback', '')}. Please revise and try again."
            )
            return {
                "task_messages": state["task_messages"] + [feedback_msg],
                "retry_count": retry_count + 1,
                "last_verdict": verdict,
            }

        # Accept result (PASS, or retries exhausted)
        results = dict(state.get("task_results", {}))
        results[task["id"]] = final_content
        summary_msg = AIMessage(
            content=f"[{task['agent_role']}] completed '{task['description']}':\n{final_content[:400]}"
        )
        return {
            "task_results": results,
            "current_task_idx": idx + 1,
            "task_messages": [],
            "retry_count": 0,
            "last_verdict": verdict,
            "messages": [summary_msg],
        }

    def _assembler_node(self, state: OrchestratorState):
        results = state.get("task_results", {})
        plan = state.get("task_plan", [])

        summary_prompt = f"Original goal: {state['goal']}\n\nTask results:\n"
        for t in plan:
            summary_prompt += f"- {t['description']}: {results.get(t['id'], 'N/A')}\n"
        summary_prompt += "\nWrite a single, coherent final answer to the original goal, using the results above."

        final = llm.invoke(summary_prompt)
        print("📦 Final result assembled")
        return {"messages": [final]}

    # ---------- routing ----------
    @staticmethod
    def _route_after_planner(state: OrchestratorState):
        if state["current_task_idx"] >= len(state["task_plan"]):
            return "assembler"
        return "agent_executor"

    @staticmethod
    def _route_after_agent(state: OrchestratorState):
        last = state["task_messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "evaluator"

    @staticmethod
    def _route_after_evaluator(state: OrchestratorState):
        if state.get("task_messages"):  # retry pending, still has scratch messages
            return "agent_executor"
        if state["current_task_idx"] >= len(state["task_plan"]):
            return "assembler"
        return "agent_executor"

    # ---------- graph build ----------
    def _build_graph(self):
        graph = StateGraph(OrchestratorState)

        graph.add_node("planner", self._planner_node)
        graph.add_node("agent_executor", self._agent_executor_node)
        graph.add_node("tools", self._tools_node)
        graph.add_node("evaluator", self._evaluator_node)
        graph.add_node("assembler", self._assembler_node)

        graph.add_edge(START, "planner")
        graph.add_conditional_edges(
            "planner", self._route_after_planner,
            {"agent_executor": "agent_executor", "assembler": "assembler"},
        )
        graph.add_conditional_edges(
            "agent_executor", self._route_after_agent,
            {"tools": "tools", "evaluator": "evaluator"},
        )
        graph.add_edge("tools", "agent_executor")
        graph.add_conditional_edges(
            "evaluator", self._route_after_evaluator,
            {"agent_executor": "agent_executor", "assembler": "assembler"},
        )
        graph.add_edge("assembler", END)

        conn = sqlite3.connect(database="DB/dynamic_chatbot.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)

        return graph.compile(checkpointer=checkpointer)

    # ---------- context sharing helper ----------
    @staticmethod
    def _build_context(task_results: dict, task: dict) -> str:
        """
        Pass only prior task results into a new task's prompt, instead of the
        full conversation transcript. Keeps each agent's context small and
        avoids unbounded context growth as more tasks/agents run.
        """
        if not task_results:
            return ""
        lines = [f"- {tid}: {str(res)[:300]}" for tid, res in task_results.items()]
        return "\n".join(lines)

    # ---------- public API ----------
    def add_tool_from_prompt(self, prompt: str, tool_name: str):
        try:
            self.tool_registry.create_tool_from_prompt(prompt, tool_name)
            self.chatbot = self._build_graph()
            print(f"✅ Graph rebuilt with new tool: {tool_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to add tool: {e}")
            return False

    def get_tool_info(self) -> str:
        return json.dumps(self.tool_registry.list_tools(), indent=2)

    def get_agent_info(self) -> str:
        return json.dumps(self.agent_factory.list_agents(), indent=2)

    def run(self, user_input: str, thread_id: str, requirements: dict = None):
        """
        requirements: {
            "new_tools": [{"name": "...", "prompt": "..."}],
            "dynamic_behavior": "concise" | "detailed" | ...,
            "preprocessing": "extra instruction text",
            "temperature": 0.0-1.0,
        }
        """
        if requirements:
            if requirements.get("new_tools"):
                for tool_spec in requirements["new_tools"]:
                    self.add_tool_from_prompt(tool_spec["prompt"], tool_spec["name"])
            if "dynamic_behavior" in requirements:
                self.set_behavior_style(requirements["dynamic_behavior"])
            if "preprocessing" in requirements:
                self.set_extra_instruction(requirements["preprocessing"])
            if "temperature" in requirements:
                self.set_temperature(requirements["temperature"])

        config = {"configurable": {"thread_id": thread_id}}
        response = self.chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        return response


# -------------------
# 5. Initialize Global Agent
# -------------------
agent_manager = DynamicAgentManager()


# -------------------
# 6. Helper Functions
# -------------------
def get_agent_tools() -> str:
    return agent_manager.get_tool_info()


def get_agent_registry() -> str:
    return agent_manager.get_agent_info()


def add_tool_dynamically(tool_name: str, tool_prompt: str) -> bool:
    return agent_manager.add_tool_from_prompt(tool_prompt, tool_name)


def run_agent_with_requirements(user_input: str, thread_id: str, requirements: dict = None):
    return agent_manager.run(user_input, thread_id, requirements)