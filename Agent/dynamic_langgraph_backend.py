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
import importlib
from datetime import datetime
from langchain_groq import ChatGroq
from langsmith import traceable

load_dotenv()

# -------------------
# LangSmith Setup
# -------------------
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Dynamic-LangGraph-Backend"
# Make sure to add LANGCHAIN_API_KEY to your .env file

if not os.environ.get("LANGCHAIN_API_KEY"):
    print(
        "⚠️ LANGCHAIN_API_KEY not set — LangSmith tracing will NOT appear in the "
        "dashboard even though tracing is enabled. Add it to your .env file."
    )

# Every custom LLM call below is tagged/named (see `config=` on llm.invoke calls
# and @traceable on the helper functions) so the LangSmith trace tree reads as:
#   orchestrator_run
#     planner ("planner_llm")
#     agent_executor ("create_agent" / "refresh_agent_tools" -> possibly
#         "auto_create_missing_tools" -> "create_tool_from_prompt", then
#         "agent_exec:<role>")
#     tools ("tool:<tool_name>")
#     evaluator ("evaluator:<task_id>")
#     assembler ("assembler_llm")
# instead of a flat list of identically-named "ChatGroq" runs.

MAX_RETRIES = 2
MAX_TASKS = 6
AUTO_TOOL_LIMIT = 2  # max new tools an agent's tool-selection step may auto-create per call
MAX_TOOL_VALIDATION_RETRIES = 1  # repair attempts if a generated tool fails its smoke test

# -------------------
# Optional module whitelist for generated tools
# -------------------
# The sandbox in create_tool_from_prompt has no __import__, so generated code
# can never "import pandas" itself. Instead we pre-import a fixed set of
# modules ONCE here — only those that actually succeed (i.e. are genuinely
# installed in this environment) end up available — and hand them to
# generated code as ready-to-use globals under their normal name. A tool
# whose code needs something outside this dict simply can't get it; that's
# treated as a "needs_package" status rather than a code bug to repair (see
# create_tool_from_prompt). To add a library, add its pip package to
# requirements.txt/install it, then list it here.
_OPTIONAL_MODULES = {
    "math": "math",
    "statistics": "statistics",
    "itertools": "itertools",
    "collections": "collections",
    "urllib_parse": "urllib.parse",   # exposed as `urllib_parse` since dotted names can't be bare identifiers
    "bs4": "bs4",
    "pandas": "pandas",
    "numpy": "numpy",
}


def _build_allowed_extra_modules() -> dict:
    mods = {}
    for alias, modpath in _OPTIONAL_MODULES.items():
        try:
            mods[alias] = importlib.import_module(modpath)
        except ImportError:
            continue  # not installed here -> simply unavailable, not an error
    return mods


ALLOWED_EXTRA_MODULES = _build_allowed_extra_modules()
_IMPORT_LINE_RE = re.compile(r"^\s*(import\s+\S+|from\s+\S+\s+import\s+.+)$", re.MULTILINE)

# -------------------
# Risk classification for auto tool creation
# -------------------
# Some tool requests should never be turned into sandboxed, LLM-generated
# code, no matter how well-worded the prompt is:
#   - direct database/SQL access: an LLM assembling queries from natural
#     language, holding live DB credentials, is an injection/data-loss risk
#     even for read-only requests (e.g. it could return an entire PII table).
#   - sensitive write/irreversible actions: sending messages, deleting data,
#     payments, shell/system commands.
# These require a human to hand-write and review a scoped, parameterized
# tool (same as stock_price/weather below) and register it manually via
# register_tool() — auto-creation refuses and explains why instead of
# silently producing something unsafe or something that just fails.
_DB_RISK_RE = re.compile(
    r"\b(sql|database|db connection|postgres|postgresql|mysql|sqlite|mongodb|"
    r"execute query|run (a |the )?query)\b", re.I,
)
_SENSITIVE_ACTION_RE = re.compile(
    r"\b(send (an? )?(email|message|sms)|smtp|delete|drop table|remove record|"
    r"transfer money|make a payment|wire transfer|shell command|subprocess|"
    r"system command|execute code)\b", re.I,
)


def classify_tool_risk(prompt: str) -> str | None:
    """Returns 'database_access', 'sensitive_action', or None (safe to auto-generate)."""
    if _DB_RISK_RE.search(prompt or ""):
        return "database_access"
    if _SENSITIVE_ACTION_RE.search(prompt or ""):
        return "sensitive_action"
    return None


_RISK_MESSAGES = {
    "database_access": (
        "This tool needs direct database access. Auto-generating live SQL "
        "execution from an LLM prompt is disabled — arbitrary or malformed "
        "queries against a real database are too risky to sandbox-generate, "
        "even read-only ones. Hand-write a parameterized tool (e.g. SQLAlchemy "
        "with a fixed, reviewed query and a read-only DB user/role) and "
        "register it directly via tool_registry.register_tool(), the same way "
        "stock_price and weather are registered below."
    ),
    "sensitive_action": (
        "This tool would take a sensitive or irreversible action (sending, "
        "deleting, paying, or running system commands). Auto-generation is "
        "disabled for this category. Hand-write and review this tool yourself, "
        "then register it via tool_registry.register_tool()."
    ),
}


class SecretsManager:
    """
    Narrow accessor for secrets, used both by hand-written tools (like your
    existing get_stock_price) and injected into the sandbox for generated
    tools via a wrapped get_secret() function (never raw os.environ).

    Checks os.environ first (so .env / real deployment secrets still work),
    then falls back to secrets set at runtime via `set()` (e.g. from a
    Streamlit sidebar field), so a user can supply a missing key without
    restarting the process.
    """
    def __init__(self):
        self._runtime_secrets: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return os.environ.get(name) or self._runtime_secrets.get(name)

    def set(self, name: str, value: str):
        self._runtime_secrets[name] = value

    def has(self, name: str) -> bool:
        val = self.get(name)
        return val is not None and val != ""

    def missing_from(self, names: list[str]) -> list[str]:
        return [n for n in (names or []) if not self.has(n)]


secrets_manager = SecretsManager()

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
        # Per-tool bookkeeping so the UI/agents can tell "works" apart from
        # "exists but isn't configured yet":
        #   status: "active" | "needs_package" | "needs_secret"
        #   missing_packages / missing_secrets: what's blocking it, if any
        #   required_secrets: full declared list (kept even once satisfied,
        #     so revalidate_tool() knows what to re-check)
        self.tool_status: dict[str, dict] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register built-in tools. stock_price/weather are 'secret-gated':
        registered immediately (visible in the UI) but their active status
        depends on whether their API key is already configured."""
        self.register_tool("search", self._search_tool())
        self.register_tool("calculator", self._calculator_tool())
        self._register_secret_gated_tool(
            "stock_price", self._stock_price_tool(), ["ALPHAVANTAGE_API_KEY"]
        )
        self._register_secret_gated_tool(
            "weather", self._weather_tool(), ["OPENWEATHER_API_KEY"]
        )

    def _register_secret_gated_tool(self, name: str, tool_obj: BaseTool, required_secrets: list[str]):
        missing = secrets_manager.missing_from(required_secrets)
        self.register_tool(
            name, tool_obj,
            status="needs_secret" if missing else "active",
            missing_secrets=missing,
            required_secrets=required_secrets,
        )

    def register_tool(
        self,
        name: str,
        tool_obj: BaseTool,
        code: str = "",
        status: str = "active",
        missing_packages: list[str] | None = None,
        missing_secrets: list[str] | None = None,
        required_secrets: list[str] | None = None,
    ):
        """
        Register a tool in the registry.

        Forces tool_obj.name to match the registry key. Without this, a tool
        whose underlying function is named differently from its registry key
        (e.g. get_stock_price registered under "stock_price", or any
        LLM-generated tool whose function name differs from the name typed
        in the UI) will bind to the LLM under its *real* name, so tool_calls
        come back with a name the registry doesn't recognize and
        `_tools_node` reports "tool not found" even though the tool exists.

        status/missing_* let a tool be registered (visible, inspectable)
        even when it can't run yet — e.g. it needs a package that isn't
        installed or a secret the user hasn't supplied. Default-registered
        built-in tools and hand-written tools are "active" with nothing missing.
        """
        tool_obj.name = name
        self.tools[name] = tool_obj
        self.tool_code[name] = code
        self.tool_status[name] = {
            "status": status,
            "missing_packages": missing_packages or [],
            "missing_secrets": missing_secrets or [],
            "required_secrets": required_secrets or [],
        }
        if status == "active":
            print(f"✅ Tool registered: {name}")
        else:
            print(f"🟡 Tool registered but not usable yet: {name} ({status}: "
                  f"{(missing_packages or []) + (missing_secrets or [])})")

    def get_all_tools(self) -> list[BaseTool]:
        """Get all registered tools as a list"""
        return list(self.tools.values())

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a specific tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> dict[str, str]:
        """List all tools with their descriptions"""
        return {name: tool.description for name, tool in self.tools.items()}

    def list_active_tools(self) -> dict[str, str]:
        """
        Descriptions of only tools that are actually runnable right now.
        Used when asking the LLM to select tools for an agent/task, so it
        never picks something still blocked on a missing package or secret
        (it would just fail at execution time otherwise).
        """
        return {
            name: tool.description
            for name, tool in self.tools.items()
            if self.tool_status.get(name, {}).get("status") == "active"
        }

    def list_tools_with_status(self) -> dict[str, dict]:
        """Full status detail per tool, for the sidebar / debugging."""
        return {
            name: {
                "description": tool.description,
                **self.tool_status.get(name, {"status": "active", "missing_packages": [], "missing_secrets": []}),
            }
            for name, tool in self.tools.items()
        }

    def list_pending_tools(self) -> dict[str, dict]:
        """Tools that were auto-created, passed their smoke test, but are
        awaiting human approval before any agent can use them."""
        return {
            name: {
                "description": self.tools[name].description,
                "code": self.tool_code.get(name, ""),
            }
            for name, info in self.tool_status.items()
            if info.get("status") == "pending_approval"
        }

    def approve_tool(self, name: str) -> dict:
        """User clicked Continue: activate a pending tool so agents can
        pick it up on their next tool-refresh."""
        info = self.tool_status.get(name)
        if info is None:
            return {"status": "unknown"}
        info["status"] = "active"
        self.tool_status[name] = info
        print(f"✅ Tool '{name}' approved by user and activated.")
        return info

    def reject_tool(self, name: str):
        """User clicked Reject: discard the generated tool entirely."""
        self.tools.pop(name, None)
        self.tool_code.pop(name, None)
        self.tool_status.pop(name, None)
        print(f"🗑️ Tool '{name}' rejected by user and removed.")

    def get_missing_secrets(self) -> list[str]:
        """Union of every secret name any registered tool is still waiting on."""
        names = set()
        for info in self.tool_status.values():
            names.update(info.get("missing_secrets", []))
        return sorted(names)

    def revalidate_tool(self, name: str) -> dict:
        """
        Re-check a tool's status against the current secrets_manager state
        (e.g. after the user supplies a key in the sidebar) without
        regenerating its code. Re-runs the smoke test if all its declared
        secrets are now present.
        """
        info = self.tool_status.get(name)
        tool_obj = self.tools.get(name)
        if info is None or tool_obj is None:
            return {"status": "unknown"}

        required_secrets = info.get("required_secrets", [])
        still_missing_secrets = secrets_manager.missing_from(required_secrets)

        if info.get("missing_packages"):
            # Packages can't be fixed at runtime by this process.
            info["status"] = "needs_package"
            self.tool_status[name] = info
            return info

        if still_missing_secrets:
            info["status"] = "needs_secret"
            info["missing_secrets"] = still_missing_secrets
            self.tool_status[name] = info
            return info

        ok, msg = self._smoke_test_tool(tool_obj)
        info["missing_secrets"] = []
        info["status"] = "active" if ok else "needs_package"
        if not ok:
            print(f"⚠️ Revalidation smoke test for '{name}' still failing: {msg}")
        self.tool_status[name] = info
        return info

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
            api_key = secrets_manager.get("ALPHAVANTAGE_API_KEY")
            if not api_key:
                return {"error": "ALPHAVANTAGE_API_KEY not set"}
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
            try:
                r = requests.get(url, timeout=5)
                return r.json()
            except Exception as e:
                return {"error": str(e)}
        return get_stock_price

    @staticmethod
    def _weather_tool() -> BaseTool:
        @tool
        def get_weather(city: str) -> dict:
            """Fetch current weather for a city name (e.g., 'Surat', 'London')"""
            api_key = secrets_manager.get("OPENWEATHER_API_KEY")
            if not api_key:
                return {"error": "OPENWEATHER_API_KEY not set"}
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            try:
                r = requests.get(url, timeout=5)
                return r.json()
            except Exception as e:
                return {"error": str(e)}
        return get_weather

    @staticmethod
    def _make_stub_tool(name: str, message: str) -> BaseTool:
        """A placeholder tool for requests that were refused auto-generation
        (see classify_tool_risk). It's registered so the name shows up in the
        UI with a clear explanation, but its status is never 'active' so
        agents can never select or call it."""
        @tool
        def stub(input_data: str = "") -> dict:
            """Placeholder — this tool requires manual integration."""
            return {"error": message}
        return stub

    @traceable(name="create_tool_from_prompt", run_type="chain")
    def create_tool_from_prompt(self, prompt: str, tool_name: str, require_approval: bool = False) -> BaseTool:
        """
        Create a new tool from a natural language prompt.
        Code is executed in a restricted-builtins sandbox: no file/network-escape
        primitives (open, os, subprocess, eval, __import__, etc.) are exposed.

        require_approval=True registers a successfully-generated tool with
        status "pending_approval" instead of "active": it's created and
        smoke-tested, but excluded from list_active_tools() (so no agent can
        be bound to it) until a human approves it via approve_tool(). Used
        for tools the system decides it needs on its own (auto-creation
        during agent execution). Manual creation via the sidebar form
        already IS the human approval step, so it's called with the default
        require_approval=False and goes active immediately.

        Wrapped in @traceable so LangSmith shows this as one named span with
        the requested tool_name + prompt as inputs, letting you inspect
        exactly what was asked for and what got generated without reading logs.
        """
        print(f"🔨 Creating tool '{tool_name}' from prompt...")

        # NOTE: hard risk-based refusal (classify_tool_risk / _RISK_MESSAGES,
        # still defined above) is intentionally NOT enforced here for this
        # demo build — everything runs in a test environment. Re-enable by
        # uncommenting the block below once this needs to run against real
        # systems/data:
        #
        # risk = classify_tool_risk(prompt)
        # if risk:
        #     message = _RISK_MESSAGES[risk]
        #     stub = self._make_stub_tool(tool_name, message)
        #     self.register_tool(tool_name, stub, status="needs_manual_integration")
        #     return stub

        available_module_names = list(ALLOWED_EXTRA_MODULES.keys())

        creation_prompt = f"""
        Create a Python tool function for the following requirement:
        {prompt}

        Requirements:
        1. Use the @tool decorator from langchain_core.tools
        2. Include a clear docstring
        3. Handle errors gracefully
        4. Return JSON-serializable data
        5. Keep it focused and simple
        6. Do not use file I/O, subprocess, eval, or exec
        7. Do NOT write import statements. The following modules are already
           available as global names if you need them: requests, json, datetime, re,
           {", ".join(available_module_names) if available_module_names else "(none extra installed)"}.
           If you genuinely need a package not in that list, still write the
           code using it under its normal name (e.g. "openpyxl") — just list it
           in required_packages below rather than importing it; the tool will
           be created but marked unusable until that package is added.
        8. If this tool needs an API key / credential / secret, do NOT use
           os.environ. Instead call get_secret("EXACT_ENV_VAR_NAME") to
           retrieve it, and list that exact name in required_secrets below.

        Return ONLY JSON in this exact shape, no prose, no markdown fences:
        {{
          "code": "the full @tool-decorated function as a string",
          "required_packages": ["pkg1"],
          "required_secrets": ["SOME_API_KEY"]
        }}
        Leave required_packages/required_secrets as empty lists if none are needed.
        """

        response = llm.invoke(
            creation_prompt,
            config={
                "run_name": "tool_codegen_llm",
                "tags": ["tool_creation", tool_name],
                "metadata": {"tool_name": tool_name},
            },
        )
        spec = parse_json_safely(response.content, default=None)
        if spec is None or not spec.get("code"):
            # Fall back to treating the whole response as raw code, for
            # models that ignore the JSON-envelope instruction.
            spec = {"code": response.content, "required_packages": [], "required_secrets": []}
        tool_code = spec.get("code", "")
        declared_packages = [p for p in (spec.get("required_packages") or []) if p]
        declared_secrets = [s for s in (spec.get("required_secrets") or []) if s]
        print(f"📝 Generated code for tool '{tool_name}' (packages={declared_packages}, "
              f"secrets={declared_secrets}):\n{tool_code}")

        # Packages the LLM asked for that we don't actually have pre-imported.
        # This is a real capability gap, not something a repair loop can fix —
        # it needs a human to `pip install` it and add it to
        # _OPTIONAL_MODULES, then recreate the tool.
        missing_packages = [p for p in declared_packages if p not in ALLOWED_EXTRA_MODULES]
        if missing_packages:
            print(f"📦 Tool '{tool_name}' needs package(s) not available in this "
                  f"environment: {missing_packages}. Registering as inactive.")

        # --- Sandboxed execution environment ---
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
            "TypeError": TypeError, "KeyError": KeyError, "RuntimeError": RuntimeError,
            "PermissionError": PermissionError, "True": True,
            "False": False, "None": None,
        }

        def _make_get_secret(declared: list[str]):
            def get_secret(name: str):
                if name not in declared:
                    raise PermissionError(
                        f"Tool '{tool_name}' tried to read secret '{name}' without "
                        f"declaring it in required_secrets."
                    )
                value = secrets_manager.get(name)
                if value is None:
                    raise RuntimeError(f"MISSING_SECRET:{name}")
                return value
            return get_secret

        safe_globals = {
            "__builtins__": safe_builtins,
            "tool": tool,
            "requests": requests,
            "json": json,
            "datetime": datetime,
            "re": re,
            "get_secret": _make_get_secret(declared_secrets),
            **{k: v for k, v in ALLOWED_EXTRA_MODULES.items() if k in declared_packages},
        }

        current_code = tool_code
        last_error = None
        missing_secret_names: list[str] = []

        # Phase 2: don't trust generated code just because exec() didn't
        # raise. Run it once against synthetic args and confirm the result
        # is JSON-serializable — the two failure modes the fixed prompt above
        # asks the LLM to avoid ("handle errors gracefully", "return
        # JSON-serializable data") but doesn't guarantee. A broken tool that
        # only fails this way would otherwise only be discovered mid-task,
        # inside a live user-facing run. On failure, we branch:
        #   - a missing secret is a config gap -> register as "needs_secret",
        #     no point burning a repair attempt rewriting code that's already
        #     correct
        #   - anything else is treated as a real code bug -> feed the exact
        #     error back to the LLM for one repair attempt before giving up
        for attempt in range(MAX_TOOL_VALIDATION_RETRIES + 1):
            # Strip any import lines the LLM wrote anyway (the sandbox has no
            # __import__, so these would only ever hard-fail; better to drop
            # them and let a genuine missing-name error surface cleanly).
            cleaned_code = _IMPORT_LINE_RE.sub("", self._extract_code_block(current_code))
            try:
                local_vars = {}
                exec(cleaned_code, safe_globals, local_vars)

                created_tool = next(
                    (v for v in local_vars.values() if isinstance(v, BaseTool)), None
                )
                if created_tool is None:
                    raise ValueError("Generated code did not define a @tool-decorated function.")

                if missing_packages:
                    # Code exists and parses, but a dependency isn't
                    # installed here -> register inactive, skip smoke test
                    # (it would just fail on the same missing name).
                    self.register_tool(
                        tool_name, created_tool, cleaned_code,
                        status="needs_package", missing_packages=missing_packages,
                        required_secrets=declared_secrets,
                    )
                    print(f"🟡 Tool '{tool_name}' registered but inactive — missing package(s): {missing_packages}")
                    return created_tool

                still_missing_secrets = secrets_manager.missing_from(declared_secrets)
                if still_missing_secrets:
                    self.register_tool(
                        tool_name, created_tool, cleaned_code,
                        status="needs_secret", missing_secrets=still_missing_secrets,
                        required_secrets=declared_secrets,
                    )
                    print(f"🔑 Tool '{tool_name}' registered but inactive — missing secret(s): {still_missing_secrets}")
                    return created_tool

                ok, smoke_message = self._smoke_test_tool(created_tool)
                if not ok:
                    if smoke_message.startswith("RuntimeError: MISSING_SECRET:"):
                        missing_secret_names = [smoke_message.split("MISSING_SECRET:", 1)[1]]
                        raise RuntimeError(f"missing_secret:{missing_secret_names}")
                    raise RuntimeError(f"Smoke test failed: {smoke_message}")

                final_status = "pending_approval" if require_approval else "active"
                self.register_tool(
                    tool_name, created_tool, cleaned_code,
                    status=final_status, required_secrets=declared_secrets,
                )
                if final_status == "pending_approval":
                    print(f"⏸️ Tool '{tool_name}' created and passed smoke test — awaiting approval.")
                else:
                    print(f"✅ Tool '{tool_name}' created and passed smoke test.")
                return created_tool

            except RuntimeError as e:
                if str(e).startswith("missing_secret:"):
                    created_tool = next(
                        (v for v in local_vars.values() if isinstance(v, BaseTool)), None
                    )
                    if created_tool is not None:
                        self.register_tool(
                            tool_name, created_tool, cleaned_code,
                            status="needs_secret", missing_secrets=missing_secret_names,
                            required_secrets=declared_secrets,
                        )
                        print(f"🔑 Tool '{tool_name}' registered but inactive — missing secret(s): {missing_secret_names}")
                        return created_tool
                last_error = str(e)
            except Exception as e:
                last_error = str(e)

            print(f"⚠️ Tool '{tool_name}' generation attempt {attempt + 1} failed: {last_error}")

            if attempt >= MAX_TOOL_VALIDATION_RETRIES:
                break

            repair_prompt = f"""
                The following tool code failed with this error:
                {last_error}

                Original code:
                {cleaned_code}

                Fix the code so it runs correctly and satisfies the original requirement:
                {prompt}

                Same rules as before: use @tool, no file I/O/subprocess/eval/exec/import
                statements, handle errors gracefully, return JSON-serializable data.
                Available pre-imported modules: requests, json, datetime, re,
                {", ".join(k for k in declared_packages if k in ALLOWED_EXTRA_MODULES) or "(none)"}.
                For secrets use get_secret("NAME"), never os.environ.
                Return ONLY the corrected function code, starting with @tool.
            """
            response = llm.invoke(
                repair_prompt,
                config={
                    "run_name": "tool_codegen_repair_llm",
                    "tags": ["tool_creation", "repair", tool_name],
                    "metadata": {"tool_name": tool_name, "error": last_error},
                },
            )
            current_code = response.content

        print(f"❌ Error creating tool '{tool_name}': {last_error}")
        raise RuntimeError(
            f"Failed to create a working tool '{tool_name}' after "
            f"{MAX_TOOL_VALIDATION_RETRIES + 1} attempt(s). Last error: {last_error}"
        )

    @staticmethod
    def _extract_code_block(raw: str) -> str:
        if "```python" in raw:
            return raw.split("```python")[1].split("```")[0]
        elif "```" in raw:
            return raw.split("```")[1].split("```")[0]
        return raw

    @staticmethod
    def _generate_dummy_args(tool_obj: BaseTool) -> dict:
        """Build plausible synthetic args from a tool's schema for smoke testing."""
        dummy = {}
        try:
            schema_props = tool_obj.args or {}
        except Exception:
            schema_props = {}
        for field_name, field_info in schema_props.items():
            field_type = (field_info or {}).get("type", "string")
            if field_type == "integer":
                dummy[field_name] = 1
            elif field_type == "number":
                dummy[field_name] = 1.0
            elif field_type == "boolean":
                dummy[field_name] = True
            elif field_type == "array":
                dummy[field_name] = []
            elif field_type == "object":
                dummy[field_name] = {}
            else:
                dummy[field_name] = "test"
        return dummy

    def _smoke_test_tool(self, tool_obj: BaseTool) -> tuple[bool, str]:
        """
        Invoke a freshly generated tool once with synthetic args to catch
        obvious runtime errors (undefined names, wrong signature, missing
        deps, non-JSON-serializable return) before it's ever handed to a
        live agent. This does NOT validate correctness of the tool's logic
        (that would need real credentials/data) — only that it runs and
        returns something usable.
        """
        dummy_args = self._generate_dummy_args(tool_obj)
        try:
            result = tool_obj.invoke(dummy_args)
            json.dumps(result, default=str)
            return True, "ok"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"


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

    @traceable(name="auto_create_missing_tools", run_type="chain")
    def _create_missing_tools(self, specs: list[dict]) -> list[str]:
        """
        Auto-create tools an LLM tool-selection step flagged as missing
        (Phase 1: closes the loop so tool creation happens as part of the
        agent's own workflow, not only via the manual sidebar form).

        Capped at AUTO_TOOL_LIMIT per call to avoid runaway generation from
        a single planning/refresh step. Failures are logged and skipped —
        a bad auto-generated tool must not crash the task in progress.
        """
        created = []
        for spec in (specs or [])[:AUTO_TOOL_LIMIT]:
            name = (spec or {}).get("name")
            prompt = (spec or {}).get("prompt")
            if not name or not prompt:
                continue
            if self.tool_registry.get_tool(name) is not None:
                # Already exists (LLM re-proposed a known tool). Only worth
                # adding to this agent's set if it's actually usable — an
                # existing tool stuck on "needs_package"/"needs_secret"
                # would just fail if the agent tried to call it.
                status = self.tool_registry.tool_status.get(name, {}).get("status")
                if status == "active":
                    created.append(name)
                else:
                    print(f"⏭️ Tool '{name}' already exists but is '{status}' — not attaching to agent.")
                continue
            try:
                print(f"⚡ Auto-creating missing tool '{name}' from agent workflow...")
                self.tool_registry.create_tool_from_prompt(prompt, name, require_approval=True)
                status = self.tool_registry.tool_status.get(name, {}).get("status")
                if status == "active":
                    created.append(name)
                else:
                    print(f"🟡 Tool '{name}' created but not yet usable ({status}) — "
                          f"it will be attached once approved/configured.")
            except Exception as e:
                print(f"❌ Auto tool creation failed for '{name}': {e}")
        return created

    @traceable(name="refresh_agent_tools", run_type="chain")
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
        available_tools = self.tool_registry.list_active_tools()

        selection_prompt = f"""
The "{role}" agent currently has these tools: {agent_conf['tool_names']}
It now needs to handle this task: "{task_description}"

Available tools (name: description): {json.dumps(available_tools)}

Return ONLY JSON in this exact shape:
{{
  "tools": ["tool_name1", "tool_name2"],
  "new_tools": [{{"name": "snake_case_tool_name", "prompt": "one self-contained instruction describing what this new tool must do"}}]
}}

Rules:
- "tools": its current tools plus any additional EXISTING tools (from the
  Available tools list) this new task genuinely needs. Do not drop a
  currently held tool unless it is clearly irrelevant.
- "new_tools": ONLY include an entry here if none of the available tools can
  do something this task clearly requires (e.g. reading email, sending a
  message, converting a file format, calling a specific API). Leave it as
  an empty list if the available tools are sufficient.
- Do not propose more than {AUTO_TOOL_LIMIT} new tools.
"""
        response = llm.invoke(
            selection_prompt,
            config={
                "run_name": "agent_tool_refresh_llm",
                "tags": ["tool_refresh", role],
                "metadata": {"role": role, "task_description": task_description},
            },
        )
        result = parse_json_safely(response.content, default=None)
        if result is None:
            print(f"⚠️ Tool refresh JSON parse failed for role '{role}'; keeping existing tools.")
            return agent_conf

        created_tool_names = self._create_missing_tools(result.get("new_tools", []))

        desired_names = (
            set(result.get("tools", []))
            | set(agent_conf["tool_names"])
            | set(created_tool_names)
        )
        if desired_names == set(agent_conf["tool_names"]):
            return agent_conf

        selected_tools = [
            self.tool_registry.get_tool(name)
            for name in desired_names
            if self.tool_registry.get_tool(name) is not None
            and self.tool_registry.tool_status.get(name, {}).get("status") == "active"
        ]
        agent_conf["llm"] = llm.bind_tools(selected_tools) if selected_tools else llm
        agent_conf["tool_names"] = [t.name for t in selected_tools]
        print(f"🔄 Agent '{role}' tools refreshed: {agent_conf['tool_names']}")
        return agent_conf

    def tool_directive(self, tool_names: list[str]) -> str:
        """
        A fixed, non-negotiable instruction appended to every agent's system
        prompt at execution time (see DynamicAgentManager._agent_executor_node).

        Left to its own devices, the LLM used here (openai/gpt-oss-120b) will
        often respond "I can't browse the internet, please provide a URL"
        even when a real `search` tool is bound and ready to call — the
        dynamically generated persona prompt doesn't reliably force tool use
        on its own. This directive is regenerated from the agent's *current*
        tool_names each time a task runs, so it stays correct even after
        refresh_tools() changes what an agent has access to.
        """
        if not tool_names:
            return ""
        descriptions = self.tool_registry.list_tools()
        lines = [f'- {name}: {descriptions.get(name, "")}' for name in tool_names]
        return (
            "\n\nTOOL USE IS MANDATORY WHEN RELEVANT. You have real, working tools:\n"
            + "\n".join(lines)
            + "\n\nIf completing this task requires information you don't already "
            "have with certainty (current events, prices, live data, computation, "
            "etc.), you MUST call the appropriate tool above to get it. Do NOT say "
            "you cannot browse the internet, do NOT ask the user to supply data "
            "that one of your tools can retrieve, and do NOT answer from memory "
            "when a tool exists to get current facts. Only ask the user a question "
            "if none of your tools can obtain what's needed."
        )

    @traceable(name="create_agent", run_type="chain")
    def create_agent(self, role: str, task_description: str) -> dict:
        print(f"🧬 Creating new agent for role: '{role}'")
        available_tools = self.tool_registry.list_active_tools()

        selection_prompt = f"""
You are configuring a specialized AI agent for the role "{role}".
It will handle tasks like: "{task_description}"

Available tools (name: description): {json.dumps(available_tools)}

Return ONLY JSON in this exact shape:
{{
  "system_prompt": "a system prompt defining this agent's persona, scope and behavior",
  "tools": ["tool_name1", "tool_name2"],
  "new_tools": [{{"name": "snake_case_tool_name", "prompt": "one self-contained instruction describing what this new tool must do"}}]
}}

Rules:
- "tools": only names that already appear in the Available tools list above.
  Select tools genuinely relevant to this role. It is fine to select zero.
- "new_tools": ONLY include an entry here if none of the available tools can
  do something this role clearly needs (e.g. reading email, sending a
  message, converting a file format, calling a specific external API).
  Leave it as an empty list if the available tools are sufficient.
- Do not propose more than {AUTO_TOOL_LIMIT} new tools.
"""
        response = llm.invoke(
            selection_prompt,
            config={
                "run_name": "agent_creation_llm",
                "tags": ["agent_creation", role],
                "metadata": {"role": role, "task_description": task_description},
            },
        )
        config = parse_json_safely(response.content, default=None)
        if config is None:
            print(f"⚠️ Tool selection JSON parse failed for role '{role}'. Raw response: {response.content[:300]!r}")
            config = {
                "system_prompt": f"You are a focused, helpful '{role}' agent. Be concise and accurate.",
                "tools": [],
                "new_tools": [],
            }

        created_tool_names = self._create_missing_tools(config.get("new_tools", []))
        all_tool_names = list(dict.fromkeys(config.get("tools", []) + created_tool_names))

        selected_tools = [
            self.tool_registry.get_tool(name)
            for name in all_tool_names
            if self.tool_registry.get_tool(name) is not None
            and self.tool_registry.tool_status.get(name, {}).get("status") == "active"
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
        response = llm.invoke(
            planning_prompt,
            config={
                "run_name": "planner_llm",
                "tags": ["planner"],
                "metadata": {"goal": goal},
            },
        )
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

            # Compose the full system prompt fresh from the agent's *current*
            # tool_names rather than using a static stored string, so the
            # mandatory tool-use directive always reflects whatever tools
            # refresh_tools() has most recently attached to this agent.
            full_system_prompt = agent["system_prompt"] + self.agent_factory.tool_directive(agent["tool_names"])
            task_messages = [
                SystemMessage(content=full_system_prompt),
                HumanMessage(content=init_prompt),
            ]

        print(f"🤖 [{role}] executing task {task['id']}: {task['description']}")
        response = agent["llm"].invoke(
            task_messages,
            config={
                "run_name": f"agent_exec:{role}",
                "tags": ["agent_executor", role, task["id"]],
                "metadata": {"role": role, "task_id": task["id"], "task_description": task["description"]},
            },
        )
        task_messages = task_messages + [response]

        return {"task_messages": task_messages}

    def _tools_node(self, state: OrchestratorState):
        """Custom tool executor operating on task_messages (not the main transcript)."""
        idx = state["current_task_idx"]
        task = state["task_plan"][idx] if idx < len(state["task_plan"]) else {"id": "?", "agent_role": "?"}
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
                    # config here is what makes each tool call show up in
                    # LangSmith as its own named "tool:<name>" span with
                    # tool_args as input and the return value as output —
                    # this is the "what tool got used, what was its output"
                    # visibility, without reading console logs.
                    result = tool_obj.invoke(
                        tool_args,
                        config={
                            "run_name": f"tool:{tool_name}",
                            "tags": ["tool_call", tool_name, task["id"]],
                            "metadata": {"task_id": task["id"], "agent_role": task.get("agent_role", "?")},
                        },
                    )
            except Exception as e:
                err_text = str(e)
                if err_text.startswith("MISSING_SECRET:"):
                    secret_name = err_text.split("MISSING_SECRET:", 1)[1]
                    result = (
                        f"Error: tool '{tool_name}' needs the secret '{secret_name}' to be "
                        f"configured before it can run. Tell the user to supply it (e.g. via "
                        f"the sidebar) rather than retrying this call."
                    )
                else:
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
        eval_response = llm.invoke(
            eval_prompt,
            config={
                "run_name": f"evaluator:{task['id']}",
                "tags": ["evaluator", task["agent_role"], task["id"]],
                "metadata": {"task_id": task["id"], "role": task["agent_role"]},
            },
        )
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

        final = llm.invoke(
            summary_prompt,
            config={
                "run_name": "assembler_llm",
                "tags": ["assembler"],
                "metadata": {"goal": state["goal"]},
            },
        )
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

    def get_tool_status_info(self) -> str:
        """Full status per tool (active / needs_package / needs_secret), for the sidebar."""
        return json.dumps(self.tool_registry.list_tools_with_status(), indent=2)

    def get_missing_secrets(self) -> list[str]:
        return self.tool_registry.get_missing_secrets()

    def get_pending_tools_info(self) -> str:
        return json.dumps(self.tool_registry.list_pending_tools(), indent=2)

    def approve_tool(self, name: str):
        return self.tool_registry.approve_tool(name)

    def reject_tool(self, name: str):
        self.tool_registry.reject_tool(name)

    def set_secret(self, name: str, value: str):
        """Supply a secret at runtime (e.g. from a sidebar field) and
        re-check every tool waiting on it so it can flip to 'active'
        immediately, without regenerating any code."""
        secrets_manager.set(name, value)
        for tool_name in list(self.tool_registry.tools.keys()):
            self.tool_registry.revalidate_tool(tool_name)

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

        config = {
            "configurable": {"thread_id": thread_id},
            "run_name": "orchestrator_run",
            "tags": ["orchestrator_run"],
            "metadata": {"goal": user_input, "thread_id": thread_id},
        }
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


def get_tool_status() -> str:
    return agent_manager.get_tool_status_info()


def get_missing_secrets() -> list[str]:
    return agent_manager.get_missing_secrets()


def set_secret_dynamically(name: str, value: str):
    agent_manager.set_secret(name, value)


def get_pending_tools() -> str:
    return agent_manager.get_pending_tools_info()


def approve_pending_tool(name: str):
    return agent_manager.approve_tool(name)


def reject_pending_tool(name: str):
    agent_manager.reject_tool(name)


def add_tool_dynamically(tool_name: str, tool_prompt: str) -> bool:
    return agent_manager.add_tool_from_prompt(tool_prompt, tool_name)


def run_agent_with_requirements(user_input: str, thread_id: str, requirements: dict = None):
    return agent_manager.run(user_input, thread_id, requirements)