"""
sandbox.py
==========
Isolated subprocess execution environment for dynamically generated tools.

Extracted from dynamic_langgraph_backend.py to keep the tool execution
infrastructure separate from the LangGraph orchestration logic.

How it works
------------
Every tool created via DynamicToolRegistry.create_tool_from_prompt() runs
inside a single shared Python venv. Each tool call is executed as a
subprocess via a thin runner script (_SANDBOX_RUNNER_SOURCE), so:

  1. Broken or hung tool code cannot crash or block the main process.
  2. Third-party pip packages (requests, pandas, etc.) can be imported
     freely by generated tools without touching the main process's env.
  3. A persistent install cache (_installed.json) avoids reinstalling
     packages that are already present in the shared venv.

Trade-off: one shared venv means two tools that need conflicting versions
of the same package can collide. Acceptable given AUTO_TOOL_LIMIT; if it
ever bites, give the offending tool its own venv by passing
base_dir="tool_envs/<tool_name>" instead of "tool_envs/shared".

Public API
----------
ToolSandboxExecutor             -- manages the shared venv + per-call subprocess
  .ensure_env(tool, reqs)       -- create venv + install missing packages
  .save_tool_module(tool, code) -- persist generated source to tools folder
  .run(tool, func, kwargs)      -- execute one tool call, return JSON result
"""

import os
import sys
import json
import venv
import subprocess
from typing import Any


# ---------------------------------------------------------------------------
# Thin runner script written into the shared venv at startup.
# The executor calls this script as a subprocess for every tool invocation.
# ---------------------------------------------------------------------------

_SANDBOX_RUNNER_SOURCE = '''
import sys, json, importlib.util

def main():
    module_path, func_name = sys.argv[1], sys.argv[2]
    kwargs = json.loads(sys.stdin.read() or "{}")

    spec = importlib.util.spec_from_file_location("generated_tool_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target = getattr(module, func_name)
    result = target(**kwargs)

    print(json.dumps({"ok": True, "result": result}, default=str))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)
'''


class ToolSandboxExecutor:
    """
    Runs every dynamically created tool's generated code through ONE shared
    venv + a subprocess per call, so tool code can `import` any pip package
    without the main process ever needing that package (or being able to be
    crashed/hung by broken generated code).

    Trade-off vs a venv per tool: one shared site-packages means two tools
    that need conflicting versions of the same package can step on each
    other. Acceptable for an initial build where AUTO_TOOL_LIMIT already
    caps tool creation; if that ever bites, give just the conflicting tool
    its own venv (base_dir=f"tool_envs/{tool_name}") rather than switching
    the whole system over.
    """

    def __init__(
        self,
        base_dir: str = "tool_envs/shared",
        timeout_s: int = 25,
        install_timeout_s: int = 180,
    ):
        self.base_dir = base_dir
        self.timeout_s = timeout_s
        self.install_timeout_s = install_timeout_s
        self.tools_dir = os.path.join(base_dir, "tools")
        os.makedirs(self.tools_dir, exist_ok=True)

        # Write the runner script into the shared env directory
        self._runner_path = os.path.join(base_dir, "_sandbox_runner.py")
        with open(self._runner_path, "w", encoding="utf-8") as f:
            f.write(_SANDBOX_RUNNER_SOURCE)

        self._venv_dir = os.path.join(base_dir, "venv")
        self._installed_cache_path = os.path.join(base_dir, "_installed.json")
        self._installed: set[str] = set()
        self._load_installed_cache()

    # ------------------------------------------------------------------
    # venv / package management
    # ------------------------------------------------------------------

    def _venv_python(self) -> str:
        rel = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
        return os.path.join(self._venv_dir, *rel)

    def _module_path(self, tool_name: str) -> str:
        return os.path.join(self.tools_dir, f"{tool_name}.py")

    def _load_installed_cache(self):
        if os.path.exists(self._installed_cache_path):
            try:
                with open(self._installed_cache_path, "r", encoding="utf-8") as f:
                    self._installed = set(json.load(f))
            except (json.JSONDecodeError, OSError):
                self._installed = set()

    def _save_installed_cache(self):
        with open(self._installed_cache_path, "w", encoding="utf-8") as f:
            json.dump(sorted(self._installed), f)

    def ensure_env(self, tool_name: str, requirements: list[str] | None = None):
        """Create the shared venv on first use, then install only the packages not already present."""
        # Import here to avoid circular dependency; logging_utils has no
        # dependency on sandbox so the import is always safe.
        from logging_utils import _log

        if not os.path.exists(self._venv_dir):
            _log("SANDBOX", "Creating shared tool virtual environment",
                 tool=tool_name, path=self._venv_dir)
            venv.EnvBuilder(with_pip=True, clear=True).create(self._venv_dir)

        missing = [r for r in (requirements or []) if r.lower() not in self._installed]
        if missing:
            _log("SANDBOX", "Installing tool dependencies", tool=tool_name, packages=missing)
            proc = subprocess.run(
                [
                    self._venv_python(), "-m", "pip", "install", "-q",
                    "--disable-pip-version-check", *missing,
                ],
                capture_output=True, text=True, timeout=self.install_timeout_s,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Failed to install requirements for tool '{tool_name}': {proc.stderr[-800:]}"
                )
            self._installed.update(r.lower() for r in missing)
            self._save_installed_cache()
        elif requirements:
            _log("SANDBOX", "Dependencies already available; install skipped",
                 tool=tool_name, packages=requirements)

    def save_tool_module(self, tool_name: str, code: str) -> str:
        """Persist the generated (import-allowed) source into the shared tools folder."""
        path = self._module_path(tool_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        return path

    # ------------------------------------------------------------------
    # Subprocess execution
    # ------------------------------------------------------------------

    def run(self, tool_name: str, func_name: str, kwargs: dict) -> Any:
        """Execute one tool call inside the shared venv subprocess and return its result."""
        module_path = self._module_path(tool_name)
        try:
            proc = subprocess.run(
                [self._venv_python(), self._runner_path, module_path, func_name],
                input=json.dumps(kwargs),
                capture_output=True, text=True, timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Tool '{tool_name}' timed out after {self.timeout_s}s")

        if not proc.stdout.strip():
            raise RuntimeError(
                f"Tool '{tool_name}' produced no output. stderr: {proc.stderr[-800:]}"
            )

        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Tool '{tool_name}' returned non-JSON output: {proc.stdout[-800:]}"
            )

        if not payload.get("ok"):
            raise RuntimeError(f"Tool '{tool_name}' error: {payload.get('error')}")
        return payload["result"]
