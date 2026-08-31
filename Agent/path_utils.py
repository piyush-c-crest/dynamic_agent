"""
path_utils.py
=============
Path confinement and workspace utilities for the Dynamic Agent Orchestration system.

Extracted from dynamic_langgraph_backend.py so that tool implementations and
the agent manager share one authoritative definition of "safe paths".

Responsibilities
----------------
- Define the active working directory (per-run via contextvars)
- Canonicalize & confine any agent-requested path to the active workdir
- Reject paths that escape the workdir via '..' or symlinks
- Reject paths that point into the application's own internal directories
  (DB, tool_envs, tools, logs)

Public API
----------
current_workdir()               -- the workdir active for the current run
current_artifacts_dir()         -- .artifacts/ sub-folder inside current_workdir
resolve_and_confine(path_str)   -- canonicalize + confinement check
PathConfinementError            -- raised when a path escapes the workdir
WorkdirSelectionError           -- raised when a user-chosen workdir is invalid
_path_touches_agent_data(path)  -- True if path is inside an internal data dir
_APP_DATA_ERROR                 -- standard error string for the above
DEFAULT_AGENT_WORKDIR           -- fallback workspace path (from AGENT_WORKDIR env)
"""

import os
import contextvars
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_AGENT_WORKDIR = Path(
    os.environ.get("AGENT_WORKDIR", os.path.join(os.getcwd(), "agent_workspace"))
).resolve()
DEFAULT_AGENT_WORKDIR.mkdir(parents=True, exist_ok=True)

# Internal directories the agent must never read from or write to
_AGENT_DATA_DIR_NAMES = ("DB", "tool_envs", "tools", "logs")


def _agent_data_roots() -> list[Path]:
    return [Path(name).resolve() for name in _AGENT_DATA_DIR_NAMES]


_APP_DATA_ERROR = (
    "Refused: this path is inside the application's internal data directory "
    "(DB, tool_envs, tools, logs) -- never a target for agent file operations."
)


# ---------------------------------------------------------------------------
# Per-run workdir context variable
# ---------------------------------------------------------------------------

# Each call to DynamicAgentManager.run() sets this so all tool closures
# automatically scope themselves to the right folder without needing it
# threaded through every function signature.
_workdir_ctx: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "workdir_ctx", default=None
)


def current_workdir() -> Path:
    """The workdir tools should confine themselves to for the run currently
    in progress: the thread's selected cowork folder if one is set, else
    DEFAULT_AGENT_WORKDIR."""
    return _workdir_ctx.get() or DEFAULT_AGENT_WORKDIR


def current_artifacts_dir() -> Path:
    """Artifacts live inside a `.artifacts` folder of whichever workdir is
    active, so a report the agent creates while cowork'd into a user's
    folder actually shows up there, rather than always landing in the
    default workspace."""
    d = current_workdir() / ".artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class PathConfinementError(Exception):
    """Raised when a tool-requested path would escape the active workdir."""


class WorkdirSelectionError(Exception):
    """Raised when a user-requested working directory is invalid or refused."""


# ---------------------------------------------------------------------------
# Confinement helpers
# ---------------------------------------------------------------------------

def resolve_and_confine(path_str: str | None, base: Path = None) -> Path:
    """Canonicalize a path relative to `base` (default: the active workdir)
    and reject any escape (via '..' or a symlink) outside it."""
    base = (base or current_workdir()).resolve()
    candidate = (base / path_str) if path_str else base
    resolved = candidate.resolve()
    if not (resolved == base or resolved.is_relative_to(base)):
        raise PathConfinementError(
            f"Path escapes the agent workdir ({base}): {path_str!r}"
        )
    return resolved


def _path_touches_agent_data(path: Path) -> bool:
    """Return True if `path` is inside one of the application's internal data
    directories (DB, tool_envs, tools, logs)."""
    return any(
        path == root or path.is_relative_to(root)
        for root in _agent_data_roots()
    )
