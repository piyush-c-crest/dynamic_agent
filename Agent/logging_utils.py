"""
logging_utils.py
================
Structured, colored console logging for the Dynamic Agent Orchestration system.

Extracted from dynamic_langgraph_backend.py so that any module (sandbox,
artifact_builder, main, etc.) can import _log / _log_block without pulling
in the entire LangGraph orchestration stack.

Public API
----------
_log(tag, msg, **fields)        -- single-line structured event
_log_block(tag, title, body)    -- multi-line payload (prompt / reply / tool output)
LOG_FILE_PATH                   -- path to the current run's plain-text log file
_Ansi                           -- ANSI color constants (for other modules that need them)
"""

import os
import re
import sys
import json
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class _Ansi:
    """ANSI styles used for human-readable terminal logs.

    The log file always has these styles removed, so it remains easy to grep,
    copy into an issue, or parse with a log collector.
    """

    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


_LOG_TAG_COLORS = {
    "SYSTEM": _Ansi.CYAN,
    "WORKFLOW": _Ansi.BLUE,
    "AI-REQUEST": _Ansi.MAGENTA,
    "AI-REPLY": _Ansi.CYAN,
    "TOOL-CALL": _Ansi.YELLOW,
    "TOOL-RESULT": _Ansi.GREEN,
    "TOOL-CREATION": _Ansi.MAGENTA,
    "REGISTRY": _Ansi.CYAN,
    "SANDBOX": _Ansi.BLUE,
    "AGENT": _Ansi.BLUE,
    "FILESYSTEM": _Ansi.CYAN,
    "TOKEN-USAGE": _Ansi.YELLOW,
    "FINAL-RESPONSE": _Ansi.GREEN,
    "WARNING": _Ansi.YELLOW,
    "ERROR": _Ansi.RED,
    "SKILL-ACQUISITION": _Ansi.MAGENTA,
    # v2 live skill acquisition (skill_acquisition.py) -- same family as
    # TOOL-CREATION (also an on-the-fly "the system just gained a new
    # capability" event), functionally unnecessary since _log falls back to
    # CYAN for any unlisted tag, but listed explicitly so its color is a
    # deliberate choice rather than an accident of the default.
}


# ---------------------------------------------------------------------------
# Color support detection
# ---------------------------------------------------------------------------

def _use_color() -> bool:
    """Enable colors for interactive terminals; override with LOG_COLOR.

    Set LOG_COLOR=always to force colors or LOG_COLOR=never (or NO_COLOR) to
    disable them. The file mirror is always plain text regardless.
    """
    setting = os.environ.get("LOG_COLOR", "auto").lower()
    if os.environ.get("NO_COLOR") is not None or setting == "never":
        return False
    if setting == "always":
        return True
    return bool(getattr(sys.__stdout__, "isatty", lambda: False)())


def _paint(text: str, color: str = "", *, bold: bool = False, dim: bool = False) -> str:
    if not _use_color() or not color:
        return text
    prefix = color
    if bold:
        prefix += _Ansi.BOLD
    if dim:
        prefix += _Ansi.DIM
    return f"{prefix}{text}{_Ansi.RESET}"


# ---------------------------------------------------------------------------
# Tee: mirrors terminal output to a plain-text log file
# ---------------------------------------------------------------------------

class _Tee:
    """Mirrors terminal output to a plain-text log file.

    ANSI escapes stay in the terminal but are stripped before writing to the
    file. This preserves colored debugging locally without corrupting saved
    logs with escape codes.
    """

    def __init__(self, console_stream, log_stream):
        self.console_stream = console_stream
        self.log_stream = log_stream

    def write(self, data):
        try:
            self.console_stream.write(data)
            self.console_stream.flush()
        except Exception:
            pass  # never let a logging failure break the app
        try:
            self.log_stream.write(_ANSI_ESCAPE_RE.sub("", data))
            self.log_stream.flush()
        except Exception:
            pass

    def flush(self):
        for stream in (self.console_stream, self.log_stream):
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        return bool(getattr(self.console_stream, "isatty", lambda: False)())


def _setup_console_logging(log_dir: str = "logs") -> str:
    """Redirect stdout/stderr so everything printed to the terminal is also written to a file."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(log_path, "a", encoding="utf-8")  # utf-8 to safely hold any generated content
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return log_path


# ---------------------------------------------------------------------------
# Structured log functions
# ---------------------------------------------------------------------------

def _format_log_value(value: Any, max_chars: int = 300) -> str:
    """Render an inline field predictably without letting it flood a log line."""
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            rendered = str(value)
    rendered = re.sub(r"\s+", " ", rendered).strip()
    if len(rendered) > max_chars:
        rendered = f"{rendered[:max_chars - 1]}..."
    return repr(rendered) if isinstance(value, str) else rendered


def _log(tag: str, msg: str = "", **fields: Any) -> None:
    """Write one colored, structured event to the console and plain log file.

    Example: ``[12:34:56.789] [TOOL-CALL] invoke | tool="search" task="T1"``.
    Keep values in ``fields`` rather than interpolating them into ``msg`` so
    important context can be scanned consistently across the entire run.
    """
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    tag_text = f"[{tag:<14}]"
    colored_tag = _paint(tag_text, _LOG_TAG_COLORS.get(tag, _Ansi.CYAN), bold=True)
    context = (
        " | " + " ".join(f"{key}={_format_log_value(value)}" for key, value in fields.items())
        if fields else ""
    )
    print(f"{_paint(f'[{ts}]', _Ansi.DIM)} {colored_tag} {msg}{_paint(context, _Ansi.DIM)}")


def _log_block(tag: str, title: str, body: str, max_chars: int = 2000) -> None:
    """Same as _log but for multi-line payloads (prompts, AI replies, tool
    output) -- prints a header line followed by the (possibly truncated)
    body so long content doesn't spam the single-line log stream."""
    body = body if body is not None else ""
    truncated = len(body) > max_chars
    shown = body[:max_chars] + (f"\n... [truncated, {len(body) - max_chars} more chars]" if truncated else "")
    _log(tag, f"BEGIN {title}", chars=len(body), truncated=truncated)
    for line in shown.splitlines() or [""]:
        print(_paint(f"    | {line}", _Ansi.DIM))
    _log(tag, f"END {title}")


# ---------------------------------------------------------------------------
# Module init: set up file mirroring and emit the first log line
# ---------------------------------------------------------------------------

LOG_FILE_PATH = _setup_console_logging()

_log(
    "SYSTEM",
    "Structured logging initialized",
    log_file=LOG_FILE_PATH,
    color_mode=os.environ.get("LOG_COLOR", "auto"),
)
