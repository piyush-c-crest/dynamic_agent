import sys
from io import StringIO
from typing import Any, Dict

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from app.config import assert_markdown_path

ddg_search = DuckDuckGoSearchRun()

# Keep REPL snippets small so tool calls stay valid for providers like Groq.
_MAX_REPL_CHARS = 2500


@tool
def web_search(query: str) -> str:
    """Search the web for short factual queries. Keep the query concise."""
    return ddg_search.run(query)


@tool
def file_read(file_path: str) -> str:
    """Read a Markdown (.md) file from the current run outputs folder. Example: 'analysis.md'."""
    try:
        path = assert_markdown_path(file_path)
    except ValueError as e:
        return f"Error: {e}"
    if not path.exists():
        return f"Error: file not found at '{path}'"
    return path.read_text(encoding="utf-8")


@tool
def file_write(file_path: str, content: str) -> str:
    """Write Markdown to the run outputs folder. Filename must end with .md (e.g. 'report.md')."""
    try:
        path = assert_markdown_path(file_path)
    except ValueError as e:
        return f"Error: {e}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Successfully wrote to '{path.name}'"


@tool
def python_repl(code: str) -> str:
    """Run a SHORT Python snippet for numeric calculations only.

    Rules:
    - A few lines of math/stats code only.
    - Do NOT paste research briefs, dependency JSON, or long documents into the code.
    - Do NOT use json.loads on large upstream text — that text is already in your prompt.
    """
    if not isinstance(code, str) or not code.strip():
        return "Error: python_repl requires non-empty code."
    if len(code) > _MAX_REPL_CHARS:
        return (
            f"Error: python_repl code too large ({len(code)} chars; max {_MAX_REPL_CHARS}). "
            "Do not embed documents or dependency JSON. Use upstream text from the prompt "
            "and call python_repl only for short calculations."
        )
    # Soft-block the common failure mode that breaks Groq tool calling.
    lowered = code.lower()
    if "json.loads" in lowered and ("research" in lowered or "output" in lowered or "'''" in code or '"""' in code):
        return (
            "Error: Do not json.loads dependency/research text in python_repl. "
            "That content is already available in the user message — analyze it directly."
        )

    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    try:
        exec(code, {})
        sys.stdout = old_stdout
        return redirected_output.getvalue() or "(no stdout)"
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error executing python code: {str(e)}"


LANGCHAIN_TOOLS: Dict[str, Any] = {
    "web_search": web_search,
    "file_read": file_read,
    "file_write": file_write,
    "python_repl": python_repl,
}


def execute_tool_by_name(tool_name: str, args: Dict[str, Any]) -> Any:
    if tool_name not in LANGCHAIN_TOOLS:
        raise ValueError(f"Tool '{tool_name}' not found in the tools registry.")
    return LANGCHAIN_TOOLS[tool_name].invoke(args)
