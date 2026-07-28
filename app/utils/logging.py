import json
from datetime import datetime
from typing import Any, Optional

from app.config import Config, active_model_name
from app.storage.run_paths import RunPathStore


def format_log_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _fmt_details(details: Optional[dict]) -> str:
    if not details:
        return ""
    parts = [f"{k}={format_log_value(v)}" for k, v in details.items()]
    return "\n  " + "\n  ".join(parts)


class AuditLogger:
    """Plain-text run.log under data/logs/{batch_id}/{run_id}/run.log"""

    def __init__(self, run_id: str, task_id: Optional[str] = None):
        self.run_id = run_id
        self.task_id = task_id
        paths = RunPathStore().require(run_id)
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = paths.log_file
        self._limits_logged = False

    def for_task(self, task_id: str) -> "AuditLogger":
        return AuditLogger(self.run_id, task_id=task_id)

    def _write(self, line: str) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_limits(self) -> None:
        if self._limits_logged:
            return
        self._limits_logged = True
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(
            f"[{ts}] [Config] provider={Config.ACTIVE_PROVIDER} model={active_model_name()}\n"
            f"  limits: max_tasks={Config.MAX_TASKS} max_replans={Config.MAX_REPLANS} "
            f"max_spawned_agents={Config.MAX_SPAWNED_AGENTS}"
        )

    def log(
        self,
        stage: str,
        event: str,
        details: Optional[dict] = None,
        task_id: Optional[str] = None,
    ) -> None:
        self.log_limits()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        resolved_task = task_id or (details or {}).get("task_id") or self.task_id
        task_part = f" task={resolved_task}" if resolved_task else ""
        self._write(f"[{ts}] [{stage}]{task_part} {event}{_fmt_details(details)}")

    def log_tool(
        self,
        stage: str,
        task_id: str,
        agent: str,
        tool_name: str,
        args: dict,
        result: Optional[str] = None,
    ) -> None:
        self.log(stage, f"tool_call name={tool_name}", {"agent": agent, "args": args}, task_id=task_id)
        if result is not None:
            self.log(stage, f"tool_result name={tool_name}", {"result": result}, task_id=task_id)

    def log_llm(
        self,
        stage: str,
        event: str,
        content: Optional[str] = None,
        usage: Optional[dict] = None,
        task_id: Optional[str] = None,
        **extra,
    ) -> None:
        details = dict(extra)
        if content is not None:
            details["content"] = content
        if usage:
            details["token_usage"] = usage
        self.log(stage, event, details, task_id=task_id)
