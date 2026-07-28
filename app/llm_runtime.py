"""Per-run LLM overrides (e.g. from frontend API mode). Falls back to Config when unset."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

_llm_runtime: ContextVar[Optional["LLMRuntimeConfig"]] = ContextVar("llm_runtime", default=None)


@dataclass(frozen=True)
class LLMRuntimeConfig:
    provider: str
    model: str
    api_key: str


def set_llm_runtime(cfg: Optional[LLMRuntimeConfig]) -> None:
    _llm_runtime.set(cfg)


def get_llm_runtime() -> Optional[LLMRuntimeConfig]:
    return _llm_runtime.get()


def clear_llm_runtime() -> None:
    _llm_runtime.set(None)
