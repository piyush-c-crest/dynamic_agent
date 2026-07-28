import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar, get_args, get_origin

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def extract_token_usage(response: Any) -> Dict[str, Any]:
    usage: Dict[str, Any] = {}
    meta = getattr(response, "usage_metadata", None)
    if meta:
        usage.update(dict(meta))
    resp_meta = getattr(response, "response_metadata", None) or {}
    for key in ("token_usage", "usage"):
        if key in resp_meta and isinstance(resp_meta[key], dict):
            usage.update(resp_meta[key])
    return usage


def log_llm_response(logger, stage: str, response: Any, event: str = "llm_response", **extra) -> None:
    if not logger:
        return
    logger.log_llm(
        stage,
        event,
        content=getattr(response, "content", None) or "",
        usage=extract_token_usage(response) or None,
        **extra,
    )


def _is_list_annotation(annotation: Any) -> bool:
    if annotation is None:
        return False
    if get_origin(annotation) is list:
        return True
    for arg in get_args(annotation):
        if _is_list_annotation(arg):
            return True
    return False


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, dict):
        return [
            f"{k}: {v}" if not isinstance(v, (dict, list)) else f"{k}: {json.dumps(v)}"
            for k, v in value.items()
        ]
    return [str(value)]


def _normalize_for_schema(data: dict, schema: Type[BaseModel]) -> dict:
    normalized = dict(data)
    for name, field_info in schema.model_fields.items():
        if name not in normalized:
            continue
        if _is_list_annotation(field_info.annotation):
            normalized[name] = _coerce_list(normalized[name])
    return normalized


def _strip_function_wrapper(text: str) -> str:
    text = (text or "").strip()
    fn_match = re.search(r"<function=[^>]+>\s*(.*?)\s*</function>", text, re.DOTALL)
    if fn_match:
        return fn_match.group(1).strip()
    return text


def _extract_json(text: str) -> dict:
    text = _strip_function_wrapper(text)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group())


def _salvage_from_error(err: Exception) -> Optional[dict]:
    msg = str(err)
    if "failed_generation" not in msg and "<function=" not in msg:
        return None

    for pattern in (
        r"failed_generation['\"]:\s*'([^']+)'",
        r'failed_generation["\']:\s*"([^"]+)"',
        r"<function=[^>]+>\s*(\{.*\})\s*</function>",
    ):
        match = re.search(pattern, msg, re.DOTALL)
        if not match:
            continue
        try:
            return _extract_json(match.group(1))
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _field_hints(schema: Type[BaseModel]) -> str:
    hints = []
    for name, field_info in schema.model_fields.items():
        kind = "list of strings" if _is_list_annotation(field_info.annotation) else "string"
        if field_info.annotation is bool:
            kind = "boolean"
        hints.append(f'"{name}" ({kind})')
    return ", ".join(hints)


def invoke_structured(
    model,
    schema: Type[T],
    messages: List[BaseMessage],
    logger=None,
    stage: str = "LLM",
) -> T:
    """Invoke an LLM and parse a Pydantic schema, with JSON fallback for Groq/tool failures."""
    try:
        result = model.with_structured_output(schema).invoke(messages)
        if logger:
            logger.log(stage, "structured_output_success", {"schema": schema.__name__})
        return result
    except Exception as primary_err:
        if logger:
            logger.log(stage, "structured_output_failed", {"schema": schema.__name__, "error": str(primary_err)})

        salvaged = _salvage_from_error(primary_err)
        if salvaged is not None:
            try:
                result = schema.model_validate(_normalize_for_schema(salvaged, schema))
                if logger:
                    logger.log(stage, "structured_output_salvaged", {"schema": schema.__name__})
                return result
            except Exception:
                pass

        fallback_messages = list(messages) + [
            HumanMessage(
                content=(
                    f"Return ONLY a valid JSON object with these fields: {_field_hints(schema)}. "
                    "List fields must be JSON arrays of strings, not objects. "
                    "No markdown fences, XML, or function tags."
                )
            )
        ]
        response = model.invoke(fallback_messages)
        log_llm_response(logger, stage, response, event="structured_output_fallback", schema=schema.__name__)

        try:
            data = _normalize_for_schema(_extract_json(response.content or ""), schema)
            return schema.model_validate(data)
        except Exception as fallback_err:
            raise RuntimeError(
                f"Structured output failed for {schema.__name__}: {primary_err}; "
                f"fallback parse failed: {fallback_err}"
            ) from fallback_err
