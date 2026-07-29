import os
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_ROOT / ".env")

_active_run_outputs_dir: ContextVar[Path | None] = ContextVar("active_run_outputs_dir", default=None)


class Config:
    # local = CLI static run; server = FastAPI gateway (see main.py ORCHESTRATOR_MODE)
    ORCHESTRATOR_MODE: str = os.getenv("ORCHESTRATOR_MODE", "local").lower()

    # local = credentials from .env | api = credentials from POST /chat llm_settings
    LLM_SOURCE: str = os.getenv("LLM_SOURCE", "local").lower()

    ACTIVE_PROVIDER: str = os.getenv("ACTIVE_PROVIDER", "groq")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

    MAX_TASKS: int = int(os.getenv("MAX_TASKS", "20"))
    MAX_REPLANS: int = int(os.getenv("MAX_REPLANS", "3"))
    MAX_SPAWNED_AGENTS: int = int(os.getenv("MAX_SPAWNED_AGENTS", "5"))
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BACKEND_ROOT / "data")))
    REGISTRY_DIR: Path = Path(os.getenv("REGISTRY_DIR", str(BACKEND_ROOT / "registry")))
    RUNS_DIR: Path = DATA_DIR / os.getenv("RUNS_DIRNAME", "runs")
    LOGS_DIR: Path = DATA_DIR / os.getenv("LOGS_DIRNAME", "logs")

    BATCH_ID_FORMAT: str = os.getenv("BATCH_ID_FORMAT", "%Y%m%d-%H%M%S")
    WORKFLOW_FILENAME: str = os.getenv("WORKFLOW_FILENAME", "workflow.json")
    LOG_FILENAME: str = os.getenv("LOG_FILENAME", "run.log")
    OUTPUTS_DIRNAME: str = os.getenv("OUTPUTS_DIRNAME", "outputs")
    RUN_INDEX_FILENAME: str = os.getenv("RUN_INDEX_FILENAME", "run_index.json")

    RUN_INDEX_FILE: Path = RUNS_DIR / RUN_INDEX_FILENAME


Config.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
Config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)


def active_model_name() -> str:
    _provider, model, _key = resolve_llm_credentials()
    return model or "unknown"


def local_llm_summary() -> dict:
    """Non-secret snapshot of server .env LLM settings (for GET /config)."""
    provider = _normalize_provider(Config.ACTIVE_PROVIDER)
    keys = {
        "openai": (Config.OPENAI_MODEL, bool(Config.OPENAI_API_KEY)),
        "groq": (Config.GROQ_MODEL, bool(Config.GROQ_API_KEY)),
        "anthropic": (Config.ANTHROPIC_MODEL, bool(Config.ANTHROPIC_API_KEY)),
        "gemini": (Config.GEMINI_MODEL, bool(Config.GEMINI_API_KEY)),
    }
    model, key_set = keys.get(provider, ("", False))
    return {"provider": provider, "model": model, "api_key_set": key_set}


def set_active_run_outputs_dir(path: Path | None) -> None:
    _active_run_outputs_dir.set(path)


def get_active_run_outputs_dir() -> Path | None:
    return _active_run_outputs_dir.get()


def resolve_workspace_path(file_path: str) -> Path:
    """Resolve agent file paths to the run outputs folder (no nested outputs/)."""
    normalized = file_path.replace("\\", "/").strip().lstrip("./")

    while normalized.startswith("outputs/"):
        normalized = normalized[len("outputs/") :]

    if "/outputs/" in normalized:
        normalized = normalized.rsplit("/outputs/", 1)[-1]

    if normalized.startswith("data/runs/"):
        normalized = Path(normalized).name

    base = get_active_run_outputs_dir()
    if base is None:
        return Path(normalized)

    return base / Path(normalized).name


def assert_markdown_path(file_path: str) -> Path:
    """Only .md files are supported for agent file I/O right now."""
    path = resolve_workspace_path(file_path)
    if path.suffix.lower() != ".md":
        raise ValueError(
            f"Only Markdown (.md) files are supported currently; got '{path.name}'. "
            "Use a filename ending in .md (html/docx/csv are not supported)."
        )
    return path


def _normalize_provider(name: str) -> str:
    key = (name or "").lower().strip()
    if key in {"claude", "anthropic"}:
        return "anthropic"
    if key in {"openai", "gpt"}:
        return "openai"
    if key == "groq":
        return "groq"
    if key == "gemini":
        return "gemini"
    return key


def uses_request_llm() -> bool:
    return Config.LLM_SOURCE == "api"


def resolve_llm_credentials() -> tuple[str, str, str]:
    """Return (provider, model, api_key) based on backend LLM_SOURCE."""
    from config.llm_runtime import get_llm_runtime

    if uses_request_llm():
        runtime = get_llm_runtime()
        if runtime and runtime.provider and runtime.model and runtime.api_key:
            return (
                _normalize_provider(runtime.provider),
                runtime.model.strip(),
                runtime.api_key.strip(),
            )
        provider = _normalize_provider(Config.ACTIVE_PROVIDER)
        return provider, "", ""

    provider = _normalize_provider(Config.ACTIVE_PROVIDER)
    keys = {
        "openai": (Config.OPENAI_MODEL, Config.OPENAI_API_KEY),
        "groq": (Config.GROQ_MODEL, Config.GROQ_API_KEY),
        "anthropic": (Config.ANTHROPIC_MODEL, Config.ANTHROPIC_API_KEY),
        "gemini": (Config.GEMINI_MODEL, Config.GEMINI_API_KEY),
    }
    model, api_key = keys.get(provider, ("", ""))
    return provider, model, api_key


def get_chat_model(temperature: float = 0.0):
    provider, model, api_key = resolve_llm_credentials()
    if not api_key:
        raise ValueError(
            f"No API key configured for provider '{provider}'. "
            "Set it in .env (local mode) or pass llm_settings from the frontend (API mode)."
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(api_key=api_key, model=model, temperature=temperature)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=api_key, model=model, temperature=temperature)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        os.environ["GOOGLE_API_KEY"] = api_key
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    raise ValueError(f"Unsupported provider: '{provider}'")
