"""Shared sidebar — header, navigation, API, model config, limits."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from api_client import OrchestratorClient

PROVIDER_LABELS = {
    "openai": "OpenAI",
    "groq": "Groq",
    "anthropic": "Claude (Anthropic)",
}

DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "llama-3.1-8b-instant"

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "groq": DEFAULT_MODEL,
    "anthropic": "claude-3-5-sonnet-20240620",
}

PROVIDERS = ["groq", "openai", "anthropic"]


def _init_model_defaults() -> None:
    st.session_state.setdefault("llm_provider", DEFAULT_PROVIDER)
    if st.session_state["llm_provider"] not in PROVIDERS:
        st.session_state["llm_provider"] = DEFAULT_PROVIDER
    st.session_state.setdefault(
        "llm_model",
        DEFAULT_MODELS.get(st.session_state["llm_provider"], DEFAULT_MODEL),
    )
    st.session_state.setdefault("llm_api_key", "")


def build_llm_settings_for_chat() -> Optional[Dict[str, str]]:
    """Build llm_settings from sidebar fields (backend decides whether to use them)."""
    provider = st.session_state.get("llm_provider", DEFAULT_PROVIDER)
    model = (st.session_state.get("llm_model") or "").strip()
    api_key = (st.session_state.get("llm_api_key") or "").strip()
    if not model or not api_key:
        return None
    return {"provider": provider, "model": model, "api_key": api_key}


def llm_settings_required(server_config: Optional[Dict[str, Any]]) -> bool:
    return (server_config or {}).get("llm_source") == "api"


def _status_indicator(online: bool) -> str:
    color = "#16a34a" if online else "#dc2626"
    label = "Online" if online else "Offline"
    return (
        f'<span class="dao-api-status">'
        f'<span class="dao-status-dot" style="background:{color};"></span>'
        f'{label}</span>'
    )


def _registry_summary(agents: List[Dict[str, Any]]) -> str:
    if not agents:
        return "Static registry: Researcher, Data Analyst, Document Generator."
    parts = []
    for agent in agents:
        tools = ", ".join(agent.get("tools") or [])
        parts.append(f"<strong>{agent.get('role', '?')}</strong> ({tools})")
    return "Registry · " + " · ".join(parts)


def render_sidebar() -> Tuple[OrchestratorClient, Optional[Dict[str, Any]]]:
    with st.sidebar:
        st.markdown(
            '<div class="dao-sidebar-header">'
            '<div class="dao-sidebar-title">Dynamic Agent</div>'
            '<div class="dao-sidebar-subtitle">Orchestration</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="dao-sidebar-section-label">Modules</div>', unsafe_allow_html=True)
        nav1, nav2 = st.columns(2, gap="small")
        with nav1:
            st.page_link("app.py", label="Chat", icon="💬", use_container_width=True)
        with nav2:
            st.page_link("pages/1_History.py", label="History", icon="🗂️", use_container_width=True)

        st.markdown('<div class="dao-sidebar-section-label">API</div>', unsafe_allow_html=True)
        api_base = st.text_input(
            "Base URL",
            value=st.session_state.get("api_base", "http://127.0.0.1:8000"),
            label_visibility="collapsed",
            placeholder="http://127.0.0.1:8000",
        )
        st.session_state["api_base"] = api_base
        client = OrchestratorClient(api_base)
        online = client.health()
        st.markdown(_status_indicator(online), unsafe_allow_html=True)
        if not online:
            st.caption("Set `ORCHESTRATOR_MODE=server` in `.env`, then run `python main.py`.")

        server_config: Optional[Dict[str, Any]] = None
        if online:
            try:
                server_config = client.get_config()
            except Exception as exc:
                st.warning(f"Config unavailable: {exc}")

        st.markdown('<div class="dao-sidebar-section-label">Model configuration</div>', unsafe_allow_html=True)
        _init_model_defaults()

        current_provider = st.session_state.get("llm_provider", DEFAULT_PROVIDER)
        provider = st.selectbox(
            "Provider",
            options=PROVIDERS,
            format_func=lambda p: PROVIDER_LABELS.get(p, p),
            index=PROVIDERS.index(current_provider),
        )
        if provider != st.session_state.get("llm_provider"):
            st.session_state["llm_model"] = DEFAULT_MODELS.get(provider, DEFAULT_MODEL)
        st.session_state["llm_provider"] = provider

        model = st.text_input(
            "Model name",
            value=st.session_state.get("llm_model") or DEFAULT_MODELS.get(provider, DEFAULT_MODEL),
        )
        st.session_state["llm_model"] = model

        api_key = st.text_input(
            "API key",
            value=st.session_state.get("llm_api_key", ""),
            type="password",
        )
        st.session_state["llm_api_key"] = api_key

        if server_config:
            st.markdown('<div class="dao-sidebar-section-label">Workflow limits</div>', unsafe_allow_html=True)
            limits = server_config.get("limits") or {}
            c1, c2, c3 = st.columns(3, gap="small")
            with c1:
                st.metric("Tasks", limits.get("max_tasks", "—"))
            with c2:
                st.metric("Replans", limits.get("max_replans", "—"))
            with c3:
                st.metric("Spawn", limits.get("max_spawned_agents", "—"))

            registry_html = _registry_summary(server_config.get("registry_agents") or [])
            st.markdown(
                f'<div class="dao-registry-note">{registry_html}</div>',
                unsafe_allow_html=True,
            )

    return client, server_config
