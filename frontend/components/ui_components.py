"""Shared UI helpers for stages, tasks, and results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

STAGE_LABELS = {
    "goal_intake": "1 · Goal Intake",
    "task_graph": "2 · Task Graph",
    "agent_resolution": "3 · Agent Resolution",
    "execution": "4 · Execution",
    "shared_memory": "5 · Shared Memory",
    "evaluation": "6 · Evaluation",
    "replanner": "7 · Replanner",
    "result_assembly": "8 · Result Assembly",
}

STAGE_ORDER = list(STAGE_LABELS.keys())

STATUS_COLORS = {
    "completed": "#16a34a",
    "success": "#16a34a",
    "running": "#2563eb",
    "starting": "#2563eb",
    "pending": "#94a3b8",
    "failed": "#dc2626",
    "partial_failure": "#d97706",
}


def status_badge(status: Optional[str]) -> str:
    label = (status or "unknown").replace("_", " ").title()
    color = STATUS_COLORS.get((status or "").lower(), "#64748b")
    return (
        f'<span style="display:inline-block;padding:0.15rem 0.55rem;border-radius:999px;'
        f'background:{color}18;color:{color};font-size:0.8rem;font-weight:600;'
        f'border:1px solid {color}33;">{label}</span>'
    )


def inject_theme_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f0f7ff 0%, #ffffff 45%, #f8fbff 100%);
        }
        [data-testid="stSidebar"] {
            background: #e8f1fc;
            border-right: 1px solid #c5daf5;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding: 1rem 1.1rem 1.25rem 1.1rem;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.5rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
            padding-bottom: 0 !important;
        }
        [data-testid="stSidebar"] .stDivider {
            display: none;
        }
        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #cfe0f5;
            border-radius: 8px;
            padding: 0.45rem 0.35rem;
            min-height: 3.25rem;
        }
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
            font-size: 0.7rem !important;
            white-space: nowrap;
        }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }
        [data-testid="stSidebar"] .stTextInput,
        [data-testid="stSidebar"] .stSelectbox {
            margin-bottom: 0.35rem;
        }
        [data-testid="stSidebar"] .stTextInput > label,
        [data-testid="stSidebar"] .stSelectbox > label {
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            color: #334155 !important;
            margin-bottom: 0.25rem !important;
            padding-top: 0.15rem;
        }
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] p[data-testid="stCaptionContainer"] {
            margin-top: 0.15rem !important;
            margin-bottom: 0 !important;
            font-size: 0.78rem !important;
        }
        .dao-sidebar-header {
            padding: 0.1rem 0 0.85rem 0;
            border-bottom: 1px solid #c5daf5;
            margin-bottom: 0.35rem;
        }
        .dao-sidebar-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #0f3d7a;
            line-height: 1.25;
            margin: 0;
        }
        .dao-sidebar-subtitle {
            font-size: 0.88rem;
            color: #64748b;
            margin-top: 0.2rem;
        }
        .dao-sidebar-section-label {
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: #64748b;
            margin: 0.85rem 0 0.5rem 0;
            padding-top: 0.1rem;
        }
        .dao-sidebar-section-label:first-of-type {
            margin-top: 0.25rem;
        }
        .dao-api-status {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.88rem;
            font-weight: 600;
            color: #1e3a5f;
            margin: 0.35rem 0 0.1rem 0;
        }
        .dao-registry-note {
            font-size: 0.76rem;
            line-height: 1.45;
            color: #475569;
            background: #ffffff;
            border: 1px solid #cfe0f5;
            border-radius: 8px;
            padding: 0.55rem 0.65rem;
            margin-top: 0.45rem;
        }
        .dao-registry-note strong {
            color: #0f3d7a;
            font-weight: 600;
        }
        .dao-status-dot {
            width: 0.55rem;
            height: 0.55rem;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 0 2px rgba(255,255,255,0.8);
        }
        h1, h2, h3 { color: #0f3d7a !important; }
        .dao-card {
            background: #ffffff;
            border: 1px solid #cfe0f5;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 2px rgba(37, 99, 235, 0.06);
        }
        .dao-stage-title {
            font-weight: 650;
            color: #1e3a5f;
            margin-bottom: 0.25rem;
        }
        .dao-muted { color: #64748b; font-size: 0.85rem; }
        .dao-chat-user {
            background: #2563eb;
            color: white;
            padding: 0.75rem 1rem;
            border-radius: 12px 12px 4px 12px;
            margin: 0.4rem 0 0.4rem auto;
            max-width: 85%;
        }
        .dao-chat-bot {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e3a5f;
            padding: 0.75rem 1rem;
            border-radius: 12px 12px 12px 4px;
            margin: 0.4rem auto 0.4rem 0;
            max-width: 92%;
        }
        /* Hide Streamlit chrome: Deploy, Rerun, cache, print, record, etc. */
        #MainMenu {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        header [data-testid="stToolbar"] {display: none !important;}
        div[data-testid="stToolbar"] {display: none !important;}
        [data-testid="stDecoration"] {display: none !important;}
        [data-testid="stStatusWidget"] {display: none !important;}
        .stDeployButton, [data-testid="stAppDeployButton"] {display: none !important;}
        button[kind="header"] {display: none !important;}
        [data-testid="stHeaderActionElements"] {display: none !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stages(stages: Dict[str, Any], order: Optional[List[str]] = None) -> None:
    order = order or STAGE_ORDER
    st.subheader("Pipeline stages")
    cols = st.columns(4)
    for i, name in enumerate(order):
        stage = (stages or {}).get(name) or {}
        with cols[i % 4]:
            st.markdown(
                f'<div class="dao-card">'
                f'<div class="dao-stage-title">{STAGE_LABELS.get(name, name)}</div>'
                f'{status_badge(stage.get("status", "pending"))}'
                f'</div>',
                unsafe_allow_html=True,
            )
            details = stage.get("details") or {}
            if details:
                with st.expander("Details", expanded=False):
                    st.json(details)


def render_tasks(tasks: List[Dict[str, Any]]) -> None:
    st.subheader("Tasks")
    if not tasks:
        st.info("No tasks yet.")
        return

    for task in tasks:
        status = task.get("status", "pending")
        with st.container():
            st.markdown(
                f'<div class="dao-card">'
                f'<div style="display:flex;justify-content:space-between;gap:0.75rem;align-items:center;">'
                f'<strong style="color:#0f3d7a;">{task.get("id", "?")}</strong>'
                f'{status_badge(status)}'
                f'</div>'
                f'<div style="margin-top:0.4rem;color:#1e3a5f;">{task.get("description", "")}</div>'
                f'<div class="dao-muted" style="margin-top:0.45rem;">'
                f'Agent: {task.get("assigned_agent") or task.get("required_capability") or "—"}'
                f' · Tools: {", ".join(task.get("assigned_tools") or []) or "—"}'
                f' · Output: {task.get("output_key") or "—"}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if task.get("error_message"):
                st.error(task["error_message"])
            evaluation = task.get("evaluation")
            if evaluation:
                with st.expander(f"Evaluation · {task.get('id')}", expanded=False):
                    st.json(evaluation)


def render_final_result(workflow: Dict[str, Any]) -> None:
    status = workflow.get("status")
    st.subheader("Final result")
    st.markdown(status_badge(status), unsafe_allow_html=True)

    final = workflow.get("final_response")
    if final:
        st.markdown(final)
    else:
        st.caption("Final response not available yet.")

    files = [f for f in (workflow.get("output_files") or []) if str(f).lower().endswith(".md")]
    if files:
        st.markdown("**Output files** (Markdown)")
        for name in files:
            st.code(name, language=None)
    elif workflow.get("output_files"):
        st.caption("No Markdown (.md) outputs for this run.")

    errors = workflow.get("errors") or []
    if errors:
        st.warning("Run errors")
        for err in errors:
            st.write(f"- {err}")


def render_goal(goal: Optional[Dict[str, Any]]) -> None:
    if not goal:
        return
    st.subheader("Goal")
    st.markdown(f"**{goal.get('title', 'Untitled')}**")
    st.write(goal.get("description", ""))
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Deliverables**")
        for item in goal.get("deliverables") or []:
            st.write(f"- {item}")
    with cols[1]:
        st.markdown("**Constraints**")
        for item in goal.get("constraints") or []:
            st.write(f"- {item}")
    with cols[2]:
        st.markdown("**Assumptions**")
        for item in goal.get("assumptions") or []:
            st.write(f"- {item}")


def render_workflow_view(workflow: Dict[str, Any], stages_order: Optional[List[str]] = None) -> None:
    if not workflow:
        st.info("Waiting for workflow data…")
        return

    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.markdown(f"**Run** `{workflow.get('run_id', '')}`")
        prompt = workflow.get("user_prompt")
        if prompt:
            st.caption(prompt)
    with header_cols[1]:
        st.markdown(status_badge(workflow.get("status")), unsafe_allow_html=True)

    render_goal(workflow.get("goal"))
    render_stages(workflow.get("stages") or {}, stages_order)
    render_tasks(workflow.get("tasks") or [])

    memory = workflow.get("memory") or {}
    if memory:
        with st.expander("Shared memory", expanded=False):
            for key, value in memory.items():
                st.markdown(f"**{key}**")
                if isinstance(value, dict) and "output" in value:
                    st.markdown(value.get("output") or "")
                else:
                    st.json(value)

    render_final_result(workflow)


def is_terminal(status: Optional[str]) -> bool:
    return (status or "").lower() in {"success", "failed", "partial_failure", "completed"}
