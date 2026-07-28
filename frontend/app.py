"""
Dynamic Agent Orchestrator — Streamlit frontend.

Chat UI for starting runs, live stage/task tracking, and final results.
History lives on a separate page.
"""

from __future__ import annotations

import html
import time

import streamlit as st

from sidebar import build_llm_settings_for_chat, llm_settings_required, render_sidebar
from ui_components import (
    inject_theme_css,
    is_terminal,
    render_workflow_view,
    status_badge,
)

st.set_page_config(
    page_title="Dynamic Agent Orchestrator",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

inject_theme_css()
client, server_config = render_sidebar()
online = client.health()

st.title("Chat")
st.caption("Submit a task, watch pipeline stages and tasks, then review the final result.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_run_id" not in st.session_state:
    st.session_state.active_run_id = None
if "active_workflow" not in st.session_state:
    st.session_state.active_workflow = None

for msg in st.session_state.messages:
    role = msg["role"]
    css = "dao-chat-user" if role == "user" else "dao-chat-bot"
    body = html.escape(msg["content"]).replace("\n", "<br>")
    st.markdown(f'<div class="{css}">{body}</div>', unsafe_allow_html=True)

prompt = st.chat_input("Describe the task for the orchestrator…")
if prompt:
    if not online:
        st.error("API is offline. Start the FastAPI server first.")
    else:
        llm_settings = build_llm_settings_for_chat()
        if llm_settings_required(server_config) and not llm_settings:
            st.error("Model configuration required: provider, model name, and API key.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            try:
                resp = client.start_chat(prompt, llm_settings=llm_settings)
                run_id = resp["run_id"]
                st.session_state.active_run_id = run_id
                st.session_state.active_workflow = None
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Started run `{run_id}`. Tracking stages and tasks…",
                    }
                )
            except Exception as exc:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Failed to start run: {exc}"}
                )
            st.rerun()

run_id = st.session_state.active_run_id
if run_id:
    st.divider()
    st.markdown(f"### Live run `{run_id}`")

    details = None
    try:
        details = client.get_run(run_id)
    except Exception as exc:
        st.warning(f"Could not fetch run: {exc}")

    if details is None:
        st.info("Run is starting — waiting for workflow registration…")
        time.sleep(1.5)
        st.rerun()
    else:
        workflow = details.get("workflow") or {}
        state = details.get("state") or {}
        status = workflow.get("status") or state.get("status") or "running"
        st.session_state.active_workflow = workflow
        st.markdown(status_badge(status), unsafe_allow_html=True)
        render_workflow_view(workflow)

        if not is_terminal(status):
            time.sleep(2)
            st.rerun()
        else:
            st.success("Run finished. Open History for a persistent record.")
