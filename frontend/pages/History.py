"""History module — browse past runs from run_index.json + workflow.json."""

from __future__ import annotations

import streamlit as st

from components.sidebar import render_sidebar
from components.ui_components import inject_theme_css, render_workflow_view, status_badge

st.set_page_config(
    page_title="History · Dynamic Agent",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

inject_theme_css()
client, _ = render_sidebar()

st.title("History")
st.caption("All runs from `backend/data/runs/run_index.json`, with details from each `workflow.json`.")

col_actions = st.columns([1, 4])
with col_actions[0]:
    if st.button("Refresh", use_container_width=True):
        st.rerun()

try:
    payload = client.list_history()
except Exception as exc:
    st.error(f"Could not load history: {exc}")
    st.stop()

runs = payload.get("runs") or []
st.markdown(f"**{payload.get('total', len(runs))}** runs")

if not runs:
    st.info("No runs yet. Start one from Chat.")
    st.stop()

left, right = st.columns([1.1, 2])

with left:
    st.subheader("Runs")
    options = []
    labels = {}
    for item in runs:
        rid = item["run_id"]
        title = item.get("goal_title") or (item.get("user_prompt") or "")[:60] or rid[:8]
        label = f"{title} · {item.get('created_at', '')[:19]}"
        options.append(rid)
        labels[rid] = label

    selected = st.radio(
        "Select a run",
        options=options,
        format_func=lambda rid: labels.get(rid, rid),
        label_visibility="collapsed",
    )

    if selected:
        meta = next(r for r in runs if r["run_id"] == selected)
        st.markdown(status_badge(meta.get("status")), unsafe_allow_html=True)
        st.caption(f"Path: `{meta.get('relative_path')}`")
        st.caption(f"Created: {meta.get('created_at')}")

with right:
    if not selected:
        st.info("Select a run to view details.")
    else:
        try:
            detail = client.get_history(selected)
        except Exception as exc:
            st.error(f"Could not load run details: {exc}")
            st.stop()

        workflow = detail.get("workflow") or {}
        stages_order = detail.get("stages_order")
        render_workflow_view(workflow, stages_order)
