# dynamic_streamlit_frontend.py

import streamlit as st
from dynamic_langgraph_backend import agent_manager
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import json
import os

# ======================================== Utility Functions ================================

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    st.session_state['dynamic_requirements'] = {}

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = agent_manager.chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

# ======================================== Session Setup ===================================

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'dynamic_requirements' not in st.session_state:
    st.session_state['dynamic_requirements'] = {}

add_thread(st.session_state['thread_id'])

# ======================================== Sidebar UI =======================================

st.sidebar.title('Dynamic Multi-Agent Orchestrator')
_langsmith_project = os.environ.get('LANGCHAIN_PROJECT')
if _langsmith_project:
    st.sidebar.caption(
        f"🔍 Full run traces (agents/tools created, tool inputs & outputs, "
        f"evaluator verdicts) are in LangSmith project **{_langsmith_project}**."
    )

# --- Chat Management ---
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.sidebar.button('➕ New Chat'):
        reset_chat()
        st.rerun()

with col2:
    if st.sidebar.button('🗑️ Clear All'):
        st.session_state['chat_threads'] = []
        reset_chat()
        st.rerun()

# --- Conversation History ---
st.sidebar.header('💬 Conversations')
for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)[:8] + '...'):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            elif isinstance(msg, AIMessage) and msg.content:
                role = 'assistant'
            else:
                continue
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
        st.rerun()

# --- Available Tools ---
st.sidebar.header('🔧 Available Tools')
try:
    tools_dict = json.loads(agent_manager.get_tool_info())
    for t_name, t_desc in tools_dict.items():
        with st.sidebar.container():
            st.markdown(f"**🛠️ {t_name}**")
            st.caption(t_desc)
except Exception:
    st.sidebar.info(agent_manager.get_tool_info())

# --- Dynamically Created Agents ---
st.sidebar.header('🧠 Active Agents')
try:
    agents_dict = json.loads(agent_manager.get_agent_info())
    if agents_dict:
        for role, cfg in agents_dict.items():
            with st.sidebar.container():
                st.markdown(f"**🧬 {role}**")
                st.caption(f"Tools: {', '.join(cfg['tools']) if cfg['tools'] else 'none'}")
    else:
        st.sidebar.caption("No agents created yet — send a message to trigger planning.")
except Exception:
    st.sidebar.caption("No agents created yet.")

# --- Dynamic Tool Creation ---
st.sidebar.header('➕ Add New Tool')
with st.sidebar.expander('Create Tool from Prompt'):
    tool_name = st.text_input('Tool name:', key='tool_name')
    tool_prompt = st.text_area(
        'Describe what this tool should do:',
        height=100,
        key='tool_prompt'
    )

    if st.button('Create Tool', key='create_tool_btn'):
        if tool_name and tool_prompt:
            with st.spinner(f'Creating tool "{tool_name}"...'):
                success = agent_manager.add_tool_from_prompt(tool_prompt, tool_name)
                if success:
                    st.success(f'✅ Tool "{tool_name}" created!')
                    st.rerun()
                else:
                    st.error(f'❌ Failed to create tool')
        else:
            st.warning('Please fill in both fields')

# --- Dynamic Behavior Settings ---
st.sidebar.header('⚙️ Dynamic Settings')
with st.sidebar.expander('Configure Agent Behavior'):

    enable_preprocessing = st.checkbox(
        'Enable Extra Planning Instruction',
        value=False,
        key='enable_preprocessing'
    )

    if enable_preprocessing:
        preprocessing_instruction = st.text_area(
            'Instruction (folded into the planning step):',
            height=50,
            key='preprocessing_instruction'
        )
        st.session_state['dynamic_requirements']['preprocessing'] = preprocessing_instruction
    else:
        st.session_state['dynamic_requirements'].pop('preprocessing', None)

    behavior_type = st.selectbox(
        'Agent Behavior:',
        ['Standard', 'Detailed', 'Concise', 'Code-Focus', 'Creative'],
        key='behavior_type'
    )
    st.session_state['dynamic_requirements']['dynamic_behavior'] = behavior_type.lower()

    temperature = st.slider(
        'Temperature (Creativity)',
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        key='temperature'
    )
    st.session_state['dynamic_requirements']['temperature'] = temperature

# ======================================== Main Chat UI =====================================

st.title('Dynamic Multi-Agent Orchestrator')

col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Tools Available', len(json.loads(agent_manager.get_tool_info())))
with col2:
    st.metric('Agents Created', len(json.loads(agent_manager.get_agent_info())))
with col3:
    st.metric('Messages', len(st.session_state['message_history']))

st.divider()

# --- Display Chat History ---
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# --- Chat Input ---
user_input = st.chat_input('Describe your goal (e.g. "research X and summarize it")...')

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # Apply dynamic settings for this run (fixes the previous bug where these
    # sidebar controls were built but never actually reached the backend)
    reqs = st.session_state.get('dynamic_requirements', {})
    if 'dynamic_behavior' in reqs:
        agent_manager.set_behavior_style(reqs['dynamic_behavior'])
    if 'preprocessing' in reqs:
        agent_manager.set_extra_instruction(reqs['preprocessing'])
    if 'temperature' in reqs:
        agent_manager.set_temperature(reqs['temperature'])

    config = {
        'configurable': {'thread_id': st.session_state['thread_id']},
        'run_name': 'orchestrator_run',
        'tags': ['orchestrator_run', 'streamlit'],
        'metadata': {'goal': user_input, 'thread_id': st.session_state['thread_id']},
    }

    with st.chat_message('assistant'):
        plan_box = st.empty()
        status_box = st.empty()
        answer_box = st.empty()

        final_answer = ""
        task_plan = []

        # stream_mode="updates" yields {node_name: node_output} after each node
        # runs, which lets us show the plan, which agent is working, and
        # evaluator verdicts live as the workflow executes.
        for update in agent_manager.chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_output in update.items():

                if node_name == "planner":
                    task_plan = node_output.get("task_plan", [])
                    with plan_box.container():
                        st.markdown("**🗂️ Task Plan**")
                        for t in task_plan:
                            st.markdown(f"- `{t['id']}` **[{t['agent_role']}]** {t['description']}")

                elif node_name == "agent_executor":
                    status_box.markdown("🤖 An agent is working on the current task...")

                elif node_name == "tools":
                    status_box.markdown("🔧 Executing tool call(s)...")

                elif node_name == "evaluator":
                    verdict = node_output.get("last_verdict", {})
                    status = verdict.get("status", "PASS")
                    if status == "RETRY":
                        status_box.markdown(f"🔁 Evaluator requested a retry: {verdict.get('reason', '')}")
                    else:
                        status_box.markdown(f"✅ Task evaluated: {verdict.get('reason', 'looks good')}")

                elif node_name == "assembler":
                    msgs = node_output.get("messages", [])
                    if msgs:
                        final_answer = msgs[-1].content
                        answer_box.markdown(final_answer)

        status_box.empty()

    if final_answer:
        st.session_state['message_history'].append({'role': 'assistant', 'content': final_answer})

# --- Footer with Instructions ---
with st.expander('📖 How to Use'):
    st.markdown("""
    ### Dynamic Orchestration Features:

    1. **Automatic Task Decomposition**
       - Every message is treated as a goal and broken into a task plan
       - Each task is tagged with the type of specialist agent it needs

    2. **Dynamic Agent Creation**
       - New specialist agents (with their own system prompt and tool subset)
         are created the first time a role is needed, and reused after that
       - See them appear live in the **🧠 Active Agents** sidebar panel

    3. **Evaluation & Self-Correction**
       - Each task's output is evaluated before being accepted
       - Failing outputs trigger an automatic retry with feedback (up to 2 retries)

    4. **Create Tools Dynamically**
       - Use "Add New Tool" in the sidebar; new agents can pick up new tools automatically

    5. **Configure Behavior**
       - Behavior style and extra instructions now actually reach the planner/agents
       - Temperature is applied to the underlying LLM before each run

    ### Example Prompts:
    - "Research the current price of AAPL stock and explain if it's a good time to buy"
    - "Calculate 15% tip on a $84.50 bill and explain the math"
    - "Search for the latest news on renewable energy and summarize the key points"
    """)
