# dynamic_streamlit_frontend.py

import streamlit as st
from dynamic_langgraph_backend import agent_manager, run_agent_with_requirements
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import json

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

if 'available_tools' not in st.session_state:
    st.session_state['available_tools'] = agent_manager.get_tool_info()

add_thread(st.session_state['thread_id'])

# ======================================== Sidebar UI =======================================

st.sidebar.title('🤖 Dynamic AI Agent')

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
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})
        
        st.session_state['message_history'] = temp_messages
        st.rerun()

# --- Available Tools ---
st.sidebar.header('🔧 Available Tools')
st.sidebar.info(agent_manager.get_tool_info())

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
                    st.session_state['available_tools'] = agent_manager.get_tool_info()
                    st.rerun()
                else:
                    st.error(f'❌ Failed to create tool')
        else:
            st.warning('Please fill in both fields')

# --- Dynamic Behavior Settings ---
st.sidebar.header('⚙️ Dynamic Settings')
with st.sidebar.expander('Configure Agent Behavior'):
    
    # Enable/Disable specific behaviors
    enable_preprocessing = st.checkbox(
        'Enable Message Preprocessing',
        value=False,
        key='enable_preprocessing'
    )
    
    if enable_preprocessing:
        preprocessing_instruction = st.text_area(
            'Preprocessing instruction:',
            height=50,
            key='preprocessing_instruction'
        )
        st.session_state['dynamic_requirements']['preprocessing'] = preprocessing_instruction
    
    # Custom behavior type
    behavior_type = st.selectbox(
        'Agent Behavior:',
        ['Standard', 'Detailed', 'Concise', 'Code-Focus', 'Creative'],
        key='behavior_type'
    )
    st.session_state['dynamic_requirements']['dynamic_behavior'] = behavior_type.lower()
    
    # Temperature/Creativity
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

st.title('🚀 Dynamic AI Agent')

# Display current configuration
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Tools Available', len(json.loads(agent_manager.get_tool_info())))
with col2:
    st.metric('Thread ID', st.session_state['thread_id'][:8] + '...')
with col3:
    st.metric('Messages', len(st.session_state['message_history']))

st.divider()

# --- Display Chat History ---
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# --- Chat Input ---
user_input = st.chat_input('Type your message or request a new tool...')

if user_input:
    
    # Check if user is requesting a new tool
    if any(phrase in user_input.lower() for phrase in ['create tool', 'add tool', 'new tool', 'make tool']):
        st.info('📝 Tool creation requested! Use the sidebar to create a new tool.')
    
    # Add user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    
    # Prepare configuration
    config = {'configurable': {'thread_id': st.session_state['thread_id']}}
    
    # Display assistant response with streaming
    with st.chat_message('assistant'):
        def ai_stream():
            for message_chunk, metadata in agent_manager.chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
        
        ai_message = st.write_stream(ai_stream())
    
    # Store assistant response
    if ai_message:
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

# --- Footer with Instructions ---
with st.expander('📖 How to Use'):
    st.markdown("""
    ### Dynamic Agent Features:
    
    1. **Create Tools Dynamically**
       - Use the "Add New Tool" section in the sidebar
       - Describe what you want the tool to do
       - Agent creates and registers it automatically
    
    2. **Configure Behavior**
       - Change agent behavior type (Detailed, Concise, etc.)
       - Adjust temperature for creativity level
       - Enable preprocessing for custom message handling
    
    3. **Multi-Tool Conversations**
       - Ask questions that require multiple tools
       - Agent automatically uses the right tools
       - Tools are persistent across conversations
    
    4. **Thread Management**
       - Each conversation is saved separately
       - Switch between conversations instantly
       - All tool usage is logged in SQLite
    
    ### Example Prompts:
    - "Create a tool that converts currencies"
    - "Add a tool for sentiment analysis"
    - "Make a weather tool for any city"
    - "Build a tool that summarizes text"
    """)
