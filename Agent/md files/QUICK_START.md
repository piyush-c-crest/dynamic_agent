# Dynamic AI Agent - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install langchain langgraph langchain-openai langgraph streamlit
```

### 2. Set Environment Variables

Create `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

### 3. Run the Agent

**Option A: Streamlit UI (Recommended for first-time users)**

```bash
streamlit run dynamic_streamlit_frontend.py
```

Then open browser to `http://localhost:8501`

**Option B: Python Script**

```python
from dynamic_langgraph_backend import agent_manager
import uuid

thread_id = str(uuid.uuid4())

# Run agent
response = agent_manager.run(
    user_input="What is 42 * 3?",
    thread_id=thread_id
)

print(response)
```

**Option C: Programmatic with Tool Creation**

```python
from dynamic_langgraph_backend import agent_manager

# Create a tool on the fly
agent_manager.add_tool_from_prompt(
    prompt="Create a tool that converts temperatures between Celsius and Fahrenheit",
    tool_name="temperature_converter"
)

# Use it immediately
result = agent_manager.run(
    user_input="Convert 25 Celsius to Fahrenheit",
    thread_id="conv_1"
)
```

---

## 📋 Common Use Cases

### Use Case 1: Create Tool from Chat

In Streamlit UI:
1. Open sidebar → "Add New Tool"
2. Enter tool name: `weather`
3. Enter description: "Create a tool that fetches weather for any city"
4. Click "Create Tool"
5. Now use it: "What's the weather in New York?"

### Use Case 2: Create Multiple Tools

```python
tools = [
    ("currency_converter", "Convert between USD, EUR, GBP, INR"),
    ("text_summarizer", "Summarize long text into bullet points"),
    ("json_validator", "Validate and format JSON"),
]

for tool_name, description in tools:
    agent_manager.add_tool_from_prompt(description, tool_name)
    print(f"✅ {tool_name} created")
```

### Use Case 3: Conditional Tool Creation

```python
def create_tools_based_on_request(user_message):
    if "weather" in user_message.lower():
        agent_manager.add_tool_from_prompt(
            "Fetch weather data",
            "weather"
        )
    
    if "stock" in user_message.lower():
        agent_manager.add_tool_from_prompt(
            "Get stock prices",
            "stock_analyzer"
        )
```

### Use Case 4: Track Tool Performance

```python
from advanced_dynamic_patterns import SelfModifyingAgent

agent = SelfModifyingAgent()

# Simulate tool usage
agent.track_tool_usage("weather", execution_time=0.5, success=True, feedback=4.8)
agent.track_tool_usage("weather", execution_time=0.6, success=True, feedback=4.9)

# Get report
print(agent.get_performance_report())
```

### Use Case 5: Learn from User Corrections

```python
from advanced_dynamic_patterns import AdaptiveLearningAgent

agent = AdaptiveLearningAgent()

# Record a correction
agent.record_feedback(
    user_input="Calculate 5 * 3",
    agent_response="The answer is 16",  # Wrong
    feedback_type="correction",
    correction="The answer is 15"  # Right
)

# Next time, agent applies learned rule
```

---

## 🎯 Key Files

| File | Purpose |
|------|---------|
| `dynamic_langgraph_backend.py` | Core agent logic, tool registry, graph building |
| `dynamic_streamlit_frontend.py` | Web UI for interacting with agent |
| `dynamic_agent_examples.py` | 7 usage examples |
| `advanced_dynamic_patterns.py` | Self-modifying, meta-learning, emergent behaviors |
| `real_world_implementation.py` | Production patterns: validation, caching, quotas |

---

## 📚 Code Examples

### Example 1: Simple Tool Creation

```python
from dynamic_langgraph_backend import agent_manager

# Add tool
success = agent_manager.add_tool_from_prompt(
    prompt="Create a tool that validates email addresses",
    tool_name="email_validator"
)

if success:
    print("✅ Tool created!")
```

### Example 2: Check Available Tools

```python
from dynamic_langgraph_backend import agent_manager
import json

tools = json.loads(agent_manager.get_tool_info())
print("Available tools:")
for tool_name, description in tools.items():
    print(f"  • {tool_name}: {description}")
```

### Example 3: Run Agent with Requirements

```python
from dynamic_langgraph_backend import run_agent_with_requirements

result = run_agent_with_requirements(
    user_input="What are the top 5 stock prices?",
    thread_id="conversation_1",
    requirements={
        "new_tools": [
            {"name": "stock_fetcher", "prompt": "Fetch stock prices"}
        ],
        "dynamic_behavior": "detailed"
    }
)
```

### Example 4: Production Usage with Validation

```python
from real_world_implementation import (
    ProductionDynamicAgent,
    ToolValidator,
    ToolRequestParser
)
from dynamic_langgraph_backend import agent_manager

# Create production agent
agent = ProductionDynamicAgent(agent_manager)

# Handle user message (automatic validation + caching)
response = agent.handle_user_message(
    user_message="Create a currency converter",
    thread_id="user_123"
)

print(response)
```

### Example 5: Self-Improving Agent

```python
from advanced_dynamic_patterns import SelfModifyingAgent

agent = SelfModifyingAgent()

# Track performance
agent.track_tool_usage("slow_calculator", 10.0, False, 2.0)
agent.track_tool_usage("slow_calculator", 9.5, False, 2.0)

# Auto-improve based on metrics
agent.auto_improve_tools()

# Get performance report
print(agent.get_performance_report())
```

---

## 🔍 Troubleshooting

### Problem: Tool not created

**Solution:**
```python
# Check Claude can access the tool
import os
print(os.getenv("OPENAI_API_KEY"))  # Make sure it's set

# Try with simpler prompt
agent_manager.add_tool_from_prompt(
    "Create a tool that adds two numbers",
    "simple_adder"
)
```

### Problem: Streamlit app crashes

**Solution:**
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/

# Reinstall
pip install --upgrade streamlit

# Run with debug mode
streamlit run dynamic_streamlit_frontend.py --logger.level=debug
```

### Problem: OpenAI rate limit

**Solution:**
```python
# Add delay between requests
import time
time.sleep(2)  # Wait 2 seconds between tool creations

# Or use quota management
from real_world_implementation import QuotaManager

quota_mgr = QuotaManager()
quota_mgr.set_quota("tool_name", daily=100, hourly=10)
```

---

## 📊 Architecture Overview

```
User Input
    ↓
┌─────────────────────────────┐
│ Tool Request Parser         │ ← Detect if user wants new tool
└──────────────┬──────────────┘
               ↓
        Is Tool Request?
           /        \
         YES        NO
          ↓          ↓
   ┌──────────┐  ┌──────────────────┐
   │Create    │  │Run Agent with    │
   │Tool      │  │existing tools    │
   │From      │  │                  │
   │Prompt    │  └────────┬─────────┘
   └────┬─────┘           ↓
        ↓           [LLM Decision]
   ┌────────────┐        /    \
   │Validate    │    USE TOOL  NO TOOL
   │Tool Code   │      /          \
   └────┬───────┘      ↓           ↓
        ↓        ┌──────────┐  ┌────────┐
   ┌────────────┐│Execute   │  │Answer  │
   │Cache       ││Tool      │  │Query   │
   │Tool        │└──────┬───┘  └───┬────┘
   └────┬───────┘       ↓          ↓
        ↓            ┌──────────────────┐
   ┌──────────┐      │ Return Response  │
   │Register  │──→   │ to User          │
   │with Agent│      └──────────────────┘
   └──────────┘
```

---

## 🎓 Learning Path

**Beginner:**
1. Run Streamlit UI
2. Create tools via sidebar
3. Try conversations

**Intermediate:**
1. Run `dynamic_agent_examples.py`
2. Understand `dynamic_langgraph_backend.py`
3. Modify examples

**Advanced:**
1. Study `advanced_dynamic_patterns.py`
2. Implement `real_world_implementation.py` patterns
3. Build production agents

---

## ⚡ Performance Tips

### Tip 1: Cache Tools
Tools are cached automatically. Check `.tool_cache/` directory.

### Tip 2: Batch Tool Creation
```python
tools = [...]
for tool_name, prompt in tools:
    agent_manager.add_tool_from_prompt(prompt, tool_name)
    # Graph rebuilds automatically
```

### Tip 3: Use Tool Quotas
```python
from real_world_implementation import QuotaManager

quota = QuotaManager()
quota.set_quota("expensive_tool", daily=100)
```

### Tip 4: Monitor Performance
```python
from advanced_dynamic_patterns import SelfModifyingAgent

agent = SelfModifyingAgent()
# Track every tool use
print(agent.get_performance_report())
```

---

## 🔐 Security Best Practices

### ✅ Do:
- Use `.env` for API keys
- Validate user prompts before creating tools
- Sandbox tool execution
- Log all tool creations

### ❌ Don't:
- Hardcode API keys in code
- Create unlimited tools (set quotas)
- Trust user prompts directly
- Run untrusted code

---

## 📞 Common Commands

```bash
# Run Streamlit
streamlit run dynamic_streamlit_frontend.py

# Run examples
python dynamic_agent_examples.py

# Test validation
python real_world_implementation.py

# Test advanced patterns
python advanced_dynamic_patterns.py

# View tool cache
ls -la .tool_cache/

# Check SQLite database
sqlite3 dynamic_chatbot.db ".tables"
```

---

## 🎯 What to Build Next

1. **Custom UI** - Build your own frontend
2. **Slack Bot** - Connect to Slack
3. **Discord Bot** - Add to Discord
4. **REST API** - Expose as API
5. **Mobile App** - React Native client
6. **Dashboard** - Analytics & monitoring

---

## 📖 Further Reading

- `DYNAMIC_AGENT_GUIDE.md` - Comprehensive guide
- `advanced_dynamic_patterns.py` - Code comments
- `real_world_implementation.py` - Production patterns

---

## ✨ Next Steps

1. **Now:** Run Streamlit UI and play with it
2. **Next:** Create tools from your use cases
3. **Then:** Modify code for your needs
4. **Finally:** Deploy to production

**Let's build amazing dynamic agents! 🚀**
