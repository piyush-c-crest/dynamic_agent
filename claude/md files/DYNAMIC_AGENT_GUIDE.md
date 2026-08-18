# Dynamic AI Agent - Complete Guide

## Overview

A **dynamic AI agent** creates tools, behaviors, and workflows on-the-fly based on:
- User prompts and requirements
- Runtime conditions
- Conversation context
- User profiles

This is different from a static agent where tools are hardcoded at startup.

---

## What Makes An Agent Dynamic?

### Static Agent (Your Original Code)
```python
tools = [search_tool, calculator, get_stock_price]  # Fixed at startup
```

### Dynamic Agent (New)
```python
# Tools created on-the-fly
user_prompt = "Create a weather tool"
agent.add_tool_from_prompt(user_prompt, "weather")

# New tool immediately available without restart
```

---

## Key Components

### 1. **DynamicToolRegistry**
Manages tool creation and registration.

```python
registry = DynamicToolRegistry()

# Register a tool
registry.register_tool("calculator", calculator_tool)

# Add tool from natural language
registry.create_tool_from_prompt(
    prompt="A tool that validates emails",
    tool_name="email_validator"
)

# Get all tools
tools = registry.get_all_tools()
```

### 2. **DynamicNodeBuilder**
Creates custom nodes on demand.

```python
builder = DynamicNodeBuilder()

# Create preprocessing node
preprocess = builder.create_preprocessing_node(
    "Remove sensitive data from messages"
)

# Create filter node
filter_node = builder.create_filter_node("sensitive_data")

# Create enrichment node
enrich = builder.create_enrichment_node("wikipedia_data")
```

### 3. **DynamicAgentManager**
Orchestrates everything - tools, nodes, graph rebuilding.

```python
manager = DynamicAgentManager()

# Add new tool and rebuild graph
manager.add_tool_from_prompt(prompt, tool_name)

# Add custom behavior node
manager.add_custom_node("preprocessor", preprocess_func)

# Run with dynamic requirements
manager.run(user_input, thread_id, requirements={
    "new_tools": [...],
    "preprocessing": "...",
    "dynamic_behavior": "detailed"
})
```

---

## Usage Patterns

### Pattern 1: User-Requested Tool Creation

**Scenario:** User says "Create a tool that converts currencies"

```python
# Backend detects request
if "create" in user_input and "tool" in user_input:
    # Extract tool description from user input
    tool_prompt = extract_tool_description(user_input)
    tool_name = extract_tool_name(user_input)
    
    # Create and register tool
    agent_manager.add_tool_from_prompt(tool_prompt, tool_name)
    
    # Confirm to user
    return f"✅ Tool '{tool_name}' created and ready!"
```

### Pattern 2: Conditional Tool Creation

**Scenario:** Analyze conversation and create needed tools

```python
def analyze_and_create_tools(conversation):
    # Map keywords to tool requirements
    keyword_tool_map = {
        "weather": "A tool that fetches weather data",
        "email": "A tool for email validation",
        "json": "A tool for JSON validation",
    }
    
    for keyword, tool_prompt in keyword_tool_map.items():
        if keyword in conversation.lower():
            tool_name = keyword
            if agent_manager.tool_registry.get_tool(tool_name) is None:
                agent_manager.add_tool_from_prompt(tool_prompt, tool_name)
```

### Pattern 3: Profile-Based Tool Loading

**Scenario:** Different tools for different user types

```python
def load_tools_for_profile(user_profile):
    if user_profile == "developer":
        tools = [
            ("code_formatter", "Format Python/JavaScript code"),
            ("regex_helper", "Build and test regex patterns"),
            ("test_generator", "Generate unit tests"),
        ]
    elif user_profile == "data_analyst":
        tools = [
            ("csv_analyzer", "Analyze CSV data"),
            ("data_visualizer", "Generate data insights"),
            ("sql_helper", "Write SQL queries"),
        ]
    
    for tool_name, prompt in tools:
        agent_manager.add_tool_from_prompt(prompt, tool_name)
```

### Pattern 4: Tool Pipeline

**Scenario:** Tools that work sequentially

```python
def create_data_processing_pipeline():
    # Tool 1: Fetch data
    agent_manager.add_tool_from_prompt(
        "Fetch JSON data from APIs",
        "api_fetcher"
    )
    
    # Tool 2: Validate data
    agent_manager.add_tool_from_prompt(
        "Validate JSON structure",
        "json_validator"
    )
    
    # Tool 3: Transform data
    agent_manager.add_tool_from_prompt(
        "Transform data to standard format",
        "data_transformer"
    )
    
    # Tool 4: Analyze data
    agent_manager.add_tool_from_prompt(
        "Generate statistics and insights",
        "data_analyzer"
    )

# User: "Process this API endpoint"
# Agent automatically chains: fetch -> validate -> transform -> analyze
```

### Pattern 5: Behavior Modification

**Scenario:** Change how agent responds based on context

```python
requirements = {
    "dynamic_behavior": "detailed",  # vs "concise", "code-focused", etc.
    "preprocessing": "Remove PII before processing",
    "tone": "professional",
    "language": "technical"
}

agent_manager.run(user_input, thread_id, requirements)
```

### Pattern 6: Real-time Tool Modification

**Scenario:** Modify existing tool behavior

```python
# Mid-conversation: User says "Make this tool faster"
# Update tool prompt/implementation
new_prompt = "Make the tool more efficient, cache results"
agent_manager.tool_registry.update_tool("weather", new_prompt)
```

---

## How It Works Under the Hood

### 1. Tool Creation from Prompt

```
User: "Create a weather tool"
      ↓
Agent receives prompt
      ↓
Send to Claude: "Generate Python code for a weather tool with @tool decorator"
      ↓
Claude returns Python code
      ↓
Execute code in safe sandbox
      ↓
Register as langchain Tool
      ↓
Rebuild graph with new tool
      ↓
Tool immediately available
```

### 2. Graph Rebuilding

When a new tool is added:

```
OLD GRAPH:
chat_node → tools_condition → [search, calculator] → chat_node

NEW TOOL ADDED:
chat_node → tools_condition → [search, calculator, weather] → chat_node
                                          ↑
                                   New tool in ToolNode
```

### 3. State Management

Each thread maintains:
- Conversation history (messages)
- Available tools at that point in time
- Custom nodes and edges
- Tool execution logs

---

## Examples

### Example 1: Basic Dynamic Tool

```python
from dynamic_langgraph_backend import agent_manager

# Create tool from prompt
agent_manager.add_tool_from_prompt(
    prompt="Convert temperature between Celsius and Fahrenheit",
    tool_name="temperature_converter"
)

# Use it immediately
result = agent_manager.run(
    user_input="Convert 25C to Fahrenheit",
    thread_id="conversation_1"
)
```

### Example 2: Multiple Tools at Once

```python
tools_to_create = [
    ("currency_converter", "Convert between currencies"),
    ("time_converter", "Convert between timezones"),
    ("unit_converter", "Convert between units"),
]

for tool_name, prompt in tools_to_create:
    agent_manager.add_tool_from_prompt(prompt, tool_name)
```

### Example 3: Conditional Creation

```python
def detect_needed_tools(user_message):
    if "weather" in user_message.lower():
        if not agent_manager.tool_registry.get_tool("weather"):
            agent_manager.add_tool_from_prompt(
                "Fetch weather data",
                "weather"
            )
    
    if "stock" in user_message.lower():
        if not agent_manager.tool_registry.get_tool("stock_analyzer"):
            agent_manager.add_tool_from_prompt(
                "Analyze stock prices",
                "stock_analyzer"
            )
```

---

## Advanced Features

### 1. Tool Versioning
```python
# Keep track of tool versions
registry.create_tool_from_prompt(prompt, "weather_v2")
# Can have multiple versions of same tool
```

### 2. Tool Composition
```python
# Create complex tool from simpler ones
composite_prompt = """
Create a data pipeline that:
1. Calls api_fetcher tool
2. Calls json_validator tool
3. Calls data_transformer tool
4. Calls data_analyzer tool
Return final analysis
"""
agent_manager.add_tool_from_prompt(composite_prompt, "data_pipeline")
```

### 3. Conditional Node Addition
```python
if needs_authentication:
    auth_node = builder.create_auth_node()
    agent_manager.add_custom_node("auth", auth_node)

if needs_logging:
    logging_node = builder.create_logging_node()
    agent_manager.add_custom_node("logger", logging_node)
```

### 4. Tool Execution Logs
```python
# Track which tools were used
logs = agent_manager.get_tool_execution_logs(thread_id)
# Returns: [
#   {"tool": "weather", "input": {...}, "output": {...}, "time": 0.5s},
#   {"tool": "calculator", "input": {...}, "output": {...}, "time": 0.1s},
# ]
```

---

## Best Practices

### ✅ Do
- Validate tool prompts before creation
- Test newly created tools with sample inputs
- Log tool creation for debugging
- Cache frequently used tools
- Version your tools

### ❌ Don't
- Create unlimited tools (can slow down graph)
- Expose API keys in tool prompts
- Create tools without error handling
- Rebuild graph too frequently
- Trust user prompts directly (validate first)

---

## Security Considerations

### 1. Sandbox Tool Execution
```python
# Tools run in restricted environment
safe_environment = {
    "tool": tool,
    "requests": requests,  # Only approved modules
    "json": json,
}
exec(tool_code, safe_environment)
```

### 2. Prompt Injection Prevention
```python
# Don't directly execute user prompts
# Instead, validate and sanitize
def safe_tool_creation(user_prompt):
    # Validate prompt doesn't contain malicious code
    if contains_dangerous_patterns(user_prompt):
        raise ValueError("Dangerous prompt")
    
    # Create tool from sanitized prompt
    return agent_manager.add_tool_from_prompt(user_prompt, tool_name)
```

### 3. Rate Limiting
```python
# Limit tool creation per user/time
rate_limiter.check(user_id, operation="create_tool")
# Prevents abuse of dynamic tool creation
```

---

## Comparison: Static vs Dynamic

| Aspect | Static Agent | Dynamic Agent |
|--------|---|---|
| Tools | Hardcoded at startup | Created at runtime |
| Flexibility | Limited | High |
| Setup Time | Fast | Slightly slower |
| Maintainability | Easier | More complex |
| User Control | None | Full control |
| Scalability | Limited | Unlimited |
| Use Case | Known workflows | Unpredictable needs |

---

## Troubleshooting

### Issue: Tool not appearing after creation
```python
# Solution: Check if tool was registered
print(agent_manager.get_tool_info())

# Rebuild graph if needed
agent_manager.chatbot = agent_manager._build_graph()
```

### Issue: Tool execution fails
```python
# Solution: Add error handling
try:
    result = agent_manager.run(user_input, thread_id)
except Exception as e:
    print(f"Tool execution error: {e}")
    # Fallback or notify user
```

### Issue: Tools not available after restart
```python
# Solution: Persist tool definitions
# Save tool code to database
# Recreate on startup from stored code
```

---

## Future Enhancements

1. **Tool Marketplace** - Share and download community tools
2. **Tool Versioning** - Maintain multiple versions
3. **Performance Optimization** - Cache frequently used tools
4. **Monitoring** - Track tool usage and performance
5. **Auto-scaling** - Create tools based on demand
6. **Tool Testing** - Automatic unit tests for generated tools
7. **Knowledge Base** - Learn from past tool usage

---

## Conclusion

Dynamic AI agents provide unprecedented flexibility. Users can:
- Request new capabilities in natural language
- Adapt agent behavior on-the-fly
- Compose tools into powerful workflows
- No code changes or restarts needed

This is the future of AI applications!
