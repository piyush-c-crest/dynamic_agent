Here is a breakdown of how the dynamic tool creation works in your `dynamic_langgraph_backend.py`, how the underlying mechanics operate, and where you can see the generated code.

### 1. How a new tool is created dynamically

The magic happens inside the `create_tool_from_prompt()` method of the `DynamicToolRegistry` class. When a request to create a new tool comes in, the system follows these steps:

1. **Prompting the LLM:** It constructs a strict prompt telling the LLM to write a Python function that fulfills the user's requirements. It instructs the LLM to decorate the function with LangChain's `@tool` decorator, include a docstring, and return *only* the Python code.
2. **Generating the Code:** The LLM (`gpt-oss-120b` via Groq in your setup) writes the code as a raw string and returns it.
3. **Cleaning the Output:** The system strips away any markdown formatting (like ```python ... ```) to get just the raw, executable Python code.
4. **Executing the String as Code:** It uses Python's built-in `exec()` function to compile and run that string of code on the fly. It passes in a restricted local environment (allowing access to `tool`, `requests`, `json`, and `datetime` so the new tool can use them).
5. **Extracting the Tool:** After `exec()` runs, the script scans the local variables for the new function that was created and decorated as a `BaseTool`.
6. **Rebuilding the Graph:** Once the new tool is registered in the `DynamicToolRegistry`, the `DynamicAgentManager` completely rebuilds the LangGraph (`self._build_graph()`). It binds the newly expanded list of tools to the LLM (`llm.bind_tools()`) so the LLM is immediately aware that it can use this new tool for subsequent messages.

### 2. How it works (The Mechanics)

Normally, Python code is parsed and compiled before the program runs. However, Python is dynamic, which means it can interpret new strings of code during runtime. 

By using `exec()`, your application is essentially doing this:
```python
code_string = """
@tool
def my_dynamic_tool(query: str) -> dict:
    '''Searches for something'''
    return {"result": "data"}
"""

# Evaluates the string into actual Python functions in memory
local_variables = {}
exec(code_string, {"tool": tool}, local_variables)

# local_variables now contains the 'my_dynamic_tool' function!
```
Because the LangGraph graph is just a StateGraph object, your backend can just throw away the old graph, create a new one with the updated tool list, and LangGraph will seamlessly pick up the new tool.

### 3. How to see the generated code

There are two ways you can view the actual Python code that the LLM generated for the tool:

**Option A: Check your terminal/console**
The script is already set up to print the code to your console whenever a tool is created. Look for this in your terminal logs:
```text
🔨 Creating tool 'my_new_tool' from prompt...
📝 Generated code for tool 'my_new_tool':
@tool
def my_new_tool(input_data: str) -> dict:
    ...
```

**Option B: Access it programmatically**
The `DynamicToolRegistry` saves the raw string of the code in a dictionary called `self.tool_code`. If you ever want to retrieve or display the code in your frontend/API, you can access it like this:

```python
# To get the code for a specific tool:
tool_name = "my_custom_tool"
code_string = agent_manager.tool_registry.tool_code.get(tool_name)
print(code_string)
```

If you want to expose this to your frontend, you could add a quick helper function to `dynamic_langgraph_backend.py`:
```python
def get_tool_code(tool_name: str) -> str:
    """Returns the raw python code for a dynamically generated tool"""
    return agent_manager.tool_registry.tool_code.get(tool_name, "Tool not found.")
```