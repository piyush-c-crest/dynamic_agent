# dynamic_langgraph_backend.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any, Callable
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, Tool, BaseTool
from dotenv import load_dotenv
import sqlite3
import requests
import json
import re
from datetime import datetime
from langchain_groq import ChatGroq

load_dotenv()

# -------------------
# 5. State Definition
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 1. LLM
# -------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# -------------------
# 2. Dynamic Tool Registry
# -------------------
class DynamicToolRegistry:
    """
    Manages dynamic tool creation and registration.
    Tools can be created from natural language prompts.
    """
    def __init__(self):
        self.tools: dict[str, BaseTool] = {}
        self.tool_code: dict[str, str] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register built-in tools"""
        self.register_tool("search", self._search_tool())
        self.register_tool("calculator", self._calculator_tool())
        self.register_tool("stock_price", self._stock_price_tool())
    
    def register_tool(self, name: str, tool_obj: BaseTool, code: str = ""):
        """Register a tool in the registry"""
        self.tools[name] = tool_obj
        self.tool_code[name] = code
        print(f"✅ Tool registered: {name}")
    
    def get_all_tools(self) -> list[BaseTool]:
        """Get all registered tools as a list"""
        return list(self.tools.values())
    
    def get_tool(self, name: str) -> BaseTool | None:
        """Get a specific tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> dict[str, str]:
        """List all tools with their descriptions"""
        return {name: tool.description for name, tool in self.tools.items()}
    
    @staticmethod
    def _search_tool() -> BaseTool:
        """Create search tool"""
        @tool
        def search(query: str) -> str:
            """Search the web for information"""
            search_tool = DuckDuckGoSearchRun(region="us-en")
            result = search_tool.run(query)
            return result[:500]  # Limit results
        return search
    
    @staticmethod
    def _calculator_tool() -> BaseTool:
        """Create calculator tool"""
        @tool
        def calculator(first_num: float, second_num: float, operation: str) -> dict:
            """Perform arithmetic: add, sub, mul, div"""
            ops = {
                "add": lambda a, b: a + b,
                "sub": lambda a, b: a - b,
                "mul": lambda a, b: a * b,
                "div": lambda a, b: a / b if b != 0 else "Error: Division by zero"
            }
            result = ops.get(operation, lambda a, b: "Error: Unknown operation")(first_num, second_num)
            return {"result": result, "operation": operation}
        return calculator
    
    @staticmethod
    def _stock_price_tool() -> BaseTool:
        """Create stock price tool"""
        @tool
        def get_stock_price(symbol: str) -> dict:
            """Fetch stock price for a symbol (e.g., AAPL, TSLA)"""
            api_key = "C9PE94QUEW9VWGFM"
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
            try:
                r = requests.get(url, timeout=5)
                return r.json()
            except Exception as e:
                return {"error": str(e)}
        return get_stock_price
    
    def create_tool_from_prompt(self, prompt: str, tool_name: str) -> BaseTool:
        """
        Create a new tool from a natural language prompt.
        The prompt should describe what the tool should do.
        """
        print(f"🔨 Creating tool '{tool_name}' from prompt...")
        
        creation_prompt = f"""
        Create a Python tool function for the following requirement:
        {prompt}
        
        Requirements:
        1. Use the @tool decorator from langchain_core.tools
        2. Include a clear docstring
        3. Handle errors gracefully
        4. Return JSON-serializable data
        5. Keep it focused and simple
                
        Example format:

        @tool
        def {tool_name}(input_data: str) -> dict:
            Description of what the tool does.
            code implementation here
            
        
        Return ONLY the function code, starting with @tool and ending with the return statement.
        No imports needed (they're already available).
        """
        
        response = llm.invoke(creation_prompt)
        tool_code = response.content

        print(f"📝 Generated code for tool '{tool_name}':\n{tool_code}")
        
        # Extract function code
        try:
            # Remove markdown formatting if present
            if "```python" in tool_code:
                tool_code = tool_code.split("```python")[1].split("```")[0]
            elif "```" in tool_code:
                tool_code = tool_code.split("```")[1].split("```")[0]
            
            # Execute the code to create the tool
            local_vars = {}
            exec(tool_code, {
                "tool": tool,
                "requests": requests,
                "json": json,
                "datetime": datetime,
            }, local_vars)
            
            # Find the function (should be the one decorated with @tool)
            for var_name, var_obj in local_vars.items():
                if isinstance(var_obj, BaseTool):
                    self.register_tool(tool_name, var_obj, tool_code)
                    return var_obj
            
            raise ValueError("No tool created from prompt")
        
        except Exception as e:
            print(f"❌ Error creating tool: {str(e)}")
            raise

# -------------------
# 3. Dynamic Node Builder
# -------------------
class DynamicNodeBuilder:
    """
    Create custom nodes dynamically based on requirements.
    """
    @staticmethod
    def create_preprocessing_node(instruction: str) -> Callable:
        """Create a node that preprocesses messages based on instruction"""
        def preprocess_node(state: dict) -> dict:
            messages = state.get("messages", [])
            if messages:
                # Apply preprocessing instruction
                last_msg = messages[-1]
                # In production, you'd call Claude to actually process it
                print(f"📝 Preprocessing with instruction: {instruction}")
            return state
        return preprocess_node
    
    @staticmethod
    def create_filter_node(filter_type: str) -> Callable:
        """Create a node that filters messages"""
        def filter_node(state: dict) -> dict:
            messages = state.get("messages", [])
            # Example: filter sensitive information
            if filter_type == "sensitive":
                print(f"🔒 Filtering sensitive information")
            return state
        return filter_node
    
    @staticmethod
    def create_enrichment_node(data_source: str) -> Callable:
        """Create a node that enriches messages with external data"""
        def enrichment_node(state: dict) -> dict:
            print(f"📊 Enriching with data from: {data_source}")
            # Add context to messages
            return state
        return enrichment_node

# -------------------
# 4. Dynamic Agent Manager
# -------------------
class DynamicAgentManager:
    """
    Manages dynamic agent configuration and graph building.
    """
    def __init__(self):
        self.tool_registry = DynamicToolRegistry()
        self.node_builder = DynamicNodeBuilder()
        self.llm_with_tools = llm.bind_tools(self.tool_registry.get_all_tools())
        self.custom_nodes: dict[str, Callable] = {}
        self.custom_edges: list[tuple] = []
        self.chatbot = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph with current tools and nodes"""
        graph = StateGraph(ChatState)
        
        # Core nodes
        graph.add_node("chat_node", self._chat_node)
        graph.add_node("tools", ToolNode(self.tool_registry.get_all_tools()))
        
        # Add custom nodes if any
        for node_name, node_func in self.custom_nodes.items():
            graph.add_node(node_name, node_func)
        
        # Edges
        graph.add_edge(START, "chat_node")
        graph.add_conditional_edges("chat_node", tools_condition)
        graph.add_edge("tools", "chat_node")
        
        # Add custom edges
        for source, destination in self.custom_edges:
            graph.add_edge(source, destination)
        
        # Compile with checkpointing
        conn = sqlite3.connect(database="dynamic_chatbot.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)
        
        return graph.compile(checkpointer=checkpointer)
    
    def _chat_node(self, state: ChatState):
        """LLM node with all available tools"""
        messages = state["messages"]
        self.llm_with_tools = llm.bind_tools(self.tool_registry.get_all_tools())
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def add_tool_from_prompt(self, prompt: str, tool_name: str):
        """Dynamically add a new tool"""
        try:
            self.tool_registry.create_tool_from_prompt(prompt, tool_name)
            self.chatbot = self._build_graph()  # Rebuild graph with new tool
            print(f"✅ Graph rebuilt with new tool: {tool_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to add tool: {e}")
            return False
    
    def add_custom_node(self, node_name: str, node_func: Callable, edges: list[tuple] = None):
        """Add a custom node to the graph"""
        self.custom_nodes[node_name] = node_func
        if edges:
            self.custom_edges.extend(edges)
        self.chatbot = self._build_graph()
        print(f"✅ Graph rebuilt with new node: {node_name}")
    
    def get_tool_info(self) -> str:
        """Get formatted info about available tools"""
        tools_info = self.tool_registry.list_tools()
        return json.dumps(tools_info, indent=2)
    
    def run(self, user_input: str, thread_id: str, requirements: dict = None):
        """
        Run the agent with optional dynamic modifications.
        
        requirements: {
            "new_tools": [
                {"name": "tool_name", "prompt": "description"}
            ],
            "preprocessing": "instruction",
            "dynamic_behavior": "behavior_type"
        }
        """
        # Handle dynamic requirements
        if requirements:
            if requirements.get("new_tools"):
                for tool_spec in requirements["new_tools"]:
                    self.add_tool_from_prompt(tool_spec["prompt"], tool_spec["name"])
        
        # Run the agent
        config = {"configurable": {"thread_id": thread_id}}
        messages_input = [HumanMessage(content=user_input)]
        
        response = self.chatbot.invoke(
            {"messages": messages_input},
            config=config
        )
        
        return response

# -------------------
# 6. Initialize Global Agent
# -------------------
agent_manager = DynamicAgentManager()

# -------------------
# 7. Helper Functions
# -------------------
def get_agent_tools() -> str:
    """Get available tools as JSON"""
    return agent_manager.get_tool_info()

def add_tool_dynamically(tool_name: str, tool_prompt: str) -> bool:
    """Add a new tool on the fly"""
    return agent_manager.add_tool_from_prompt(tool_prompt, tool_name)

def run_agent_with_requirements(user_input: str, thread_id: str, requirements: dict = None):
    """Run agent with dynamic modifications"""
    return agent_manager.run(user_input, thread_id, requirements)
