# dynamic_agent_examples.py

from dynamic_langgraph_backend import agent_manager, add_tool_dynamically
import uuid
import json

# ===================== Example 1: Add Tools on the Fly =====================

def example_1_add_tools():
    """Dynamically add new tools without restarting"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Adding Tools Dynamically")
    print("="*60)
    
    # Define new tools as natural language prompts
    tools_to_create = [
        {
            "name": "weather",
            "prompt": """
            Create a tool that fetches weather information for a city.
            It should return temperature, humidity, and weather condition.
            Use a free weather API or mock data.
            """
        },
        {
            "name": "email_validator",
            "prompt": """
            Create a tool that validates email addresses.
            Return whether the email is valid and any issues found.
            """
        },
        {
            "name": "text_summarizer",
            "prompt": """
            Create a tool that summarizes long text into key points.
            Should work with paragraphs or articles.
            Return a list of bullet points.
            """
        }
    ]
    
    print(f"\n📊 Current tools: {list(json.loads(agent_manager.get_tool_info()).keys())}")
    
    for tool_spec in tools_to_create:
        print(f"\n🔨 Creating tool: {tool_spec['name']}")
        success = add_tool_dynamically(tool_spec["name"], tool_spec["prompt"])
        
        if success:
            print(f"✅ {tool_spec['name']} created successfully!")
        else:
            print(f"❌ Failed to create {tool_spec['name']}")
    
    print(f"\n📊 Updated tools: {list(json.loads(agent_manager.get_tool_info()).keys())}")

# ===================== Example 2: Dynamic Behavior Change =====================

def example_2_dynamic_behavior():
    """Change agent behavior based on context"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Dynamic Behavior Modification")
    print("="*60)
    
    thread_id = str(uuid.uuid4())
    
    behaviors = {
        "detailed": "Provide very detailed responses with examples",
        "concise": "Keep responses short and to the point",
        "code_focus": "Focus on code examples and technical details"
    }
    
    user_query = "Explain Python list comprehensions"
    
    for behavior_name, behavior_desc in behaviors.items():
        print(f"\n\n--- Behavior: {behavior_name.upper()} ---")
        print(f"Instruction: {behavior_desc}")
        
        requirements = {
            "dynamic_behavior": behavior_name,
            "preprocessing": behavior_desc
        }
        
        # Run agent with specific behavior
        # Note: In production, you'd modify the LLM system prompt
        print(f"User: {user_query}")

# ===================== Example 3: Conditional Tool Creation =====================

def example_3_conditional_tools():
    """Create tools based on conversation content"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Conditional Tool Creation")
    print("="*60)
    
    # Simulate analyzing conversation to determine needed tools
    user_messages = [
        "I need to convert between currencies",
        "Can you help with temperature conversion?",
        "I need to validate JSON data"
    ]
    
    # Map user requests to tool creation prompts
    tool_mapping = {
        "convert between currencies": {
            "name": "currency_converter",
            "prompt": "Create a tool that converts amounts between different currencies using current exchange rates"
        },
        "temperature conversion": {
            "name": "temperature_converter",
            "prompt": "Create a tool that converts temperature between Celsius, Fahrenheit, and Kelvin"
        },
        "validate JSON": {
            "name": "json_validator",
            "prompt": "Create a tool that validates JSON strings and reports errors with line numbers"
        }
    }
    
    for msg in user_messages:
        print(f"\n📨 User message: {msg}")
        
        # Check which tool is needed
        for keyword, tool_spec in tool_mapping.items():
            if keyword.lower() in msg.lower():
                print(f"✅ Detected need for: {tool_spec['name']}")
                
                # Check if tool already exists
                existing_tools = json.loads(agent_manager.get_tool_info()).keys()
                if tool_spec["name"] not in existing_tools:
                    print(f"🔨 Creating tool...")
                    add_tool_dynamically(tool_spec["name"], tool_spec["prompt"])

# ===================== Example 4: Multi-Step Tool Building =====================

def example_4_tool_pipeline():
    """Create tools that depend on each other"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Tool Pipeline (Tools Using Tools)")
    print("="*60)
    
    print("\n🔗 Creating a pipeline of tools that work together...")
    
    # First tool: Data fetcher
    fetcher_prompt = """
    Create a tool that fetches data from a public API (like JSONPlaceholder).
    It should handle errors gracefully and return structured data.
    """
    
    # Second tool: Data processor
    processor_prompt = """
    Create a tool that processes and cleans data.
    It should remove duplicates, handle missing values, and format results.
    """
    
    # Third tool: Data analyzer
    analyzer_prompt = """
    Create a tool that analyzes processed data.
    Return statistics like count, average, min, max, and unique values.
    """
    
    tools_pipeline = [
        ("api_data_fetcher", fetcher_prompt),
        ("data_processor", processor_prompt),
        ("data_analyzer", analyzer_prompt),
    ]
    
    for tool_name, tool_prompt in tools_pipeline:
        print(f"\n📍 Step: Creating {tool_name}...")
        success = add_tool_dynamically(tool_name, tool_prompt)
        if success:
            print(f"✅ {tool_name} ready")

# ===================== Example 5: Runtime Tool Modification =====================

def example_5_runtime_modification():
    """Modify tools and agent based on runtime conditions"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Runtime Modifications")
    print("="*60)
    
    thread_id = str(uuid.uuid4())
    
    # Scenario 1: User is a developer
    print("\n👨‍💻 User Profile: Developer")
    print("📌 Enabling dev-focused tools...")
    
    dev_tools = [
        ("code_formatter", "Create a tool that formats Python/JavaScript code"),
        ("test_generator", "Create a tool that generates unit tests from code"),
        ("regex_helper", "Create a tool that helps build and test regex patterns"),
    ]
    
    # Scenario 2: User is a student
    print("\n👨‍🎓 User Profile: Student")
    print("📌 Enabling educational tools...")
    
    student_tools = [
        ("study_guide", "Create a tool that generates study guides from topics"),
        ("quiz_generator", "Create a tool that generates quiz questions"),
        ("definition_lookup", "Create a tool that explains complex terms"),
    ]
    
    print("\n📊 Profile-specific tools can be loaded based on user context")

# ===================== Example 6: Real-time Tool Addition =====================

def example_6_realtime_addition():
    """Show how tools are added in real conversations"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Real-time Tool Addition During Conversation")
    print("="*60)
    
    thread_id = str(uuid.uuid4())
    conversation = [
        "What's the weather?",
        "Can you create a weather tool?",
        "Now check weather for New York",
        "Can you also add timezone conversion?",
        "Convert 3 PM EST to IST"
    ]
    
    for i, user_msg in enumerate(conversation, 1):
        print(f"\n[Turn {i}] User: {user_msg}")
        
        # Detect if user is asking for new tool
        if "create" in user_msg.lower() and "tool" in user_msg.lower():
            print("🔨 Tool creation request detected!")
            # Extract tool name and create it
        else:
            print("💬 Processing with existing tools...")

# ===================== Example 7: Advanced - Tool Composition =====================

def example_7_tool_composition():
    """Create complex tools by composing simpler ones"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Tool Composition")
    print("="*60)
    
    print("\n🧩 Creating composite tools from simpler ones...")
    
    composite_prompt = """
    Create a sophisticated tool called "data_pipeline" that:
    1. Fetches data from an API
    2. Validates the data structure
    3. Transforms it into a standard format
    4. Returns metadata about the transformation
    
    The tool should handle errors at each step and provide detailed logging.
    """
    
    print("\n📌 Creating composite tool...")
    # In a real scenario, this would orchestrate multiple existing tools
    add_tool_dynamically("data_pipeline", composite_prompt)

# ===================== Main Execution =====================

if __name__ == "__main__":
    print("🚀 Dynamic Agent Examples")
    print("="*60)
    
    # Run examples
    try:
        example_1_add_tools()
        # Uncomment to run other examples
        # example_2_dynamic_behavior()
        # example_3_conditional_tools()
        # example_4_tool_pipeline()
        # example_5_runtime_modification()
        # example_6_realtime_addition()
        # example_7_tool_composition()
        
        print("\n" + "="*60)
        print("✅ All examples completed!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
