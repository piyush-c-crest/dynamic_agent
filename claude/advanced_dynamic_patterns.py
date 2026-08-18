# advanced_dynamic_patterns.py

"""
Advanced patterns for dynamic AI agents:
- Self-modifying agents
- Learning from interactions
- Meta-programming capabilities
- Tool discovery and optimization
- Emergent behaviors
"""

from typing import Callable, Any, Dict, List
from dataclasses import dataclass
from datetime import datetime
import json
from dynamic_langgraph_backend import agent_manager, llm
import sqlite3

# ===================== 1. Self-Modifying Agents =====================

@dataclass
class ToolMetadata:
    """Track tool usage and performance"""
    name: str
    created_at: datetime
    usage_count: int = 0
    success_rate: float = 1.0
    avg_execution_time: float = 0.0
    last_used: datetime = None
    error_count: int = 0
    feedback_score: float = 0.0

class SelfModifyingAgent:
    """
    An agent that modifies itself based on:
    - User feedback
    - Performance metrics
    - Conversation patterns
    - Error rates
    """
    
    def __init__(self):
        self.tool_metadata: Dict[str, ToolMetadata] = {}
        self.tool_performance_db = self._init_db()
        self.learning_history = []
    
    def _init_db(self):
        """Initialize SQLite for tracking metrics"""
        conn = sqlite3.connect("agent_metrics.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_metrics (
                tool_name TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                usage_count INTEGER,
                success_rate REAL,
                avg_execution_time REAL,
                error_count INTEGER,
                feedback_score REAL
            )
        """)
        conn.commit()
        return conn
    
    def track_tool_usage(self, tool_name: str, execution_time: float, 
                        success: bool, feedback: float = None):
        """Track tool performance"""
        if tool_name not in self.tool_metadata:
            self.tool_metadata[tool_name] = ToolMetadata(
                name=tool_name,
                created_at=datetime.now()
            )
        
        metadata = self.tool_metadata[tool_name]
        metadata.usage_count += 1
        metadata.last_used = datetime.now()
        
        # Update success rate
        if not success:
            metadata.error_count += 1
        metadata.success_rate = (metadata.usage_count - metadata.error_count) / metadata.usage_count
        
        # Update execution time
        metadata.avg_execution_time = (
            (metadata.avg_execution_time * (metadata.usage_count - 1) + execution_time) 
            / metadata.usage_count
        )
        
        # Update feedback if provided
        if feedback is not None:
            metadata.feedback_score = feedback
        
        # Save to database
        self._save_metrics(tool_name, metadata)
        
        print(f"📊 {tool_name}: {metadata.usage_count} uses, {metadata.success_rate:.1%} success")
    
    def _save_metrics(self, tool_name: str, metadata: ToolMetadata):
        """Persist metrics to database"""
        cursor = self.tool_performance_db.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO tool_metrics VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            tool_name,
            metadata.created_at,
            metadata.usage_count,
            metadata.success_rate,
            metadata.avg_execution_time,
            metadata.error_count,
            metadata.feedback_score
        ))
        self.tool_performance_db.commit()
    
    def auto_improve_tools(self):
        """Automatically improve tools based on performance"""
        print("\n🔄 Auto-Improvement Cycle Started")
        
        for tool_name, metadata in self.tool_metadata.items():
            # Remove underperforming tools
            if metadata.success_rate < 0.5 and metadata.usage_count > 10:
                print(f"❌ Removing underperforming tool: {tool_name}")
                agent_manager.tool_registry.tools.pop(tool_name, None)
                continue
            
            # Optimize slow tools
            if metadata.avg_execution_time > 5.0:
                print(f"⚡ Optimizing slow tool: {tool_name}")
                self._optimize_tool(tool_name)
            
            # Enhance poorly rated tools
            if metadata.feedback_score < 3.0 and metadata.usage_count > 5:
                print(f"📈 Enhancing poorly rated tool: {tool_name}")
                self._enhance_tool(tool_name)
    
    def _optimize_tool(self, tool_name: str):
        """Improve tool performance"""
        prompt = f"""
        The tool '{tool_name}' is running slow (avg: {self.tool_metadata[tool_name].avg_execution_time}s).
        Create an optimized version that:
        1. Caches results when possible
        2. Uses async operations
        3. Reduces unnecessary API calls
        4. Returns results faster
        """
        # Generate optimized version
        response = llm.invoke(prompt)
        print(f"💡 Optimization suggestion: {response.content[:100]}...")
    
    def _enhance_tool(self, tool_name: str):
        """Improve tool quality based on feedback"""
        metadata = self.tool_metadata[tool_name]
        prompt = f"""
        Users gave tool '{tool_name}' a score of {metadata.feedback_score}/5.
        Feedback indicates the tool needs improvement.
        Suggest enhancements to:
        1. Better error handling
        2. Improved accuracy
        3. Enhanced user experience
        4. Additional features
        """
        response = llm.invoke(prompt)
        print(f"📝 Enhancement suggestions: {response.content[:100]}...")
    
    def get_performance_report(self) -> str:
        """Generate performance report"""
        report = {
            "total_tools": len(self.tool_metadata),
            "total_uses": sum(m.usage_count for m in self.tool_metadata.values()),
            "avg_success_rate": sum(m.success_rate for m in self.tool_metadata.values()) / len(self.tool_metadata) if self.tool_metadata else 0,
            "tools": {
                name: {
                    "uses": m.usage_count,
                    "success_rate": f"{m.success_rate:.1%}",
                    "avg_time": f"{m.avg_execution_time:.2f}s",
                    "feedback": f"{m.feedback_score:.1f}/5"
                }
                for name, m in self.tool_metadata.items()
            }
        }
        return json.dumps(report, indent=2)

# ===================== 2. Meta-Learning Agent =====================

class MetaLearningAgent:
    """
    Agent that learns patterns about tool usage and creates new tools
    based on discovered patterns.
    """
    
    def __init__(self):
        self.usage_patterns: List[Dict] = []
        self.tool_combinations: Dict[str, int] = {}
        self.discovered_workflows: List[str] = []
    
    def record_interaction(self, user_input: str, tools_used: List[str], result: str):
        """Record interactions for learning"""
        pattern = {
            "timestamp": datetime.now(),
            "input": user_input,
            "tools_used": tools_used,
            "result": result
        }
        self.usage_patterns.append(pattern)
        
        # Track tool combinations
        combo = tuple(sorted(tools_used))
        self.tool_combinations[str(combo)] = self.tool_combinations.get(str(combo), 0) + 1
    
    def discover_workflows(self) -> List[Dict]:
        """Discover common workflows from patterns"""
        workflows = []
        
        # Find frequently used tool combinations
        sorted_combos = sorted(
            self.tool_combinations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("\n🔍 Discovered Workflows:")
        for combo_str, count in sorted_combos[:5]:
            if count > 2:  # Only workflows used multiple times
                tools = eval(combo_str)
                workflow = {
                    "tools": list(tools),
                    "frequency": count,
                    "pattern": f"Use {', '.join(tools)} together"
                }
                workflows.append(workflow)
                print(f"  • {workflow['pattern']} (used {count} times)")
        
        return workflows
    
    def create_workflow_tool(self, workflow_tools: List[str]) -> bool:
        """
        Create a composite tool that chains workflow tools together.
        Example: weather + calendar + notification tools → "plan_day_tool"
        """
        workflow_name = "_".join(workflow_tools) + "_workflow"
        
        prompt = f"""
        Create a composite tool that uses these tools together:
        {', '.join(workflow_tools)}
        
        The tool should:
        1. Orchestrate these tools in a logical sequence
        2. Pass results from one tool to the next
        3. Combine final results meaningfully
        4. Handle errors from any tool
        
        Use case: Common workflow combining these tools
        """
        
        print(f"🔧 Creating workflow tool: {workflow_name}")
        success = agent_manager.add_tool_from_prompt(prompt, workflow_name)
        
        if success:
            self.discovered_workflows.append(workflow_name)
            print(f"✅ Workflow tool created: {workflow_name}")
        
        return success
    
    def auto_create_workflows(self):
        """Automatically create tools for discovered workflows"""
        workflows = self.discover_workflows()
        
        for workflow in workflows:
            if workflow["frequency"] > 3:  # Only for frequent workflows
                self.create_workflow_tool(workflow["tools"])

# ===================== 3. Context-Aware Agent =====================

class ContextAwareAgent:
    """
    Agent that adapts behavior based on conversation context.
    """
    
    def __init__(self):
        self.context_history = []
        self.user_profiles = {}
        self.context_tools = {}
    
    def analyze_context(self, conversation: List[str]) -> Dict[str, Any]:
        """Analyze conversation to extract context"""
        context = {
            "domain": self._detect_domain(conversation),
            "complexity": self._assess_complexity(conversation),
            "user_expertise": self._assess_expertise(conversation),
            "technical_level": self._assess_technical_level(conversation),
            "required_tools": []
        }
        return context
    
    def _detect_domain(self, conversation: List[str]) -> str:
        """Detect which domain the conversation is about"""
        domains = {
            "finance": ["stock", "price", "investment", "portfolio", "crypto"],
            "science": ["research", "experiment", "hypothesis", "data", "analysis"],
            "coding": ["code", "debug", "function", "error", "algorithm"],
            "health": ["health", "medical", "doctor", "symptom", "disease"],
            "education": ["learn", "study", "explain", "course", "assignment"],
        }
        
        conv_text = " ".join(conversation).lower()
        
        for domain, keywords in domains.items():
            if any(kw in conv_text for kw in keywords):
                return domain
        
        return "general"
    
    def _assess_complexity(self, conversation: List[str]) -> str:
        """Assess conversation complexity"""
        avg_length = sum(len(msg.split()) for msg in conversation) / len(conversation)
        
        if avg_length < 10:
            return "simple"
        elif avg_length < 30:
            return "moderate"
        else:
            return "complex"
    
    def _assess_expertise(self, conversation: List[str]) -> str:
        """Assess user expertise level"""
        expertise_keywords = {
            "beginner": ["explain", "how do", "what is", "basics"],
            "intermediate": ["implement", "configure", "integrate"],
            "advanced": ["optimize", "architecture", "performance", "scalability"],
        }
        
        conv_text = " ".join(conversation).lower()
        
        for level, keywords in expertise_keywords.items():
            if any(kw in conv_text for kw in keywords):
                return level
        
        return "intermediate"
    
    def _assess_technical_level(self, conversation: List[str]) -> str:
        """Assess technical requirements"""
        tech_keywords = {
            "non_technical": ["simple", "easy", "user-friendly"],
            "technical": ["api", "sdk", "database", "architecture"],
            "highly_technical": ["kernel", "memory management", "concurrency"],
        }
        
        conv_text = " ".join(conversation).lower()
        
        for level, keywords in tech_keywords.items():
            if any(kw in conv_text for kw in keywords):
                return level
        
        return "technical"
    
    def create_context_specific_tools(self, context: Dict[str, Any]):
        """Create tools based on context"""
        domain = context["domain"]
        expertise = context["user_expertise"]
        
        print(f"\n📌 Creating {expertise}-level tools for {domain} domain")
        
        domain_tools = {
            "finance": [
                ("stock_analyzer", "Analyze stock prices and trends"),
                ("portfolio_tracker", "Track investment portfolio"),
                ("financial_calculator", "Calculate financial metrics"),
            ],
            "science": [
                ("data_plotter", "Create plots and visualizations"),
                ("statistical_analyzer", "Perform statistical analysis"),
                ("hypothesis_tester", "Test scientific hypotheses"),
            ],
            "coding": [
                ("code_formatter", "Format and beautify code"),
                ("bug_finder", "Identify bugs in code"),
                ("test_generator", "Generate unit tests"),
            ],
        }
        
        if domain in domain_tools:
            for tool_name, tool_desc in domain_tools[domain]:
                if not agent_manager.tool_registry.get_tool(tool_name):
                    print(f"  🔧 Creating {tool_name}...")
                    agent_manager.add_tool_from_prompt(tool_desc, tool_name)

# ===================== 4. Adaptive Learning Agent =====================

class AdaptiveLearningAgent:
    """
    Agent that learns from corrections and user feedback.
    """
    
    def __init__(self):
        self.corrections = []
        self.feedback_db = self._init_feedback_db()
        self.learned_rules = []
    
    def _init_feedback_db(self):
        """Initialize feedback database"""
        conn = sqlite3.connect("agent_feedback.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                user_input TEXT,
                agent_response TEXT,
                feedback_type TEXT,
                correction TEXT,
                severity TEXT
            )
        """)
        conn.commit()
        return conn
    
    def record_feedback(self, user_input: str, agent_response: str, 
                       feedback_type: str, correction: str = None):
        """Record user feedback"""
        feedback = {
            "timestamp": datetime.now(),
            "user_input": user_input,
            "agent_response": agent_response,
            "feedback_type": feedback_type,  # "correction", "praise", "suggestion"
            "correction": correction
        }
        self.corrections.append(feedback)
        
        # Assess severity
        severity = "low" if feedback_type == "praise" else "high" if feedback_type == "correction" else "medium"
        
        # Save to database
        cursor = self.feedback_db.cursor()
        cursor.execute("""
            INSERT INTO feedback VALUES (NULL, ?, ?, ?, ?, ?, ?)
        """, (
            feedback["timestamp"],
            user_input,
            agent_response,
            feedback_type,
            correction,
            severity
        ))
        self.feedback_db.commit()
        
        # Learn from feedback
        self._learn_from_feedback(feedback)
    
    def _learn_from_feedback(self, feedback: Dict):
        """Extract learning rules from feedback"""
        if feedback["feedback_type"] == "correction":
            rule = {
                "pattern": feedback["user_input"],
                "incorrect_response": feedback["agent_response"],
                "correct_response": feedback["correction"],
                "learned_at": datetime.now()
            }
            self.learned_rules.append(rule)
            print(f"📚 Learned rule: {len(self.learned_rules)} rules now in database")
    
    def apply_learned_rules(self, user_input: str) -> str:
        """Apply learned rules to improve responses"""
        for rule in self.learned_rules:
            if rule["pattern"].lower() in user_input.lower():
                print(f"💡 Applying learned rule...")
                return rule["correct_response"]
        
        return None
    
    def get_learning_report(self) -> str:
        """Report on learning progress"""
        report = {
            "total_feedback": len(self.corrections),
            "corrections": sum(1 for c in self.corrections if c["feedback_type"] == "correction"),
            "suggestions": sum(1 for c in self.corrections if c["feedback_type"] == "suggestion"),
            "rules_learned": len(self.learned_rules),
            "accuracy_improvement": f"{(len(self.learned_rules) / max(1, len(self.corrections))) * 100:.1f}%"
        }
        return json.dumps(report, indent=2)

# ===================== 5. Emergent Behavior Agent =====================

class EmergentBehaviorAgent:
    """
    Agent that exhibits emergent behaviors through tool composition
    and unexpected tool combinations.
    """
    
    def __init__(self):
        self.tool_graph = {}
        self.emergence_events = []
    
    def discover_tool_relationships(self):
        """Build graph of tool relationships"""
        tools = list(agent_manager.tool_registry.tools.keys())
        
        # Ask Claude to identify relationships
        prompt = f"""
        Given these tools: {', '.join(tools)}
        
        Identify:
        1. Which tools could work together
        2. What emergent capabilities arise from combinations
        3. Novel use cases enabled by composition
        
        Return as JSON.
        """
        
        response = llm.invoke(prompt)
        print(f"🔗 Tool relationship analysis:\n{response.content[:200]}...")
    
    def trigger_emergence(self, tool_combination: List[str]):
        """
        Trigger emergent behavior by combining tools in unexpected ways.
        """
        combination_name = "_".join(tool_combination) + "_emergence"
        
        prompt = f"""
        Create an innovative tool that combines these in an unexpected way:
        {', '.join(tool_combination)}
        
        Think about:
        1. Novel use cases
        2. Creative combinations
        3. Emergent properties
        4. Surprising capabilities
        
        The resulting tool should do something neither tool alone could do.
        """
        
        print(f"\n✨ Triggering emergence with: {', '.join(tool_combination)}")
        success = agent_manager.add_tool_from_prompt(prompt, combination_name)
        
        if success:
            event = {
                "timestamp": datetime.now(),
                "combination": tool_combination,
                "emergent_tool": combination_name
            }
            self.emergence_events.append(event)
            print(f"✨ Emergent behavior discovered: {combination_name}")

# ===================== 6. Usage Examples =====================

def example_self_modifying_agent():
    """Example: Self-modifying agent that improves itself"""
    print("\n" + "="*60)
    print("EXAMPLE: Self-Modifying Agent")
    print("="*60)
    
    agent = SelfModifyingAgent()
    
    # Simulate tool usage with metrics
    agent.track_tool_usage("weather", 0.5, True, 4.5)
    agent.track_tool_usage("weather", 0.6, True, 4.8)
    agent.track_tool_usage("calculator", 0.1, True, 5.0)
    agent.track_tool_usage("slow_tool", 15.0, False, 2.0)
    agent.track_tool_usage("slow_tool", 14.5, False, 2.0)
    
    print("\n📊 Performance Report:")
    print(agent.get_performance_report())
    
    print("\n🔄 Running auto-improvement...")
    agent.auto_improve_tools()

def example_meta_learning():
    """Example: Agent that learns from interactions"""
    print("\n" + "="*60)
    print("EXAMPLE: Meta-Learning Agent")
    print("="*60)
    
    agent = MetaLearningAgent()
    
    # Record interactions
    agent.record_interaction("What's the weather?", ["weather"], "Sunny, 72F")
    agent.record_interaction("Show me my schedule and weather", ["calendar", "weather"], "Busy day, nice weather")
    agent.record_interaction("Plan my day", ["calendar", "weather", "notification"], "Day planned")
    agent.record_interaction("Get weather and check calendar", ["calendar", "weather"], "All set")
    agent.record_interaction("Weather check", ["weather"], "Clear skies")
    
    print("\n🔍 Discovering workflows...")
    agent.auto_create_workflows()

def example_context_aware():
    """Example: Context-aware tool creation"""
    print("\n" + "="*60)
    print("EXAMPLE: Context-Aware Agent")
    print("="*60)
    
    agent = ContextAwareAgent()
    
    # Analyze conversation
    conversation = [
        "How do I optimize my stock portfolio?",
        "What's the best allocation strategy?",
        "How do I minimize risk?"
    ]
    
    context = agent.analyze_context(conversation)
    print(f"\n📊 Detected Context:")
    for key, value in context.items():
        print(f"  {key}: {value}")
    
    print(f"\n🔧 Creating context-specific tools...")
    agent.create_context_specific_tools(context)

def example_adaptive_learning():
    """Example: Agent learning from corrections"""
    print("\n" + "="*60)
    print("EXAMPLE: Adaptive Learning Agent")
    print("="*60)
    
    agent = AdaptiveLearningAgent()
    
    # Record feedback
    agent.record_feedback(
        "Calculate 5 * 3",
        "The answer is 16",
        "correction",
        "The answer is 15"
    )
    
    agent.record_feedback(
        "What's 10 / 2?",
        "The answer is 4",
        "correction",
        "The answer is 5"
    )
    
    agent.record_feedback(
        "Your code formatting is great!",
        "Applied formatting rules",
        "praise"
    )
    
    print("\n📚 Learning Report:")
    print(agent.get_learning_report())
    
    # Test learned rules
    test_input = "Calculate 5 * 3"
    result = agent.apply_learned_rules(test_input)
    if result:
        print(f"\n✅ Applied learned rule for: {test_input}")

def example_emergence():
    """Example: Discovering emergent behaviors"""
    print("\n" + "="*60)
    print("EXAMPLE: Emergent Behavior Discovery")
    print("="*60)
    
    agent = EmergentBehaviorAgent()
    
    # Discover relationships
    agent.discover_tool_relationships()
    
    # Trigger emergence
    agent.trigger_emergence(["weather", "calendar", "notification"])
    agent.trigger_emergence(["calculator", "stock_price", "portfolio_tracker"])

# ===================== Main =====================

if __name__ == "__main__":
    print("🚀 Advanced Dynamic Agent Patterns")
    print("="*60)
    
    try:
        example_self_modifying_agent()
        example_meta_learning()
        example_context_aware()
        example_adaptive_learning()
        example_emergence()
        
        print("\n" + "="*60)
        print("✅ All advanced examples completed!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
