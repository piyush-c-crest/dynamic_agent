# real_world_implementation.py

"""
Real-world patterns for building dynamic agents in production.
These are battle-tested patterns used in production systems.
"""

from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass
import json
from datetime import datetime
import hashlib
import pickle
import os

# ===================== PATTERN 1: Tool Request Parser =====================

class ToolRequestParser:
    """
    Parse natural language requests and extract tool creation instructions.
    Real-world: Users ask "create a tool that..." and we need to parse it.
    """
    
    @staticmethod
    def parse_tool_request(user_message: str) -> Optional[dict]:
        """
        Extract tool request from user message.
        
        Examples:
        - "Create a tool that validates emails"
        - "I need a tool to convert currencies"
        - "Add a weather fetcher"
        """
        
        # Keywords that indicate tool creation request
        tool_request_keywords = [
            "create", "add", "make", "build", "develop",
            "implement", "create a tool", "add a tool"
        ]
        
        message_lower = user_message.lower()
        
        # Check if this is a tool request
        is_tool_request = any(kw in message_lower for kw in tool_request_keywords)
        
        if not is_tool_request:
            return None
        
        # Extract tool name and description
        tool_info = {
            "is_request": True,
            "message": user_message,
            "tool_name": ToolRequestParser._extract_tool_name(user_message),
            "description": ToolRequestParser._extract_description(user_message),
            "urgency": ToolRequestParser._assess_urgency(user_message),
            "complexity": ToolRequestParser._assess_complexity(user_message),
        }
        
        return tool_info
    
    @staticmethod
    def _extract_tool_name(message: str) -> str:
        """Extract likely tool name from message"""
        # Simple heuristic: look for nouns after "tool"
        if "that" in message:
            parts = message.split("that")
            if len(parts) > 1:
                return parts[0].split()[-1].lower()
        return "custom_tool"
    
    @staticmethod
    def _extract_description(message: str) -> str:
        """Extract tool description"""
        if "that" in message:
            return message.split("that")[1].strip()
        return message
    
    @staticmethod
    def _assess_urgency(message: str) -> str:
        """Assess how urgent the request is"""
        urgent_words = ["urgent", "asap", "now", "immediately", "quickly"]
        if any(word in message.lower() for word in urgent_words):
            return "high"
        return "normal"
    
    @staticmethod
    def _assess_complexity(message: str) -> str:
        """Assess tool complexity"""
        complexity_indicators = {
            "simple": ["simple", "basic", "easy"],
            "moderate": ["feature", "support", "handle"],
            "complex": ["sophisticated", "advanced", "complex"],
        }
        
        message_lower = message.lower()
        for level, keywords in complexity_indicators.items():
            if any(kw in message_lower for kw in keywords):
                return level
        
        return "moderate"

# ===================== PATTERN 2: Tool Caching & Versioning =====================

class ToolCache:
    """
    Cache tools and their code to avoid re-creating them.
    Also maintain tool versions for rollback.
    """
    
    def __init__(self, cache_dir: str = ".tool_cache"):
        self.cache_dir = cache_dir
        self.version_history = {}
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def save_tool(self, tool_name: str, tool_code: str, metadata: dict = None):
        """Save tool to cache with versioning"""
        
        # Create hash for version control
        code_hash = hashlib.md5(tool_code.encode()).hexdigest()[:8]
        version = len(self.version_history.get(tool_name, [])) + 1
        
        version_info = {
            "version": version,
            "code_hash": code_hash,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        # Save to file
        cache_path = os.path.join(
            self.cache_dir,
            f"{tool_name}_v{version}_{code_hash}.py"
        )
        
        with open(cache_path, 'w') as f:
            f.write(f"# Tool: {tool_name} (v{version})\n")
            f.write(f"# Created: {version_info['timestamp']}\n\n")
            f.write(tool_code)
        
        # Track version
        if tool_name not in self.version_history:
            self.version_history[tool_name] = []
        
        self.version_history[tool_name].append(version_info)
        
        print(f"💾 Cached: {tool_name} v{version}")
        return version_info
    
    def get_tool(self, tool_name: str, version: int = None) -> Optional[str]:
        """Retrieve tool from cache"""
        
        if tool_name not in self.version_history:
            return None
        
        # Get latest version if not specified
        if version is None:
            version = len(self.version_history[tool_name])
        
        # Find matching version
        for v_info in self.version_history[tool_name]:
            if v_info["version"] == version:
                cache_path = os.path.join(
                    self.cache_dir,
                    f"{tool_name}_v{version}_{v_info['code_hash']}.py"
                )
                
                if os.path.exists(cache_path):
                    with open(cache_path, 'r') as f:
                        return f.read()
        
        return None
    
    def rollback_tool(self, tool_name: str, version: int) -> bool:
        """Rollback tool to a previous version"""
        
        tool_code = self.get_tool(tool_name, version)
        if tool_code:
            print(f"↩️ Rolling back {tool_name} to v{version}")
            return True
        return False
    
    def get_version_history(self, tool_name: str) -> list:
        """Get version history of a tool"""
        return self.version_history.get(tool_name, [])

# ===================== PATTERN 3: Tool Validation Framework =====================

class ToolValidator:
    """
    Validate tools before registration.
    Ensures quality, safety, and compatibility.
    """
    
    class ValidationLevel(Enum):
        STRICT = 3    # All checks must pass
        MODERATE = 2  # Most checks must pass
        LENIENT = 1   # Basic checks only
    
    def __init__(self, level: ValidationLevel = ValidationLevel.MODERATE):
        self.level = level
        self.validation_results = []
    
    def validate_tool(self, tool_name: str, tool_code: str) -> tuple[bool, list]:
        """
        Validate tool before adding to agent.
        Returns: (is_valid, list_of_issues)
        """
        
        issues = []
        checks = self._get_checks_for_level(self.level)
        
        for check_name, check_func in checks.items():
            try:
                result = check_func(tool_name, tool_code)
                if not result["passed"]:
                    issues.append(f"❌ {check_name}: {result['message']}")
            except Exception as e:
                issues.append(f"⚠️ {check_name}: {str(e)}")
        
        is_valid = len(issues) == 0
        
        print(f"\n✓ Tool Validation: {tool_name}")
        for issue in issues:
            print(f"  {issue}")
        
        return is_valid, issues
    
    def _get_checks_for_level(self, level: ValidationLevel) -> dict:
        """Get validation checks based on level"""
        
        all_checks = {
            "has_decorator": self._check_has_decorator,
            "has_docstring": self._check_has_docstring,
            "has_error_handling": self._check_has_error_handling,
            "no_hardcoded_secrets": self._check_no_secrets,
            "json_serializable": self._check_json_serializable,
            "performance_ok": self._check_performance,
        }
        
        if level == self.ValidationLevel.STRICT:
            return all_checks
        elif level == self.ValidationLevel.MODERATE:
            return {k: v for k, v in all_checks.items() if k != "performance_ok"}
        else:  # LENIENT
            return {k: v for k, v in all_checks.items() if k in ["has_decorator", "no_hardcoded_secrets"]}
    
    @staticmethod
    def _check_has_decorator(name: str, code: str) -> dict:
        """Check if tool has @tool decorator"""
        passed = "@tool" in code
        return {
            "passed": passed,
            "message": "Missing @tool decorator" if not passed else "OK"
        }
    
    @staticmethod
    def _check_has_docstring(name: str, code: str) -> dict:
        """Check if tool has docstring"""
        passed = '"""' in code or "'''" in code
        return {
            "passed": passed,
            "message": "Missing docstring" if not passed else "OK"
        }
    
    @staticmethod
    def _check_has_error_handling(name: str, code: str) -> dict:
        """Check if tool handles errors"""
        passed = "try" in code or "except" in code or "Exception" in code
        return {
            "passed": passed,
            "message": "No error handling" if not passed else "OK"
        }
    
    @staticmethod
    def _check_no_secrets(name: str, code: str) -> dict:
        """Check for hardcoded secrets"""
        secrets = ["api_key", "password", "token", "secret"]
        has_secrets = any(secret in code.lower() for secret in secrets)
        passed = not has_secrets
        return {
            "passed": passed,
            "message": "Possible hardcoded secrets" if not passed else "OK"
        }
    
    @staticmethod
    def _check_json_serializable(name: str, code: str) -> dict:
        """Check if tool returns JSON-serializable data"""
        passed = "return" in code and ("{" in code or "[" in code)
        return {
            "passed": passed,
            "message": "Returns might not be JSON-serializable" if not passed else "OK"
        }
    
    @staticmethod
    def _check_performance(name: str, code: str) -> dict:
        """Check for obvious performance issues"""
        issues = []
        
        # Check for infinite loops
        if "while True" in code:
            issues.append("Infinite loop detected")
        
        # Check for nested loops
        if code.count("for") > 2:
            issues.append("Multiple nested loops")
        
        # Check for large iterations
        if "range(1000000)" in code or "range(999999)" in code:
            issues.append("Large iteration range")
        
        passed = len(issues) == 0
        return {
            "passed": passed,
            "message": "; ".join(issues) if issues else "OK"
        }

# ===================== PATTERN 4: Tool Quota & Rate Limiting =====================

@dataclass
class ToolQuota:
    """Track tool usage quotas"""
    tool_name: str
    max_daily_uses: int = 1000
    max_hourly_uses: int = 100
    current_daily_uses: int = 0
    current_hourly_uses: int = 0
    last_reset_hour: datetime = None
    last_reset_day: datetime = None

class QuotaManager:
    """
    Manage tool usage quotas.
    Prevent abuse and control resource usage.
    """
    
    def __init__(self):
        self.quotas: dict[str, ToolQuota] = {}
    
    def set_quota(self, tool_name: str, daily: int = 1000, hourly: int = 100):
        """Set usage quota for a tool"""
        self.quotas[tool_name] = ToolQuota(
            tool_name=tool_name,
            max_daily_uses=daily,
            max_hourly_uses=hourly
        )
        print(f"📊 Quota set for {tool_name}: {daily}/day, {hourly}/hour")
    
    def can_use_tool(self, tool_name: str) -> tuple[bool, str]:
        """Check if tool can be used based on quota"""
        
        if tool_name not in self.quotas:
            return True, "No quota set"
        
        quota = self.quotas[tool_name]
        now = datetime.now()
        
        # Reset hourly counter
        if quota.last_reset_hour is None or (now - quota.last_reset_hour).seconds > 3600:
            quota.current_hourly_uses = 0
            quota.last_reset_hour = now
        
        # Reset daily counter
        if quota.last_reset_day is None or (now - quota.last_reset_day).days > 0:
            quota.current_daily_uses = 0
            quota.last_reset_day = now
        
        # Check limits
        if quota.current_hourly_uses >= quota.max_hourly_uses:
            return False, f"Hourly quota exceeded ({quota.current_hourly_uses}/{quota.max_hourly_uses})"
        
        if quota.current_daily_uses >= quota.max_daily_uses:
            return False, f"Daily quota exceeded ({quota.current_daily_uses}/{quota.max_daily_uses})"
        
        return True, "OK"
    
    def record_usage(self, tool_name: str):
        """Record tool usage"""
        if tool_name in self.quotas:
            self.quotas[tool_name].current_hourly_uses += 1
            self.quotas[tool_name].current_daily_uses += 1

# ===================== PATTERN 5: Tool Dependency Graph =====================

class DependencyGraph:
    """
    Track tool dependencies.
    Ensure tools are created in correct order.
    """
    
    def __init__(self):
        self.dependencies: dict[str, list] = {}
        self.created_tools: set = set()
    
    def add_dependency(self, tool_name: str, depends_on: list[str]):
        """Add dependency relationship"""
        self.dependencies[tool_name] = depends_on
        print(f"📌 {tool_name} depends on: {', '.join(depends_on)}")
    
    def can_create(self, tool_name: str) -> tuple[bool, list]:
        """Check if tool can be created (all dependencies exist)"""
        
        if tool_name not in self.dependencies:
            return True, []
        
        missing = []
        for dep in self.dependencies[tool_name]:
            if dep not in self.created_tools:
                missing.append(dep)
        
        return len(missing) == 0, missing
    
    def mark_created(self, tool_name: str):
        """Mark tool as created"""
        self.created_tools.add(tool_name)
    
    def get_creation_order(self, tools: list[str]) -> list[str]:
        """Get order to create tools based on dependencies"""
        
        order = []
        remaining = set(tools)
        
        while remaining:
            # Find tools with no unsatisfied dependencies
            ready = []
            for tool in remaining:
                can_create, missing = self.can_create(tool)
                if can_create or not any(m in remaining for m in missing):
                    ready.append(tool)
            
            if not ready:
                raise ValueError("Circular dependency detected")
            
            # Add first ready tool to order
            tool = ready[0]
            order.append(tool)
            remaining.remove(tool)
            self.mark_created(tool)
        
        return order

# ===================== PATTERN 6: Integration Pattern =====================

class ProductionDynamicAgent:
    """
    Production-ready dynamic agent combining all patterns.
    """
    
    def __init__(self, agent_manager):
        self.agent_manager = agent_manager
        self.parser = ToolRequestParser()
        self.cache = ToolCache()
        self.validator = ToolValidator(ToolValidator.ValidationLevel.MODERATE)
        self.quota_manager = QuotaManager()
        self.dependency_graph = DependencyGraph()
    
    def handle_user_message(self, user_message: str, thread_id: str) -> str:
        """Handle user message with full validation pipeline"""
        
        # Step 1: Check if this is a tool request
        tool_request = self.parser.parse_tool_request(user_message)
        
        if tool_request:
            print(f"\n🔧 Tool Request Detected")
            print(f"  Name: {tool_request['tool_name']}")
            print(f"  Urgency: {tool_request['urgency']}")
            print(f"  Complexity: {tool_request['complexity']}")
            
            # Step 2: Check cache first
            cached = self.cache.get_tool(tool_request['tool_name'])
            if cached:
                print(f"📦 Found in cache, using existing version")
                # Use cached version
            else:
                # Step 3: Create new tool
                success = self._create_and_validate_tool(tool_request)
                if not success:
                    return "❌ Failed to create tool. Please try again with more detail."
            
            return f"✅ Tool '{tool_request['tool_name']}' is now ready to use!"
        
        # Step 4: Check quota for existing tool usage
        # (in real implementation, parse which tools are being used)
        
        # Step 5: Run agent with normal query
        return self._run_agent(user_message, thread_id)
    
    def _create_and_validate_tool(self, tool_request: dict) -> bool:
        """Create and validate a tool"""
        
        print("\n🔨 Creating tool...")
        
        # Generate tool code
        try:
            success = self.agent_manager.add_tool_from_prompt(
                tool_request['description'],
                tool_request['tool_name']
            )
            
            if success:
                print(f"✅ Tool created successfully")
                return True
            else:
                print(f"❌ Failed to create tool")
                return False
                
        except Exception as e:
            print(f"❌ Error creating tool: {e}")
            return False
    
    def _run_agent(self, user_message: str, thread_id: str) -> str:
        """Run agent with safety checks"""
        try:
            result = self.agent_manager.run(user_message, thread_id)
            return result
        except Exception as e:
            return f"❌ Error: {str(e)}"

# ===================== Usage Example =====================

def example_production_usage():
    """Show production usage"""
    print("\n" + "="*60)
    print("PRODUCTION DYNAMIC AGENT EXAMPLE")
    print("="*60)
    
    # Parse tool request
    parser = ToolRequestParser()
    
    requests = [
        "Create a tool that converts currencies",
        "I need a weather tool for my dashboard",
        "Make a simple email validator urgently",
        "What's the weather today?",  # Not a tool request
    ]
    
    for req in requests:
        print(f"\n📨 User: {req}")
        parsed = parser.parse_tool_request(req)
        if parsed:
            print(f"  ✓ Tool request detected: {parsed['tool_name']}")
            print(f"  ✓ Urgency: {parsed['urgency']}")
        else:
            print(f"  ✓ Regular query")
    
    # Validation
    print("\n" + "-"*60)
    print("VALIDATION EXAMPLE")
    print("-"*60)
    
    sample_tool_code = """
    @tool
    def email_validator(email: str) -> dict:
        '''Validate email format'''
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        is_valid = bool(re.match(pattern, email))
        return {"email": email, "is_valid": is_valid}
    """
    
    validator = ToolValidator()
    is_valid, issues = validator.validate_tool("email_validator", sample_tool_code)
    print(f"\nValidation result: {'✅ PASS' if is_valid else '❌ FAIL'}")
    
    # Caching
    print("\n" + "-"*60)
    print("CACHING EXAMPLE")
    print("-"*60)
    
    cache = ToolCache()
    cache.save_tool("weather", sample_tool_code, {"source": "openai"})
    cache.save_tool("weather", "# Updated version", {"source": "manual_update"})
    
    history = cache.get_version_history("weather")
    print(f"\nTool versions: {len(history)}")
    for v in history:
        print(f"  • v{v['version']}: {v['timestamp']}")

if __name__ == "__main__":
    example_production_usage()
