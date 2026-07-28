import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from app.models.schemas import AgentDefinition
from app.config import Config

# Map common planner phrasings onto the three static registry roles.
CAPABILITY_ALIASES = {
    "researcher": "Researcher",
    "research": "Researcher",
    "web researcher": "Researcher",
    "web research": "Researcher",
    "information gatherer": "Researcher",
    "data analyst": "Data Analyst",
    "data analysis": "Data Analyst",
    "analyst": "Data Analyst",
    "analysis": "Data Analyst",
    "document generator": "Document Generator",
    "document writer": "Document Generator",
    "report generator": "Document Generator",
    "report writer": "Document Generator",
    "writer": "Document Generator",
    "documentation": "Document Generator",
}


class RegistryStore:
    """Manages reading the Agent Registry and Tool Catalog JSON files."""

    def __init__(self, data_dir: Path = Config.REGISTRY_DIR):
        self.data_dir = data_dir
        self.agent_registry_path = self.data_dir / "agent_registry.json"
        self.tool_catalog_path = self.data_dir / "tool_catalog.json"

    def load_agent_registry(self) -> List[AgentDefinition]:
        """Loads all static agents from agent_registry.json."""
        if not self.agent_registry_path.exists():
            return []
        with open(self.agent_registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [AgentDefinition(**item) for item in data]

    def load_tool_catalog(self) -> List[Dict[str, Any]]:
        """Loads all allowed tool definitions from tool_catalog.json."""
        if not self.tool_catalog_path.exists():
            return []
        with open(self.tool_catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def find_agent_by_capability(self, capability: str) -> Optional[AgentDefinition]:
        """Finds an agent in the registry that matches the required capability."""
        agents = self.load_agent_registry()
        if not agents:
            return None

        cap_lower = (capability or "").lower().strip()
        if not cap_lower:
            return None

        canonical = CAPABILITY_ALIASES.get(cap_lower)
        if canonical:
            for agent in agents:
                if agent.role.lower() == canonical.lower():
                    return agent

        for agent in agents:
            if agent.role.lower() == cap_lower:
                return agent

        for agent in agents:
            role_lower = agent.role.lower()
            if cap_lower in role_lower or role_lower in cap_lower:
                return agent

        for agent in agents:
            if cap_lower in agent.description.lower():
                return agent

        return None
