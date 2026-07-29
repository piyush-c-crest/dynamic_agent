from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Goal(BaseModel):
    """Stage 1: Goal Intake representation."""
    title: str = Field(..., description="Summary title of the overall goal")
    description: str = Field(..., description="Detailed description of the user request")
    deliverables: List[str] = Field(default_factory=list, description="Concrete outputs to be produced")
    constraints: List[str] = Field(default_factory=list, description="Key constraints and limitations")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made about the request")

class Task(BaseModel):
    """Atomic task element in the TaskGraph DAG."""
    id: str = Field(..., description="Unique identifier for the task, e.g. task_01")
    description: str = Field(..., description="Detailed description of what needs to be done")
    required_capability: str = Field(..., description="Required agent capability or role type")
    dependencies: List[str] = Field(default_factory=list, description="IDs of tasks that must complete first")
    status: str = Field("pending", description="Status of task: pending, running, success, failed")
    assigned_agent: Optional[str] = Field(None, description="The specific resolved agent name")
    assigned_tools: List[str] = Field(default_factory=list, description="List of tools allowed for this task")
    output_key: Optional[str] = Field(None, description="Key where the result is stored in shared memory")
    error_message: Optional[str] = Field(None, description="Error details if the task failed")

class TaskGraph(BaseModel):
    """Stage 2: DAG representation of the execution plan."""
    tasks: List[Task] = Field(default_factory=list, description="List of tasks in the graph")

class AgentDefinition(BaseModel):
    """Stage 3: Definition of an agent (static or dynamically created)."""
    role: str = Field(..., description="The unique role or specialist name")
    description: str = Field(..., description="The agent's focus area and scope")
    tools: List[str] = Field(default_factory=list, description="Names of tools allowed for this agent")
    system_prompt: Optional[str] = Field(None, description="Detailed system instructions / prompt for the agent's behavior")
    is_dynamic: bool = Field(False, description="Whether this agent was created on the fly by the factory")

class EvaluationResult(BaseModel):
    """Stage 6: Output of the task evaluator."""
    task_id: str = Field(..., description="ID of the task being evaluated")
    passed: bool = Field(..., description="Whether the deliverables are satisfactory")
    reason: str = Field(..., description="Detailed explanation of the pass/fail judgment")

class RunState(BaseModel):
    """Run-level persistence state (guardrails and counters)."""
    run_id: str = Field(..., description="Unique ID for this orchestrator execution run")
    storage_path: str = Field("", description="Relative path under data/runs and data/logs")
    status: str = Field("running", description="Status: running, success, partial_failure, failed")
    replans_used: int = Field(0, description="Counter for replan operations")
    agents_spawned: int = Field(
        0,
        description="Count of dynamically factory-spawned agents (not static registry lookups).",
    )
    created_at: str = Field(..., description="ISO timestamp of run creation")
    updated_at: str = Field(..., description="ISO timestamp of last update")
