"""
Agent layer for the codebase RAG assistant.

Public API:

    from src.agent import (
        PlannerAgent,
        QueryCategory,
        QueryClassifier,
        HeuristicQueryClassifier,
        PlanStep,
        ExecutionPlan,
        PlanningStrategy,
        AgentOrchestrator,
        OrchestratorResult,
        IterationTrace,
    )
"""

from .orchestrator import AgentOrchestrator, IterationTrace, OrchestratorResult
from .planner import (
    ExecutionPlan,
    HeuristicQueryClassifier,
    PlanningStrategy,
    PlannerAgent,
    PlanStep,
    QueryCategory,
    QueryClassifier,
    extract_probable_filename,
    split_into_subquestions,
)

__all__ = [
    "PlannerAgent",
    "QueryCategory",
    "QueryClassifier",
    "HeuristicQueryClassifier",
    "PlanStep",
    "ExecutionPlan",
    "PlanningStrategy",
    "extract_probable_filename",
    "split_into_subquestions",
    "AgentOrchestrator",
    "OrchestratorResult",
    "IterationTrace",
]
