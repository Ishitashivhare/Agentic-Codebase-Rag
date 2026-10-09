"""
planner.py - the Planner Agent.

Given a natural-language question about the codebase, the planner:

  1. Classifies the question into one of six categories.
  2. Builds an ExecutionPlan: an ordered list of tool calls (by name, from
     the existing src/tools layer), each with a stated purpose, and a
     flag for whether the step depends on a previous step's result.
  3. Decides whether the question needs a single retrieval pass or
     multiple passes.

The planner does NOT execute tools itself and does not import
ToolRegistry - it only needs tool *names*, which it takes from the tool
classes' `name` attributes so the two layers can never drift out of
sync. Execution (actually calling registry.run(...), and re-planning a
dependent step using a prior step's output) is deliberately left to an
orchestrator.

Design notes (SOLID):
  - Single responsibility: this module plans; it never calls a tool.
  - Open/closed: adding a 7th category means adding one QueryCategory
    value + one PlanningStrategy subclass + one registry entry - nothing
    existing changes.
  - Liskov substitution: PlannerAgent only calls `strategy.build_plan(query)`
    on whatever implements PlanningStrategy; every strategy is interchangeable.
  - Dependency inversion: PlannerAgent depends on the QueryClassifier
    Protocol, not on HeuristicQueryClassifier specifically - a future
    LLM-backed classifier can be swapped in without touching this class.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Protocol, runtime_checkable

from src.tools import (
    CodeRelationshipTool,
    FindFileTool,
    GetRepositoryStructureTool,
    ReadFileTool,
    SearchCodeTool,
    SearchSymbolTool,
)


# --------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------


class QueryCategory(str, Enum):
    """The six question categories the planner classifies into."""

    CODE_SEARCH = "code_search"
    FILE_SEARCH = "file_search"
    REPOSITORY_ANALYSIS = "repository_analysis"
    ARCHITECTURE_QUESTION = "architecture_question"
    FLOW_QUESTION = "flow_question"
    MULTI_STEP = "multi_step"
    SYMBOL_SEARCH = "symbol_search"
    RELATIONSHIP_ANALYSIS = "relationship_analysis"


# --------------------------------------------------------------------------
# Plan data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanStep:
    """One tool call in a plan.

    Attributes:
        tool_name: Name of the tool to call, matching a name registered
            in ToolRegistry (e.g. "search_code").
        arguments: Keyword arguments to pass to `registry.run(tool_name, **arguments)`.
        purpose: Human-readable reason this step exists - useful for
            logging, debugging, and for showing the user "why" the agent
            did something.
        requires_previous_result: If True, this step's `arguments`
            (typically the query text) should be refined by the
            orchestrator using the previous step's output before
            execution, rather than run verbatim. The planner flags that
            a second pass is needed without prescribing the exact
            follow-up query - it doesn't have the first pass's results
            yet, only the orchestrator does at execution time.
    """

    tool_name: str
    arguments: Dict[str, Any]
    purpose: str
    requires_previous_result: bool = False


@dataclass(frozen=True)
class ExecutionPlan:
    """The planner's complete output for one question.

    Attributes:
        query: The original question.
        category: The classified QueryCategory.
        steps: Ordered tool calls to execute.
        multi_pass: True if this plan requires more than one retrieval
            pass (either multiple steps against the same tool, or a step
            marked `requires_previous_result`).
        reasoning: Human-readable explanation of why this category and
            this plan were chosen - includes classifier evidence.
        sub_plans: For MULTI_STEP queries, the individual per-sub-question
            plans that were merged into `steps`. Empty for all other
            categories.
    """

    query: str
    category: QueryCategory
    steps: List[PlanStep]
    multi_pass: bool
    reasoning: str
    sub_plans: List["ExecutionPlan"] = field(default_factory=list)


# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------


@runtime_checkable
class QueryClassifier(Protocol):
    """Anything that can map a question to a QueryCategory.

    PlannerAgent depends on this Protocol, not on a concrete
    implementation, so the heuristic classifier below can later be
    replaced by an LLM-backed one without changing PlannerAgent.
    """

    def classify(self, query: str) -> QueryCategory: ...

    def explain(self, query: str) -> Dict[QueryCategory, int]: ...


# Sub-question boundary markers:
#   - a literal "?" (end of a question)
#   - a semicolon
#   - "then" / "and then" as a connective between two asks
#   - "and" immediately followed by a wh-word or auxiliary verb, e.g.
#     "...chunking work, and where is the embeddings file?" - the
#     lookahead keeps the wh-word attached to the start of the next piece
#     instead of consuming it.
# This is a heuristic, not a parser. Known, accepted false negative:
# "explain the architecture and the request flow" is NOT split, since it
# has no "?", ";", "then", or wh-word after "and" - a real limitation,
# not an oversight.
_SUBQUERY_SPLIT_PATTERN: Pattern[str] = re.compile(
    r"\?+|;|\band\s+then\b|\bthen\b"
    r"|\band\s+(?=(?:where|how|what|which|who|why|does|is|are|can|could|should)\b)",
    re.IGNORECASE,
)

# One or more filename-shaped tokens, e.g. "auth.py", "vector_db.py".
_FILENAME_PATTERN: Pattern[str] = re.compile(r"\b[\w\-]+\.[A-Za-z0-9]{1,6}\b")

_SOURCE_CAPABILITY_PATTERN: Pattern[str] = re.compile(
    r"\b(?:ingest|upload|support(?:ed)?|data\s+source|file\s+type|database|csv|sqlite|mysql|sql)\b",
    re.IGNORECASE,
)


def split_into_subquestions(query: str) -> List[str]:
    """Split a compound question into individual sub-questions.

    Returns a list with a single element (the original, trimmed query)
    if no split boundary is found.
    """
    parts = _SUBQUERY_SPLIT_PATTERN.split(query)
    cleaned = [p.strip(" ,.") for p in parts if p and p.strip(" ,.")]
    return cleaned if cleaned else [query.strip()]


def extract_probable_filename(query: str) -> Optional[str]:
    """Best-effort extraction of a filename-shaped token from a query,
    e.g. "where is auth.py" -> "auth.py". Returns None if no such token
    is present."""
    match = _FILENAME_PATTERN.search(query)
    return match.group(0) if match else None


class HeuristicQueryClassifier:
    """Deterministic, keyword/regex-based QueryClassifier.

    No LLM call, no network I/O - fast and fully reproducible, which
    matters for a planning step that runs on every query. Trades recall
    for these guarantees: a phrase the keyword lists don't anticipate
    falls back to CODE_SEARCH (the most general, most common intent for
    a codebase assistant) rather than misfiring into a narrower category.
    """

    # Checked in this order when two categories tie on score; earlier
    # entries are more specific and win ties over the general-purpose
    # CODE_SEARCH.
    _PRIORITY_ORDER: List[QueryCategory] = [
        QueryCategory.ARCHITECTURE_QUESTION,
        QueryCategory.FLOW_QUESTION,
        QueryCategory.RELATIONSHIP_ANALYSIS,
        QueryCategory.SYMBOL_SEARCH,
        QueryCategory.REPOSITORY_ANALYSIS,
        QueryCategory.FILE_SEARCH,
        QueryCategory.CODE_SEARCH,
    ]

    _LEAF_PATTERNS: Dict[QueryCategory, List[Pattern[str]]] = {
        QueryCategory.FILE_SEARCH: [
            re.compile(r"\bfind\b.*\bfile\b", re.IGNORECASE),
            re.compile(r"\bwhere\s+is\b", re.IGNORECASE),
            re.compile(r"\blocate\b", re.IGNORECASE),
            re.compile(r"\bwhich file\b", re.IGNORECASE),
            re.compile(r"\bfilename\b", re.IGNORECASE),
            re.compile(_FILENAME_PATTERN.pattern),
        ],
        QueryCategory.CODE_SEARCH: [
            re.compile(r"\bhow does\b", re.IGNORECASE),
            re.compile(r"\bwhat does\b", re.IGNORECASE),
            re.compile(r"\bimplementation of\b", re.IGNORECASE),
            re.compile(r"\bshow me the code\b", re.IGNORECASE),
            re.compile(r"\bhow is\b.*\bimplemented\b", re.IGNORECASE),
        ],
        QueryCategory.REPOSITORY_ANALYSIS: [
            re.compile(r"\bstructure\b", re.IGNORECASE),
            re.compile(r"\blayout\b", re.IGNORECASE),
            re.compile(r"\boverview\b", re.IGNORECASE),
            re.compile(r"\borgani[sz]ed\b", re.IGNORECASE),
            re.compile(r"\bwhat'?s in (this|the) repo\b", re.IGNORECASE),
            re.compile(r"\bproject layout\b", re.IGNORECASE),
        ],
        QueryCategory.ARCHITECTURE_QUESTION: [
            re.compile(r"\barchitecture\b", re.IGNORECASE),
            re.compile(r"\bdesign\b", re.IGNORECASE),
            re.compile(r"\bcomponents?\b", re.IGNORECASE),
            re.compile(r"\bsystem design\b", re.IGNORECASE),
            re.compile(r"\bhigh.level\b", re.IGNORECASE),
        ],
        QueryCategory.FLOW_QUESTION: [
            re.compile(r"\bflow\b", re.IGNORECASE),
            re.compile(r"\bwalk me through\b", re.IGNORECASE),
            re.compile(r"\bwhat happens when\b", re.IGNORECASE),
            re.compile(r"\bstep by step\b", re.IGNORECASE),
            re.compile(r"\btrace\b", re.IGNORECASE),
            re.compile(r"\bsequence of\b", re.IGNORECASE),
            re.compile(r"\blifecycle\b", re.IGNORECASE),
        ],
        QueryCategory.RELATIONSHIP_ANALYSIS: [
            re.compile(r"\bcall(?:er|ee| graph|s)?\b", re.IGNORECASE),
            re.compile(r"\bcallees?\b", re.IGNORECASE),
            re.compile(r"\bdependenc(?:y|ies)\b", re.IGNORECASE),
            re.compile(r"\bwho invokes?\b", re.IGNORECASE),
            re.compile(r"\bwhat calls?\b", re.IGNORECASE),
            re.compile(r"\brelationship(?:s)?\b", re.IGNORECASE),
        ],
        QueryCategory.SYMBOL_SEARCH: [
            re.compile(r"\bexact symbol\b", re.IGNORECASE),
            re.compile(r"\bexact\b.*\bdefinition\b", re.IGNORECASE),
            re.compile(r"\bfunction definition\b", re.IGNORECASE),
            re.compile(r"\bclass definition\b", re.IGNORECASE),
            re.compile(r"\bwhere is .*\b(?:defined|implemented)\b", re.IGNORECASE),
        ],
    }

    def classify(self, query: str) -> QueryCategory:
        if self._looks_multi_step(query):
            return QueryCategory.MULTI_STEP

        scores = self.explain(query)
        max_score = max(scores.values())
        if max_score == 0:
            return QueryCategory.CODE_SEARCH

        for category in self._PRIORITY_ORDER:
            if scores[category] == max_score:
                return category
        return QueryCategory.CODE_SEARCH  # unreachable in practice

    def explain(self, query: str) -> Dict[QueryCategory, int]:
        """Return the number of matched patterns per leaf category -
        useful for logging and for building a human-readable `reasoning`
        string in the plan."""
        return {
            category: sum(1 for pattern in patterns if pattern.search(query))
            for category, patterns in self._LEAF_PATTERNS.items()
        }

    @staticmethod
    def _looks_multi_step(query: str) -> bool:
        return len(split_into_subquestions(query)) >= 2


# --------------------------------------------------------------------------
# Planning strategies (one per leaf category)
# --------------------------------------------------------------------------


class PlanningStrategy(ABC):
    """Builds an ExecutionPlan for one QueryCategory.

    Each concrete strategy is single-purpose and independent of the
    others - PlannerAgent never needs to know how a given category
    builds its plan, only that it can call `build_plan(query)`.
    """

    category: QueryCategory

    @abstractmethod
    def build_plan(self, query: str) -> ExecutionPlan: ...


class CodeSearchStrategy(PlanningStrategy):
    """"How does X work" / "what does X do" -style questions: one
    semantic search pass is usually sufficient."""

    category = QueryCategory.CODE_SEARCH

    def __init__(self, default_k: int = 5) -> None:
        self._default_k = default_k

    def build_plan(self, query: str) -> ExecutionPlan:
        steps = [
            PlanStep(
                tool_name=SearchCodeTool.name,
                arguments={"query": query, "k": self._default_k},
                purpose="Semantic search for code relevant to the question.",
            )
        ]
        if _SOURCE_CAPABILITY_PATTERN.search(query):
            steps.append(
                PlanStep(
                    tool_name=SearchCodeTool.name,
                    arguments={
                        "query": (
                            "upload and connection endpoints, supported data sources, "
                            "file extension allowlists, database adapters, CSV, SQLite, "
                            "MySQL, SQL files, and raw script files"
                        ),
                        "k": self._default_k,
                    },
                    purpose=(
                        "Follow up on source-capability questions by retrieving adapter, "
                        "endpoint, and file-extension definitions."
                    ),
                )
            )
        return ExecutionPlan(
            query=query,
            category=self.category,
            steps=steps,
            multi_pass=len(steps) > 1,
            reasoning=(
                "Classified as code_search: the question asks how something "
                "works or is implemented. A single semantic search over the "
                "indexed code should surface the relevant chunks."
                + (
                    " A second targeted search retrieves source adapters and "
                    "upload/file-extension definitions because the question "
                    "asks about supported data sources."
                    if len(steps) > 1
                    else ""
                )
            ),
        )


class FileSearchStrategy(PlanningStrategy):
    """"Where is X" / "find file Y" -style questions: locate the file,
    and if a specific filename was detected, read it immediately since
    that's almost always the actual intent behind naming a file."""

    category = QueryCategory.FILE_SEARCH

    def build_plan(self, query: str) -> ExecutionPlan:
        filename = extract_probable_filename(query)
        steps = [
            PlanStep(
                tool_name=FindFileTool.name,
                arguments={
                    "filename": filename if filename else query,
                    "exact_match": bool(filename),
                },
                purpose="Locate the file(s) matching the requested name.",
            )
        ]
        multi_pass = False
        if filename:
            steps.append(
                PlanStep(
                    tool_name=ReadFileTool.name,
                    arguments={"path": filename},
                    purpose=(
                        f"Read '{filename}' now that it has been located - "
                        "naming a specific file usually means the user wants "
                        "to see its contents, not just confirm it exists."
                    ),
                    requires_previous_result=True,
                )
            )
            multi_pass = True

        detected = f" (detected filename: '{filename}')" if filename else ""
        return ExecutionPlan(
            query=query,
            category=self.category,
            steps=steps,
            multi_pass=multi_pass,
            reasoning=(
                f"Classified as file_search: the question is asking to "
                f"locate a file{detected}. find_file resolves the path; "
                "read_file follows up only when a filename was confidently detected."
            ),
        )


class RepositoryAnalysisStrategy(PlanningStrategy):
    """"What's the project layout" -style questions: the repo structure
    tool directly answers this, no code-level retrieval needed."""

    category = QueryCategory.REPOSITORY_ANALYSIS

    def build_plan(self, query: str) -> ExecutionPlan:
        steps = [
            PlanStep(
                tool_name=GetRepositoryStructureTool.name,
                arguments={},
                purpose="Retrieve the overall directory/file layout of the repository.",
            )
        ]
        return ExecutionPlan(
            query=query,
            category=self.category,
            steps=steps,
            multi_pass=False,
            reasoning=(
                "Classified as repository_analysis: the question is about "
                "overall project layout, which get_repository_structure "
                "answers directly."
            ),
        )


class ArchitectureQuestionStrategy(PlanningStrategy):
    """"What's the architecture / how is this designed" -style questions:
    need both the structural map (what modules exist) and semantic detail
    (how they relate) - one pass alone would answer only half of it."""

    category = QueryCategory.ARCHITECTURE_QUESTION

    def __init__(self, default_k: int = 5) -> None:
        self._default_k = default_k

    def build_plan(self, query: str) -> ExecutionPlan:
        steps = [
            PlanStep(
                tool_name=GetRepositoryStructureTool.name,
                arguments={},
                purpose="Establish the module/directory layout as structural context.",
            ),
            PlanStep(
                tool_name=SearchCodeTool.name,
                arguments={"query": query, "k": self._default_k},
                purpose="Retrieve code describing the components/relationships the question asks about.",
            ),
        ]
        return ExecutionPlan(
            query=query,
            category=self.category,
            steps=steps,
            multi_pass=True,
            reasoning=(
                "Classified as architecture_question: architecture answers "
                "need both the structural map and semantic code detail, so "
                "two passes (structure, then search) run rather than one."
            ),
        )


class FlowQuestionStrategy(PlanningStrategy):
    """"Walk me through what happens when X" -style questions: usually
    span multiple hops across files, so an initial search almost always
    only finds the entry point - a follow-up pass, refined using what
    the first pass found, is planned in advance."""

    category = QueryCategory.FLOW_QUESTION

    def __init__(self, default_k: int = 5) -> None:
        self._default_k = default_k

    def build_plan(self, query: str) -> ExecutionPlan:
        steps = [
            PlanStep(
                tool_name=SearchCodeTool.name,
                arguments={"query": query, "k": self._default_k},
                purpose="Initial search to find the entry point of the described flow.",
            ),
            PlanStep(
                tool_name=SearchCodeTool.name,
                arguments={"query": query, "k": self._default_k},
                purpose=(
                    "Follow-up search to continue tracing the flow past the "
                    "entry point - refined by the orchestrator using the "
                    "first pass's top result before running."
                ),
                requires_previous_result=True,
            ),
        ]
        return ExecutionPlan(
            query=query,
            category=self.category,
            steps=steps,
            multi_pass=True,
            reasoning=(
                "Classified as flow_question: tracing 'what happens when...' "
                "typically crosses multiple functions/files, so a single "
                "retrieval pass tends to stop at the first hop. A second, "
                "context-refined pass is planned up front."
            ),
        )


def extract_probable_symbol(query: str) -> Optional[str]:
    """Extract a qualified or backtick-quoted symbol when one is visible."""
    quoted = re.search(r"`([^`]+)`", query)
    if quoted:
        return quoted.group(1).strip()
    candidates = re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b|\b[A-Za-z_]\w*\b", query)
    ignored = {"where", "what", "which", "function", "class", "method", "symbol", "defined", "implemented"}
    meaningful = [candidate for candidate in candidates if candidate.lower() not in ignored]
    return meaningful[-1] if meaningful else None


class SymbolSearchStrategy(PlanningStrategy):
    category = QueryCategory.SYMBOL_SEARCH

    def build_plan(self, query: str) -> ExecutionPlan:
        symbol = extract_probable_symbol(query) or query
        return ExecutionPlan(
            query=query,
            category=self.category,
            steps=[PlanStep(
                tool_name=SearchSymbolTool.name,
                arguments={"symbol": symbol, "exact": bool(extract_probable_symbol(query))},
                purpose="Locate exact symbol definitions and their source ranges.",
            )],
            multi_pass=False,
            reasoning="Classified as symbol_search: use the AST index for exact definitions before semantic retrieval.",
        )


class RelationshipAnalysisStrategy(PlanningStrategy):
    category = QueryCategory.RELATIONSHIP_ANALYSIS

    def build_plan(self, query: str) -> ExecutionPlan:
        symbol = extract_probable_symbol(query) or query
        return ExecutionPlan(
            query=query,
            category=self.category,
            steps=[
                PlanStep(
                    tool_name=CodeRelationshipTool.name,
                    arguments={"symbol": symbol, "mode": "relationships", "direction": "both"},
                    purpose="Identify direct callers, callees, imports, and class-method relationships.",
                ),
                PlanStep(
                    tool_name=CodeRelationshipTool.name,
                    arguments={"symbol": symbol, "mode": "traverse", "direction": "both", "max_depth": 3},
                    purpose="Traverse the bounded call graph to connect the symbol across files.",
                ),
            ],
            multi_pass=True,
            reasoning="Classified as relationship_analysis: inspect direct edges, then traverse the bounded call graph.",
        )


# --------------------------------------------------------------------------
# Planner agent
# --------------------------------------------------------------------------


class PlannerAgent:
    """Classifies a question and produces its ExecutionPlan.

    Composition, not inheritance: PlannerAgent owns a QueryClassifier and
    a category -> PlanningStrategy mapping, and defers all classification
    and per-category planning logic to them (strategy pattern). This
    class's only job is: classify, then either decompose (multi-step) or
    delegate to the matching strategy.
    """

    def __init__(
        self,
        classifier: Optional[QueryClassifier] = None,
        strategies: Optional[Dict[QueryCategory, PlanningStrategy]] = None,
    ) -> None:
        """
        Args:
            classifier: Anything implementing QueryClassifier. Defaults
                to HeuristicQueryClassifier - swap in an LLM-backed
                classifier here without changing anything else.
            strategies: Maps each of the five leaf QueryCategory values
                to a PlanningStrategy. Defaults to the five strategies
                defined in this module. MULTI_STEP has no strategy entry
                by design - it's handled by decomposition in `plan()`,
                not by delegation, since it needs to call back into this
                same classify-then-delegate logic for each sub-question.
        """
        self._classifier = classifier or HeuristicQueryClassifier()
        self._strategies: Dict[QueryCategory, PlanningStrategy] = strategies or {
            QueryCategory.CODE_SEARCH: CodeSearchStrategy(),
            QueryCategory.FILE_SEARCH: FileSearchStrategy(),
            QueryCategory.REPOSITORY_ANALYSIS: RepositoryAnalysisStrategy(),
            QueryCategory.ARCHITECTURE_QUESTION: ArchitectureQuestionStrategy(),
            QueryCategory.FLOW_QUESTION: FlowQuestionStrategy(),
            QueryCategory.SYMBOL_SEARCH: SymbolSearchStrategy(),
            QueryCategory.RELATIONSHIP_ANALYSIS: RelationshipAnalysisStrategy(),
        }
        missing = set(QueryCategory) - {QueryCategory.MULTI_STEP} - self._strategies.keys()
        if missing:
            raise ValueError(f"Missing planning strategies for categories: {missing}")

    def plan(self, query: str) -> ExecutionPlan:
        """Classify `query` and build its ExecutionPlan.

        Raises:
            ValueError: If `query` is empty or whitespace-only.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        category = self._classifier.classify(query)
        if category == QueryCategory.MULTI_STEP:
            return self._plan_multi_step(query)
        return self._plan_single(query, category)

    def _plan_single(self, query: str, category: QueryCategory) -> ExecutionPlan:
        strategy = self._strategies[category]
        return strategy.build_plan(query)

    def _plan_multi_step(self, query: str) -> ExecutionPlan:
        subquestions = split_into_subquestions(query)

        # Guard: the split boundary matched but produced fewer than two
        # usable pieces (e.g. a trailing "?" with nothing after it) -
        # fall back to treating this as a single question rather than
        # returning a degenerate multi-step plan.
        if len(subquestions) < 2:
            category = self._classifier.classify(query)
            return self._plan_single(query, category)

        sub_plans: List[ExecutionPlan] = []
        for subquestion in subquestions:
            sub_category = self._classifier.classify(subquestion)
            if sub_category == QueryCategory.MULTI_STEP:
                # A sub-question that itself looks compound - classify it
                # as code_search rather than recursing indefinitely; this
                # keeps plans flat and predictable.
                sub_category = QueryCategory.CODE_SEARCH
            sub_plans.append(self._plan_single(subquestion, sub_category))

        merged_steps = [step for sub_plan in sub_plans for step in sub_plan.steps]
        categories_used = ", ".join(sp.category.value for sp in sub_plans)

        return ExecutionPlan(
            query=query,
            category=QueryCategory.MULTI_STEP,
            steps=merged_steps,
            multi_pass=True,
            reasoning=(
                f"Classified as multi_step: the question decomposes into "
                f"{len(sub_plans)} sub-questions ({categories_used}), each "
                "independently classified and planned; their steps are run in order."
            ),
            sub_plans=sub_plans,
        )
