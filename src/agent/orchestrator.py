"""
orchestrator.py - the runtime agent loop.

PlannerAgent produces a fixed ExecutionPlan before anything runs.
AgentOrchestrator adds the missing feedback loop:

    User Query -> Planner -> Tool Execution -> Observe Result
                -> Evaluate Sufficiency
                -> If insufficient: select/refine next action -> Execute -> Observe
                -> Repeat (bounded by max_iterations)
                -> Final Answer (context + citations)

What this module deliberately does NOT do:
    - It does not replace or modify PlannerAgent - it calls
      `planner.plan(query)` exactly once, at the start, and never
      consults the planner again during a run.
    - It does not modify ToolRegistry or any tool - all execution goes
      through `registry.run(tool_name, **arguments)`, unchanged.
    - It does not modify the citation system - it builds `Citation`
      objects using the same `Citation.from_metadata` /
      `format_context_label` functions rag_pipeline.py uses.
    - It is a single class with no sub-agents and no persistent memory:
      each call to `run()` is a fresh, independent execution with its
      own local state (cache, context, trace) that is discarded when
      `run()` returns.

Design note on WHY the planner's own steps always execute before any
dynamic follow-up is generated: the planner already decides multi-step
sequences for some categories (e.g. architecture questions get
[get_repository_structure, search_code]). Interrupting that sequence
mid-way to chase a coverage gap would second-guess a plan the planner
built for a reason. So this loop is two phases in sequence, not one
undifferentiated loop: (1) run every planner-supplied step, then (2)
only once that queue is empty, keep evaluating sufficiency and adding
one targeted step at a time for as long as something is missing and
iterations remain.
"""

from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.agent.planner import ExecutionPlan, PlannerAgent, PlanStep, extract_probable_filename
from src.citations import Citation, dedupe_citations, format_context_label
from src.tools import (
    FindFileTool,
    GetRepositoryStructureTool,
    ReadFileTool,
    CodeRelationshipTool,
    SearchCodeTool,
    SearchSymbolTool,
    ToolExecutionError,
    ToolInputError,
    ToolRegistry,
    ToolResult,
)

logger = logging.getLogger(__name__)

_NO_CONTEXT_MARKER = "(no relevant context retrieved)"

# Directive/question words and generic project nouns filtered out when
# extracting the significant terms of an information need - what's left
# is what the orchestrator actually checks for in retrieved content.
# This is a heuristic list, not a real NLP stopword corpus - consistent
# with HeuristicQueryClassifier in planner.py.
_STOPWORDS: frozenset = frozenset(
    {
        "explain", "identify", "describe", "show", "tell", "find", "locate",
        "determine", "walk", "trace",
        "where", "how", "what", "which", "who", "why", "does", "do", "did",
        "is", "are", "was", "were", "be", "been", "being",
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
        "this", "that", "it", "its", "with", "from", "by", "as",
        "works", "work", "working", "repository", "repo", "project",
        "codebase", "system", "me", "please", "you",
    }
)

# Two words are considered the same "root" if they share at least this
# many leading characters - cheap stand-in for stemming, catching pairs
# like "authentication"/"auth" or "generated"/"generate" without pulling
# in a real NLP dependency for a heuristic coverage check.
_ROOT_MATCH_LENGTH = 4


# --------------------------------------------------------------------------
# Result data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IterationTrace:
    """Record of one loop iteration - what was run, what it returned,
    and what the sufficiency check decided afterward. Exists for
    testability and transparency (e.g. showing an execution trace), not
    used internally for anything beyond being returned to the caller."""

    iteration: int
    action: str  # "planned_step" | "targeted_followup"
    step: PlanStep
    arguments: Dict[str, Any]
    result: ToolResult
    covered_needs: List[str]
    uncovered_needs: List[str]


@dataclass(frozen=True)
class OrchestratorResult:
    """Everything `AgentOrchestrator.run()` produces for one query.

    Attributes:
        stopped_reason: One of:
            "sufficient" - every detected information need was covered.
            "max_iterations" - the iteration cap was hit first.
            "exhausted_no_further_action" - the queue emptied, something
                is still uncovered, but no tool could be selected for it
                (e.g. search_code isn't registered).
    """

    query: str
    context_text: str
    citations: List[Citation]
    plan: ExecutionPlan
    trace: List[IterationTrace]
    iterations_used: int
    stopped_reason: str


# --------------------------------------------------------------------------
# Information-need extraction (orchestrator-local, independent of
# PlannerAgent's own, differently-tuned split logic)
# --------------------------------------------------------------------------


def _extract_information_needs(query: str) -> List[str]:
    """Split the query into distinct informational asks for sufficiency
    checking. Deliberately broader than
    `planner.split_into_subquestions` (which must avoid over-splitting
    to keep classification accurate) - this splits on plain "and" and
    on ".", not just "?"/";"/"then"/"and <wh-word>", because
    over-splitting here only means checking a couple of harmless extra
    clauses, not misclassifying the query.
    """
    clauses = re.split(r"\band\b|[;.?]|\bthen\b", query, flags=re.IGNORECASE)
    return [c.strip(" ,") for c in clauses if c.strip(" ,")]


def _key_terms(text: str) -> List[str]:
    """Significant, lowercased words from `text`, with directive words
    and generic project nouns filtered out via `_STOPWORDS`."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]*", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _shares_root(term: str, word: str) -> bool:
    if term == word:
        return True
    if len(term) < _ROOT_MATCH_LENGTH or len(word) < _ROOT_MATCH_LENGTH:
        return False
    return term[:_ROOT_MATCH_LENGTH] == word[:_ROOT_MATCH_LENGTH]


def _strip_label_lines(text: str) -> str:
    """Remove '[path: ... | ...]'-style label lines before tokenizing
    content for the sufficiency haystack.

    Without this, the label header itself - which legitimately embeds
    the FULL path, e.g. "[path: src/auth/jwt_service.py | ...]" - would
    leak the parent directory name ("auth") into the haystack the same
    way raw citation.path tokenization did (see `_collect_haystack_words`
    using `file_basename` instead of `.path` for exactly this reason).
    Only the actual retrieved code text should count as "evidence".
    """
    return "\n".join(line for line in text.splitlines() if not (line.strip().startswith("[") and line.strip().endswith("]")))


def _collect_haystack_words(context_sections: List[str], citations: List[Citation]) -> Set[str]:
    """All the words the orchestrator is willing to treat as "evidence
    retrieved so far" - retrieved CODE content text (label headers
    stripped out), plus every citation's file BASENAME (not the full
    path) and function/class name.

    Two related leaks are deliberately guarded against here, both
    stemming from the same root cause (a full file path embeds its
    parent directory name, which can accidentally satisfy an unrelated
    coverage check): citation.path is reduced to its basename below, and
    `_strip_label_lines` removes the "[path: ...]" header line each
    context block starts with before its content is tokenized. A file
    merely sitting in an "auth/" folder should not count as evidence for
    "authentication" - only its own filename, class/function names, or
    actual code content should.
    """
    cleaned_content = " ".join(_strip_label_lines(section) for section in context_sections)
    words: Set[str] = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", cleaned_content.lower()))
    for citation in citations:
        basename = citation.file_basename
        words.update(re.findall(r"[A-Za-z][A-Za-z0-9_]*", basename.lower()))
        if citation.function_name:
            words.update(re.findall(r"[A-Za-z][A-Za-z0-9_]*", citation.function_name.lower()))
        if citation.class_name:
            words.update(re.findall(r"[A-Za-z][A-Za-z0-9_]*", citation.class_name.lower()))
    return words


def _is_need_covered(need_terms: List[str], haystack_words: Set[str]) -> bool:
    if not need_terms:
        # Nothing significant survived stopword filtering (e.g. the
        # clause was just "in this repository") - there's nothing
        # meaningful to check, so treat it as trivially satisfied
        # rather than blocking the loop on an empty requirement.
        return True
    matched_terms = sum(
        1 for term in need_terms
        if any(_shares_root(term, word) for word in haystack_words)
    )
    return matched_terms == len(need_terms)


# --------------------------------------------------------------------------
# The orchestrator
# --------------------------------------------------------------------------


class AgentOrchestrator:
    """Executes a PlannerAgent's plan, then keeps retrieving - bounded
    by `max_iterations` - for as long as the query's information needs
    aren't yet covered by what's been retrieved.

    Single responsibility: run the loop. It does not classify queries
    (PlannerAgent's job), does not implement tools (ToolRegistry's job),
    and does not format citations (citations.py's job) - it calls all
    three unchanged.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        planner: Optional[PlannerAgent] = None,
        max_iterations: int = 4,
        default_k: int = 5,
    ) -> None:
        """
        Args:
            tool_registry: The (already-built) registry to execute
                against - unmodified `src.tools.ToolRegistry`.
            planner: Injectable PlannerAgent, defaulting to a fresh one.
            max_iterations: Hard cap on total tool executions in one
                `run()` call - the loop always terminates by this point
                regardless of whether sufficiency was reached.
            default_k: Fallback `k` for `run()` calls that don't pass one.
        """
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        self._registry = tool_registry
        self._planner = planner or PlannerAgent()
        self._max_iterations = max_iterations
        self._default_k = default_k

    def run(self, query: str, k: Optional[int] = None) -> OrchestratorResult:
        """Run the full loop for `query` and return the aggregated result.

        Raises:
            ValueError: If `query` is empty or whitespace-only.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        effective_k = k if k is not None else self._default_k

        plan = self._planner.plan(query)
        _apply_k_to_plan(plan, effective_k)

        needs = _extract_information_needs(query)
        need_terms_by_need: Dict[str, List[str]] = {need: _key_terms(need) for need in needs}

        pending: List[Tuple[PlanStep, str]] = [(step, "planned_step") for step in plan.steps]
        call_cache: Dict[_CacheKey, ToolResult] = {}
        attempted_keys: Set[_CacheKey] = set()
        context_sections: List[str] = []
        citations: List[Citation] = []
        trace: List[IterationTrace] = []
        previous_result: Optional[ToolResult] = None
        iteration = 0
        stopped_reason = "sufficient"

        while pending:
            if iteration >= self._max_iterations:
                stopped_reason = "max_iterations"
                break

            iteration += 1
            step, action = pending.pop(0)

            arguments = dict(step.arguments)
            if step.requires_previous_result and previous_result is not None:
                arguments = _refine_arguments(step, arguments, previous_result, query)

            cache_key = _make_cache_key(step.tool_name, arguments)
            attempted_keys.add(cache_key)
            if cache_key in call_cache:
                logger.debug("Reusing cached result for %s%r (duplicate call avoided)", step.tool_name, arguments)
                result = call_cache[cache_key]
            else:
                try:
                    result = self._registry.run(step.tool_name, **arguments)
                except (ToolInputError, ToolExecutionError) as exc:
                    logger.warning("Step '%s' failed: %s", step.tool_name, exc)
                    result = ToolResult.fail(str(exc))
                else:
                    call_cache[cache_key] = result

            previous_result = result if result.success else None

            section, step_citations = _format_step_result(step, result)
            if section:
                context_sections.append(section)
            citations.extend(step_citations)

            covered, uncovered = _evaluate_sufficiency(need_terms_by_need, context_sections, citations)
            trace.append(
                IterationTrace(
                    iteration=iteration,
                    action=action,
                    step=step,
                    arguments=arguments,
                    result=result,
                    covered_needs=covered,
                    uncovered_needs=uncovered,
                )
            )

            if not pending and uncovered:
                if iteration >= self._max_iterations:
                    stopped_reason = "max_iterations"
                else:
                    next_step = self._select_novel_action(uncovered, query, effective_k, attempted_keys)
                    if next_step is not None:
                        pending.append((next_step, "targeted_followup"))
                    # else: every uncovered need's action either has no
                    # available tool or was already attempted with no
                    # new coverage - the post-loop check below
                    # classifies this as "exhausted_no_further_action".

        final_covered, final_uncovered = _evaluate_sufficiency(need_terms_by_need, context_sections, citations)
        if stopped_reason != "max_iterations":
            stopped_reason = "sufficient" if not final_uncovered else "exhausted_no_further_action"

        context_text = "\n\n---\n\n".join(context_sections) if context_sections else _NO_CONTEXT_MARKER

        return OrchestratorResult(
            query=query,
            context_text=context_text,
            citations=dedupe_citations(citations),
            plan=plan,
            trace=trace,
            iterations_used=iteration,
            stopped_reason=stopped_reason,
        )

    def _select_novel_action(
        self,
        uncovered_needs: List[str],
        original_query: str,
        k: int,
        attempted_keys: Set["_CacheKey"],
    ) -> Optional[PlanStep]:
        """Find the first uncovered need whose proposed action hasn't
        already been attempted this run, trying them in order.

        This exists because a single stuck need (one `_select_next_action`
        keeps proposing the same already-tried query for) shouldn't cause
        the loop to give up on OTHER, still-untried needs - e.g. for
        "explain apples and oranges and bananas", if "apples" turns out
        to be unretrievable, the loop should still go on to try
        "oranges" rather than stopping the moment apples' action repeats.
        """
        for need in uncovered_needs:
            candidate = self._select_next_action(need, original_query, k)
            if candidate is None:
                continue
            candidate_key = _make_cache_key(candidate.tool_name, candidate.arguments)
            if candidate_key in attempted_keys:
                logger.debug(
                    "Skipping already-attempted action %s%r for need %r.",
                    candidate.tool_name, candidate.arguments, need,
                )
                continue
            return candidate
        return None

    def _select_next_action(self, uncovered_need: str, original_query: str, k: int) -> Optional[PlanStep]:
        """Decide the next tool call aimed at closing one specific
        coverage gap. This is the "select/refine the next action" step -
        it looks at what's still missing, not at fixed, pre-declared
        arguments.

        Prefers find_file when the uncovered need names something
        filename-shaped (e.g. "the config.yaml settings"); otherwise
        falls back to a refined search_code query built from the
        need's significant terms (e.g. "identify where JWT tokens are
        generated" -> "jwt tokens generated"). Returns None if neither
        tool is registered, in which case the loop has no further
        action to take for this need.
        """
        filename = extract_probable_filename(uncovered_need)
        if filename and self._registry.has(FindFileTool.name):
            return PlanStep(
                tool_name=FindFileTool.name,
                arguments={"filename": filename, "exact_match": True},
                purpose=f"Targeted follow-up: locate '{filename}' to cover missing need '{uncovered_need}'.",
            )

        if self._registry.has(SearchCodeTool.name):
            terms = _key_terms(uncovered_need)
            refined_query = " ".join(terms) if terms else uncovered_need
            return PlanStep(
                tool_name=SearchCodeTool.name,
                arguments={"query": refined_query, "k": k},
                purpose=f"Targeted follow-up search for missing information need: '{uncovered_need}'.",
            )

        return None


def _evaluate_sufficiency(
    need_terms_by_need: Dict[str, List[str]],
    context_sections: List[str],
    citations: List[Citation],
) -> Tuple[List[str], List[str]]:
    haystack = _collect_haystack_words(context_sections, citations)
    covered: List[str] = []
    uncovered: List[str] = []
    for need, terms in need_terms_by_need.items():
        (covered if _is_need_covered(terms, haystack) else uncovered).append(need)
    return covered, uncovered


# --------------------------------------------------------------------------
# Step execution helpers
#
# Executing a plan against the tool registry is the orchestration
# layer's job.
# --------------------------------------------------------------------------

_CacheKey = Tuple[str, Tuple[Tuple[str, Any], ...]]


def _apply_k_to_plan(plan: ExecutionPlan, k: int) -> None:
    """Overrides every planned search_code step's k with the caller-
    supplied value. `PlanStep` is frozen, but `arguments` is a plain
    dict - mutating it in place is safe since each plan is single-use."""
    for step in plan.steps:
        if step.tool_name == SearchCodeTool.name and "k" in step.arguments:
            step.arguments["k"] = k


def _make_cache_key(tool_name: str, arguments: Dict[str, Any]) -> _CacheKey:
    return tool_name, tuple(sorted(arguments.items()))


def _refine_arguments(
    step: PlanStep,
    arguments: Dict[str, Any],
    previous_result: ToolResult,
    original_query: str,
) -> Dict[str, Any]:
    """Fill in a dependent step's arguments using the previous step's
    output - the orchestration-time counterpart to the planner flagging
    `requires_previous_result=True` without prescribing exact values."""
    refined = dict(arguments)

    if step.tool_name == ReadFileTool.name:
        if previous_result.success and previous_result.data:
            refined["path"] = previous_result.data[0]

    elif step.tool_name == SearchCodeTool.name:
        if previous_result.success and previous_result.data:
            top_hit = previous_result.data[0]
            snippet = str(top_hit.get("content", ""))[:160].replace("\n", " ").strip()
            if snippet:
                refined["query"] = f"{original_query} (continuing after: {snippet})"

    return refined


def _format_step_result(step: PlanStep, result: ToolResult) -> Tuple[Optional[str], List[Citation]]:
    if not result.success:
        logger.info("Step '%s' returned no result: %s", step.tool_name, result.error)
        return None, []

    if step.tool_name == SearchCodeTool.name:
        return _format_search_results(result.data or [])

    if step.tool_name == SearchSymbolTool.name:
        return _format_symbol_results(result.data or [])

    if step.tool_name == CodeRelationshipTool.name:
        return _format_relationship_results(result.data)

    if step.tool_name == ReadFileTool.name:
        path = result.metadata.get("path", "unknown file")
        start_line, end_line = result.metadata.get("returned_lines", (None, None))
        label = f"[path: {path} | lines {start_line}-{end_line}]" if start_line and end_line else f"[path: {path}]"
        citation = Citation(path=path, start_line=start_line, end_line=end_line)
        return f"{label}\n{result.data}", [citation]

    if step.tool_name == FindFileTool.name:
        matches: List[str] = result.data or []
        if not matches:
            return None, []
        filename = result.metadata.get("filename", "?")
        return f"Files matching '{filename}': {', '.join(matches)}", []

    if step.tool_name == GetRepositoryStructureTool.name:
        return f"Repository structure:\n{_tree_to_text(result.data)}", []

    return str(result.data), []


def _format_search_results(chunks: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Citation]]:
    if not chunks:
        return None, []

    lines: List[str] = []
    citations: List[Citation] = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        content = chunk.get("content", "")
        lines.append(f"{format_context_label(metadata)}\n{content}")

        citation = Citation.from_metadata(metadata)
        if citation:
            citations.append(citation)

    return "\n\n".join(lines), citations


def _format_symbol_results(symbols: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Citation]]:
    if not symbols:
        return None, []
    lines: List[str] = []
    citations: List[Citation] = []
    for symbol in symbols:
        path = symbol.get("path", "unknown")
        lines.append(
            f"Symbol {symbol.get('qualified_name', symbol.get('name', '?'))} "
            f"({symbol.get('kind', 'symbol')}) in {path}, lines "
            f"{symbol.get('start_line', '?')}-{symbol.get('end_line', '?')}"
        )
        citation = Citation.from_metadata({"source": path, **symbol})
        if citation:
            citations.append(citation)
    return "\n".join(lines), citations


def _format_relationship_results(data: Any) -> Tuple[Optional[str], List[Citation]]:
    if not data:
        return None, []
    payload = data if isinstance(data, list) else data
    citations: List[Citation] = []
    records = payload if isinstance(payload, list) else (
        payload.get("callers", []) + payload.get("callees", [])
    )
    for record in records:
        for key in ("caller_metadata", "callee_metadata"):
            metadata = record.get(key) if isinstance(record, dict) else None
            if metadata:
                citation = Citation.from_metadata(metadata)
                if citation:
                    citations.append(citation)
        if isinstance(record, dict) and record.get("path"):
            citation = Citation(
                path=record["path"],
                start_line=record.get("line"),
                end_line=record.get("line"),
            )
            citations.append(citation)
    if isinstance(payload, dict):
        for symbol in payload.get("symbols", []):
            citation = Citation.from_metadata({"source": symbol.get("path"), **symbol})
            if citation:
                citations.append(citation)
    return f"Code relationships:\n{json.dumps(payload, indent=2, default=str)}", citations


def _tree_to_text(node: Optional[Dict[str, Any]], prefix: str = "") -> str:
    if not node:
        return "(empty)"

    label = f"{node.get('name', '?')}/" if node.get("type") == "directory" else node.get("name", "?")
    lines = [f"{prefix}{label}"]

    for child in node.get("children", []):
        if child.get("type") == "directory":
            lines.append(_tree_to_text(child, prefix + "  "))
        else:
            lines.append(f"{prefix}  {child.get('name', '?')}")

    return "\n".join(lines)
