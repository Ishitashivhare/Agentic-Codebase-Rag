"""AST-backed definitions, imports, and call relationships for a repository."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

@dataclass(frozen=True)
class SymbolRecord:
    name: str
    qualified_name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    class_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "kind": self.kind,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "class_name": self.class_name,
        }

    def metadata(self) -> dict:
        return {
            "source": self.path,
            "function_name": self.name if self.kind in {"function", "method"} else None,
            "class_name": self.class_name or (self.name if self.kind == "class" else None),
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


@dataclass(frozen=True)
class RelationshipRecord:
    caller: str
    callee: str
    resolved_callee: Optional[str]
    path: str
    line: int
    caller_record: Optional[SymbolRecord] = None
    callee_record: Optional[SymbolRecord] = None

    def to_dict(self) -> dict:
        return {
            "caller": self.caller,
            "callee": self.callee,
            "resolved_callee": self.resolved_callee,
            "path": self.path,
            "line": self.line,
            "caller_metadata": self.caller_record.metadata() if self.caller_record else None,
            "callee_metadata": self.callee_record.metadata() if self.callee_record else None,
        }


class CodeRelationshipIndex:
    """Build and query a deterministic, Python-only relationship index."""

    def __init__(self, repo_root: str | Path):
        self.repo_root = Path(repo_root).resolve()
        self.symbols: List[SymbolRecord] = []
        self.relationships: List[RelationshipRecord] = []
        self.imports: List[dict] = []
        self._by_qualified: Dict[str, SymbolRecord] = {}
        self._by_name: Dict[str, List[SymbolRecord]] = {}
        self._build()

    def _build(self) -> None:
        from src.loader import load_repo

        documents = load_repo(str(self.repo_root))
        raw_calls: List[tuple] = []
        for document in documents:
            try:
                tree = ast.parse(document["content"], filename=document["path"])
            except SyntaxError:
                continue
            visitor = _ModuleVisitor(document["path"])
            visitor.visit(tree)
            self.symbols.extend(visitor.symbols)
            self.imports.extend(visitor.imports)
            raw_calls.extend(visitor.calls)

        for symbol in self.symbols:
            self._by_qualified.setdefault(symbol.qualified_name, symbol)
            self._by_name.setdefault(symbol.name, []).append(symbol)

        for caller, callee, path, line in raw_calls:
            resolved = self.resolve(callee, caller)
            self.relationships.append(
                RelationshipRecord(
                    caller=caller,
                    callee=callee,
                    resolved_callee=resolved.qualified_name if resolved else None,
                    path=path,
                    line=line,
                    caller_record=self._by_qualified.get(caller),
                    callee_record=resolved,
                )
            )

    def search_symbols(self, query: str, exact: bool = True) -> List[SymbolRecord]:
        normalized = query.strip()
        if not normalized:
            return []
        if exact:
            direct = self._by_qualified.get(normalized) or self._by_name.get(normalized, [])
            if isinstance(direct, list):
                return sorted(direct, key=lambda item: (item.path, item.start_line))
            return [direct]
        lowered = normalized.lower()
        return sorted(
            [
                symbol
                for symbol in self.symbols
                if lowered in symbol.name.lower() or lowered in symbol.qualified_name.lower()
            ],
            key=lambda item: (item.path, item.start_line),
        )

    def resolve(self, callee: str, caller: Optional[str] = None) -> Optional[SymbolRecord]:
        candidates: List[str] = [callee]
        if caller and "." in caller:
            candidates.insert(0, f"{caller.rsplit('.', 1)[0]}.{callee}")
        for candidate in candidates:
            if candidate in self._by_qualified:
                return self._by_qualified[candidate]
        name = callee.rsplit(".", 1)[-1]
        matches = self._by_name.get(name, [])
        return matches[0] if len(matches) == 1 else None

    def relationships_for(self, symbol: str) -> dict:
        records = self.search_symbols(symbol, exact=True)
        names = {record.qualified_name for record in records}
        if records:
            names.update(record.name for record in records)
        related = [
            relation
            for relation in self.relationships
            if relation.caller in names or relation.resolved_callee in names
        ]
        return {
            "symbols": [record.to_dict() for record in records],
            "imports": [item for item in self.imports if item["path"] in {record.path for record in records}],
            "callers": [relation.to_dict() for relation in related if relation.resolved_callee in names],
            "callees": [relation.to_dict() for relation in related if relation.caller in names],
        }

    def traverse(self, symbol: str, direction: str = "both", max_depth: int = 2) -> List[dict]:
        starts = self.search_symbols(symbol, exact=True)
        if not starts:
            return []
        frontier = [(record.qualified_name, 0) for record in starts]
        visited: Set[tuple] = set(frontier)
        paths: List[dict] = []
        while frontier:
            current, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for relation in self.relationships:
                next_symbol = None
                relation_kind = None
                if direction in {"both", "callees"} and relation.caller == current and relation.resolved_callee:
                    next_symbol, relation_kind = relation.resolved_callee, "callee"
                elif direction in {"both", "callers"} and relation.resolved_callee == current:
                    next_symbol, relation_kind = relation.caller, "caller"
                if not next_symbol:
                    continue
                path_key = (next_symbol, depth + 1)
                paths.append({"depth": depth + 1, "relation": relation_kind, **relation.to_dict()})
                if path_key not in visited:
                    visited.add(path_key)
                    frontier.append(path_key)
        return paths


class _ModuleVisitor(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path = path
        self.symbols: List[SymbolRecord] = []
        self.imports: List[dict] = []
        self.calls: List[tuple] = []
        self.scope: List[str] = []

    def _qualified(self, name: str) -> str:
        return ".".join(self.scope + [name]) if self.scope else name

    def _record(self, node: ast.AST, name: str, kind: str, class_name: Optional[str] = None) -> None:
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start) or start
        self.symbols.append(SymbolRecord(name, self._qualified(name), kind, self.path, start, end, class_name))

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend({"module": alias.name, "name": alias.asname, "path": self.path, "line": node.lineno} for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        self.imports.extend({"module": module, "name": alias.name, "path": self.path, "line": node.lineno} for alias in node.names)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node, node.name, "class")
        self.scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = "method" if self.scope else "function"
        class_name = self.scope[-1] if self.scope else None
        self._record(node, node.name, kind, class_name)
        self.scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._record(node, target.id, "variable")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _expression_name(node.func)
        if callee:
            caller = ".".join(self.scope) if self.scope else f"module:{self.path}"
            self.calls.append((caller, callee, self.path, node.lineno))
        self.generic_visit(node)


def _expression_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: List[str] = []
        current: Optional[ast.AST] = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
    return None
