import tempfile
import unittest
from pathlib import Path

from src.code_index import CodeRelationshipIndex
from src.tools import CodeRelationshipTool, SearchSymbolTool
from src.tools.citation_validator import validate_citations
from src.citations import Citation


class RelationshipFixtures:
    @staticmethod
    def write(root: Path, ambiguous: bool = False) -> None:
        (root / "entry.py").write_text(
            "from service import process\n\n"
            "def entry(adapter):\n"
            "    return process(adapter)\n",
            encoding="utf-8",
        )
        (root / "service.py").write_text(
            "from adapter import Adapter\n\n"
            "def process(adapter):\n"
            "    return adapter.execute_query()\n",
            encoding="utf-8",
        )
        (root / "adapter.py").write_text(
            "class Adapter:\n"
            "    def execute_query(self):\n"
            "        return 1\n\n"
            "def dynamic(obj):\n"
            "    return obj.run()\n",
            encoding="utf-8",
        )
        if ambiguous:
            (root / "other.py").write_text(
                "class OtherAdapter:\n"
                "    def execute_query(self):\n"
                "        return 2\n",
                encoding="utf-8",
            )
        (root / "broken.py").write_text("def broken(:\n", encoding="utf-8")


class CodeRelationshipTests(unittest.TestCase):
    def test_exact_symbol_search_and_missing_symbol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            RelationshipFixtures.write(root)
            index = CodeRelationshipIndex(root)
            result = SearchSymbolTool(index).run("execute_query")
            self.assertTrue(result.success)
            self.assertEqual(1, result.metadata["result_count"])
            self.assertEqual("adapter.py", result.data[0]["path"])
            self.assertEqual([], SearchSymbolTool(index).run("does_not_exist").data)

    def test_syntax_error_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            RelationshipFixtures.write(root)
            index = CodeRelationshipIndex(root)
            self.assertEqual([], SearchSymbolTool(index).run("broken").data)

    def test_cross_file_traversal_resolves_concrete_method(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            RelationshipFixtures.write(root)
            index = CodeRelationshipIndex(root)
            result = CodeRelationshipTool(index).run(
                "entry", mode="traverse", direction="callees", max_depth=3
            )
            self.assertTrue(result.success)
            resolved = {edge["resolved_callee"] for edge in result.data}
            self.assertIn("process", resolved)
            self.assertIn("Adapter.execute_query", resolved)

    def test_dynamic_dispatch_remains_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            RelationshipFixtures.write(root)
            index = CodeRelationshipIndex(root)
            result = CodeRelationshipTool(index).run("dynamic", mode="relationships")
            self.assertTrue(result.success)
            self.assertIsNone(result.data["callees"][0]["resolved_callee"])

    def test_ambiguous_method_is_not_claimed_as_concrete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            RelationshipFixtures.write(root, ambiguous=True)
            index = CodeRelationshipIndex(root)
            result = CodeRelationshipTool(index).run("process", mode="relationships")
            self.assertTrue(result.success)
            self.assertIsNone(result.data["callees"][0]["resolved_callee"])
            self.assertEqual(2, len(SearchSymbolTool(index).run("execute_query").data))

    def test_citation_validation_rejects_invalid_ranges_and_unsupported_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            RelationshipFixtures.write(root)
            context = "[path: entry.py | lines 3-4]\ndef entry(adapter):"
            valid = Citation("entry.py", start_line=3, end_line=4)
            invalid_range = Citation("entry.py", start_line=0, end_line=4)
            unsupported = Citation("adapter.py", start_line=1, end_line=3)
            result = validate_citations(
                [valid, invalid_range, unsupported], root, context
            )
            self.assertEqual([valid], result)


if __name__ == "__main__":
    unittest.main()
