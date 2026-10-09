import tempfile
import unittest
from pathlib import Path

from src.agent.orchestrator import AgentOrchestrator, _is_need_covered
from src.agent.planner import PlannerAgent, QueryCategory
from src.citations import Citation
from src.rag_pipeline import ask_codebase
from src.tools import CodeRelationshipTool, SearchSymbolTool, ToolRegistry
from src.code_index import CodeRelationshipIndex


class FakeVectorStore:
    def similarity_search(self, query, k=4):
        class Document:
            page_content = "Semantic evidence for the requested code path."
            metadata = {
                "source": "entry.py",
                "function_name": "entry",
                "start_line": 3,
                "end_line": 4,
            }

        return [Document()]


class FakeLlm:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return "The requested execution path is supported by the retrieved evidence."


class AgentWorkflowTests(unittest.TestCase):
    def _write_repository(self, root: Path) -> None:
        (root / "entry.py").write_text(
            "from service import process\n\n"
            "def entry(adapter):\n"
            "    return process(adapter)\n",
            encoding="utf-8",
        )
        (root / "service.py").write_text(
            "def process(adapter):\n"
            "    return adapter.execute_query()\n",
            encoding="utf-8",
        )
        (root / "adapter.py").write_text(
            "class Adapter:\n"
            "    def execute_query(self):\n"
            "        return 1\n",
            encoding="utf-8",
        )

    def test_planner_routes_new_capabilities(self):
        planner = PlannerAgent()
        self.assertEqual(
            QueryCategory.SYMBOL_SEARCH,
            planner.plan("Find the exact definition of execute_query").category,
        )
        self.assertEqual(
            QueryCategory.RELATIONSHIP_ANALYSIS,
            planner.plan("Who calls process?").category,
        )

    def test_sufficiency_requires_all_meaningful_terms(self):
        self.assertTrue(_is_need_covered(["execute", "query"], {"execute", "query"}))
        self.assertFalse(_is_need_covered(["execute", "query"], {"execute"}))

    def test_registry_orchestrator_and_answer_generation_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_repository(root)
            index = CodeRelationshipIndex(root)
            registry = ToolRegistry()
            registry.register(SearchSymbolTool(index))
            registry.register(CodeRelationshipTool(index))
            orchestrator = AgentOrchestrator(registry, planner=PlannerAgent(), max_iterations=4)
            result = orchestrator.run("Who calls process?")
            self.assertEqual("relationship_analysis", result.plan.category.value)
            self.assertEqual(2, result.iterations_used)
            self.assertTrue(result.trace[0].result.success)
            self.assertTrue(result.trace[1].result.success)

            fake_llm = FakeLlm()
            answer = ask_codebase(
                "Who calls process?",
                FakeVectorStore(),
                repo_root=str(root),
                llm_client=fake_llm,
                max_iterations=4,
            )
            self.assertIn("Source:", answer)
            self.assertEqual(1, len(fake_llm.prompts))
            self.assertIn("Code relationships:", fake_llm.prompts[0])


if __name__ == "__main__":
    unittest.main()
