import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "promt.py"
spec = importlib.util.spec_from_file_location("promt", MODULE_PATH)
promt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promt)


class ToolCatalogTests(unittest.TestCase):
    def test_tool_catalog_lists_all_registered_tools(self):
        catalog = promt.get_tool_catalog()

        self.assertEqual(len(catalog), len(promt.TOOLS))
        self.assertEqual([item["name"] for item in catalog], list(promt.TOOLS.keys()))

    def test_semantic_memory_helpers_store_and_search(self):
        promt.STATE = {"episodes": [], "semantic_memories": []}

        entry = promt.add_semantic_memory("Brazil is the top wins country", topic="country_stats", session_id="s1")

        self.assertEqual(entry["topic"], "country_stats")
        self.assertEqual(promt.get_semantic_memory(1)[0]["content"], "Brazil is the top wins country")
        self.assertGreaterEqual(len(promt.search_semantic_memory("Brazil")), 1)


if __name__ == "__main__":
    unittest.main()
