import tempfile
import unittest
from pathlib import Path

from immunos_mcp.agents.memory_agent import MemoryAgent


class TestMemoryAgent(unittest.TestCase):
    def test_store_retrieve_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "memory.json"
            agent = MemoryAgent(path=memory_path)

            agent.store("key", {"value": 1})
            self.assertEqual(agent.retrieve("key")["value"], 1)

            agent.clear()
            self.assertIsNone(agent.retrieve("key"))


if __name__ == "__main__":
    unittest.main()
