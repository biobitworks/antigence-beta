import unittest

from immunos_mcp.orchestrator import ImmunosOrchestrator
from immunos_mcp.core.antigen import Antigen


class TestImmunosOrchestrator(unittest.TestCase):
    def test_orchestrator_runs(self):
        orchestrator = ImmunosOrchestrator()
        antigen = Antigen.from_text("test input")
        result = orchestrator.analyze(antigen)
        self.assertIsNotNone(result)
        self.assertIn("danger", result.signals)
        self.assertTrue(len(result.agents) > 0)


if __name__ == "__main__":
    unittest.main()
