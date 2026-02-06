from immunos_mcp.agents.dendritic_agent import DendriticAgent
from immunos_mcp.core.antigen import Antigen


def test_dendritic_feature_vector_shape():
    agent = DendriticAgent()
    antigen = Antigen.from_text("This study reports a 100% improvement in outcomes.")
    vector = agent.get_feature_vector(antigen)
    assert len(vector) == 20
    assert all(isinstance(v, float) for v in vector)


def test_dendritic_feature_bounds():
    agent = DendriticAgent()
    antigen = Antigen.from_text("According to studies, results are mixed and uncertain.")
    features = agent.extract_features(antigen)
    credibility = features.get("source_credibility", 0.0)
    assert 0.0 <= credibility <= 1.0
