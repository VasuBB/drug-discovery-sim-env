from drug_discovery_env.config.settings import get_settings
from drug_discovery_env.retrieval.hybrid import HybridRetriever


def test_hybrid_retriever_returns_ranked_docs() -> None:
    settings = get_settings()
    docs = [
        {"id": "1", "title": "INSR signaling", "abstract": "PI3K pathway safety and selectivity"},
        {"id": "2", "title": "oncology kinase", "abstract": "EGFR resistance"},
    ]
    retriever = HybridRetriever(settings, docs)
    out = retriever.retrieve("INSR selectivity safety")
    assert out
    assert out[0]["grounding"]["snippet_id"]
