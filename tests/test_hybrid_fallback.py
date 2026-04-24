from drug_discovery_env.config.settings import get_settings
from drug_discovery_env.data_provider.hybrid_provider import HybridDataProvider


class AlwaysFailLive:
    def get_targets_for_disease(self, disease: str):
        raise RuntimeError("forced")

    def get_compounds(self, query):
        raise RuntimeError("forced")

    def search_literature(self, query: str):
        raise RuntimeError("forced")


def test_hybrid_falls_back_to_local() -> None:
    settings = get_settings().model_copy(deep=True)
    provider = HybridDataProvider(settings)
    provider.live = AlwaysFailLive()
    target = provider.get_targets_for_disease("Type 2 Diabetes")
    assert target["source"] == "local"
    assert target["resolution"] == "local"
