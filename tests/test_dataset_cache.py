import json
from pathlib import Path

import pytest

from drug_discovery_env.config.settings import DataSourceMode, Settings, get_settings
from drug_discovery_env.data_provider.cached_provider import CachedTargetsLiveToolsProvider
from drug_discovery_env.data_provider.dataset import _CACHE, load_dataset


@pytest.fixture()
def stub_dataset(tmp_path: Path) -> Path:
    rows = [
        {
            "disease": "Type 2 Diabetes",
            "efo_id": "EFO_0001",
            "target": "INSR",
            "target_class": "kinase",
            "druggability": 0.82,
            "confidence": 0.9,
            "known_drugs": ["CCO", "COC1=CC=CC=C1O"],
            "split": "train",
        },
        {
            "disease": "Alzheimer disease",
            "efo_id": "EFO_0002",
            "target": "APP",
            "target_class": "membrane",
            "druggability": 0.40,
            "confidence": 0.6,
            "known_drugs": ["NCCC1=CC=CC=C1"],
            "split": "test",
        },
    ]
    path = tmp_path / "diseases.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    _CACHE.clear()
    return path


def test_load_dataset_splits(stub_dataset: Path) -> None:
    ds = load_dataset(stub_dataset)
    assert ds.n_train == 1
    assert ds.n_test == 1
    assert ds.train[0].target == "INSR"
    assert ds.test[0].known_drugs == ["NCCC1=CC=CC=C1"]


def test_cached_provider_uses_cache(stub_dataset: Path) -> None:
    settings: Settings = get_settings().model_copy(deep=True)
    settings.data.mode = DataSourceMode.CACHED_TARGETS_LIVE_TOOLS
    settings.dataset.cache_path = str(stub_dataset)
    provider = CachedTargetsLiveToolsProvider(settings)
    payload = provider.get_targets_for_disease("type 2 diabetes")
    assert payload["target"] == "INSR"
    assert payload["source"] == "cache"
    assert payload["known_drugs"] == ["CCO", "COC1=CC=CC=C1O"]
