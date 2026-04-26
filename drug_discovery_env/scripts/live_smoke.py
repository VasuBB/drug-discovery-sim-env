from __future__ import annotations

import argparse
import json

from drug_discovery_env.config.settings import DataSourceMode, get_settings
from drug_discovery_env.data_provider.factory import build_data_provider


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the live provider")
    parser.add_argument("--disease", type=str, required=True)
    parser.add_argument("--literature-query", type=str, required=True)
    args = parser.parse_args()

    settings = get_settings().model_copy(deep=True)
    settings.data.mode = DataSourceMode.LIVE_ONLY
    provider = build_data_provider(settings)

    target = provider.get_targets_for_disease(args.disease)
    compounds = provider.get_compounds({"min_qed": 0.45, "target": target})
    docs = provider.search_literature(args.literature_query)

    print(json.dumps({
        "target": target,
        "compounds_count": len(compounds),
        "sample_compound": compounds[0] if compounds else None,
        "docs_count": len(docs),
        "sample_doc": docs[0] if docs else None,
    }, indent=2))


if __name__ == "__main__":
    main()
