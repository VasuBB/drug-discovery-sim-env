"""One-time disease/target/known-drugs dataset builder.

Pages Open Targets GraphQL for thousands of diseases, picks the top associated
target per disease, fetches up to N known compounds from ChEMBL for that
target, deterministically splits into train/test, and writes the cache that
the env / GRPO trainer / evaluator all read from.

Usage:
    python -m drug_discovery_env.scripts.prepare_dataset
    python -m drug_discovery_env.scripts.prepare_dataset --num-diseases 200 \
        --output data/diseases_small.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from drug_discovery_env.config.runtime import resolve_path
from drug_discovery_env.config.settings import Settings, get_settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ot_post(session: requests.Session, url: str, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    response = session.post(url, json={"query": query, "variables": variables}, timeout=30.0)
    response.raise_for_status()
    return response.json()


def _iterate_diseases(session: requests.Session, settings: Settings) -> Iterable[Dict[str, Any]]:
    """Yield disease rows from Open Targets, page by page."""
    page_size = settings.dataset.page_size
    target_count = settings.dataset.num_diseases
    url = settings.data.endpoints.open_targets_url

    query = """
    query SearchDiseases($queryString: String!, $page: Pagination!) {
      search(queryString: $queryString, entityNames: ["disease"], page: $page) {
        total
        hits {
          id
          name
          entity
        }
      }
    }
    """

    seeds = [
        "disease", "syndrome", "cancer", "infection", "deficiency", "disorder",
        "diabetes", "alzheimer", "parkinson", "leukemia", "lymphoma", "sclerosis",
        "fibrosis", "anemia", "arthritis", "asthma", "epilepsy", "neuropathy",
        "carcinoma", "melanoma", "tumor", "ischemia", "hepatitis", "nephritis",
        "myopathy", "dystrophy", "hypertension", "obesity", "depression", "psychosis",
    ]

    seen: set[str] = set()
    yielded = 0
    for seed in seeds:
        if yielded >= target_count:
            break
        page_index = 0
        while yielded < target_count:
            payload = _ot_post(
                session,
                url,
                query,
                {"queryString": seed, "page": {"index": page_index, "size": page_size}},
            )
            search = payload.get("data", {}).get("search", {}) or {}
            hits = search.get("hits") or []
            if not hits:
                break
            for hit in hits:
                if hit.get("entity") != "disease":
                    continue
                efo_id = hit.get("id")
                name = hit.get("name")
                if not efo_id or not name or efo_id in seen:
                    continue
                seen.add(efo_id)
                yield {"efo_id": efo_id, "name": str(name)}
                yielded += 1
                if yielded >= target_count:
                    break
            page_index += 1
            if page_index * page_size >= int(search.get("total", 0) or 0):
                break


_TOP_TARGET_QUERY = """
query DiseaseAssoc($efoId: String!) {
  disease(efoId: $efoId) {
    associatedTargets(page: {index: 0, size: 1}) {
      rows {
        score
        target {
          approvedSymbol
          targetClass {
            label
          }
        }
      }
    }
  }
}
"""


def _fetch_top_target(session: requests.Session, settings: Settings, efo_id: str) -> Optional[Dict[str, Any]]:
    try:
        payload = _ot_post(session, settings.data.endpoints.open_targets_url, _TOP_TARGET_QUERY, {"efoId": efo_id})
    except requests.HTTPError:
        return None
    rows = (
        payload.get("data", {})
        .get("disease", {})
        .get("associatedTargets", {})
        .get("rows", [])
    )
    return rows[0] if rows else None


def _fetch_known_drugs(session: requests.Session, settings: Settings, target_symbol: str, k: int) -> List[str]:
    if not target_symbol:
        return []
    url = f"{settings.data.endpoints.chembl_url}/molecule"
    params = {
        "limit": max(20, k * 2),
        "molecule_structures__isnull": False,
        "format": "json",
        "pref_name__icontains": target_symbol,
    }
    try:
        response = session.get(url, params=params, timeout=settings.data.request_timeout_seconds * 4)
        response.raise_for_status()
        molecules = response.json().get("molecules", []) or []
    except Exception:
        return []
    out: List[str] = []
    for row in molecules:
        smiles = ((row.get("molecule_structures") or {}).get("canonical_smiles") or "").strip()
        if smiles and smiles not in out:
            out.append(smiles)
        if len(out) >= k:
            break
    return out


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build(
    settings: Settings,
    *,
    num_diseases: Optional[int] = None,
    output_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    test_fraction: Optional[float] = None,
    min_druggability: Optional[float] = None,
    seed: Optional[int] = None,
    chembl_known_drugs_per_target: Optional[int] = None,
    sleep_s: float = 0.05,
) -> Dict[str, Any]:
    ds = settings.dataset
    n = num_diseases or ds.num_diseases
    out_path = output_path or resolve_path(ds.cache_path)
    manifest = manifest_path or resolve_path(ds.manifest_path)
    test_frac = test_fraction if test_fraction is not None else ds.test_fraction
    min_drug = min_druggability if min_druggability is not None else ds.min_druggability
    seed_val = seed if seed is not None else ds.seed
    k_drugs = chembl_known_drugs_per_target or ds.chembl_known_drugs_per_target

    # Override the count we ask Open Targets for
    settings = settings.model_copy(deep=True)
    settings.dataset.num_diseases = n

    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": settings.data.user_agent})

    rows: List[Dict[str, Any]] = []
    print(f"[prepare_dataset] fetching up to {n} diseases from Open Targets...")
    for i, disease in enumerate(_iterate_diseases(session, settings)):
        top = _fetch_top_target(session, settings, disease["efo_id"])
        if not top:
            continue
        target = top.get("target") or {}
        symbol = str(target.get("approvedSymbol") or "").strip()
        if not symbol:
            continue
        druggability = max(0.0, min(1.0, float(top.get("score", 0.0))))
        if druggability < min_drug:
            continue
        target_class_list = target.get("targetClass") or []
        target_class = str(target_class_list[0].get("label")) if target_class_list else "unknown"
        confidence = max(0.0, min(1.0, 0.35 + druggability * 0.6))
        known = _fetch_known_drugs(session, settings, symbol, k_drugs)
        rows.append(
            {
                "disease": disease["name"],
                "efo_id": disease["efo_id"],
                "target": symbol,
                "target_class": target_class,
                "druggability": druggability,
                "confidence": confidence,
                "known_drugs": known,
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}] kept={len(rows)} latest='{disease['name']}' target={symbol}")
        time.sleep(sleep_s)

    if not rows:
        raise RuntimeError("No disease rows produced; check network/Open Targets availability.")

    rng = random.Random(seed_val)
    rng.shuffle(rows)
    n_test = max(1, int(round(len(rows) * test_frac)))
    for idx, row in enumerate(rows):
        row["split"] = "test" if idx < n_test else "train"

    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_payload = {
        "rows": len(rows),
        "n_train": sum(1 for r in rows if r["split"] == "train"),
        "n_test": sum(1 for r in rows if r["split"] == "test"),
        "test_fraction": test_frac,
        "seed": seed_val,
        "min_druggability": min_drug,
        "chembl_known_drugs_per_target": k_drugs,
        "fetched_at": _now_iso(),
        "sha256": _sha256_of(out_path),
        "cache_path": str(out_path),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[prepare_dataset] wrote {out_path} ({manifest_payload['n_train']} train + "
        f"{manifest_payload['n_test']} test) and manifest {manifest}"
    )
    return manifest_payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the cached disease/target dataset")
    parser.add_argument("--num-diseases", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--min-druggability", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--known-drugs-per-target", type=int, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    build(
        settings,
        num_diseases=args.num_diseases,
        output_path=args.output,
        manifest_path=args.manifest,
        test_fraction=args.test_fraction,
        min_druggability=args.min_druggability,
        seed=args.seed,
        chembl_known_drugs_per_target=args.known_drugs_per_target,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
