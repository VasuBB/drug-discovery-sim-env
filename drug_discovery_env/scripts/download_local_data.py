from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

from drug_discovery_env.config.settings import get_settings
from drug_discovery_env.utils.logging import get_logger

DEFAULT_HEADERS = {"Accept": "application/json"}


def _request_json_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: float,
    retries: int,
    retry_backoff_seconds: float,
    logger: Any,
    **kwargs: Any,
) -> dict[str, Any] | None:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = session.request(method, url, timeout=timeout, **kwargs)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # pragma: no cover - depends on network
            last_exc = exc
            if attempt >= retries:
                break
            wait_s = retry_backoff_seconds * (2**attempt)
            logger.warning("Request failed (%s %s), retry in %.1fs: %s", method, url, wait_s, exc)
            time.sleep(wait_s)
    logger.error("Request failed after retries (%s %s): %s", method, url, last_exc)
    return None


def _request_text_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: float,
    retries: int,
    retry_backoff_seconds: float,
    logger: Any,
    **kwargs: Any,
) -> str | None:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = session.request(method, url, timeout=timeout, **kwargs)
            r.raise_for_status()
            return r.text
        except Exception as exc:  # pragma: no cover - depends on network
            last_exc = exc
            if attempt >= retries:
                break
            wait_s = retry_backoff_seconds * (2**attempt)
            logger.warning("Request failed (%s %s), retry in %.1fs: %s", method, url, wait_s, exc)
            time.sleep(wait_s)
    logger.error("Request failed after retries (%s %s): %s", method, url, last_exc)
    return None


def fetch_top_target_for_disease(
    session: requests.Session,
    opentargets_url: str,
    disease: str,
    timeout: float,
    retries: int,
    retry_backoff_seconds: float,
    logger: Any,
) -> dict[str, Any] | None:
    search = {
        "query": """
        query SearchDisease($name: String!) {
          search(queryString: $name, entityNames: [\"disease\"]) {
            hits { id name entity }
          }
        }
        """,
        "variables": {"name": disease},
    }
    payload = _request_json_with_retry(
        session,
        "POST",
        opentargets_url,
        timeout=timeout,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
        logger=logger,
        json=search,
        headers=DEFAULT_HEADERS,
    )
    if not payload:
        return None
    hits = payload.get("data", {}).get("search", {}).get("hits", [])
    if not hits:
        return None
    disease_id = hits[0]["id"]

    assoc = {
        "query": """
        query DiseaseAssoc($efoId: String!) {
          disease(efoId: $efoId) {
            associatedTargets(page: {index: 0, size: 1}) {
              rows {
                score
                target {
                  approvedSymbol
                  targetClass { label }
                }
              }
            }
          }
        }
        """,
        "variables": {"efoId": disease_id},
    }
    payload = _request_json_with_retry(
        session,
        "POST",
        opentargets_url,
        timeout=timeout,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
        logger=logger,
        json=assoc,
        headers=DEFAULT_HEADERS,
    )
    if not payload:
        return None
    rows = payload.get("data", {}).get("disease", {}).get("associatedTargets", {}).get("rows", [])
    if not rows:
        return None
    top = rows[0]
    classes = top["target"].get("targetClass", [])
    return {
        "disease": disease,
        "target": top["target"].get("approvedSymbol", "UNKNOWN"),
        "target_class": classes[0]["label"] if classes else "unknown",
        "druggability": float(top.get("score", 0.5)),
        "confidence": min(1.0, 0.35 + float(top.get("score", 0.5)) * 0.6),
    }


def fetch_compounds(
    session: requests.Session,
    chembl_url: str,
    limit: int,
    timeout: float,
    retries: int,
    retry_backoff_seconds: float,
    max_consecutive_failures: int,
    logger: Any,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    page_size = 100
    consecutive_failures = 0
    while len(out) < limit:
        params = {
            "limit": page_size,
            "offset": offset,
            "format": "json",
            "molecule_structures__isnull": False,
            "molecule_properties__isnull": False,
        }
        payload = _request_json_with_retry(
            session,
            "GET",
            f"{chembl_url}/molecule",
            timeout=timeout,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
            logger=logger,
            params=params,
            headers=DEFAULT_HEADERS,
        )
        if not payload:
            consecutive_failures += 1
            offset += page_size
            if consecutive_failures >= max_consecutive_failures:
                logger.error("Stopping ChEMBL fetch after %d consecutive failures", consecutive_failures)
                break
            continue
        consecutive_failures = 0
        rows = payload.get("molecules", [])
        if not rows:
            break
        for row in rows:
            s = row.get("molecule_structures", {}).get("canonical_smiles")
            if not s:
                continue
            props = row.get("molecule_properties", {})
            alogp = float(props.get("alogp") or 2.5)
            mw = float(props.get("full_mwt") or 350)
            qed = max(0.0, min(1.0, 0.92 - abs(alogp - 2.2) * 0.10 - abs(mw - 360) * 0.0009))
            out.append({"smiles": s, "qed": qed})
            if len(out) >= limit:
                break
        offset += page_size
    return out


def fetch_pubmed_docs(
    session: requests.Session,
    esearch_url: str,
    efetch_url: str,
    queries: list[str],
    per_query: int,
    timeout: float,
    retries: int,
    retry_backoff_seconds: float,
    logger: Any,
) -> list[dict[str, Any]]:
    import xml.etree.ElementTree as ET

    docs: dict[str, dict[str, Any]] = {}
    for query in queries:
        payload = _request_json_with_retry(
            session,
            "GET",
            esearch_url,
            timeout=timeout,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
            logger=logger,
            params={"db": "pubmed", "term": query, "retmax": per_query, "retmode": "json", "sort": "relevance"},
            headers=DEFAULT_HEADERS,
        )
        if not payload:
            continue
        ids = payload.get("esearchresult", {}).get("idlist", [])
        if not ids:
            continue
        text = _request_text_with_retry(
            session,
            "GET",
            efetch_url,
            timeout=timeout,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
            logger=logger,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        )
        if not text:
            continue
        root = ET.fromstring(text)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            title_el = article.find(".//ArticleTitle")
            abs_nodes = article.findall(".//Abstract/AbstractText")
            year_el = article.find(".//PubDate/Year")
            if pmid_el is None or pmid_el.text is None or title_el is None:
                continue
            pmid = pmid_el.text.strip()
            title = "".join(title_el.itertext()).strip()
            abstract = " ".join("".join(a.itertext()).strip() for a in abs_nodes).strip()
            if not abstract:
                continue
            docs[pmid] = {
                "id": f"pmid_{pmid}",
                "title": title[:280],
                "abstract": abstract[:3000],
                "year": int(year_el.text) if year_el is not None and year_el.text and year_el.text.isdigit() else 2020,
            }
    return list(docs.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download large local datasets for local-only training")
    parser.add_argument("--compound-limit", type=int, default=5000)
    parser.add_argument("--pubmed-per-query", type=int, default=250)
    parser.add_argument("--timeout-seconds", type=float, default=None, help="Override request timeout")
    parser.add_argument("--retries", type=int, default=None, help="Override retries per request")
    parser.add_argument("--max-diseases", type=int, default=None, help="Cap number of disease target queries")
    parser.add_argument("--skip-targets", action="store_true", help="Skip OpenTargets target fetch")
    parser.add_argument("--skip-compounds", action="store_true", help="Skip ChEMBL compound fetch")
    parser.add_argument("--skip-literature", action="store_true", help="Skip PubMed literature fetch")
    args = parser.parse_args()

    settings = get_settings().model_copy(deep=True)
    root = Path(settings.data.local_data_dir)
    root.mkdir(parents=True, exist_ok=True)

    logger = get_logger("drug_discovery_env.download", log_file=f"{settings.app.log_dir}/data_download.log")
    logger.info("Starting local data download")

    timeout = args.timeout_seconds if args.timeout_seconds is not None else settings.data.request_timeout_seconds
    retries = args.retries if args.retries is not None else settings.data.retries
    retry_backoff_seconds = settings.data.retry_backoff_seconds

    session = requests.Session()
    session.headers.update(
        {
            **DEFAULT_HEADERS,
            "User-Agent": settings.data.user_agent,
        }
    )

    diseases = settings.data.disease_queries[: args.max_diseases] if args.max_diseases else settings.data.disease_queries

    targets: list[dict[str, Any]] = []
    if args.skip_targets:
        logger.info("Skipping OpenTargets fetch")
    else:
        for disease in diseases:
            row = fetch_top_target_for_disease(
                session,
                settings.data.endpoints.open_targets_url,
                disease,
                timeout,
                retries,
                retry_backoff_seconds,
                logger,
            )
            if row:
                targets.append(row)

    compounds: list[dict[str, Any]] = []
    if args.skip_compounds:
        logger.info("Skipping ChEMBL fetch")
    else:
        compounds = fetch_compounds(
            session,
            settings.data.endpoints.chembl_url,
            limit=args.compound_limit,
            timeout=timeout,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
            max_consecutive_failures=settings.data.max_consecutive_failures,
            logger=logger,
        )

    literature: list[dict[str, Any]] = []
    if args.skip_literature:
        logger.info("Skipping PubMed fetch")
    else:
        literature = fetch_pubmed_docs(
            session,
            settings.data.endpoints.pubmed_esearch_url,
            settings.data.endpoints.pubmed_efetch_url,
            settings.data.literature_queries,
            per_query=args.pubmed_per_query,
            timeout=timeout,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
            logger=logger,
        )

    (root / "targets_snapshot.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")
    (root / "compound_library.json").write_text(json.dumps(compounds, indent=2), encoding="utf-8")
    (root / "literature_snapshot.json").write_text(json.dumps(literature, indent=2), encoding="utf-8")

    logger.info("Saved targets=%d compounds=%d literature=%d", len(targets), len(compounds), len(literature))
    print(json.dumps({"targets": len(targets), "compounds": len(compounds), "literature": len(literature)}, indent=2))


if __name__ == "__main__":
    main()
