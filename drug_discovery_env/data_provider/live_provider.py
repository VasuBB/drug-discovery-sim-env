from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import requests

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.data_provider.base import DataProvider


class LiveAPIProvider(DataProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.data.user_agent})

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        timeout = self.settings.data.request_timeout_seconds
        response = self.session.request(method=method, url=url, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response

    def _ot_search_disease_id(self, disease: str) -> str | None:
        query = {
            "query": """
            query SearchDisease($name: String!) {
              search(queryString: $name, entityNames: ["disease"]) {
                hits { id name entity }
              }
            }
            """,
            "variables": {"name": disease},
        }
        response = self._request("POST", self.settings.data.endpoints.open_targets_url, json=query)
        hits = response.json().get("data", {}).get("search", {}).get("hits", [])
        return hits[0].get("id") if hits else None

    def _ot_top_target(self, disease_id: str) -> dict[str, Any] | None:
        assoc_query = {
            "query": """
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
            """,
            "variables": {"efoId": disease_id},
        }
        try:
            assoc_response = self._request("POST", self.settings.data.endpoints.open_targets_url, json=assoc_query)
            rows = (
                assoc_response.json()
                .get("data", {})
                .get("disease", {})
                .get("associatedTargets", {})
                .get("rows", [])
            )
            return rows[0] if rows else None
        except requests.HTTPError:
            # Open Targets schema can change; tolerate failures and continue with search-only data.
            return None

    def get_targets_for_disease(self, disease: str) -> dict[str, Any]:
        disease_id = self._ot_search_disease_id(disease)
        target_name = "UNKNOWN_TARGET"
        target_class = "unknown"
        druggability = 0.45
        confidence = 0.40

        if disease_id:
            best = self._ot_top_target(disease_id)
            if best:
                target = best.get("target", {})
                target_name = str(target.get("approvedSymbol") or target_name)
                classes = target.get("targetClass") or []
                target_class = str(classes[0].get("label")) if classes else target_class
                druggability = max(0.0, min(1.0, float(best.get("score", 0.45))))
                confidence = max(0.0, min(1.0, 0.35 + druggability * 0.6))

        return {
            "disease": disease,
            "target": target_name,
            "target_class": target_class,
            "druggability": druggability,
            "source": "live",
            "confidence": confidence,
            "timestamp": self._timestamp(),
        }

    def get_compounds(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        min_qed = float(query.get("min_qed", 0.0))
        target_symbol = str(query.get("target", {}).get("target", ""))

        url = f"{self.settings.data.endpoints.chembl_url}/molecule"
        params = {
            "limit": 60,
            "molecule_structures__isnull": False,
            "molecule_properties__isnull": False,
            "format": "json",
        }
        if target_symbol and target_symbol != "UNKNOWN_TARGET":
            params["pref_name__icontains"] = target_symbol

        response = self._request("GET", url, params=params)
        try:
            molecules = response.json().get("molecules", [])
        except ValueError:
            molecules = []
        if not molecules and "pref_name__icontains" in params:
            # Fallback: target filter can be too restrictive for this endpoint.
            params.pop("pref_name__icontains", None)
            response = self._request("GET", url, params=params)
            try:
                molecules = response.json().get("molecules", [])
            except ValueError:
                molecules = []

        out: list[dict[str, Any]] = []
        for row in molecules:
            structures = row.get("molecule_structures", {})
            smiles = structures.get("canonical_smiles")
            if not smiles:
                continue
            props = row.get("molecule_properties", {})
            alogp = float(props.get("alogp") or 2.5)
            full_mwt = float(props.get("full_mwt") or 350)
            psa = float(props.get("psa") or 65)
            hbd = float(props.get("hbd") or 1)
            hba = float(props.get("hba") or 4)

            # Simple developability surrogate derived from common med-chem heuristics.
            penalties = abs(alogp - 2.2) * 0.10 + abs(full_mwt - 360) * 0.0009 + abs(psa - 70) * 0.002
            penalties += max(0.0, hbd - 3.0) * 0.03 + max(0.0, hba - 8.0) * 0.02
            qed = max(0.0, min(1.0, 0.92 - penalties))

            if qed < min_qed:
                continue
            out.append(
                {
                    "smiles": smiles,
                    "qed": qed,
                    "source": "live",
                    "confidence": 0.58,
                    "chembl_id": row.get("molecule_chembl_id"),
                }
            )
            if len(out) >= 20:
                break
        return out

    def _pubmed_fetch_xml(self, ids: list[str]) -> str:
        response = self._request(
            "GET",
            self.settings.data.endpoints.pubmed_efetch_url,
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
                "tool": self.settings.data.pubmed_tool,
                "email": self.settings.data.pubmed_email,
            },
        )
        return response.text

    def _parse_pubmed_xml(self, xml_payload: str) -> list[dict[str, Any]]:
        root = ET.fromstring(xml_payload)
        docs: list[dict[str, Any]] = []
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            title_el = article.find(".//ArticleTitle")
            abs_nodes = article.findall(".//Abstract/AbstractText")
            year_el = article.find(".//PubDate/Year")

            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""
            abstract = " ".join("".join(n.itertext()).strip() for n in abs_nodes).strip()
            if not pmid or not title:
                continue
            docs.append(
                {
                    "id": f"pmid_{pmid}",
                    "title": title[:280],
                    "abstract": abstract[:3000],
                    "year": int(year_el.text) if year_el is not None and year_el.text and year_el.text.isdigit() else None,
                    "score": 0.5,
                    "source": "live",
                    "confidence": 0.62,
                }
            )
        return docs

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        return {tok for tok in cleaned.split() if len(tok) >= 4}

    def search_literature(self, query: str) -> list[dict[str, Any]]:
        base_params = {
            "db": "pubmed",
            "retmax": 10,
            "retmode": "json",
            "sort": "relevance",
            "tool": self.settings.data.pubmed_tool,
            "email": self.settings.data.pubmed_email,
        }
        esearch = self._request(
            "GET",
            self.settings.data.endpoints.pubmed_esearch_url,
            params={**base_params, "term": query},
        ).json()
        ids = esearch.get("esearchresult", {}).get("idlist", [])
        if not ids:
            tokens = [t for t in query.replace(",", " ").split() if len(t) > 3]
            fallback_term = " OR ".join(tokens[:4]) if tokens else query
            esearch = self._request(
                "GET",
                self.settings.data.endpoints.pubmed_esearch_url,
                params={**base_params, "term": fallback_term},
            ).json()
            ids = esearch.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        xml_payload = self._pubmed_fetch_xml(ids[:10])
        docs = self._parse_pubmed_xml(xml_payload)
        q_tokens = self._tokenize(query)
        min_overlap = self.settings.data.min_pubmed_token_overlap
        scored: list[tuple[int, dict[str, Any]]] = []
        for doc in docs:
            text = f"{doc.get('title', '')} {doc.get('abstract', '')}"
            d_tokens = self._tokenize(text)
            overlap = len(q_tokens.intersection(d_tokens))
            if overlap >= min_overlap:
                row = dict(doc)
                row["score"] = max(float(doc.get("score", 0.0)), min(1.0, overlap / 10.0))
                scored.append((overlap, row))
        if not scored:
            return docs[:10]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [row for _, row in scored[:10]]
