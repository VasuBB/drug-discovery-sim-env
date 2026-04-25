"""Action parser — accepts both JSON (main) and XML-tag (akshat) formats.

Tries JSON first because it's more robust to whitespace/escaping; falls back
to XML tags so older training corpora still work. On total parse failure
returns a safe `pause_and_review_all` action with the raw text retained as
`reasoning` so reward shaping can still see something.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from drug_discovery_env.core.models import DrugDiscoveryAction


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
_PARAMS_RE = re.compile(r"<params>(.*?)</params>", re.DOTALL)
_REASON_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
_EVID_RE = re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL)
_TARGET_ID_RE = re.compile(r"<target_compound_id>(.*?)</target_compound_id>", re.DOTALL)


class ActionParser:
    def __init__(self, allowed_tools: set[str]) -> None:
        self.allowed_tools = allowed_tools

    def parse(self, raw_text: Any) -> DrugDiscoveryAction:
        # already-typed action — pass through
        if isinstance(raw_text, DrugDiscoveryAction):
            return raw_text
        if isinstance(raw_text, dict):
            return self._from_dict(raw_text)

        text = str(raw_text or "")

        # 1) Try JSON
        json_match = _JSON_RE.search(text)
        if json_match:
            try:
                obj = json.loads(json_match.group(0))
                if isinstance(obj, dict) and "tool" in obj:
                    return self._from_dict(obj, fallback_text=text)
            except Exception:
                pass

        # 2) Try XML tags
        tool_match = _TOOL_RE.search(text)
        if tool_match:
            tool = tool_match.group(1).strip()
            params: Dict[str, Any] = {}
            params_match = _PARAMS_RE.search(text)
            if params_match:
                payload = params_match.group(1).strip()
                if payload:
                    try:
                        params = json.loads(payload)
                    except Exception:
                        params = {}
            reason_match = _REASON_RE.search(text)
            reasoning = reason_match.group(1).strip() if reason_match else ""
            evid_match = _EVID_RE.search(text)
            evidence_ids = (
                [x.strip() for x in evid_match.group(1).split(",") if x.strip()]
                if evid_match
                else []
            )
            target_match = _TARGET_ID_RE.search(text)
            target_compound_id = target_match.group(1).strip() if target_match else None
            return DrugDiscoveryAction(
                tool=tool,
                params=params,
                target_compound_id=target_compound_id,
                reasoning=reasoning,
                evidence_ids=evidence_ids,
            )

        # 3) Total fallback
        return DrugDiscoveryAction(
            tool="pause_and_review_all",
            params={},
            reasoning=text,
        )

    def _from_dict(self, obj: Dict[str, Any], fallback_text: str = "") -> DrugDiscoveryAction:
        tool = str(obj.get("tool", "pause_and_review_all"))
        params = obj.get("params", {}) or {}
        if not isinstance(params, dict):
            params = {}
        reasoning = str(obj.get("reasoning", "") or fallback_text)
        evidence_ids = obj.get("evidence_ids", []) or []
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        return DrugDiscoveryAction(
            tool=tool,
            params=params,
            target_compound_id=obj.get("target_compound_id"),
            reasoning=reasoning,
            evidence_ids=[str(x) for x in evidence_ids],
        )
