from __future__ import annotations

import json
import re

from drug_discovery_env.core.models import DrugDiscoveryAction


TOOL_PATTERN = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
PARAM_PATTERN = re.compile(r"<params>(.*?)</params>", re.DOTALL)
REASON_PATTERN = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
EVIDENCE_PATTERN = re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL)


class ActionParser:
    def __init__(self, allowed_tools: set[str]) -> None:
        self.allowed_tools = allowed_tools

    def parse(self, raw_text: str) -> DrugDiscoveryAction:
        tool_match = TOOL_PATTERN.search(raw_text)
        if not tool_match:
            raise ValueError("Missing <tool> tag")
        tool = tool_match.group(1).strip()
        if tool not in self.allowed_tools:
            raise ValueError(f"Unknown tool: {tool}")

        reason_match = REASON_PATTERN.search(raw_text)
        reasoning = reason_match.group(1).strip() if reason_match else ""

        params: dict[str, object] = {}
        params_match = PARAM_PATTERN.search(raw_text)
        if params_match:
            payload = params_match.group(1).strip()
            params = json.loads(payload) if payload else {}

        evidence_ids: list[str] = []
        evidence_match = EVIDENCE_PATTERN.search(raw_text)
        if evidence_match:
            evidence_ids = [x.strip() for x in evidence_match.group(1).split(",") if x.strip()]

        return DrugDiscoveryAction(tool=tool, params=params, reasoning=reasoning, evidence_ids=evidence_ids)
