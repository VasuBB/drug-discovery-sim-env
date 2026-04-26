"""Prompt rendering + action parsing for the trained policy.

Single source of truth for:
  - the system prompt (what the LLM is told to do)
  - the per-turn user message (rendered observation)
  - the action JSON parser (with `pause_and_review_all` fallback on malformed output)

All training / evaluation / inference paths import from here so the contract
is uniform.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from drug_discovery_env.core.models import DrugDiscoveryAction, DrugDiscoveryObservation

SYSTEM_PROMPT = """You are the Project Lead in a simulated drug discovery campaign.

Goal: take the given disease through 5 stages
(target_selection -> hit_id -> hit_to_lead -> admet -> lead_validation) and
nominate the best lead compound before budget or step cap runs out. You have
sub-agents (chemist, toxicologist, budget, oversight) advising you each step.
Listen to toxicology BLOCK warnings. Conserve budget; validate_compound is the
most expensive tool, reserve it for verified candidates.

You MUST reply with a single JSON object on one line, with these keys:
  "tool":  one of select_target, search_compounds, predict_affinity,
           evaluate_admet, modify_molecule, synthesize, validate_compound,
           search_literature, advance_stage, abandon_compound,
           pause_and_review_all, delegate_to_subagent, request_subagent_summary
  "params": object of tool-specific arguments. For search_literature, pass
           {"query": "<your scientific query>"}; the query string is itself a
           learned decision and is logged.
  "target_compound_id": the C### id of the active compound this acts on, or null
  "reasoning": one or two sentences justifying the decision in scientific terms
               (mention binding, ADMET, PAINS, Lipinski, scaffold, selectivity,
               novelty, budget, hypothesis, uncertainty, tradeoff)

Do not output anything outside the JSON object."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def render_observation(obs: DrugDiscoveryObservation | Dict[str, Any]) -> str:
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
    lines: list[str] = []
    lines.append(
        f"Stage: {obs_dict.get('stage')}  Step: {obs_dict.get('step_index')}/{obs_dict.get('max_steps')}"
    )
    lines.append(
        f"Disease: {obs_dict.get('disease')}  Target: {obs_dict.get('selected_target')}"
    )
    lines.append(
        f"Budget: {obs_dict.get('budget_remaining', 0):.0f}/{obs_dict.get('budget_total', 0):.0f}"
        f"  (last cost {obs_dict.get('last_tool_cost', 0):.0f})"
    )
    if obs_dict.get("last_tool"):
        last_blob = json.dumps(obs_dict.get("last_result", {}), default=str)[:240]
        lines.append(f"Last tool: {obs_dict['last_tool']} -> {last_blob}")
    msgs = obs_dict.get("subagent_messages") or []
    if msgs:
        lines.append("Sub-agent messages:")
        for msg in msgs[:8]:
            lines.append(f"  [{msg.get('agent')}/{msg.get('severity')}] {msg.get('message')}")
    actives = obs_dict.get("active_compounds") or []
    if actives:
        lines.append(f"Active compounds ({len(actives)}):")
        for c in actives[:8]:
            admet = c.get("admet") or {}
            smiles = (c.get("smiles") or "")[:40]
            lines.append(
                f"  {c.get('id')}  smiles={smiles}"
                f"  aff={c.get('binding_affinity_nM')}"
                f"  dock={c.get('docking_score')}"
                f"  ro5={admet.get('ro5_pass')} pains={admet.get('pains')} tox={admet.get('tox_score')}"
            )
    lines.append(f"Status: {obs_dict.get('message', '')}")
    return "\n".join(lines)


def parse_action_text(text: str) -> Dict[str, Any]:
    match = _JSON_RE.search(text or "")
    if not match:
        return {"tool": "pause_and_review_all", "params": {}, "reasoning": text or ""}
    try:
        payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            return {"tool": "pause_and_review_all", "params": {}, "reasoning": text or ""}
        return payload
    except Exception:
        return {"tool": "pause_and_review_all", "params": {}, "reasoning": text or ""}


def action_from_text(text: str) -> DrugDiscoveryAction:
    payload = parse_action_text(text)
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    evidence = payload.get("evidence_ids") or []
    if not isinstance(evidence, list):
        evidence = []
    return DrugDiscoveryAction(
        tool=str(payload.get("tool", "pause_and_review_all")),
        params=params,
        target_compound_id=payload.get("target_compound_id"),
        reasoning=str(payload.get("reasoning", "")),
        evidence_ids=[str(x) for x in evidence],
    )


def initial_user_message(disease: str) -> str:
    return (
        f"Begin a new campaign for {disease}. Stage: target_selection. "
        "Choose your first tool call."
    )
