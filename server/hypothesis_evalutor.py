"""Sub-agent panel: Chemist, Toxicologist, Budget Manager, Oversight.

Filename retained for backwards-compatibility with the original repo skeleton
(originally a typo of `evaluator`). All sub-agents are rule-based — no LLMs —
so the environment is deterministic given a seed and easy to reason about.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _msg(agent: str, severity: str, message: str) -> Dict[str, str]:
    return {"agent": agent, "severity": severity, "message": message}


def chemist_review(
    last_tool: Optional[str],
    last_result: Dict[str, Any],
    active_compounds: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not active_compounds:
        return out
    smiles_list = [c["smiles"] for c in active_compounds if c.get("smiles")]
    unique = len(set(smiles_list))
    if unique < max(1, len(smiles_list) // 2):
        out.append(_msg(
            "chemist", "warn",
            "Active pool has low structural diversity; consider modifying scaffolds before nominating a lead.",
        ))
    if last_tool == "modify_molecule":
        out.append(_msg(
            "chemist", "info",
            "Modification recorded. Re-screen the variant for binding and ADMET before advancing.",
        ))
    if last_tool == "predict_binding_affinity":
        score = (last_result or {}).get("score")
        if score is not None and score > 0.7:
            out.append(_msg(
                "chemist", "info",
                f"Strong predicted potency (score {score:.2f}); consider scaffold-hopping for selectivity.",
            ))
    return out


def toxicologist_review(
    active_compounds: List[Dict[str, Any]],
    advanced_compound_id: Optional[str],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for c in active_compounds:
        admet = c.get("admet") or {}
        if not admet:
            continue
        cid = c.get("id", "?")
        if admet.get("pains"):
            out.append(_msg("toxicologist", "block", f"Compound {cid} hits a PAINS substructure — do not advance."))
        if admet.get("tox_flag"):
            out.append(_msg(
                "toxicologist", "warn",
                f"Compound {cid} carries a toxic substructure (score {admet.get('tox_score', 0):.2f}).",
            ))
        if admet.get("ro5_pass") is False:
            out.append(_msg("toxicologist", "warn", f"Compound {cid} fails Lipinski Rule-of-Five."))
    if advanced_compound_id:
        adv = next((c for c in active_compounds if c.get("id") == advanced_compound_id), None)
        if adv:
            admet = adv.get("admet") or {}
            if not admet:
                out.append(_msg("toxicologist", "warn", "Lead nominated without ADMET evaluation."))
            elif admet.get("pains") or admet.get("tox_flag"):
                out.append(_msg("toxicologist", "block",
                                 f"Nominated lead {advanced_compound_id} has unresolved tox liability."))
    return out


def budget_review(
    budget_remaining: float, budget_total: float, last_cost: float
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if budget_total <= 0:
        return out
    frac_used = 1.0 - (budget_remaining / budget_total)
    if frac_used >= 0.9:
        out.append(_msg(
            "budget", "block",
            f"Budget critically low ({budget_remaining:.1f} of {budget_total:.0f} left). Stop spending; nominate a lead now.",
        ))
    elif frac_used >= 0.7:
        out.append(_msg(
            "budget", "warn",
            f"70%+ of budget consumed ({budget_remaining:.1f} left). Prefer cheap tools (search, ADMET) over docking.",
        ))
    if last_cost >= 50:
        out.append(_msg(
            "budget", "info",
            f"Last action cost {last_cost:.0f} units — high-cost tool; reserve for verified candidates.",
        ))
    return out


def oversight_review(
    last_action_tool: Optional[str],
    last_action_target_id: Optional[str],
    prior_warnings: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Detects: (1) advancing/run-docking on a compound after a 'block' tox warning,
    (2) running expensive docking after a budget block warning."""
    out: List[Dict[str, str]] = []
    block_msgs = [w for w in prior_warnings if w.get("severity") == "block"]
    if not block_msgs:
        return out
    if last_action_tool in ("advance_stage", "run_docking") and any(
        w["agent"] == "toxicologist" for w in block_msgs
    ):
        out.append(_msg("oversight", "block",
                        "Project Lead acted on a compound despite an active toxicology BLOCK warning."))
    if last_action_tool == "run_docking" and any(w["agent"] == "budget" for w in block_msgs):
        out.append(_msg("oversight", "block",
                        "Project Lead ran high-cost docking despite an active budget BLOCK warning."))
    return out
