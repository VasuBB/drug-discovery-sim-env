"""The Drug Discovery campaign environment.

Implements the OpenEnv `Environment` interface. One `reset` starts a fresh
50-step research campaign in a sampled disease scenario. Each `step` consumes
one decision from the Project Lead agent (a tool call or stage decision).
The environment:

  - debits the agent's budget by the tool cost,
  - executes the simulated tool,
  - lets four rule-based sub-agents review the new state,
  - records reasoning traces and warning compliance,
  - emits a structured Observation,
  - and terminates when stages are cleared, the budget hits zero, or the
    step cap is reached.

Reward is sparse-ish: each step's reward is `0.0` until termination, at
which point the full composable rubric is summed and returned in the final
Observation's `reward` field.
"""

from __future__ import annotations

import os
import random
import sys
import uuid
from typing import Any, Dict, List, Optional

# Allow `python -m server.app` style imports as well as direct execution.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
_REPO_DIR = os.path.dirname(_PKG_DIR)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from openenv.core.env_server import Environment

from models import DrugDiscoveryAction, DrugDiscoveryObservation, DrugDiscoveryState  # type: ignore  # noqa: E402

from hypothesis_evalutor import (  # noqa: E402
    budget_review,
    chemist_review,
    oversight_review,
    toxicologist_review,
)
from lab_simulator import LabSimulator  # noqa: E402
from reward_calculator import compute_reward  # noqa: E402
from scenario_generator import Scenario, find_scenario, sample_scenario  # noqa: E402


STAGE_ORDER = ["target_selection", "hit_id", "hit_to_lead", "admet", "lead_validation", "finished"]


class DrugDiscoveryEnvironment(Environment):
    """50-step simulated drug discovery research campaign."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    DEFAULT_MAX_STEPS = 50
    DEFAULT_BUDGET = 1000.0

    def __init__(self) -> None:
        self._lab = LabSimulator()
        self._state = DrugDiscoveryState()
        self._scenario: Optional[Scenario] = None

        self._compounds: Dict[str, Dict[str, Any]] = {}
        self._compound_order: List[str] = []
        self._next_compound_idx = 0

        self._stages_completed: List[str] = []
        self._reasoning_traces: List[str] = []

        self._warnings_issued = 0
        self._warnings_ignored = 0
        self._open_block_warnings: List[Dict[str, str]] = []

        self._last_tool: Optional[str] = None
        self._last_result: Dict[str, Any] = {}
        self._last_cost: float = 0.0
        self._last_message: str = ""

        self._rng = random.Random()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        disease: Optional[str] = None,
        max_steps: Optional[int] = None,
        budget: Optional[float] = None,
        **kwargs: Any,
    ) -> DrugDiscoveryObservation:
        if seed is not None:
            self._rng = random.Random(seed)

        scenario: Optional[Scenario] = None
        if disease:
            scenario = find_scenario(disease)
        if scenario is None:
            scenario = sample_scenario(seed=seed)
        self._scenario = scenario

        max_steps = int(max_steps or self.DEFAULT_MAX_STEPS)
        budget = float(budget if budget is not None else self.DEFAULT_BUDGET)

        self._state = DrugDiscoveryState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            disease=scenario.disease,
            selected_target=None,
            max_steps=max_steps,
            budget_total=budget,
            budget_remaining=budget,
            stage="target_selection",
            advanced_compound_id=None,
            terminated_reason=None,
        )
        self._compounds = {}
        self._compound_order = []
        self._next_compound_idx = 0
        self._stages_completed = []
        self._reasoning_traces = []
        self._warnings_issued = 0
        self._warnings_ignored = 0
        self._open_block_warnings = []
        self._last_tool = None
        self._last_result = {}
        self._last_cost = 0.0
        self._last_message = (
            f"New campaign: find a lead compound for {scenario.disease}. "
            f"Budget {budget:.0f}, max {max_steps} steps. Begin with stage 'target_selection'."
        )

        return self._build_observation(done=False, reward=None)

    def step(
        self,
        action: DrugDiscoveryAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DrugDiscoveryObservation:
        self._state.step_count += 1
        if action.reasoning:
            self._reasoning_traces.append(action.reasoning)

        prior_blocks = list(self._open_block_warnings)
        ignored_now = self._was_warning_ignored(action, prior_blocks)
        if ignored_now:
            self._warnings_ignored += ignored_now

        cost, result_payload, message, terminated_reason = self._dispatch(action)

        self._last_tool = action.tool
        self._last_result = result_payload
        self._last_cost = cost
        self._last_message = message

        if terminated_reason is None and self._state.budget_remaining <= 0:
            terminated_reason = "budget_exhausted"
        if terminated_reason is None and self._state.step_count >= self._state.max_steps:
            terminated_reason = "max_steps"

        # Sub-agent panel: regenerate every step from current state.
        active = self._active_compounds()
        msgs: List[Dict[str, str]] = []
        msgs.extend(chemist_review(self._last_tool, self._last_result, active))
        msgs.extend(toxicologist_review(active, self._state.advanced_compound_id))
        msgs.extend(budget_review(self._state.budget_remaining, self._state.budget_total, cost))
        msgs.extend(oversight_review(self._last_tool, action.target_compound_id, prior_blocks))
        self._warnings_issued += sum(1 for m in msgs if m["severity"] in ("warn", "block"))
        self._open_block_warnings = [m for m in msgs if m["severity"] == "block"]

        done = terminated_reason is not None
        reward: Optional[float] = None
        if done:
            self._state.terminated_reason = terminated_reason
            advanced = self._compounds.get(self._state.advanced_compound_id) if self._state.advanced_compound_id else None
            seen = [c["smiles"] for c in active]
            breakdown = compute_reward(
                advanced_compound=advanced,
                target=self._state.selected_target,
                seen_smiles=seen,
                stages_completed=self._stages_completed,
                budget_remaining=self._state.budget_remaining,
                budget_total=self._state.budget_total,
                reasoning_traces=self._reasoning_traces,
                warnings_issued=self._warnings_issued,
                warnings_ignored=self._warnings_ignored,
            )
            reward = breakdown.total()
            self._last_result = {**self._last_result, "reward_breakdown": breakdown.to_dict()}
            self._last_message = (
                f"Campaign ended ({terminated_reason}). Reward={reward:.3f}. "
                f"Breakdown: {breakdown.to_dict()}"
            )

        obs = self._build_observation(done=done, reward=reward, subagent_messages=msgs)
        return obs

    @property
    def state(self) -> DrugDiscoveryState:
        return self._state

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, action: DrugDiscoveryAction):
        """Return (cost, result_payload, message, terminated_reason)."""
        tool = (action.tool or "").strip()
        params = action.params or {}
        cost = self._lab.COSTS.get(tool, 0.0)

        if tool not in (
            "search_chembl", "predict_binding_affinity", "compute_admet",
            "modify_molecule", "run_docking", "literature_search",
            "delegate_to_subagent", "advance_stage", "abandon_compound",
            "pause_and_review_all", "request_subagent_summary",
        ):
            return 0.0, {"error": f"unknown tool '{tool}'"}, f"Rejected: unknown tool '{tool}'.", None

        # Charge first, then check whether we can afford it.
        if cost > self._state.budget_remaining:
            return 0.0, {"error": "insufficient_budget", "cost": cost}, (
                f"Cannot afford '{tool}' (cost {cost:.0f}, remaining {self._state.budget_remaining:.0f})."
            ), "budget_exhausted"

        self._state.budget_remaining = max(0.0, self._state.budget_remaining - cost)

        if tool == "search_chembl":
            query = params.get("query") or params.get("target") or self._state.disease
            results = self._lab.search_chembl(query, max_results=int(params.get("max_results", 5)))
            for r in results:
                self._add_compound(r["smiles"], origin="chembl")
            return cost, {"hits": results}, f"ChEMBL-sim returned {len(results)} candidate(s) for '{query}'.", None

        if tool == "literature_search":
            query = params.get("query") or self._state.disease
            results = self._lab.literature_search(query, max_results=int(params.get("max_results", 3)))
            return cost, {"abstracts": results}, f"Literature returned {len(results)} abstract(s) for '{query}'.", None

        if tool == "predict_binding_affinity":
            cid = action.target_compound_id
            smiles = params.get("smiles") or self._smiles_for(cid)
            if not smiles:
                return cost, {"error": "no_smiles"}, "predict_binding_affinity requires a smiles or target_compound_id.", None
            target = params.get("target") or self._state.selected_target
            if not target:
                return cost, {"error": "no_target"}, "Select a target first (advance from target_selection).", None
            res = self._lab.predict_binding_affinity(smiles, target)
            cid = self._upsert_compound(smiles, cid, binding_affinity_nM=res["affinity_nM"])
            return cost, res, (
                f"Predicted affinity for {cid}: {res['affinity_nM']:.1f} nM (score {res['score']:.2f})."
            ), None

        if tool == "compute_admet":
            cid = action.target_compound_id
            smiles = params.get("smiles") or self._smiles_for(cid)
            if not smiles:
                return cost, {"error": "no_smiles"}, "compute_admet requires a smiles or target_compound_id.", None
            res = self._lab.compute_admet(smiles)
            cid = self._upsert_compound(smiles, cid, admet=res.to_dict())
            return cost, res.to_dict(), (
                f"ADMET for {cid}: RO5={'pass' if res.ro5_pass else 'fail'}, "
                f"PAINS={'YES' if res.pains else 'no'}, tox_score={res.tox_score:.2f}, qed={res.qed:.2f}."
            ), None

        if tool == "modify_molecule":
            cid = action.target_compound_id
            smiles = params.get("smiles") or self._smiles_for(cid)
            if not smiles:
                return cost, {"error": "no_smiles"}, "modify_molecule requires a smiles or target_compound_id.", None
            instruction = params.get("instruction", "add methyl")
            res = self._lab.modify_molecule(smiles, instruction)
            new_id = self._add_compound(res["smiles"], origin="modified", parent_id=cid, history=f"modify({instruction})")
            return cost, {**res, "new_compound_id": new_id}, (
                f"Created {new_id} from {cid or '?'} via '{instruction}'."
            ), None

        if tool == "run_docking":
            cid = action.target_compound_id
            smiles = params.get("smiles") or self._smiles_for(cid)
            if not smiles:
                return cost, {"error": "no_smiles"}, "run_docking requires a smiles or target_compound_id.", None
            target = params.get("target") or self._state.selected_target
            if not target:
                return cost, {"error": "no_target"}, "Select a target first.", None
            res = self._lab.run_docking(smiles, target)
            cid = self._upsert_compound(smiles, cid, docking_score=res["docking_score"])
            return cost, res, (
                f"Docking for {cid}: score={res['docking_score']:.2f} (quality {res['quality']:.2f})."
            ), None

        if tool == "delegate_to_subagent":
            who = params.get("subagent", "chemist")
            return 0.0, {"delegated_to": who}, f"Delegated to sub-agent '{who}'. See subagent_messages.", None

        if tool == "request_subagent_summary":
            return 0.0, {"summary_requested": True}, "Requested sub-agent summary; see subagent_messages.", None

        if tool == "pause_and_review_all":
            return 0.0, {"paused": True, "active_count": len(self._compounds)}, (
                f"Paused. {len(self._compounds)} compounds in pool; review and decide next stage."
            ), None

        if tool == "abandon_compound":
            cid = action.target_compound_id
            if cid and cid in self._compounds:
                del self._compounds[cid]
                self._compound_order = [c for c in self._compound_order if c != cid]
                if self._state.advanced_compound_id == cid:
                    self._state.advanced_compound_id = None
                return 0.0, {"abandoned": cid}, f"Abandoned compound {cid}.", None
            return 0.0, {"error": "no_such_compound"}, "No such compound to abandon.", None

        if tool == "advance_stage":
            return self._handle_advance_stage(action, params)

        return 0.0, {"error": "unhandled"}, f"Unhandled tool '{tool}'.", None

    # ------------------------------------------------------------------
    # Stage advancement (the core of long-horizon control)
    # ------------------------------------------------------------------

    def _handle_advance_stage(self, action: DrugDiscoveryAction, params: Dict[str, Any]):
        current = self._state.stage
        if current == "target_selection":
            target = params.get("target")
            if not target:
                return 0.0, {"error": "no_target"}, "advance_stage from target_selection requires params.target.", None
            self._state.selected_target = target
            self._stages_completed.append("target_selection")
            self._state.stage = "hit_id"
            return 0.0, {"selected_target": target, "new_stage": "hit_id"}, (
                f"Selected target '{target}'. Advanced to hit_id."
            ), None

        if current == "hit_id":
            if not self._compounds:
                return 0.0, {"error": "no_compounds"}, "Cannot advance from hit_id without any compounds.", None
            self._stages_completed.append("hit_id")
            self._state.stage = "hit_to_lead"
            return 0.0, {"new_stage": "hit_to_lead"}, "Advanced to hit_to_lead.", None

        if current == "hit_to_lead":
            self._stages_completed.append("hit_to_lead")
            self._state.stage = "admet"
            return 0.0, {"new_stage": "admet"}, "Advanced to admet.", None

        if current == "admet":
            without_admet = [c["id"] for c in self._compounds.values() if not c.get("admet")]
            if not any(c.get("admet") for c in self._compounds.values()):
                return 0.0, {"error": "no_admet"}, "Cannot advance from admet without ADMET data on any compound.", None
            self._stages_completed.append("admet")
            self._state.stage = "lead_validation"
            note = ""
            if without_admet:
                note = f" ({len(without_admet)} compound(s) untested)"
            return 0.0, {"new_stage": "lead_validation", "untested_compounds": without_admet}, (
                f"Advanced to lead_validation{note}."
            ), None

        if current == "lead_validation":
            cid = action.target_compound_id or params.get("compound_id")
            if not cid or cid not in self._compounds:
                return 0.0, {"error": "no_lead"}, (
                    "advance_stage from lead_validation requires target_compound_id of the nominated lead."
                ), None
            self._state.advanced_compound_id = cid
            self._stages_completed.append("lead_validation")
            self._state.stage = "finished"
            return 0.0, {"nominated_lead": cid, "new_stage": "finished"}, (
                f"Nominated {cid} as the lead compound. Campaign complete."
            ), "completed"

        return 0.0, {"error": "already_finished"}, "Campaign already finished.", "completed"

    # ------------------------------------------------------------------
    # Compound bookkeeping
    # ------------------------------------------------------------------

    def _new_id(self) -> str:
        self._next_compound_idx += 1
        return f"C{self._next_compound_idx:03d}"

    def _add_compound(
        self,
        smiles: str,
        origin: str = "chembl",
        parent_id: Optional[str] = None,
        history: Optional[str] = None,
    ) -> str:
        for cid, c in self._compounds.items():
            if c["smiles"] == smiles:
                if history:
                    c["history"].append(history)
                return cid
        cid = self._new_id()
        self._compounds[cid] = {
            "id": cid,
            "smiles": smiles,
            "origin": origin,
            "parent_id": parent_id,
            "binding_affinity_nM": None,
            "admet": None,
            "docking_score": None,
            "history": [history] if history else [],
        }
        self._compound_order.append(cid)
        return cid

    def _upsert_compound(self, smiles: str, cid: Optional[str], **fields: Any) -> str:
        if cid and cid in self._compounds:
            self._compounds[cid].update({k: v for k, v in fields.items() if v is not None})
            return cid
        new_id = self._add_compound(smiles)
        self._compounds[new_id].update({k: v for k, v in fields.items() if v is not None})
        return new_id

    def _smiles_for(self, cid: Optional[str]) -> Optional[str]:
        if not cid:
            return None
        c = self._compounds.get(cid)
        return c["smiles"] if c else None

    def _active_compounds(self) -> List[Dict[str, Any]]:
        return [self._compounds[cid] for cid in self._compound_order if cid in self._compounds]

    # ------------------------------------------------------------------
    # Warning compliance
    # ------------------------------------------------------------------

    def _was_warning_ignored(
        self, action: DrugDiscoveryAction, prior_blocks: List[Dict[str, str]]
    ) -> int:
        if not prior_blocks:
            return 0
        ignored = 0
        for w in prior_blocks:
            if w["agent"] == "toxicologist" and action.tool in ("advance_stage", "run_docking"):
                ignored += 1
            elif w["agent"] == "budget" and action.tool in ("run_docking", "predict_binding_affinity"):
                ignored += 1
        return ignored

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        done: bool,
        reward: Optional[float],
        subagent_messages: Optional[List[Dict[str, str]]] = None,
    ) -> DrugDiscoveryObservation:
        return DrugDiscoveryObservation(
            done=done,
            reward=reward,
            stage=self._state.stage,
            step_index=self._state.step_count,
            max_steps=self._state.max_steps,
            disease=self._state.disease,
            selected_target=self._state.selected_target,
            budget_total=self._state.budget_total,
            budget_remaining=self._state.budget_remaining,
            last_tool_cost=self._last_cost,
            last_tool=self._last_tool,
            last_result=self._last_result,
            active_compounds=self._active_compounds(),
            advanced_compound_id=self._state.advanced_compound_id,
            subagent_messages=subagent_messages or [],
            message=self._last_message,
        )
