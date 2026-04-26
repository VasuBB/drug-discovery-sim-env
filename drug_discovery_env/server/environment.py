"""DrugDiscoveryEnv — 50-step simulated drug discovery campaign.

OpenEnv Environment subclass implementing the MCP-style tool surface.

  - reset() starts a fresh campaign for an explicit disease provided by the caller
  - step() consumes one action: a tool call + reasoning + optional compound id
  - sub-agents (chemist, toxicologist, budget, oversight) review every step
    and emit severity-tagged messages (info/warn/block)
  - the campaign terminates when stages are cleared, budget hits zero, or the
    step cap is reached. Reward is sparse: step rewards are 0.0 until terminal.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from drug_discovery_env.agents import (
    BudgetManagerAgent,
    ChemistAgent,
    OversightAgent,
    ToxicologistAgent,
)
from drug_discovery_env.chemistry.rdkit_lab import LabSimulator
from drug_discovery_env.config.settings import Settings, get_settings
from drug_discovery_env.core.action_parser import ActionParser
from drug_discovery_env.core.budget_manager import BudgetManager
from drug_discovery_env.core.models import (
    STAGE_ORDER,
    DrugDiscoveryAction,
    DrugDiscoveryObservation,
    DrugDiscoveryState,
    ToolProvenance,
)
from drug_discovery_env.core.serializer import summarize_state
from drug_discovery_env.core.stage_manager import StageManager
from drug_discovery_env.core.state import GameState
from drug_discovery_env.core.topology import TopologyEngine
from drug_discovery_env.data_provider import build_data_provider
from drug_discovery_env.openenv_compat import Environment
from drug_discovery_env.retrieval.hybrid import HybridRetriever
from drug_discovery_env.rewards.aggregator import RewardEngine
from drug_discovery_env.tools import (
    AbandonCompoundTool,
    AdvanceStageTool,
    DelegateToSubagentTool,
    EvaluateAdmetTool,
    ModifyMoleculeTool,
    PauseAndReviewAllTool,
    PredictAffinityTool,
    RequestSubagentSummaryTool,
    SearchCompoundsTool,
    SearchLiteratureTool,
    SelectTargetTool,
    SynthesizeTool,
    ValidateCompoundTool,
)


class DrugDiscoveryEnv(Environment[DrugDiscoveryAction, DrugDiscoveryObservation, DrugDiscoveryState]):
    """50-step simulated drug discovery research campaign."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, settings: Optional[Settings] = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.lab = LabSimulator()
        self.provider = build_data_provider(self.settings)
        self.topology = TopologyEngine(self.settings)
        self.stage_manager = StageManager(self.settings)
        self.budget_manager = BudgetManager(self.settings)
        self.reward_engine = RewardEngine(self.settings)

        self.tools = {
            "select_target": SelectTargetTool(self.settings, self.provider),
            "search_compounds": SearchCompoundsTool(self.provider),
            "predict_affinity": PredictAffinityTool(self.settings, self.topology, lab=self.lab),
            "evaluate_admet": EvaluateAdmetTool(lab=self.lab),
            "modify_molecule": ModifyMoleculeTool(self.settings, lab=self.lab),
            "synthesize": SynthesizeTool(self.settings),
            "validate_compound": ValidateCompoundTool(lab=self.lab),
            "search_literature": SearchLiteratureTool(
                self.provider,
                retriever_factory=lambda docs: HybridRetriever(self.settings, docs),
            ),
            "advance_stage": AdvanceStageTool(self.settings, self.stage_manager),
            "abandon_compound": AbandonCompoundTool(),
            "pause_and_review_all": PauseAndReviewAllTool(),
            "delegate_to_subagent": DelegateToSubagentTool(),
            "request_subagent_summary": RequestSubagentSummaryTool(),
        }

        self.parser = ActionParser(set(self.tools.keys()))
        self.toxicologist = ToxicologistAgent(self.settings)
        self.chemist = ChemistAgent()
        self.budget_agent = BudgetManagerAgent()
        self.oversight = OversightAgent(self.settings)
        self._game_state: Optional[GameState] = None
        self._episode_id: Optional[str] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        disease: Optional[str] = None,
        *,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DrugDiscoveryObservation:
        if disease is None:
            disease = kwargs.get("disease")
        if not disease:
            raise ValueError("disease is required when running in live_only mode")

        max_steps = int(kwargs.get("max_steps") or self.settings.app.max_steps)
        budget = float(kwargs.get("budget") or self.settings.budget.initial_credits)

        gs = GameState(
            disease=disease,
            stage="target_selection",
            step=0,
            max_steps=max_steps,
            budget_initial=budget,
            budget_remaining=budget,
            pathway_graph={},
            disease_nodes=[],
        )
        gs.last_message = (
            f"New campaign: find a lead compound for {disease}. "
            f"Budget {budget:.0f}, max {max_steps} steps. Begin with stage 'target_selection'."
        )
        self._game_state = gs
        self._episode_id = episode_id or str(uuid.uuid4())

        return self._build_observation(done=False, reward=None, subagent_messages=[])

    def step(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DrugDiscoveryObservation:
        del timeout_s, kwargs
        state = self._ensure_state()

        parsed: DrugDiscoveryAction = self.parser.parse(action)

        # Bookkeeping: capture reasoning trace before tool dispatch
        if parsed.reasoning:
            state.reasoning_traces.append(parsed.reasoning)

        prior_blocks = list(state.open_block_warnings)
        ignored = self._was_warning_ignored(parsed, prior_blocks)
        if ignored:
            state.warnings_ignored += ignored

        state.step += 1

        # Tool dispatch — unknown tool returns a structured error and zero cost
        tool = self.tools.get(parsed.tool)
        if tool is None:
            result_payload: Dict[str, Any] = {
                "error": "unknown_tool",
                "tool": parsed.tool,
            }
            cost = 0.0
            message = f"Rejected: unknown tool '{parsed.tool}'."
        else:
            try:
                result_payload = tool.execute(state, dict(parsed.params or {}))
            except Exception as exc:  # tool failures shouldn't kill the rollout
                result_payload = {"error": "tool_exception", "message": str(exc)}
            information_gain = self._estimate_information_gain(tool, result_payload)
            uncertainty = self._mean_uncertainty(state)
            cost = self.budget_manager.pay(state, parsed.tool, information_gain, uncertainty)
            message = f"Executed '{parsed.tool}'."

        state.last_tool = parsed.tool
        state.last_result = result_payload if isinstance(result_payload, dict) else {"value": result_payload}
        state.last_cost = cost
        state.last_message = message

        # Termination check — agent's advance_stage may have set terminated_reason
        terminated_reason = state.terminated_reason
        if terminated_reason is None and state.budget_remaining <= 0:
            terminated_reason = "budget_exhausted"
        if terminated_reason is None and state.step >= state.max_steps:
            terminated_reason = "max_steps"
        done = terminated_reason is not None

        # Sub-agent panel — regenerate every step
        msgs: List[Dict[str, str]] = []
        msgs.extend(self.chemist.run(state))
        msgs.extend(self.toxicologist.run(state))
        msgs.extend(self.budget_agent.run(state))
        msgs.extend(self.oversight.run(state))

        state.warnings_issued += sum(1 for m in msgs if m.get("severity") in ("warn", "block"))
        state.open_block_warnings = [m for m in msgs if m.get("severity") == "block"]
        state.sub_agent_inbox = self._inbox_by_agent(msgs)

        # Reward — terminal sums every component; non-terminal exposes the
        # process/strategy/reasoning breakdown for shaping (reward = 0.0).
        breakdown = self.reward_engine.compute(state, parsed, terminal=done)
        reward: Optional[float] = breakdown.total if done else 0.0
        if done:
            state.terminated_reason = terminated_reason
            state.last_message = (
                f"Campaign ended ({terminated_reason}). Reward={reward:.3f}. "
                f"Breakdown: {breakdown.to_dict()}"
            )
            state.last_result = {**state.last_result, "reward_breakdown": breakdown.to_dict()}

        obs = self._build_observation(
            done=done,
            reward=reward,
            subagent_messages=msgs,
            reward_breakdown=breakdown,
        )
        return obs

    @property
    def state(self) -> DrugDiscoveryState:
        gs = self._ensure_state()
        best = gs.best_compound()
        best_score = 0.0
        if best is not None:
            best_score = (best.potency + best.selectivity + best.safety + best.developability) / 4
        return DrugDiscoveryState(
            episode_id=self._episode_id,
            step_count=gs.step,
            disease=gs.disease,
            selected_target=gs.selected_target,
            max_steps=gs.max_steps,
            budget_total=gs.budget_initial,
            budget_remaining=gs.budget_remaining,
            stage=gs.stage,
            compounds_tested=len(gs.compound_ledger),
            best_score=best_score,
            advanced_compound_id=gs.advanced_compound_id,
            terminated_reason=gs.terminated_reason,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_state(self) -> GameState:
        if self._game_state is None:
            raise RuntimeError("Environment not reset")
        return self._game_state

    def _estimate_information_gain(self, tool: Any, result: Dict[str, Any]) -> float:
        base = float(getattr(tool, "default_information_gain", 0.4))
        if isinstance(result, dict):
            if result.get("error"):
                return max(0.05, base * 0.3)
            count = result.get("count")
            if isinstance(count, (int, float)) and count == 0:
                return max(0.05, base * 0.5)
        return base

    def _mean_uncertainty(self, state: GameState) -> float:
        if not state.compound_ledger:
            return 0.7  # high prior when nothing's been measured
        return sum(c.uncertainty for c in state.compound_ledger.values()) / len(state.compound_ledger)

    def _was_warning_ignored(
        self,
        action: DrugDiscoveryAction,
        prior_blocks: List[Dict[str, str]],
    ) -> int:
        if not prior_blocks:
            return 0
        ignored = 0
        for w in prior_blocks:
            agent = w.get("agent")
            if agent == "toxicologist" and action.tool in ("advance_stage", "validate_compound"):
                ignored += 1
            elif agent == "budget" and action.tool in ("validate_compound", "predict_affinity"):
                ignored += 1
        return ignored

    @staticmethod
    def _inbox_by_agent(msgs: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        out: Dict[str, List[Dict[str, str]]] = {}
        for m in msgs:
            agent = m.get("agent", "unknown")
            out.setdefault(agent, []).append(m)
        return out

    @staticmethod
    def _resolve_provenance(result: Dict[str, Any]) -> tuple[str, float]:
        if not isinstance(result, dict):
            return "simulation", 0.6
        if "provenance" in result and isinstance(result["provenance"], dict):
            prov = result["provenance"]
            return str(prov.get("source", "simulation")), float(prov.get("confidence", 0.6))
        if isinstance(result.get("hits"), list) and result["hits"]:
            first = result["hits"][0]
            return str(first.get("source", "simulation")), float(first.get("confidence", 0.6))
        if isinstance(result.get("ranked_docs"), list) and result["ranked_docs"]:
            first = result["ranked_docs"][0]
            return str(first.get("source", "simulation")), float(first.get("confidence", 0.6))
        return str(result.get("source", "simulation")), float(result.get("confidence", 0.6))

    def _build_observation(
        self,
        done: bool,
        reward: Optional[float],
        subagent_messages: List[Dict[str, str]],
        reward_breakdown: Any = None,
    ) -> DrugDiscoveryObservation:
        state = self._ensure_state()
        active = [
            {
                "id": c.id,
                "smiles": c.smiles,
                "origin": c.origin,
                "parent_id": c.parent_id,
                "potency": c.potency,
                "selectivity": c.selectivity,
                "safety": c.safety,
                "developability": c.developability,
                "uncertainty": c.uncertainty,
                "binding_affinity_nM": c.binding_affinity_nM,
                "docking_score": c.docking_score,
                "admet": c.admet,
                "history": list(c.history),
            }
            for c in state.compound_ledger.values()
        ]

        source, confidence = self._resolve_provenance(state.last_result)
        provenance = ToolProvenance(
            source=source,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=confidence,
        )

        return DrugDiscoveryObservation(
            done=done,
            reward=reward if reward is not None else 0.0,
            state_summary=summarize_state(state),
            stage=state.stage,
            step_index=state.step,
            max_steps=state.max_steps,
            disease=state.disease,
            selected_target=state.selected_target,
            budget_total=state.budget_initial,
            budget_remaining=state.budget_remaining,
            last_tool_cost=state.last_cost,
            last_tool=state.last_tool,
            last_result=state.last_result,
            tool_result=state.last_result,
            active_compounds=active,
            advanced_compound_id=state.advanced_compound_id,
            subagent_messages=subagent_messages,
            provenance=provenance,
            uncertainty={"global": self._mean_uncertainty(state)},
            reward_breakdown=reward_breakdown,
            info={"stage": state.stage, "step": state.step, "terminated_reason": state.terminated_reason},
            metadata={"episode_id": self._episode_id},
            message=state.last_message,
        )
