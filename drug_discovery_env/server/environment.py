from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from drug_discovery_env.openenv_compat import Environment

from drug_discovery_env.agents import BudgetManagerAgent, ChemistAgent, OversightAgent, ToxicologistAgent
from drug_discovery_env.chemistry import RDKitLab
from drug_discovery_env.config.settings import Settings, get_settings
from drug_discovery_env.core.action_parser import ActionParser
from drug_discovery_env.core.budget_manager import BudgetManager
from drug_discovery_env.core.models import (
    DrugDiscoveryAction,
    DrugDiscoveryObservation,
    DrugDiscoveryState,
    ToolProvenance,
)
from drug_discovery_env.core.scenarios import (
    find_scenario,
    sample_scenario,
    stage_index_to_name,
)
from drug_discovery_env.core.serializer import summarize_state
from drug_discovery_env.core.stage_manager import StageManager
from drug_discovery_env.core.state import GameState
from drug_discovery_env.core.topology import TopologyEngine
from drug_discovery_env.data_provider import build_data_provider
from drug_discovery_env.retrieval.hybrid import HybridRetriever
from drug_discovery_env.rewards.aggregator import RewardEngine
from drug_discovery_env.tools import (
    EvaluateAdmetTool,
    ModifyMoleculeTool,
    PredictAffinityTool,
    SearchCompoundsTool,
    SearchLiteratureTool,
    SelectTargetTool,
    SynthesizeTool,
    ValidateCompoundTool,
)


class DrugDiscoveryEnv(Environment[DrugDiscoveryAction, DrugDiscoveryObservation, DrugDiscoveryState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.provider = build_data_provider(self.settings)
        self.topology = TopologyEngine(self.settings)
        self.stage_manager = StageManager(self.settings)
        self.budget_manager = BudgetManager(self.settings)
        self.reward_engine = RewardEngine(self.settings)

        self.lab = RDKitLab()
        self.tools = {
            "select_target": SelectTargetTool(self.settings, self.provider),
            "search_compounds": SearchCompoundsTool(self.provider),
            "predict_affinity": PredictAffinityTool(self.settings, self.topology),
            "evaluate_admet": EvaluateAdmetTool(self.lab),
            "modify_molecule": ModifyMoleculeTool(self.settings, self.lab),
            "synthesize": SynthesizeTool(self.settings),
            "validate_compound": ValidateCompoundTool(self.lab),
            "search_literature": SearchLiteratureTool(
                self.provider,
                retriever_factory=lambda docs: HybridRetriever(self.settings, docs),
            ),
        }
        self.parser = ActionParser(set(self.tools.keys()))
        self.toxicologist = ToxicologistAgent(self.settings)
        self.chemist = ChemistAgent()
        self.budget_agent = BudgetManagerAgent()
        self.oversight = OversightAgent(self.settings)
        self._game_state: GameState | None = None
        self._open_block_warnings: list[dict[str, str]] = []

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> DrugDiscoveryObservation:
        disease = kwargs.get("disease")
        # Tolerate the legacy positional form `env.reset("Type 2 Diabetes")`
        # used by tests by treating a non-int seed as a disease name.
        if disease is None and isinstance(seed, str):
            disease = seed
            seed = None
        scenario = find_scenario(disease) if disease else None
        if scenario is None and disease:
            # Unknown disease -- use it as the campaign name but no scenario seed.
            scenario = sample_scenario(seed=seed)
        if scenario is None:
            scenario = sample_scenario(seed=seed)
        self._game_state = GameState(
            disease=scenario.disease,
            stage=1,
            step=0,
            budget_initial=self.settings.budget.initial_credits,
            budget_remaining=self.settings.budget.initial_credits,
            pathway_graph=dict(scenario.pathway_graph) or {
                "INSR": ["PI3K", "AKT1"],
                "PI3K": ["MTOR", "AKT1"],
                "AKT1": ["MTOR", "FOXO3"],
            },
            disease_nodes=list(scenario.disease_nodes) or ["INSR", "PI3K", "AKT1"],
        )
        # warning compliance counters surface through reward.oversight_penalty.
        self._game_state.uncertainty_estimates["warnings_issued"] = 0
        self._game_state.uncertainty_estimates["warnings_ignored"] = 0
        self._open_block_warnings: list[dict[str, str]] = []
        obs = DrugDiscoveryObservation(
            state_summary=summarize_state(self._game_state),
            done=False,
            reward=0.0,
            metadata={
                "episode_id": episode_id,
                "seed": seed,
                "disease": scenario.disease,
                "stage_name": stage_index_to_name(self._game_state.stage),
                "canonical_target": scenario.canonical_target,
                "alternative_targets": scenario.alternative_targets,
            },
        )
        return obs

    def _ensure_state(self) -> GameState:
        if self._game_state is None:
            raise RuntimeError("Environment not reset")
        return self._game_state

    def _run_sub_agents(self, state: GameState) -> dict[str, list[str]]:
        # Severity-aware messages from each rule-based sub-agent.
        all_msgs: list[dict[str, str]] = []
        all_msgs.extend(self.toxicologist.messages(state))
        all_msgs.extend(self.chemist.messages(state))
        all_msgs.extend(self.budget_agent.messages(state))
        all_msgs.extend(self.oversight.messages(state))

        warnings_count = sum(1 for m in all_msgs if m["severity"] in ("warn", "block"))
        state.uncertainty_estimates["warnings_issued"] = (
            int(state.uncertainty_estimates.get("warnings_issued", 0)) + warnings_count
        )
        # Hold over the new BLOCK warnings -- next step we check whether the
        # Project Lead acted on them anyway.
        self._open_block_warnings = [m for m in all_msgs if m["severity"] == "block"]

        # Group by agent for the legacy text-only inbox view used in summaries.
        grouped: dict[str, list[str]] = {
            "toxicologist": [],
            "chemist": [],
            "budget_manager": [],
            "oversight": [],
            "messages": [],
        }
        for m in all_msgs:
            agent = m["agent"]
            key = "budget_manager" if agent == "budget" else agent
            grouped.setdefault(key, []).append(f"[{m['severity']}] {m['message']}")
            grouped["messages"].append(f"[{agent}/{m['severity']}] {m['message']}")
        state.sub_agent_inbox = grouped
        return grouped

    def _record_ignored_blocks(self, parsed_action: DrugDiscoveryAction) -> int:
        """Count prior BLOCK warnings that the Project Lead ignored."""

        if not self._open_block_warnings:
            return 0
        ignored = 0
        risky_advance = parsed_action.tool in ("validate_compound",)
        risky_spend = parsed_action.tool in (
            "validate_compound",
            "predict_affinity",
            "synthesize",
        )
        for w in self._open_block_warnings:
            if w["agent"] == "toxicologist" and risky_advance:
                ignored += 1
            elif w["agent"] == "budget" and risky_spend:
                ignored += 1
            elif w["agent"] == "oversight" and risky_advance:
                ignored += 1
        return ignored

    @staticmethod
    def _resolve_provenance(result: dict[str, Any]) -> tuple[str, float]:
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

    def step(
        self,
        action: DrugDiscoveryAction | str,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> DrugDiscoveryObservation:
        _ = timeout_s, kwargs
        state = self._ensure_state()

        parsed_action = self.parser.parse(action) if isinstance(action, str) else action

        ignored = self._record_ignored_blocks(parsed_action)
        if ignored:
            state.uncertainty_estimates["warnings_ignored"] = (
                int(state.uncertainty_estimates.get("warnings_ignored", 0)) + ignored
            )

        tool = self.tools[parsed_action.tool]
        result = tool.execute(state, parsed_action.params)

        information_gain = float(result.get("count", 1)) / 20 if isinstance(result, dict) else 0.2
        if parsed_action.tool in {"evaluate_admet", "predict_affinity", "validate_compound"}:
            information_gain = max(information_gain, 0.6)
        uncertainty = sum(c.uncertainty for c in state.compound_ledger.values()) / max(1, len(state.compound_ledger))
        self.budget_manager.pay(state, parsed_action.tool, information_gain, uncertainty)

        state.step += 1
        self.stage_manager.update_stage(state)
        reward_breakdown = self.reward_engine.compute(state, parsed_action)
        messages = self._run_sub_agents(state)

        done = state.step >= self.settings.app.max_steps or state.budget_remaining <= 0 or state.stage >= 5
        source, confidence = self._resolve_provenance(result) if isinstance(result, dict) else ("simulation", 0.6)

        return DrugDiscoveryObservation(
            state_summary=summarize_state(state),
            tool_result=result,
            provenance=ToolProvenance(
                source=str(source),
                timestamp=datetime.now(timezone.utc).isoformat(),
                confidence=float(confidence),
            ),
            uncertainty={"global": uncertainty},
            sub_agent_messages=messages,
            reward_breakdown=reward_breakdown,
            done=done,
            reward=reward_breakdown.total,
            info={
                "stage": state.stage,
                "stage_name": stage_index_to_name(state.stage),
                "step": state.step,
                "ignored_blocks_this_step": ignored,
            },
            metadata={
                "state_stage": state.stage,
                "stage_name": stage_index_to_name(state.stage),
                "state_step": state.step,
            },
        )

    @property
    def state(self) -> DrugDiscoveryState:
        gs = self._ensure_state()
        best = gs.best_compound()
        best_score = 0.0
        if best is not None:
            best_score = (best.potency + best.selectivity + best.safety + best.developability) / 4
        return DrugDiscoveryState(
            episode_id=None,
            step_count=gs.step,
            stage=gs.stage,
            budget_remaining=gs.budget_remaining,
            compounds_tested=len(gs.compound_ledger),
            best_score=best_score,
        )
