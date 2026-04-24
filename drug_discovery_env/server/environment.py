from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from drug_discovery_env.agents import BudgetManagerAgent, ChemistAgent, OversightAgent, ToxicologistAgent
from drug_discovery_env.config.settings import Settings, get_settings
from drug_discovery_env.core.action_parser import ActionParser
from drug_discovery_env.core.budget_manager import BudgetManager
from drug_discovery_env.core.models import DrugDiscoveryAction, DrugDiscoveryObservation, ToolProvenance
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


class DrugDiscoveryEnv:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.provider = build_data_provider(self.settings)
        self.topology = TopologyEngine(self.settings)
        self.stage_manager = StageManager(self.settings)
        self.budget_manager = BudgetManager(self.settings)
        self.reward_engine = RewardEngine(self.settings)

        self.tools = {
            "select_target": SelectTargetTool(self.settings, self.provider),
            "search_compounds": SearchCompoundsTool(self.provider),
            "predict_affinity": PredictAffinityTool(self.settings, self.topology),
            "evaluate_admet": EvaluateAdmetTool(),
            "modify_molecule": ModifyMoleculeTool(self.settings),
            "synthesize": SynthesizeTool(self.settings),
            "validate_compound": ValidateCompoundTool(),
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
        self.state: GameState | None = None

    def reset(self, disease: str = "Type 2 Diabetes") -> DrugDiscoveryObservation:
        self.state = GameState(
            disease=disease,
            stage=1,
            step=0,
            budget_initial=self.settings.budget.initial_credits,
            budget_remaining=self.settings.budget.initial_credits,
            pathway_graph={
                "INSR": ["PI3K", "AKT1"],
                "PI3K": ["MTOR", "AKT1"],
                "AKT1": ["MTOR", "FOXO3"],
            },
            disease_nodes=["INSR", "PI3K", "AKT1"],
        )
        return DrugDiscoveryObservation(state_summary=summarize_state(self.state))

    def _ensure_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        return self.state

    def _run_sub_agents(self, state: GameState) -> dict[str, list[str]]:
        messages = {
            "toxicologist": self.toxicologist.run(state),
            "chemist": self.chemist.run(state),
            "budget_manager": self.budget_agent.run(state),
            "oversight": self.oversight.run(state),
        }
        state.sub_agent_inbox = messages
        return messages

    def step(self, action: DrugDiscoveryAction | str) -> DrugDiscoveryObservation:
        state = self._ensure_state()
        parsed = self.parser.parse(action) if isinstance(action, str) else action

        tool = self.tools[parsed.tool]
        result = tool.execute(state, parsed.params)

        information_gain = float(result.get("count", 1)) / 20 if isinstance(result, dict) else 0.2
        if parsed.tool in {"evaluate_admet", "predict_affinity", "validate_compound"}:
            information_gain = max(information_gain, 0.6)
        uncertainty = sum(c.uncertainty for c in state.compound_ledger.values()) / max(1, len(state.compound_ledger))
        self.budget_manager.pay(state, parsed.tool, information_gain, uncertainty)

        state.step += 1
        self.stage_manager.update_stage(state)
        reward_breakdown = self.reward_engine.compute(state, parsed)
        messages = self._run_sub_agents(state)

        done = state.step >= self.settings.app.max_steps or state.budget_remaining <= 0 or state.stage >= 5
        source = result.get("provenance", {}).get("source", result.get("source", "simulation")) if isinstance(result, dict) else "simulation"
        confidence = (
            result.get("provenance", {}).get("confidence", 0.6)
            if isinstance(result, dict)
            else 0.6
        )

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
            info={"stage": state.stage, "step": state.step},
        )
