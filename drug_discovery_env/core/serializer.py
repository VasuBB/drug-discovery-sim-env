from __future__ import annotations

from drug_discovery_env.core.state import GameState


def summarize_state(state: GameState) -> str:
    best = state.best_compound()
    best_text = "none"
    if best:
        best_text = (
            f"{best.smiles[:24]} potency={best.potency:.2f} safety={best.safety:.2f} "
            f"selectivity={best.selectivity:.2f}"
        )
    return (
        f"Disease={state.disease}; stage={state.stage}; step={state.step}; "
        f"budget={state.budget_remaining:.1f}/{state.budget_initial:.1f}; "
        f"compounds={len(state.compound_ledger)}; best={best_text}; "
        f"evidence={len(state.evidence_ledger)}"
    )
