"""Per-run episode trace logger.

All episodes for a given training/eval/inference run share **one** JSONL file
(`episodes.jsonl`) plus a one-line-per-episode summary CSV (`runs.csv`). Each
JSONL line is one of three record types:

  - ``episode_start`` — episode_id, disease, optional model completion (single
    line of plain text — no nested raw blobs)
  - ``turn``         — compact per-turn record (tool, params, reasoning,
    reward breakdown, budget). No duplicated "raw output" / "rendered
    observation" / sub-agent message blobs.
  - ``episode_end``  — terminal/total reward, stage_completed, nominated
    compound id + SMILES, terminated reason.

The same logger object is used by `train.py`, `evaluate.py`, and `infer.py`,
so trace format is uniform across modes.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from drug_discovery_env.config.runtime import resolve_path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TurnRecord:
    step: int
    stage: str
    tool: str
    params: Dict[str, Any]
    target_compound_id: Optional[str]
    reasoning: str
    evidence_ids: List[str]
    reward_breakdown: Dict[str, float]
    budget_remaining: float
    budget_total: float
    last_tool_cost: float
    done: bool
    reward: float
    timestamp: str = field(default_factory=_now)


_CSV_HEADER = (
    "episode_id,disease,steps,terminal_reward,total_reward,stage_completed,"
    "budget_remaining_frac,oversight_violations,nominated_compound_id,"
    "nominated_smiles,terminated_reason\n"
)


class EpisodeLogger:
    """Single-file JSONL logger shared by training / eval / inference."""

    def __init__(self, log_dir: str | Path, run_id: Optional[str] = None) -> None:
        self.run_id = run_id or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
        self.root = resolve_path(log_dir) / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._episodes_path = self.root / "episodes.jsonl"
        self._summary_path = self.root / "runs.csv"
        self._open: set[str] = set()
        if not self._episodes_path.exists():
            self._episodes_path.touch()
        if not self._summary_path.exists():
            self._summary_path.write_text(_CSV_HEADER, encoding="utf-8")

    @property
    def episodes_path(self) -> Path:
        return self._episodes_path

    def _append(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, default=str, ensure_ascii=False)
        with self._lock, self._episodes_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def start_episode(
        self,
        disease: str,
        episode_id: Optional[str] = None,
        model_completion: str = "",
    ) -> str:
        eid = episode_id or f"{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._open.add(eid)
        header: Dict[str, Any] = {
            "type": "episode_start",
            "episode_id": eid,
            "disease": disease,
            "timestamp": _now(),
        }
        if model_completion:
            header["model_completion"] = model_completion
        self._append(header)
        return eid

    def log_turn(self, episode_id: str, record: TurnRecord) -> None:
        if episode_id not in self._open:
            raise KeyError(f"unknown episode_id {episode_id!r}; call start_episode first")
        payload = {"type": "turn", "episode_id": episode_id, **record.__dict__}
        self._append(payload)

    def end_episode(
        self,
        episode_id: str,
        *,
        disease: str,
        steps: int,
        terminal_reward: float,
        total_reward: float,
        stage_completed: bool,
        budget_remaining_frac: float,
        oversight_violations: int,
        terminated_reason: str,
        nominated_compound_id: Optional[str] = None,
        nominated_smiles: Optional[str] = None,
        nominated_admet: Optional[Dict[str, Any]] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        if episode_id not in self._open:
            return
        with self._lock:
            self._open.discard(episode_id)

        footer: Dict[str, Any] = {
            "type": "episode_end",
            "episode_id": episode_id,
            "disease": disease,
            "steps": steps,
            "terminal_reward": terminal_reward,
            "total_reward": total_reward,
            "stage_completed": stage_completed,
            "budget_remaining_frac": budget_remaining_frac,
            "oversight_violations": oversight_violations,
            "nominated_compound_id": nominated_compound_id,
            "nominated_smiles": nominated_smiles,
            "nominated_admet": nominated_admet or {},
            "terminated_reason": terminated_reason,
            "extras": extras or {},
            "timestamp": _now(),
        }
        self._append(footer)

        compound_id_csv = nominated_compound_id or ""
        smiles_csv = (nominated_smiles or "").replace(",", " ").replace("\n", " ")
        with self._lock, self._summary_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{episode_id},{json.dumps(disease)},{steps},{terminal_reward:.4f},"
                f"{total_reward:.4f},{int(stage_completed)},"
                f"{budget_remaining_frac:.4f},{oversight_violations},"
                f"{compound_id_csv},{json.dumps(smiles_csv)},"
                f"{terminated_reason}\n"
            )
