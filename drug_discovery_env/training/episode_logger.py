"""Per-episode JSONL trace logger.

One JSONL file per episode, one line per agent turn. Captures everything we
need to inspect deep reasoning behaviour:

  - rendered observation handed to the policy
  - raw model output (full text)
  - parsed action (tool, params — including the literature `query` the agent
    learned to ask, target_compound_id, reasoning, evidence_ids)
  - environment response: tool result summary, sub-agent messages, reward
    breakdown, budget remaining
  - per-step `process` / `reasoning_depth` snapshots from the env

Used identically by `train.py`, `evaluate.py`, and `infer.py` so trace format
is uniform across modes.
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
    model_raw_output: str
    rendered_observation: str
    tool_result: Dict[str, Any]
    subagent_messages: List[Dict[str, Any]]
    reward_breakdown: Dict[str, float]
    budget_remaining: float
    budget_total: float
    last_tool_cost: float
    done: bool
    reward: float
    timestamp: str = field(default_factory=_now)


class EpisodeLogger:
    def __init__(self, log_dir: str | Path, run_id: Optional[str] = None) -> None:
        self.run_id = run_id or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
        self.root = resolve_path(log_dir) / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._episodes: Dict[str, Path] = {}
        self._summary_path = self.root / "runs.csv"
        if not self._summary_path.exists():
            self._summary_path.write_text(
                "episode_id,disease,steps,terminal_reward,total_reward,stage_completed,"
                "budget_remaining_frac,oversight_violations,terminated_reason\n",
                encoding="utf-8",
            )

    def start_episode(self, disease: str, episode_id: Optional[str] = None) -> str:
        eid = episode_id or f"{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
        path = self.root / f"{eid}.jsonl"
        with self._lock:
            self._episodes[eid] = path
        header = {
            "type": "episode_start",
            "episode_id": eid,
            "disease": disease,
            "timestamp": _now(),
        }
        path.write_text(json.dumps(header) + "\n", encoding="utf-8")
        return eid

    def log_turn(self, episode_id: str, record: TurnRecord) -> None:
        path = self._episodes.get(episode_id)
        if path is None:
            raise KeyError(f"unknown episode_id {episode_id!r}; call start_episode first")
        payload = {"type": "turn", "episode_id": episode_id, **record.__dict__}
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

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
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        path = self._episodes.get(episode_id)
        if path is None:
            return
        footer = {
            "type": "episode_end",
            "episode_id": episode_id,
            "disease": disease,
            "steps": steps,
            "terminal_reward": terminal_reward,
            "total_reward": total_reward,
            "stage_completed": stage_completed,
            "budget_remaining_frac": budget_remaining_frac,
            "oversight_violations": oversight_violations,
            "terminated_reason": terminated_reason,
            "extras": extras or {},
            "timestamp": _now(),
        }
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(footer, default=str) + "\n")
            with self._summary_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{episode_id},{json.dumps(disease)},{steps},{terminal_reward:.4f},"
                    f"{total_reward:.4f},{int(stage_completed)},"
                    f"{budget_remaining_frac:.4f},{oversight_violations},"
                    f"{terminated_reason}\n"
                )
