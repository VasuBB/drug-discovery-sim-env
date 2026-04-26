"""Evaluation entrypoint — runs a trained checkpoint over the cached test split.

For each held-out disease the agent runs one full campaign over HTTP against
the env server. We collect the env's reward breakdown plus a richer panel
(stage completion, ADMET pass, oversight violations, budget remaining, mean
reasoning depth, ChEMBL Tanimoto-to-known-drug similarity, precision@1) and
write a per-disease JSONL plus an aggregate `report.json`.

Usage:
    python -m drug_discovery_env.scripts.evaluate \
        --checkpoint outputs/grpo \
        [--limit 100] [--config path/to/override.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from drug_discovery_env.config.runtime import load_settings, resolve_path
from drug_discovery_env.core.models import STAGE_ORDER
from drug_discovery_env.data_provider.dataset import DiseaseRow, load_dataset
from drug_discovery_env.rewards.reasoning import ReasoningReward
from drug_discovery_env.training.episode_logger import EpisodeLogger
from drug_discovery_env.training.metrics import (
    DiseaseMetricRecord,
    aggregate_report,
    best_tanimoto,
    mean_tanimoto,
    write_per_disease,
)
from drug_discovery_env.training.policy import TransformersToolPolicy
from drug_discovery_env.training.rollout import run_episode


def _evaluate_disease(
    *,
    row: DiseaseRow,
    policy: TransformersToolPolicy,
    base_url: str,
    logger: EpisodeLogger,
    max_turns: int,
    radius: int,
    n_bits: int,
    pass_threshold: float,
    reasoning_scorer: ReasoningReward,
) -> DiseaseMetricRecord:
    result = run_episode(
        base_url=base_url,
        disease=row.disease,
        generate_text=policy.generate_text,
        logger=logger,
        max_turns=max_turns,
    )

    admet = result.nominated_admet or {}
    admet_pass = bool(admet.get("ro5_pass") and not admet.get("pains") and not admet.get("tox_flag"))
    reasoning_traces = [a.reasoning for a in result.turn_actions if a.reasoning]
    mean_reasoning_depth = reasoning_scorer.score(reasoning_traces) if reasoning_traces else 0.0

    smiles = result.nominated_smiles or ""
    refs = list(row.known_drugs or [])
    t_max = best_tanimoto(smiles, refs, radius=radius, n_bits=n_bits) if smiles and refs else 0.0
    t_mean = mean_tanimoto(smiles, refs, radius=radius, n_bits=n_bits) if smiles and refs else 0.0
    p_at_1 = 1.0 if t_max >= pass_threshold else 0.0

    return DiseaseMetricRecord(
        disease=row.disease,
        target=row.target,
        nominated_smiles=smiles or None,
        terminal_reward=result.terminal_reward,
        total_reward=result.total_reward,
        reward_breakdown=result.reward_breakdown,
        final_stage_idx=result.final_stage_idx,
        stage_completed=result.stage_completed,
        budget_remaining_frac=result.budget_remaining_frac,
        admet_pass=admet_pass,
        oversight_violations=result.oversight_violations,
        mean_reasoning_depth=mean_reasoning_depth,
        tanimoto_to_known_max=t_max,
        tanimoto_to_known_mean=t_mean,
        precision_at_1=p_at_1,
        n_known_drugs=len(refs),
        steps=result.steps,
        terminated_reason=result.terminated_reason,
        extras={"nominated_id": result.nominated_compound_id},
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Server-client evaluator")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap test diseases (debug)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    base_url = args.base_url or settings.training.base_url
    eval_cfg = settings.evaluation

    dataset = load_dataset(settings.dataset.cache_path)
    test_rows = dataset.test
    if args.limit is not None:
        test_rows = test_rows[: args.limit]
    if not test_rows:
        raise RuntimeError("Test split is empty; re-run prepare_dataset.")

    sys.stdout.reconfigure(line_buffering=True)  # ensure live progress on Kaggle/Colab

    print("=" * 78, flush=True)
    print(
        f"[evaluate] checkpoint   : {args.checkpoint}\n"
        f"[evaluate] base_url     : {base_url}\n"
        f"[evaluate] test diseases: {len(test_rows)}"
        f"  (limit={args.limit if args.limit is not None else 'none'})\n"
        f"[evaluate] max turns/ep : {settings.training.max_turns_per_episode}\n"
        f"[evaluate] morgan       : radius={eval_cfg.morgan_radius} n_bits={eval_cfg.morgan_n_bits}",
        flush=True,
    )
    print("=" * 78, flush=True)

    t_load = time.time()
    policy = TransformersToolPolicy(
        args.checkpoint,
        device=args.device,
        max_new_tokens=settings.training.max_new_tokens_per_turn,
        temperature=settings.training.generation_temperature,
        top_p=settings.training.generation_top_p,
    )
    print(f"[evaluate] policy loaded in {time.time() - t_load:.1f}s", flush=True)

    log_dir = resolve_path(eval_cfg.output_dir) / "logs"
    logger = EpisodeLogger(log_dir, run_id=args.run_id)
    reasoning_scorer = ReasoningReward()

    records: List[DiseaseMetricRecord] = []
    sum_terminal = 0.0
    sum_total = 0.0
    sum_tan_max = 0.0
    n_admet_pass = 0
    n_stage_complete = 0
    t_run_start = time.time()

    for idx, row in enumerate(test_rows, start=1):
        t0 = time.time()
        pct = 100.0 * (idx - 1) / max(1, len(test_rows))
        print(
            f"\n[evaluate] [{idx:>3}/{len(test_rows)}  {pct:5.1f}%]"
            f"  disease={row.disease!r}  target={row.target}",
            flush=True,
        )
        try:
            record = _evaluate_disease(
                row=row,
                policy=policy,
                base_url=base_url,
                logger=logger,
                max_turns=settings.training.max_turns_per_episode,
                radius=eval_cfg.morgan_radius,
                n_bits=eval_cfg.morgan_n_bits,
                pass_threshold=eval_cfg.tanimoto_pass_threshold,
                reasoning_scorer=reasoning_scorer,
            )
        except Exception as exc:  # don't let one disease kill the whole eval
            print(f"  [error] {type(exc).__name__}: {exc}", flush=True)
            continue

        records.append(record)
        sum_terminal += record.terminal_reward
        sum_total += record.total_reward
        sum_tan_max += record.tanimoto_to_known_max
        n_admet_pass += int(record.admet_pass)
        n_stage_complete += int(record.stage_completed)
        n = len(records)
        elapsed = time.time() - t0
        run_elapsed = time.time() - t_run_start
        eta = (run_elapsed / idx) * (len(test_rows) - idx)

        print(
            f"  this:    terminal={record.terminal_reward:+.3f}"
            f"  total={record.total_reward:+.3f}"
            f"  tanimoto_max={record.tanimoto_to_known_max:.3f}"
            f"  admet={'PASS' if record.admet_pass else 'fail'}"
            f"  stage={record.final_stage_idx}/{len(STAGE_ORDER)-1}"
            f"  steps={record.steps}  reason={record.terminated_reason or 'plan_end'}\n"
            f"  running: terminal={sum_terminal/n:+.3f}  total={sum_total/n:+.3f}"
            f"  tanimoto_max={sum_tan_max/n:.3f}"
            f"  admet_pass={n_admet_pass}/{n}  stage_complete={n_stage_complete}/{n}\n"
            f"  timing:  this={elapsed:.1f}s  total={run_elapsed:.1f}s  eta={eta:.0f}s",
            flush=True,
        )

    out_dir = resolve_path(eval_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_disease_path = write_per_disease(records, out_dir / "per_disease.jsonl")
    report = aggregate_report(records, per_disease_path)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n" + "=" * 78, flush=True)
    print(f"[evaluate] DONE  evaluated={len(records)}/{len(test_rows)}"
          f"  total_time={time.time() - t_run_start:.1f}s", flush=True)
    print(f"[evaluate] wrote {per_disease_path}", flush=True)
    print(f"[evaluate] wrote {report_path}", flush=True)
    print("=" * 78, flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
