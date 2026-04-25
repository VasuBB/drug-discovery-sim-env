# Write-up: Long-Horizon Multi-Agent Drug Discovery (OpenEnv)

> Short hackathon write-up. Replace the placeholder URLs with the real ones once the
> blog / video / slides are published.

## TL;DR

We built an OpenEnv environment that simulates a 50-step pharma research campaign
and trained an LLM (Qwen2.5) with **GRPO** (Group Relative Policy Optimization) to
play the role of a Project Lead. The agent has to take a disease through five
canonical stages -- target_selection &rarr; hit_id &rarr; hit_to_lead &rarr; admet &rarr;
lead_validation -- under a hard budget, advised by four rule-based sub-agents
(chemist, toxicologist, budget manager, oversight). The composable rubric
rewards a safe and potent terminal compound, efficient budget usage, scientific
reasoning depth, and compliance with sub-agent `block` warnings.

## Why this matters

Pharma R&D wastes billions on bad early-stage decisions: wrong target, toxic lead,
budget burned on dead-end assays. Training an LLM that **sequentially trades off
potency vs safety vs cost** with awareness of advisory feedback is a transferable
skill: the same loop applies to materials science, agricultural biotech, and any
multi-stage experimental design problem.

## What we built

- **OpenEnv-native environment** (`Environment / Action / Observation / State /
  HTTPEnvServer / EnvClient`), no reinvention -- see `drug_discovery_env/openenv_compat.py`.
- **5-stage pipeline** with explicit canonical names (`target_selection`,
  `hit_id`, `hit_to_lead`, `admet`, `lead_validation`).
- **8 OpenEnv MCP tools** backed by RDKit (Lipinski / PAINS / TOX SMARTS / hERG
  proxy / QED / docking surrogate) and a hybrid data provider that can run
  `live_only` (Open Targets + ChEMBL + PubMed), `local_only` (snapshots), or
  `hybrid` (live with snapshot fallback).
- **Rule-based multi-agent panel** emitting `info` / `warn` / `block` messages.
- **Composable rubric** -- terminal + process + reasoning + strategy + oversight
  penalty -- with a hard hERG safety floor.
- **Two GRPO trainers**: an in-process one (rolls out inside the trainer) and a
  live-rollout one that talks to a running OpenEnv server, both on TRL with
  optional Unsloth 4-bit loading.
- A **Colab notebook** that runs the full loop on a free T4.

## How we trained

- Group size 8, learning rate 5e-6, KL beta 0.04, clip epsilon 0.2.
- Reward functions are decomposed: `reward_total`, `reward_terminal`,
  `reward_process`, `reward_reasoning`, `reward_strategy`. TRL averages them.
- Dataset is generated from the 15-disease scenario library; each prompt seeds
  a fresh campaign.

## Results

Loss curve, reward curve, and baseline-vs-trained comparison are in
[`artifacts/training/`](artifacts/training/). Re-run with:

```bash
python -m drug_discovery_env.scripts.run_training_experiment \
    --episodes 1 --device auto --model sshleifer/tiny-gpt2 \
    --max-train-steps 3 --out-dir artifacts/training
```

For a full Qwen2.5-3B run on Colab, use
[`notebooks/02_train_grpo_colab.ipynb`](notebooks/02_train_grpo_colab.ipynb).

## Lessons

1. **Severity-tiered sub-agents** (`info` / `warn` / `block`) are a more
   useful signal than a single text channel: they let us define a clean
   "compliance" reward component without leaking the policy.
2. **Variable cost multipliers** (low-information, high-uncertainty,
   late-stage redundancy) make a fixed tool catalogue *self-regulating* --
   the agent learns to stop spamming a tool when it stops yielding info.
3. **Composable rubric** is the most important design choice: it stops the
   model from gaming any single component (e.g. writing eloquent reasoning
   about a toxic compound).

## Links

- Hugging Face Space: `TODO_ADD_SPACE_URL`
- Hugging Face blog: `TODO_ADD_BLOG_URL`
- Demo video (<2 min): `TODO_ADD_VIDEO_URL`
- Slides: `TODO_ADD_SLIDES_URL`
- WandB run: `TODO_ADD_WANDB_RUN_URL`

(Do **not** commit large video binaries to the repo or HF Space package -- use
external links above.)
