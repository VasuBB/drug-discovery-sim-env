---
title: Drug Discovery Sim Env
emoji: 🧪
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Multi-disease drug discovery RL environment (OpenEnv + GRPO)
---

# Drug Discovery Simulation Environment

> **OpenEnv India Hackathon 2026 submission.** A multi-disease, long-horizon
> drug discovery RL environment with a GRPO training pipeline (HF TRL + optional
> Unsloth 4-bit). The agent — a "Project Lead" LLM — drives a simulated 50-step
> campaign per disease, choosing among 13 cheminformatics / literature /
> sub-agent tools. Trained on ~90 cached diseases and evaluated on a held-out
> test split, including ChEMBL Tanimoto similarity to known approved drugs for
> the same target.

## 🔗 Submission links (for judges)

| Resource | URL |
|---|---|
| 🤗 **Live env on Hugging Face Space** | <https://huggingface.co/spaces/vasuboda/drug-discovery-sim-env> |
| 🌐 **Live env runtime endpoint** | <https://vasuboda-drug-discovery-sim-env.hf.space> |
| ✍️ **Blog post (writeup)** | <https://huggingface.co/spaces/vasuboda/drug-discovery-sim-env/blob/main/blog.md> |
| 📓 **Reproducible Kaggle notebook (end-to-end)** | `https://www.kaggle.com/code/<YOUR_KAGGLE_USERNAME>/drug-discovery-grpo` |
| 🎥 **Video walkthrough (<2 min)** | `https://youtu.be/<YOUR_VIDEO_ID>` |

> The HF Space link and the live runtime endpoint above are the canonical
> references for the hackathon submission. Health-check the runtime with
> `curl https://vasuboda-drug-discovery-sim-env.hf.space/health`.

### How a judge can re-run everything

1. Open the **Kaggle notebook** above → "Run all". It clones this repo, installs
   deps, builds the dataset, boots the env, runs GRPO training (~50 optimizer
   steps × 32 episodes ≈ 1,600 rollouts), evaluates on the held-out split, and
   downloads a single zip with all artefacts (plots + report + traces).
2. Or hit the **HF Space** URL — the env server is already running and
   responds to `/health`, `/reset`, `/step`, etc.

## 📊 Training evidence (from the real Kaggle run)

The notebook produces three plots after training, all written to
`outputs/plots/`:

| Plot | What it shows |
|---|---|
| `training_loss.png` | GRPO loss per optimizer step (TRL `log_history`, `logging_steps=1`) |
| `training_reward.png` | Group mean reward ± 1 std per step |
| `episode_rewards.png` | Per-episode terminal & total reward across all ~1,600 rollouts (rolling-10 mean) |

> See `outputs/plots/` in the notebook bundle (zip) for the actual PNGs from
> the run we submitted.

## 🧬 What the environment models

The agent plans a full drug discovery campaign for a real disease (Open Targets
EFO id + ChEMBL-cached known drugs) across four stages:

```
target_validation -> hit_identification -> lead_optimization -> lead_validation
```

At each turn it picks one of 13 tools:

`select_target`, `search_chembl`, `search_compounds`, `search_literature`,
`predict_affinity`, `evaluate_admet`, `modify_molecule`, `validate_compound`
(docking), `synthesize`, `delegate_to_subagent` (chemist / toxicologist /
oversight), `request_subagent_summary`, `pause_and_review_all`, `advance_stage`,
`abandon_compound`.

Each tool has a budget cost; the env tracks budget, stage progression,
oversight violations, ADMET / PAINS / hERG flags, reasoning depth, and emits a
multi-component reward.

## ⚙️ Pipeline

```
                           +-------------------+
prepare_dataset --> data/diseases.jsonl <----+ scripts/train.py     (GRPO over HTTP)
                              |               + scripts/evaluate.py (held-out test split)
FastAPI env (server/app.py) <-+               + scripts/infer.py    (any unseen disease)
```

Three concerns are strictly separated:

1. **Dataset** — `data/diseases.jsonl` is built once by `prepare_dataset.py`.
   Disease, EFO id, top associated target, druggability, target class, ChEMBL
   known-drug SMILES (used by Tanimoto evaluation), deterministic
   `train`/`test` split.
2. **Environment** — FastAPI server in `drug_discovery_env/server/`. No
   training logic. Loads cached targets via `cached_provider.py`; proxies live
   ChEMBL / PubMed via `LiveAPIProvider`.
3. **Trainer / evaluator / inference** — three thin scripts under
   `drug_discovery_env/scripts/` that all talk to the env over HTTP. Every turn
   (rendered observation, raw model output, parsed action, tool result, reward
   breakdown, sub-agent messages, budget) is JSONL-logged by `EpisodeLogger`.

## 🎯 Reward (smooth, single-step formulas)

| Component | Form |
|-----------|------|
| `terminal_compound` | `sigmoid(w_p · (pIC50 - 6)) · exp(-herg) · (0.5 + 0.5·ro5_pass) · w_n / (1 + dup_count)` (hard hERG / PAINS floor zeroes) |
| `process` | `mean( info_gain / (1 + log1p(cost)) )` over the last 5 actions |
| `strategy` | `0.5·(1 - exp(-N_compounds/3)) + 0.5·(stage_idx / (S-1))` |
| `reasoning_depth` | `tanh(unique_concepts/6) · tanh(mean_len/200)` |
| `budget_efficiency` | `(remaining/total)^0.7` |
| `oversight_penalty` | `1 - exp(-ignored)` |
| `stage_progression` | `cleared / 4` |

All weights live in [`config/defaults.yaml`](config/defaults.yaml); the
aggregator is a single weighted sum minus oversight penalty.

## 🏋️ How training works (concrete example)

GRPO is single-turn from the LLM's POV: one prompt = one disease, one
completion = a whole JSON action plan that's replayed against the env (up to
50 tool calls inside) to compute a reward.

Per **optimizer step** (with the Kaggle defaults):

- 8 prompts (8 random diseases from the 90-disease train pool)
- × `group_size = 4` completions per prompt
- = **32 env episodes per step**
- GRPO computes group-relative advantages (no value network)
- One AdamW step on LoRA adapters (4-bit base via Unsloth when GPU allows)

50 optimizer steps × 32 episodes ≈ **1,600 episodes**, ~17 visits per disease.
See the *Training evidence* section above for the actual loss / reward curves.

## 🚀 Install & run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[training,test,chem]

# 1. Build the dataset (one-time; pages Open Targets + ChEMBL).
python -m drug_discovery_env.scripts.prepare_dataset --num-diseases 140

# 2. Start the env server.
uvicorn drug_discovery_env.server.app:app --host 0.0.0.0 --port 8000 &

# 3. Train. Checkpoints -> outputs/grpo, JSONL traces -> outputs/grpo/logs/<run-id>/.
python -m drug_discovery_env.scripts.train

# 4. Evaluate on the held-out test split.
python -m drug_discovery_env.scripts.evaluate --checkpoint outputs/grpo

# 5. Run inference on a brand-new disease.
python -m drug_discovery_env.scripts.infer --checkpoint outputs/grpo \
    --disease "Idiopathic pulmonary fibrosis"
```

All four commands accept `--config path/to/override.yaml`; CLI flags override
config values for everything that's commonly tweaked.

## 📈 Evaluation metrics

`scripts/evaluate.py` writes `outputs/eval/report.json` with:

- `mean_terminal_reward`, `mean_total_reward`, full reward breakdown
- `stage_completion_rate`, `mean_steps`, `mean_budget_remaining_frac`
- `admet_pass_rate` (RO5 ∧ ¬PAINS ∧ ¬tox)
- `oversight_violation_rate`, `mean_oversight_violations`
- `mean_reasoning_depth`
- `mean_tanimoto_to_known`, `mean_tanimoto_to_known_max`, `precision_at_1`
  (Tanimoto over Morgan fingerprints between the agent's nominated SMILES and
  the cached ChEMBL known drugs for that disease's target)

Per-disease records in `outputs/eval/per_disease.jsonl`.

## 🐳 Hugging Face Space (this README is the Space card)

The `Dockerfile` at the repo root builds a slim Python 3.11 image, installs
RDKit + FastAPI (no torch — Space only serves the env), and runs:

```bash
uvicorn drug_discovery_env.server.app:app --host 0.0.0.0 --port 7860
```

To deploy the Space yourself:

```bash
huggingface-cli login                     # paste a write token
git remote add hf https://huggingface.co/spaces/vasuboda/drug-discovery-sim-env
git push hf main
```

Once the build turns green, judges can hit:

```
https://vasuboda-drug-discovery-sim-env.hf.space/health
https://vasuboda-drug-discovery-sim-env.hf.space/reset
```

## 🧪 Tests

```bash
pytest -q
```

## 📜 License

MIT.
