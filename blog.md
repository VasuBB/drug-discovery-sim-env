# 🧬 Teaching an LLM to Run a Drug Discovery Campaign — From Scratch, With RL

> *What if an LLM wasn't just a chemistry assistant — but the **Project Lead** of an entire drug discovery campaign?*

---

## The Big Idea in 30 Seconds

We built **`drug-discovery-sim-env`**: a full multi-stage, budget-constrained drug discovery simulation environment where a  LLM agent acts as **Project Director** — selecting targets, querying databases, running virtual assays, managing a $1,000 credit budget, and ultimately nominating a novel drug candidate — all trained end-to-end with **Reinforcement Learning via GRPO**.

No human in the loop. No single-shot generation. **Real multi-step decision-making**, grounded in real disease data.

| What | How |
|------|-----|
| 🧪 **Environment** | OpenEnv-compatible FastAPI server |
| 🤖 **Agent** | Qwen2.5-0.5B-Instruct (4-bit, Unsloth) |
| 🎓 **Training** | TRL GRPO on a single Kaggle T4 GPU |
| 📚 **Data** | ~5,500 disease/target pairs (Open Targets + ChEMBL) |
| 🎯 **Objective** | Nominate a potent, novel, ADMET-safe compound in ≤50 steps |

---

## Why We Built This (The Problem With Existing Approaches)

Most "LLM + drug discovery" demos boil down to one of two things:

**❌ Pattern A — Single-shot generation:**
> "Generate a SMILES string for inhibiting EGFR."

There's no cost, no noise, no ADMET filter, no strategy. The LLM just hallucinates a molecule and the demo ends.

**❌ Pattern B — Toy tool-use:**
> "Call this calculator tool, then this lookup tool, done."

Short horizon, no budget pressure, no branching tradeoffs, no real environment state.

**Real pharmaceutical preclinical discovery is neither.**

It is a long, expensive, branching search campaign where:

- 🔬 **Partial observability** — assays are noisy, targets have uncertain biology, literature is incomplete
- 💸 **Hard budget** — every assay, docking run, and synthesis attempt costs real money
- 🚧 **Stage gates** — you can't validate a compound before passing Hit ID → Lead Opt → ADMET screening
- 🧑‍⚕️ **Oversight requirements** — the toxicologist sub-agent's flags must be respected, not ignored
- 🆕 **Novelty matters** — rediscovering a known drug is worth far less than finding a new chemotype

**Our environment models all five simultaneously** — and is small enough to train a 0.5B model on a free Kaggle GPU while being faithful enough to force the agent to genuinely *plan and decide*.

---

## Architecture: Three Decoupled Concerns

One of our core engineering choices was **strict separation of the dataset, the environment, and the training loop**. Mixing them is how RL pipelines rot.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   prepare_dataset.py ──► data/diseases.jsonl                   │
│                                  │                             │
│                    ┌─────────────┘                             │
│                    ▼                                            │
│       FastAPI Env Server (server/app.py)                       │
│               │  ▲  (HTTP/JSON)                                │
│               │  │                                             │
│    ┌──────────┴──┴──────────────────────────┐                  │
│    │  scripts/train.py     (GRPO training)  │                  │
│    │  scripts/evaluate.py  (held-out split) │                  │
│    │  scripts/infer.py     (any new disease)│                  │
│    └────────────────────────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1️⃣ The Dataset — Frozen Ground Truth

`prepare_dataset.py` queries **Open Targets** and **ChEMBL** once, then freezes the result as `data/diseases.jsonl`. Each record contains:

- Disease name + EFO identifier
- Top associated protein target + druggability score
- Target class (GPCR, kinase, nuclear receptor, etc.)
- ChEMBL known approved drug SMILES (for Tanimoto evaluation at test time)
- Deterministic `train` / `test` split flag

This "build once, train forever" approach means **training is reproducible and cheap** — no live API calls during the RL loop.

### 2️⃣ The Environment — A Real FastAPI Server

The environment runs as an independent HTTP service with no training logic. Per step it:

- Loads the cached disease target
- Resolves tool calls (with noise, costs, and stage-gating)
- Proxies live ChEMBL/PubMed queries if `LiveAPIProvider` is enabled
- Returns the next observation, reward breakdown, and sub-agent messages

This is what makes it genuinely **OpenEnv-shaped**: swap the trainer, keep the environment — or vice versa.

### 3️⃣ The Training Loop — GRPO Over HTTP

Three thin scripts each talk to the env via `DrugDiscoveryClient`. Every turn is logged as JSONL by `EpisodeLogger` — observation, raw model output, parsed action, tool result, reward breakdown, sub-agent messages, budget remaining. Post-hoc analysis is trivial.

---

## The 13-Tool Action Space

For each step, the agent receives a **rendered natural-language observation** (target info, stage, budget, recent history, sub-agent alerts) and emits a JSON action selecting one of 13 tools:

| Category | Tools | Cost |
|----------|-------|------|
| **Target Selection** | `select_target` | Low |
| **Search** | `search_compounds`, `search_literature` | Low–Medium |
| **Assays** | `predict_affinity`, `evaluate_admet` | Medium |
| **Optimization** | `modify_molecule`, `synthesize`, `validate_compound` | Medium–High |
| **Process** | `advance_stage`, `abandon_compound`, `pause_and_review_all` | Low |
| **Sub-agents** | `delegate_to_subagent`, `request_subagent_summary` | Low |

Each tool has a **credit cost**, **assay noise** (simulated biological variance), and **stage-gated availability** — you can't call `validate_compound` before advancing through Hit ID and Lead Opt.

The campaign ends when: budget exhausted · 50 steps reached · compound validated through all 4 stages.

---

## The Reward Function: Dense, Smooth, Multi-Objective

> **Design philosophy:** Every component is a smooth, bounded function with clear physical meaning — so GRPO sees a dense signal, not a sparse end-of-episode spike.

### Reward Components at a Glance

```
Total Reward = w₁·terminal_compound
             + w₂·process
             + w₃·strategy
             + w₄·reasoning_depth
             + w₅·budget_efficiency
             + w₆·stage_progression
             − oversight_penalty
```

#### 🔬 Terminal Compound (Potency × Safety × Novelty)

```
sigmoid(w_p · (pIC50 − 6)) · exp(−hERG_score) · (0.5 + 0.5·ro5_pass) · w_n / (1 + dup_count)
```

- **`sigmoid(pIC50 − 6)`** — smooth reward for IC50 below 1 µM; hard cliff below baseline
- **`exp(−hERG_score)`** — exponential penalty for cardiac toxicity; zeroes the entire term above threshold
- **`ro5_pass`** — drug-likeness (Lipinski) multiplier
- **`/ (1 + dup_count)`** — novelty divisor: rediscovering known compounds scores progressively less

#### 🔎 Process Information Gain

```
mean( info_gain / (1 + log1p(cost)) )  over last 5 actions
```

Rewards queries that actually *teach you something* relative to their cost. This prevents the agent from mindlessly spamming cheap tools.

#### 🗺️ Strategy (Breadth × Progress)

```
0.5 · (1 − exp(−N_compounds / 3)) + 0.5 · (stage_idx / (S−1))
```

Balances exploring multiple chemical scaffolds vs. advancing through stage gates.

#### 🧠 Reasoning Depth

```
tanh(unique_concepts / 6) · tanh(mean_len / 200)
```

Rewards the agent for producing distinct, sufficiently developed chains of thought — parsed from its raw text output. Unique concepts × adequate length. Neither padding nor one-liners score well.

#### 💰 Budget Efficiency

```
(remaining / total)^0.7
```

Concave function — finishing with budget left is rewarded, burning it all is penalized.

#### 🚨 Oversight Penalty

```
1 − exp(−n_ignored_flags)
```

Exponentially increasing penalty for ignoring sub-agent alerts (toxicologist, PAINS filter, off-target flags). One ignored flag is a small hit; repeatedly ignoring is devastating.

#### 🏁 Stage Progression

```
cleared_stages / 4
```

Linear credit for actually advancing through gates — ensures partial credit even if the campaign doesn't terminate.

> **All weights** live in [`config/defaults.yaml`](config/defaults.yaml) and can be overridden via layered YAML. Ablating is one line: set `w_process: 0.0` and watch the policy collapse into "spam validate_compound."

---

## Training on a Free Kaggle T4

We trained **Qwen2.5-0.5B-Instruct** in 4-bit via Unsloth using TRL's `GRPOTrainer` — the whole run fits on a single Kaggle T4 GPU (16 GB VRAM).

### Per Training Step

1. Sample **4 rollouts** from the current policy against the live env server over HTTP
2. Compute scalar episode rewards with the aggregator above
3. Run a GRPO policy update

### Key Hyperparameters

```yaml
model_name:                  Qwen/Qwen2.5-0.5B-Instruct
use_unsloth:                 true
load_in_4bit:                true
group_size:                  4
per_device_batch_size:       1
gradient_accumulation_steps: 8
max_prompt_length:           2048
max_completion_length:       384
learning_rate:               5.0e-6
beta:                        0.04      # KL coefficient
epsilon:                     0.20      # clip ratio
num_train_steps:             2000
```

The full run is end-to-end reproducible from the [Kaggle notebook](notebooks/kaggle_drug_discovery_grpo.ipynb):

```
clone repo → install deps → build dataset → start FastAPI env → GRPO train → evaluate → infer
```

---

## Results

Evaluated on the **held-out test split** (diseases the model never saw during training):

| Metric | Observation |
|--------|-------------|
| **Mean total reward** | Improved steadily over 2,000 steps |
| **Stage completion rate** | More campaigns reach Validation vs. baseline |
| **ADMET pass rate** | Higher fraction of nominated compounds pass RO5 ∧ ¬PAINS ∧ ¬tox |
| **Tanimoto to known drugs** | Stays in a useful middle band — not nonsense, not just regurgitating known drugs |
| **Oversight violation rate** | Drops as the exponential penalty takes effect |
| **Reasoning depth** | Longer, more concept-diverse chains of thought emerge |

> 📈 Loss/reward curves from the real Kaggle run are in `outputs/grpo/plots/`.
> 📊 Full eval breakdown is in `outputs/eval/report.json` and `outputs/eval/per_disease.jsonl`.

---

## Failure Modes We Caught (And Fixed)

Building RL environments is as much about breaking them as building them.

### ❌ "Spam validate_compound"
**What happened:** Early policies skipped directly to the most expensive validation tool because it had the highest immediate reward signal.

**Fix:** Stage-gating (you can't call validate_compound before advancing stages) + `budget_efficiency` penalizes reckless spending.

---

### ❌ Reasoning-Depth Gaming
**What happened:** The agent padded its chain-of-thought with repetitive filler text to inflate `mean_len`.

**Fix:** Combined length with **unique concept count** — `tanh(unique_concepts/6) · tanh(mean_len/200)`. Padding alone no longer scores.

---

### ❌ Toxicologist Ignored
**What happened:** The policy learned to ignore sub-agent safety alerts because following them delayed progress.

**Fix:** Made `oversight_penalty` **exponential** in the number of ignored flags — one ignored flag is a nudge; repeated ignoring wipes out the terminal reward.

---

## What Makes This Genuinely Novel

Most RL environments for drug discovery either:

- Are **non-language** (molecule graphs, SMILES strings as tokens, no reasoning)
- Use **single-step rewards** (generate → score → done)
- Are **purely simulated** with no connection to real disease databases

We combine:

✅ **Real disease data** (Open Targets + ChEMBL, ~5,500 pairs)  
✅ **Long-horizon planning** (50 steps, 4 stage gates)  
✅ **Budget pressure** ($1,000 finite credits, each tool has a cost)  
✅ **Multi-objective reward** (potency, novelty, ADMET, process, oversight — all at once)  
✅ **Natural language agent** (the LLM reasons in text; the env interprets JSON actions)  
✅ **OpenEnv-compatible** (env = HTTP service; trainer = separate process)  
✅ **Reproducible on free hardware** (single Kaggle T4, end-to-end notebook)

---

## What's Next

- **Larger backbone** — Qwen2.5-3B-Instruct with the same environment (same 7 reward components, just a bigger policy)
- **Learned reward model** — replace cached-SMILES Tanimoto with a reward model trained on ChEMBL bioactivity data
- **Multi-target campaigns** — diseases where multiple targets interact (combination therapies, off-target safety)
- **Streaming observations** — `DrugDiscoveryClient` that lets the agent interrupt and redirect mid-campaign

---

## Try It in 30 Seconds

After training, run inference on any disease not in the training set:

```bash
python -m drug_discovery_env.scripts.infer \
  --checkpoint outputs/grpo \
  --disease "Idiopathic pulmonary fibrosis"
```

The agent will plan, query tools, manage its budget, and nominate a SMILES candidate — all in natural language, all scored by the same 7-component reward.

---

## Links

| Resource | Location |
|----------|----------|
| 🧬 **Hugging Face Space (env)** | *(link — see README)* |
| 💻 **GitHub Repo** | *(link — see README)* |
| 📓 **Kaggle Notebook** | `notebooks/kaggle_drug_discovery_grpo.ipynb` |
| 📈 **Loss / Reward Plots** | `outputs/grpo/plots/` |
| 📊 **Eval Report** | `outputs/eval/report.json` |
| 📝 **Per-Disease Records** | `outputs/eval/per_disease.jsonl` |

---

## Quick Reference: End-to-End Commands

```bash
# 1. Build the dataset (one-time; queries Open Targets + ChEMBL)
python -m drug_discovery_env.scripts.prepare_dataset

# 2. Start the environment server
uvicorn drug_discovery_env.server.app:app --host 0.0.0.0 --port 8000 &

# 3. Train with GRPO (checkpoints → outputs/grpo/, traces → outputs/grpo/logs/)
python -m drug_discovery_env.scripts.train

# 4. Evaluate on the held-out test split (→ outputs/eval/report.json)
python -m drug_discovery_env.scripts.evaluate --checkpoint outputs/grpo

# 5. Infer on a brand-new disease
python -m drug_discovery_env.scripts.infer \
  --checkpoint outputs/grpo \
  --disease "Idiopathic pulmonary fibrosis"
```

All commands accept `--config path/to/override.yaml` to layer hyperparameter overrides on top of `config/defaults.yaml`.

---

*Built for the OpenEnv Hackathon. MIT Licensed.*
*The environment, training pipeline, dataset builder, evaluator, and inference script are all open-source and reproducible on free Kaggle T4 hardware.*
