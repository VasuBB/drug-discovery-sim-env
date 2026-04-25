# Long-Horizon Multi-Agent Drug Discovery (OpenEnv)

A simulated 50-step drug discovery campaign for training LLMs with **GRPO** to reason
across a long horizon, manage budget, listen to a panel of rule-based sub-agents,
and nominate a safe and potent lead compound. Built on
[OpenEnv](https://github.com/huggingface/openenv) for the **OpenEnv Hackathon (India 2026)**.

This branch is the **merged submission**: it combines the modular architecture of
the `akshat-dev` branch (config-driven Settings, hybrid data providers, hybrid
retrieval, structured CompoundRecord, topology engine, snapshots, tests) with
the polished OpenEnv compliance, RDKit-backed lab simulator, severity-based
sub-agent panel, scenario library, composable reward rubric and live-rollout
GRPO trainer of the `main` branch.

> Nothing in this environment is real chemistry. RDKit, a small ChEMBL/ZINC-like
> compound library and a PubMed-style snapshot produce *plausible* feedback so the
> agent can learn systematic scientific reasoning under resource constraints.

---

## Why This Problem

Real drug discovery costs **~$2B** and takes **>10 years**. A huge fraction of that
waste comes from bad early decisions: picking the wrong protein target, advancing
toxic molecules, blowing the research budget on dead-end experiments. We are training
an LLM to make better *sequential, multi-objective* decisions in this exact loop --
a transferable skill that applies to materials science, agricultural biotech, and any
multi-stage experimental design problem.

This environment hits four hackathon themes simultaneously:

- **#2 Long-Horizon Planning** (primary): 50 steps, sparse reward at termination.
- **#3.1 World Modeling**: tool use over RDKit-backed cheminformatics simulators with
  partial observability and uncertainty propagation.
- **#1 Multi-Agent Interactions**: a Project Lead LLM advised by 4 rule-based
  sub-agents (chemist, toxicologist, budget, oversight).
- **Fleet AI bonus / Mercor bonus**: the oversight agent + a reasoning-depth
  reward component.

---

## What the agent does

The Project Lead drives a research campaign through 5 stages and at each step
chooses one tool call:

| Stage | Goal | Typical tools |
|-------|------|---------------|
| `target_selection` | Pick a protein target for the disease | `select_target`, `search_literature` |
| `hit_id` | Find candidate compounds | `search_compounds`, `predict_affinity` |
| `hit_to_lead` | Modify hits for better potency | `modify_molecule`, `synthesize`, `predict_affinity` |
| `admet` | Filter for drug-likeness / toxicity | `evaluate_admet` |
| `lead_validation` | Final docking + selectivity panel | `validate_compound` |

Each tool call costs budget. The four sub-agents review state every step and may
emit `info` / `warn` / `block` messages. **Ignoring a `block` warning incurs an
oversight penalty in the reward.**

### Tool catalogue (8 OpenEnv MCP tools)

| Tool | Action | Cost (credits) |
|------|--------|----------------|
| `select_target` | Pick a protein target via Open Targets / local snapshot | 5 |
| `search_compounds` | ChEMBL-like compound search | 10 |
| `predict_affinity` | RDKit-derived binding-affinity surrogate | 30 |
| `evaluate_admet` | RDKit Lipinski / PAINS / TOX SMARTS / hERG proxy / QED | 10 |
| `modify_molecule` | RDKit SMILES modification | 20 |
| `synthesize` | Two-molecule reaction with synthesis-failure model | 25 |
| `validate_compound` | RDKit docking + off-target panel | 80 |
| `search_literature` | PubMed-style retrieval (BM25 + dense + reranker, 3 modes) | 5 |

Costs are scaled at runtime by **variable-cost multipliers** (low information,
high uncertainty, late-stage redundancy) so the same tool gets *more expensive*
when used wastefully.

### The team (sub-agents)

| Agent | Role | Severities emitted |
|-------|------|--------------------|
| Toxicologist | Flags PAINS/hERG/RO5 violations | `warn`, `block` |
| Chemist | Suggests SAR moves, scaffold diversity | `info`, `warn` |
| Budget Manager | Watches credit burn | `info`, `warn`, `block` |
| Oversight | Detects low-info loops & late-stage acceleration | `warn`, `block` |

These are deterministic rule-based scripts -- only the Project Lead is trained.

---

## Reward (composable rubric)

| Component | Range | Targets |
|-----------|-------|---------|
| Terminal compound quality | `0..1.0` (weight 0.6) | potency, selectivity, safety, synthesizability, novelty, developability |
| Process reward | `0..1.0` (weight 0.2) | per-step information gain / cost efficiency |
| Reasoning depth (Mercor) | `0..1.0` (weight 0.15) | structural signal *and* scientific keyword density |
| Strategy reward | `0..1.0` (weight 0.05) | stage progress, diversity, budget efficiency |
| Oversight penalty | `-0.2..0` (raw) | **for ignoring `block` warnings** |

The components are coupled to *different* dimensions of behaviour, so an agent
can't max one component without losing on another. A toxic compound (hERG > 0.5)
zeroes the entire terminal reward -- safety is non-negotiable.

All weights, floors, and budgets live in [`config/defaults.yaml`](config/defaults.yaml)
and are reloaded by the `Settings` class with environment-variable overrides
(`DD_<SECTION>__<KEY>`).

---

## Repo layout

```
.
├── openenv.yaml                                  # OpenEnv manifest
├── config/defaults.yaml                          # All thresholds / weights / costs
├── drug_discovery_env/
│   ├── server/
│   │   ├── app.py                                # FastAPI + HTTPEnvServer entrypoint
│   │   ├── environment.py                        # The Environment (reset/step/state)
│   │   └── openenv_adapter.py                    # MCP-style adapter
│   ├── core/
│   │   ├── models.py                             # Action / Observation / State / RewardBreakdown
│   │   ├── state.py                              # GameState + CompoundRecord ledger
│   │   ├── stage_manager.py                      # Integer stage gates
│   │   ├── scenarios.py                          # 15 disease/target seeds + stage-name map (NEW, ported from main)
│   │   ├── action_parser.py                      # <tool>/<params>/<reasoning>/<evidence> XML parser
│   │   ├── budget_manager.py                     # Variable cost engine
│   │   ├── topology.py                           # Pathway neighbourhood + risk propagation
│   │   └── serializer.py                         # Compact state -> text for LLM
│   ├── chemistry/                                # RDKit lab (NEW, ported from main)
│   │   └── rdkit_lab.py                          # PAINS/TOX/hERG SMARTS, ADMET, modify, docking
│   ├── tools/                                    # 8 OpenEnv tools (RDKit-backed where relevant)
│   ├── agents/                                   # Severity-aware sub-agent panel
│   │   ├── toxicologist.py / chemist.py
│   │   ├── budget_manager.py / oversight.py
│   ├── rewards/                                  # Composable rubric
│   │   ├── terminal.py / process.py
│   │   ├── reasoning.py                          # Structural + scientific-keyword density
│   │   ├── strategy.py
│   │   ├── oversight.py                          # NEW -- penalty for ignored 'block' warnings
│   │   └── aggregator.py                         # RewardEngine
│   ├── data_provider/                            # live / hybrid / local-snapshot modes
│   ├── retrieval/                                # BM25 + dense + reranker
│   ├── data/                                     # Local snapshots (compounds / targets / literature)
│   ├── training/                                 # GRPO dataset / trainer / readiness probe
│   ├── client.py                                 # Typed OpenEnv EnvClient
│   └── scripts/
│       ├── prepare_snapshots.py
│       ├── live_smoke.py
│       ├── run_local_rollout.py
│       ├── run_evaluation.py
│       ├── run_baseline.py                       # NEW -- random-policy floor
│       ├── run_grpo.py                           # In-process GRPO readiness / training
│       ├── train_grpo_live.py                    # NEW -- live-rollout GRPO (Unsloth + TRL)
│       ├── run_training_experiment.py            # Trains + saves loss/reward plots
│       └── setup_training_env.py
├── notebooks/
│   └── 02_train_grpo_colab.ipynb                 # End-to-end Colab notebook
├── tests/                                        # 14 pytest files
├── artifacts/training/                           # Loss / reward / baseline-vs-trained plots
├── WRITEUP.md                                    # Short hackathon write-up (links to blog/video)
├── OVERVIEW.md                                   # Friendly explainer
├── implementation_plan.md
└── pyproject.toml
```

---

## Quick start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .[ml,chem,training,test]
```

(Or run the bootstrap script: `python -m drug_discovery_env.scripts.setup_training_env`.)

### 2. Validate snapshots and run the test suite

```bash
python -m drug_discovery_env.scripts.prepare_snapshots
pytest -q
```

### 3. Run the OpenEnv server

```bash
uvicorn drug_discovery_env.server.app:app --host 0.0.0.0 --port 8000 --reload
```

OpenAPI docs at `http://localhost:8000/docs`, health at `/health`.

### 4. Random-policy baseline (no GPU / model required)

```bash
python -m drug_discovery_env.scripts.run_baseline --episodes 10
```

Writes `outputs/baseline/baseline_summaries.json` and prints the mean reward
that GRPO has to beat.

### 5. GRPO training

**In-process (no server needed; modular trainer):**

```bash
python -m drug_discovery_env.scripts.run_grpo --train --episodes 2 --device auto \
    --model Qwen/Qwen2.5-0.5B-Instruct --max-train-steps 50
```

**Live-rollout (talks to a running OpenEnv server -- closer to a real RL loop):**

```bash
# terminal 1
uvicorn drug_discovery_env.server.app:app --host 0.0.0.0 --port 8000

# terminal 2
python -m drug_discovery_env.scripts.train_grpo_live \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen2.5-3B-Instruct
```

**Colab (recommended for first end-to-end run):** open
[`notebooks/02_train_grpo_colab.ipynb`](notebooks/02_train_grpo_colab.ipynb).
It installs Unsloth + TRL, starts the env in a background process, and runs ~500
rollouts on a free T4.

### 6. Reproduce loss / reward / baseline-vs-trained figures

```bash
python -m drug_discovery_env.scripts.run_training_experiment \
    --episodes 1 --data-mode local_only --device auto \
    --model sshleifer/tiny-gpt2 --max-train-steps 3 \
    --out-dir artifacts/training
```

Outputs:

- `artifacts/training/loss_curve.png`
- `artifacts/training/reward_curve.png`
- `artifacts/training/baseline_vs_trained.png`
- `artifacts/training/training_summary.json`

### 7. Deploy to a Hugging Face Space

```bash
openenv push --repo-id <your-username>/drug-discovery-sim-env
```

The Space exposes:
- API: `https://<user>-drug-discovery-sim-env.hf.space`
- Docs: `/docs`
- Health: `/health`

---

## Results

A reproducible tiny-model GRPO run produces these artifacts in `artifacts/training/`:

![Loss curve](artifacts/training/loss_curve.png)
![Reward curve](artifacts/training/reward_curve.png)
![Baseline vs trained](artifacts/training/baseline_vs_trained.png)

| Metric | Random baseline | GRPO trained |
|--------|----------------:|-------------:|
| Mean total reward | _filled in by `run_training_experiment`_ | _filled in by `run_training_experiment`_ |
| Campaigns reaching `lead_validation` | _fill in_ | _fill in_ |
| ADMET pass rate | _fill in_ | _fill in_ |
| Budget remaining at end | _fill in_ | _fill in_ |
| Oversight `block` warnings ignored | _fill in_ | _fill in_ |

> The Colab notebook produces real curves; tiny-model artifacts in `artifacts/training/`
> are committed as a reproducibility proof and **not** as the final run.

---

## OpenEnv compliance checklist

- [x] Uses **OpenEnv classes directly** (`Environment`, `Action`, `Observation`,
      `State`, `HTTPEnvServer`, `EnvClient`) via the `openenv_compat`
      shim that supports both `openenv.core` and the legacy `openenv_core`
      package paths -- no reinvention.
- [x] Gym-style `reset / step / state` API.
- [x] Valid manifest at [`openenv.yaml`](openenv.yaml).
- [x] `MCPEnvironment`-style server: 8 tools advertised through the registered
      action schema; tool dispatch is server-side and side-effect-isolated.
- [x] Working **GRPO training** via Hugging Face TRL (with optional Unsloth)
      and a Colab notebook.
- [x] Proof of training: loss / reward / baseline-vs-trained plots in
      `artifacts/training/`.
- [x] Short write-up: see [`WRITEUP.md`](WRITEUP.md) for blog / slides / video links.
- [x] README explains problem, architecture, and results (this document).

---

## Links

- Hugging Face Space: _add after `openenv push`_
- Hugging Face blog post: _link to HF blog post_
- Demo video (<2 min) / slides: _link to YouTube or slides_
- WandB run: _link to WandB run_

> **Do not commit large video binaries into this repo or the HF Space package.
> Use external URLs in the table above.**
