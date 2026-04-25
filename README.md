# Long-Horizon Multi-Agent Drug Discovery (OpenEnv)

A simulated 50-step drug discovery campaign for training LLM agents with GRPO.
Built on [OpenEnv](https://github.com/huggingface/openenv) for the OpenEnv
Hackathon (India 2026). Targets four hackathon themes simultaneously:

- **#2 Long-Horizon Planning** (primary): 50 steps, sparse reward at termination.
- **#3.1 World Modeling**: tool use over RDKit-backed cheminformatics simulators.
- **#1 Multi-Agent Interactions**: a Project Lead LLM advised by 4 rule-based sub-agents.
- **Fleet AI bonus** & **Mercor bonus**: an Oversight agent + reward shaping for
  reasoning-trace depth.

> Nothing in this environment is real chemistry. RDKit + a small ChEMBL/ZINC-like
> compound library produce plausible feedback so the agent can learn systematic
> scientific reasoning under resource constraints.

## What the agent does

The Project Lead drives a research campaign through 5 stages and at each step
chooses one tool call:

| Stage | Goal | Typical tools |
|-------|------|---------------|
| `target_selection` | Pick a protein target for the disease | `search_chembl`, `literature_search`, `advance_stage` |
| `hit_id` | Find candidate compounds | `search_chembl`, `predict_binding_affinity` |
| `hit_to_lead` | Modify hits for better potency | `modify_molecule`, `predict_binding_affinity` |
| `admet` | Filter for drug-likeness / toxicity | `compute_admet` |
| `lead_validation` | Final docking and lead nomination | `run_docking`, `advance_stage` |

Each tool call costs budget. Sub-agents (chemist, toxicologist, budget,
oversight) react every step and may emit `info`/`warn`/`block` messages.
Ignoring a `block` warning incurs an oversight penalty.

## Reward (composable rubric)

| Component | Range | Targets |
|-----------|-------|---------|
| Terminal compound quality | `0..1.0` | binding × ADMET × novelty of the nominated lead |
| Stage progression | `0..0.3` | cleared canonical stages without skipping |
| Budget efficiency | `0..0.2` | budget remaining at the end |
| Reasoning depth (Mercor) | `0..0.5` | substantive scientific reasoning per step |
| Oversight penalty | `-0.2..0` | for ignoring `block` warnings |

The components are coupled to *different* dimensions of behaviour, so an agent
can't max one component without losing on another (e.g. high reasoning depth on
a toxic compound still scores zero on terminal quality).

## Repo layout

```
.
├── models.py                       # Action / Observation / State (Pydantic)
├── client.py                       # DrugDiscoveryEnv — what training code imports
├── openenv.yaml                    # Manifest
├── server/
│   ├── app.py                      # FastAPI entrypoint (create_fastapi_app)
│   ├── drug_discovery_env.py       # The Environment (reset/step/state)
│   ├── lab_simulator.py            # RDKit-backed tool implementations
│   ├── scenario_generator.py       # Disease/target seed library
│   ├── reward_calculator.py        # Composable rubric components
│   ├── hypothesis_evalutor.py      # Sub-agent panel (chemist/tox/budget/oversight)
│   ├── requirements.txt
│   └── Dockerfile
├── training/
│   └── train_model.py              # GRPO trainer + random baseline runner
├── notebooks/
│   └── train_colab.ipynb           # End-to-end Colab notebook
└── plots/                          # Reward curves committed as PNG
```

## Run locally

```bash
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Hit the OpenAPI docs at `http://localhost:8000/docs`, the web UI at `/web`,
and the WebSocket at `/ws`.

## Run a baseline

The included random baseline plays valid stage-aware actions and gives the
floor that GRPO has to beat.

```bash
python training/train_model.py --baseline --baseline-episodes 10
```

Writes `outputs/baseline/baseline_summaries.json`.

## Train with GRPO

Run the env (locally or on HF Spaces), then:

```bash
python training/train_model.py \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --output-dir outputs/grpo
```

For Colab the [training notebook](notebooks/train_colab.ipynb) is the easiest
path: it installs Unsloth + TRL, starts the env in a background process,
trains for ~500 rollouts on a T4, and saves the reward curves.

## Deploy to a HF Space

```bash
openenv push --repo-id <your-username>/drug-discovery-sim-env
```

The Space exposes:
- API: `https://<user>-drug-discovery-sim-env.hf.space`
- Web UI: `/web`
- Docs: `/docs`
- Health: `/health`

## Results

> Plots are committed to `plots/` as PNG. Replace these placeholders with
> your real run after training.

| Metric | Random baseline | GRPO trained |
|--------|----------------:|-------------:|
| Mean total reward | _fill in_ | _fill in_ |
| Budget remaining at end | _fill in_ | _fill in_ |
| ADMET pass rate | _fill in_ | _fill in_ |
| Campaigns completed end-to-end | _fill in_ | _fill in_ |
| Oversight `block` warnings ignored | _fill in_ | _fill in_ |

![Reward curve](plots/reward_curve.png)

## Why this matters

An LLM that can reason systematically about scientific trade-offs under resource
constraints is useful far beyond drug discovery — the same loop applies to
materials science, agricultural biotech, and any multi-stage experimental design
problem. This environment is small enough to train on a free T4 yet rich enough
to demand genuine planning.

## Links

- HF Space: _add after `openenv push`_
- Mini blog post: _link to HF blog post_
- Demo video: _link to YouTube_
