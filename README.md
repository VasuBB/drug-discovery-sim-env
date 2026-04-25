# Drug Discovery Sim Environment (OpenEnv)

An advanced OpenEnv environment where an LLM learns long-horizon scientific decision-making for drug discovery under uncertainty, safety constraints, and budget pressure.

## Links (Judge-Facing)
- Hugging Face Space (environment): `TODO_ADD_SPACE_URL`
- Hugging Face blog / writeup: `TODO_ADD_BLOG_URL`
- Video (<2 min) or slides: `TODO_ADD_VIDEO_OR_SLIDES_URL`
- WandB run (optional): `TODO_ADD_WANDB_RUN_URL`

## Why This Problem
LLMs are weak at sequential scientific planning with tradeoffs (potency vs safety vs cost). This environment teaches exactly that through a 50-step campaign with realistic multi-objective rewards and hard safety floors.

## OpenEnv Compliance
- Uses OpenEnv classes directly:
  - `Environment`: [environment.py](/Users/akshatvaja/Documents/akshat/work/Hackathon/drug-discovery-sim-env/drug_discovery_env/server/environment.py)
  - `Action/Observation/State`: [models.py](/Users/akshatvaja/Documents/akshat/work/Hackathon/drug-discovery-sim-env/drug_discovery_env/core/models.py)
  - `HTTPEnvServer`: [app.py](/Users/akshatvaja/Documents/akshat/work/Hackathon/drug-discovery-sim-env/drug_discovery_env/server/app.py)
  - `EnvClient` typed client (client/server separation): [client.py](/Users/akshatvaja/Documents/akshat/work/Hackathon/drug-discovery-sim-env/drug_discovery_env/client.py)
- Gym-style API implemented: `reset`, `step`, `state`
- Valid OpenEnv manifest: [openenv.yaml](/Users/akshatvaja/Documents/akshat/work/Hackathon/drug-discovery-sim-env/openenv.yaml)

## Environment Design
- 8 tools (actions): target selection, compound search, affinity prediction, ADMET evaluation, molecule modification, synthesis, compound validation, literature search
- Data modes:
  - `live_only`: full live fetch (Open Targets + ChEMBL + PubMed)
  - `hybrid`: live with fallback to local snapshots
  - `local_only`: fully offline snapshots
- Topology-aware state dynamics:
  - disease pathway neighborhood
  - compensatory/off-target propagation
  - assay uncertainty and evidence confidence decay
- Budget economics:
  - fixed + variable cost multipliers
  - opportunity cost
  - late-stage low-information penalties

## Reward Signal
- Composite reward (`terminal + process + reasoning + strategy`) with configured weights
- Hard safety floors (e.g., hERG threshold)
- Anti-gaming structure using staged progress + evidence grounding + budget efficiency

## Training (TRL / GRPO)
- GRPO training script: [run_grpo.py](/Users/akshatvaja/Documents/akshat/work/Hackathon/drug-discovery-sim-env/drug_discovery_env/scripts/run_grpo.py)
- Repro experiment + plots script: [run_training_experiment.py](/Users/akshatvaja/Documents/akshat/work/Hackathon/drug-discovery-sim-env/drug_discovery_env/scripts/run_training_experiment.py)
- Colab notebook: [02_train_grpo_colab.ipynb](/Users/akshatvaja/Documents/akshat/work/Hackathon/drug-discovery-sim-env/notebooks/02_train_grpo_colab.ipynb)

## Results Snapshot
Latest local reproducible tiny-model GRPO run produced artifacts in `artifacts/training`:
- Loss curve: `artifacts/training/loss_curve.png`
- Reward curve: `artifacts/training/reward_curve.png`
- Baseline vs trained reward: `artifacts/training/baseline_vs_trained.png`

![Loss Curve](/Users/akshatvaja/Documents/akshat/work/Hackathon/drug-discovery-sim-env/artifacts/training/loss_curve.png)
![Reward Curve](/Users/akshatvaja/Documents/akshat/work/Hackathon/drug-discovery-sim-env/artifacts/training/reward_curve.png)
![Baseline vs Trained](/Users/akshatvaja/Documents/akshat/work/Hackathon/drug-discovery-sim-env/artifacts/training/baseline_vs_trained.png)

Caption:
- Loss plot shows optimization dynamics over GRPO steps.
- Reward plot shows training reward signal trend.
- Baseline-vs-trained bar chart compares mean reward before and after training.

## Step-by-Step Run Guide

1. Create and activate env:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install base:
```bash
python3 -m pip install -e .
```

3. Install training stack:
- macOS / Apple Silicon (MPS):
```bash
python3 -m drug_discovery_env.scripts.setup_training_env
```
- Linux + NVIDIA CUDA:
```bash
python3 -m drug_discovery_env.scripts.setup_training_env --cuda
```

4. Validate snapshots + tests:
```bash
python3 -m drug_discovery_env.scripts.prepare_snapshots
pytest -q
```

5. Run with live data:
```bash
python3 -m drug_discovery_env.scripts.live_smoke
python3 -m drug_discovery_env.scripts.run_local_rollout --data-mode live_only
python3 -m drug_discovery_env.scripts.run_evaluation --episodes 5 --data-mode live_only
```

6. GRPO readiness and training:
```bash
python3 -m drug_discovery_env.scripts.run_grpo --episodes 1 --data-mode live_only --device auto
python3 -m drug_discovery_env.scripts.run_grpo --train --episodes 2 --data-mode live_only --device mps
```
For NVIDIA GPU:
```bash
python3 -m drug_discovery_env.scripts.run_grpo --train --episodes 2 --data-mode live_only --device cuda
```

7. Generate reproducible train/plot artifacts:
```bash
python3 -m drug_discovery_env.scripts.run_training_experiment --episodes 1 --data-mode live_only --device auto --model sshleifer/tiny-gpt2 --max-train-steps 3 --out-dir artifacts/training
```

8. Run OpenEnv server:
```bash
uvicorn drug_discovery_env.server.app:app --host 0.0.0.0 --port 8000 --reload
```

## Notes
- Do not commit large video binaries into repo/HF env package; use external URLs.
- For full-scale training, run on GPU machine or Colab and use the provided notebook.
