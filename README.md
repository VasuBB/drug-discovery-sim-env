# Drug Discovery Simulation Environment

A multi-disease, long-horizon drug discovery RL environment with a single
server-client GRPO training pipeline. The agent (Project Lead LLM) drives a
simulated 50-step campaign per disease, choosing among 13 cheminformatics /
literature / sub-agent tools. It is trained on thousands of cached diseases
and evaluated on a held-out test split — including ChEMBL Tanimoto similarity
to known approved drugs for the same target.

## Pipeline

```
                          +-------------------+
prepare_dataset --> data/diseases.jsonl <----+ scripts/train.py     (GRPO over HTTP)
                              |               + scripts/evaluate.py (held-out test split)
FastAPI env (server/app.py) <-+               + scripts/infer.py    (any unseen disease)
```

Three concerns are strictly separated:

1. **Dataset** — `data/diseases.jsonl` is built once by
   `prepare_dataset.py`. It contains disease, EFO id, top associated target,
   druggability, target class, ChEMBL known-drug SMILES (used by the
   evaluation Tanimoto metric), and a deterministic `train`/`test` split.
2. **Environment** — the FastAPI server in `drug_discovery_env/server/` holds
   no training logic. It loads cached targets via
   `data_provider/cached_provider.py` and proxies live ChEMBL / PubMed
   queries through the existing `LiveAPIProvider` so the agent really
   learns *what* to query.
3. **Trainer / evaluator / inference** — three thin scripts under
   `drug_discovery_env/scripts/` that all talk to the env over HTTP via
   `DrugDiscoveryClient`. Every turn (rendered observation, raw model
   output, parsed action, tool result, reward breakdown, sub-agent
   messages, budget) is written as JSONL by `EpisodeLogger`.

## Reward (smooth, single-step formulas)

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

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[training,test,chem]
```

## End-to-end

```bash
# 1. Build the dataset (one-time; pages Open Targets + ChEMBL).
python -m drug_discovery_env.scripts.prepare_dataset

# 2. Start the env server.
uvicorn drug_discovery_env.server.app:app --host 0.0.0.0 --port 8000 &

# 3. Train. Checkpoints go to outputs/grpo, JSONL traces to outputs/grpo/logs/<run-id>/.
python -m drug_discovery_env.scripts.train

# 4. Evaluate on the held-out test split. Produces outputs/eval/report.json.
python -m drug_discovery_env.scripts.evaluate --checkpoint outputs/grpo

# 5. Run inference on a brand-new disease.
python -m drug_discovery_env.scripts.infer --checkpoint outputs/grpo \
    --disease "Idiopathic pulmonary fibrosis"
```

All four commands accept `--config path/to/override.yaml` to layer overrides
on top of `config/defaults.yaml` (hardware, hyperparameters, paths). CLI flags
override config values for everything that's commonly tweaked.

## Evaluation metrics

`scripts/evaluate.py` writes `outputs/eval/report.json` with:

- `mean_terminal_reward`, `mean_total_reward`, full reward breakdown
- `stage_completion_rate`, `mean_steps`, `mean_budget_remaining_frac`
- `admet_pass_rate` (RO5 ∧ ¬PAINS ∧ ¬tox)
- `oversight_violation_rate`, `mean_oversight_violations`
- `mean_reasoning_depth`
- `mean_tanimoto_to_known`, `mean_tanimoto_to_known_max`, `precision_at_1`
  (Tanimoto over Morgan fingerprints between the agent's nominated SMILES
  and the cached ChEMBL known drugs for that disease's target)

Per-disease records live in `outputs/eval/per_disease.jsonl`.

## Tests

```bash
pytest -q
```

## License

MIT.
