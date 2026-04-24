# Drug Discovery Sim Environment

Advanced RL environment for simulated drug discovery with:
- dataset source modes: `live_only`, `local_only`, `hybrid`
- 8-tool action surface with topology-aware transitions
- hybrid literature retrieval (`dense + bm25 + rerank`) with lexical fallback
- centralized constants via `config/defaults.yaml`

## Quickstart

```bash
python3 -m drug_discovery_env.scripts.prepare_snapshots
pytest
python3 -m drug_discovery_env.scripts.run_local_rollout
python3 -m drug_discovery_env.scripts.run_evaluation
```
