# Training artifacts

Plots and summary JSON from the most recent reproducibility run of
`drug_discovery_env.scripts.run_training_experiment`.

| File | Source |
|------|--------|
| `loss_curve.png` | per-step training loss |
| `reward_curve.png` | per-step mean reward |
| `baseline_vs_trained.png` | mean reward, random-policy baseline vs GRPO-trained |
| `training_summary.json` | full TRL `log_history` + `train_metrics` from the run |

To regenerate (CPU/MPS-friendly, takes ~5 min on a small model):

```bash
python -m drug_discovery_env.scripts.run_training_experiment \
  --episodes 2 --data-mode hybrid --device auto \
  --model Qwen/Qwen2.5-0.5B-Instruct --max-train-steps 10 \
  --out-dir artifacts/training
```

For the submission run, switch the model to `Qwen/Qwen2.5-3B-Instruct` and
increase `--max-train-steps` (T4 can comfortably do 100+ steps in <30 min).

> Note: the model checkpoints (`checkpoint-*/` directories) are intentionally
> NOT committed — they bloat the repo with multi-MB tokenizer.json dumps. Only
> the plots and summary JSON are tracked.
