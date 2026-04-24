# What Are We Building?

## The One-Liner

A **simulated drug discovery lab** where an AI agent learns to find medicines — by playing through 50-step research campaigns, making mistakes, and getting better through reinforcement learning.

---

## The Problem

Real drug discovery costs **$2 billion** and takes **12 years**. Most of that waste comes from bad early decisions — picking the wrong protein target, advancing toxic molecules, blowing the research budget on dead-end experiments. What if we could train an AI to avoid these mistakes *before* it ever touches a real lab?

---

## What We're Actually Doing

We're building a **video-game-like environment** that simulates a pharmaceutical research campaign. Think of it as a strategy game where:

- **The player** is an LLM (large language model) acting as a "Project Lead" at a pharma company
- **The goal** is to start with a disease (e.g., "Type 2 Diabetes") and end up with a promising drug molecule
- **The constraint** is a limited budget — every experiment costs credits, and you run out after ~50 steps
- **The challenge** is making smart decisions under uncertainty, balancing potency vs. safety vs. cost

The AI doesn't know real chemistry. Everything is **simulated** using open-source cheminformatics tools (RDKit, ChEMBL data). But the simulation is realistic enough that the reasoning skills transfer.

---

## How the Game Works

### The 5 Stages

```
Disease → Target → Hits → Optimization → Safety Check → Validated Lead
  (1)      (2)      (3)        (4)            (5)
```

1. **Target Selection** — Pick which protein to attack. Bad choice = everything downstream fails.
2. **Hit Identification** — Screen thousands of molecules to find ones that bind to your target.
3. **Hit-to-Lead Optimization** — Modify the best molecules to make them more potent.
4. **ADMET Evaluation** — Check safety: Is it toxic? Can it dissolve? Does it get metabolized too fast?
5. **Lead Validation** — Final scoring: docking simulation, selectivity check, synthesizability.

### The Tools (8 actions the agent can take)

| Tool | What it does | Cost |
|------|-------------|------|
| `select_target` | Pick a protein target for the disease | 5 credits |
| `search_compounds` | Search the molecule library | 10 credits |
| `predict_affinity` | Test how well a molecule binds | 30 credits |
| `evaluate_admet` | Run safety/drug-likeness checks | 10 credits |
| `modify_molecule` | Chemically tweak a molecule | 20 credits |
| `synthesize` | Combine two molecules via a reaction | 25 credits |
| `validate_compound` | Full docking + selectivity analysis | 80 credits |
| `search_literature` | Look up relevant research papers | 5 credits |

### The Team (sub-agents that advise the Project Lead)

- **Toxicologist** — Flags dangerous molecules ("This compound will cause heart problems")
- **Chemist** — Suggests improvements ("The solubility is weak, try adding a hydroxyl group")
- **Budget Manager** — Warns about spending ("You've used 70% of budget and you're only in Stage 3")
- **Oversight Agent** — Catches bad patterns ("You're ignoring toxicity warnings")

These are **rule-based scripts**, not LLMs. Only the Project Lead agent is trained.

---

## How We Train the AI

### The Reward Function

The agent gets scored on **four things**:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| **Terminal Reward** | 60% | How good is the final molecule? (potency, safety, novelty, synthesizability) |
| **Process Reward** | 20% | Did the agent follow good scientific practice? (check ADMET before docking, diversify scaffolds) |
| **Reasoning Reward** | 15% | Quality of the agent's written reasoning (does it reference actual data, not just generic text?) |
| **Strategy Reward** | 5% | Did the agent manage the whole campaign well? (visit all stages, improve progressively) |

**Key rule**: If the molecule causes cardiac toxicity (hERG positive), the entire terminal reward is **zero**. No exceptions. This teaches the agent that safety is non-negotiable.

### The Training Algorithm: GRPO

- **GRPO** (Group Relative Policy Optimization) — samples 8 different responses per situation, compares them, and updates the model to produce more of the good ones
- No separate critic model needed → fits on a single T4 GPU
- Training data is generated live through rollouts (no pre-collected dataset)

### Two Models

| | Testing (Now) | Training (With GPU) |
|---|---|---|
| **Model** | Qwen2.5-0.5B via Ollama | Qwen2.5-3B + LoRA via Unsloth |
| **Runs on** | Your Mac CPU | Free Colab T4 |
| **Purpose** | Debug the environment | Actually learn drug discovery reasoning |

---

## What Success Looks Like

| Metric | Before Training | After Training |
|--------|----------------|----------------|
| Campaigns completed (out of 10) | 2 | 8 |
| Compound quality score | 0.15–0.22 | 0.55–0.70 |
| ADMET safety pass rate | ~20% | ~65% |
| Toxicity warnings ignored | ~80% | ~15% |
| Budget remaining at end | ~0% | 28–40% |
| Cardiac toxicity triggered | ~60% | ~12% |

The most impressive result: the agent learns to **avoid cardiac toxicity** (hERG) — dropping from 60% to 12% — purely from the reward signal, without being explicitly programmed to do so.

---

## Tech Stack

- **Python 3.12** — everything is Python
- **RDKit** — molecular chemistry (SMILES parsing, fingerprints, reactions, descriptors)
- **scikit-learn** — ADMET prediction models (trained on TDC datasets)
- **OpenEnv** (Meta/PyTorch) — RL environment framework (step/reset/state API)
- **TRL + Unsloth** — GRPO training on HuggingFace models
- **Ollama** — local LLM inference for testing
- **PyTDC** — Therapeutics Data Commons for training data
- **FastAPI** — environment server

---

## Why This Matters

This isn't about replacing chemists. It's about proving that an LLM can learn **structured scientific reasoning** — weighing trade-offs, managing resources, heeding safety warnings, and building knowledge systematically across a long campaign.

The same pattern applies to materials science, agricultural biotech, and any domain where you make sequential experimental decisions under uncertainty and budget constraints.

**We're not training a chemist. We're training a scientific decision-maker.**
