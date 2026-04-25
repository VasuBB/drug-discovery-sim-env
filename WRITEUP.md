# Writeup — Long-Horizon Multi-Agent Drug Discovery (OpenEnv)

> A short post explaining what we built, why, and what the GRPO-trained model
> learned. Submit one of the following formats and link it from the README:
>
> - Hugging Face blog post: `TODO`
> - YouTube video (<2 min): `TODO`
> - Slides (PDF or Google Slides link): `TODO`
>
> Do **not** commit large video binaries — link externally.

## Problem

Real drug discovery costs ~$2B and takes >12 years. Most of that waste comes
from bad early decisions: picking the wrong protein target, advancing toxic
molecules, blowing budget on dead-end experiments. We can't train a real LLM
on a real wet lab, but we *can* train it on a faithful simulation that demands
the same reasoning skills.

## Environment

A 50-step, 5-stage research campaign:

  1. `target_selection` — pick a protein target for the disease.
  2. `hit_id` — find candidate compounds.
  3. `hit_to_lead` — modify hits for better potency.
  4. `admet` — filter for drug-likeness / toxicity.
  5. `lead_validation` — final docking + nominate a lead.

The agent (Project Lead) chooses one tool call per step from a 13-tool
inventory. Sub-agents (Chemist, Toxicologist, Budget Manager, Oversight)
review every step and emit `info` / `warn` / `block` messages. Ignoring a
`block` warning incurs an oversight penalty.

## Reward (composable rubric, 7 components)

`terminal_compound`, `stage_progression`, `budget_efficiency`, `reasoning_depth`,
`process`, `strategy`, `oversight_penalty`. The terminal component has a hard
hERG/PAINS safety floor: a toxic compound scores zero regardless of potency.

## What GRPO learns

_Fill in after the real Qwen2.5-3B run:_
  - mean reward: random baseline `X` → trained `Y`
  - typical campaign trajectory before vs after
  - one or two qualitative examples of the trained model declining a high-potency
    PAINS hit, or pausing for ADMET before nominating a lead.

## Architecture diagram

_Optional — paste from a Mermaid block or a slide here._

## Limitations & next steps

  - All chemistry is simulated; binding/ADMET/docking scores are RDKit-flavoured
    deterministic-pseudo-random functions. Training transfers reasoning patterns,
    not chemistry.
  - Sub-agents are rule-based. A future version could swap any of them for a
    smaller LLM critic.
  - No molecular-graph generative head — `modify_molecule` is string-level
    SMILES editing with RDKit validity check. SMARTS-based reaction templates
    would be a natural upgrade.
