# Drug Discovery RL Environment — Implementation Plan

## Goal

Build a simulated drug discovery pipeline as an OpenEnv RL environment for the Meta PyTorch OpenEnv Hackathon (National Finale: April 25–26, 2026). An LLM agent (Project Lead) navigates a 50-step, 5-stage campaign — from disease→target selection to validated lead compound — under budget constraints, guided by rule-based sub-agents, and trained via GRPO.

---

## Model Strategy

> [!IMPORTANT]
> **Two model tracks as requested:**

| Track | Model | Purpose | Hardware | When |
|-------|-------|---------|----------|------|
| **Testing (NOW)** | `Qwen2.5-0.5B-Instruct` via **Ollama** | CPU inference for environment debugging, integration testing, reward function validation | Local Mac (no GPU) | During development |
| **Training (GPU)** | `Qwen2.5-3B-Instruct` + LoRA (rank=16, 4-bit NF4) via **Unsloth + TRL** | GRPO fine-tuning on rollouts | Free Colab T4 / HF GPU credits | After environment is validated |

**Rationale:** 0.5B runs at ~15 tok/s on CPU, enough to test full 50-step rollouts in ~5 min. The 3B model needs GPU but shows real reasoning improvement after GRPO training.

---

## Open Questions

> [!IMPORTANT]
> 1. **Hackathon deadline**: Finale is April 25–26. Do you want me to prioritize a working demo (Phases 1–5) over full GRPO training (Phases 6–7)?
> 2. **SCScore model**: DeepChem's SCScore requires a pretrained neural net. Should I use SA Score alone for synthesizability to reduce dependencies, or include SCScore?
> 3. **Open Targets API**: The GraphQL API requires internet. Should I create a local JSON snapshot of ~50 disease-target mappings instead for offline operation?
> 4. **Docker deployment**: OpenEnv expects Docker containers. Should I set up Docker now or focus on local server first?

---

## Project Structure

```
drug-discovery-sim-env/
├── pyproject.toml                    # Package config + all dependencies
├── openenv.yaml                     # OpenEnv manifest
├── README.md
├── .gitignore
│
├── drug_discovery_env/              # Main package
│   ├── __init__.py                  # Exports: Action, Observation, DrugDiscoveryEnv
│   ├── models.py                    # Pydantic models: Action, Observation, State
│   ├── client.py                    # EnvClient subclass for OpenEnv
│   │
│   ├── server/
│   │   ├── app.py                   # FastAPI app creation
│   │   ├── environment.py           # Core Environment(OpenEnv base) — step/reset/state
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── core/                        # Environment internals
│   │   ├── state.py                 # GameState dataclass (full MDP state)
│   │   ├── stage_manager.py         # Stage transitions (1→5) + progression logic
│   │   ├── budget_manager.py        # Budget tracking + cost enforcement
│   │   ├── compound_ledger.py       # Compound storage + measurement tracking
│   │   └── action_parser.py         # Parse <reasoning>/<tool>/<params> from LLM output
│   │
│   ├── tools/                       # The 8 MDP tools
│   │   ├── __init__.py
│   │   ├── base_tool.py             # Abstract base: execute(params, state) → result
│   │   ├── select_target.py         # Stage 1: target selection + druggability
│   │   ├── search_compounds.py      # Stage 2: compound library search
│   │   ├── predict_affinity.py      # Binding affinity prediction (simulated)
│   │   ├── evaluate_admet.py        # ADMET evaluation using trained models
│   │   ├── modify_molecule.py       # RDKit-based molecular modification
│   │   ├── synthesize.py            # Reaction execution via SMARTS
│   │   ├── validate_compound.py     # Docking simulation + selectivity panel
│   │   └── search_literature.py     # PubMed abstract retrieval (local JSON)
│   │
│   ├── chemistry/                   # Cheminformatics utilities
│   │   ├── validator.py             # 4-layer SMILES validation pipeline
│   │   ├── fingerprints.py          # Morgan FP generation + Tanimoto similarity
│   │   ├── descriptors.py           # RDKit descriptor calculations (QED, LogP, etc.)
│   │   ├── reactions.py             # SMARTS reaction catalog + execution
│   │   └── sa_scorer.py             # SA Score wrapper
│   │
│   ├── models_ml/                   # Trained sklearn models for ADMET
│   │   ├── train_models.py          # Script: train all ADMET models from TDC data
│   │   ├── predictor.py             # Load + inference wrapper for all models
│   │   └── saved/                   # Joblib-serialized trained models
│   │       ├── herg_classifier.joblib
│   │       ├── caco2_regressor.joblib
│   │       ├── solubility_regressor.joblib
│   │       ├── cyp3a4_classifier.joblib
│   │       └── tox21_classifier.joblib
│   │
│   ├── agents/                      # Rule-based sub-agents
│   │   ├── toxicologist.py          # PAINS, Tox21, hERG, reactive groups
│   │   ├── chemist.py               # QED analysis, scaffold diversity, suggestions
│   │   ├── budget_manager_agent.py  # Spend-rate alerts at 50%/70%/85%
│   │   └── oversight.py             # Sklearn classifier for bad decision detection
│   │
│   ├── rewards/                     # Reward function components
│   │   ├── terminal.py              # R_terminal: compound quality (0.60 weight)
│   │   ├── process.py               # R_process: per-step shaped rewards (0.20)
│   │   ├── reasoning.py             # R_reasoning: trace quality scoring (0.15)
│   │   ├── strategy.py              # R_strategy: campaign-level bonus (0.05)
│   │   └── aggregator.py            # Master formula: R(τ) combination
│   │
│   └── data/                        # Static data files
│       ├── seed_scenarios.json      # 50 disease-target pairs
│       ├── compound_library.csv     # 5,000 SMILES from ZINC20/ChEMBL
│       ├── literature_abstracts.json # ~200 curated PubMed abstracts
│       ├── off_target_binders.json  # Known binders for 8 off-target panels
│       ├── approved_drugs.csv       # ~2,500 approved drug SMILES (novelty ref)
│       └── reaction_catalog.json    # SMARTS reaction definitions
│
├── scripts/
│   ├── download_data.py             # Download ZINC20, ChEMBL, TDC datasets
│   ├── prepare_seed_scenarios.py    # Generate 50 seed scenarios from ChEMBL
│   ├── train_admet_models.py        # Train all sklearn ADMET models
│   └── run_evaluation.py            # 10-rollout before/after comparison
│
├── training/                        # GRPO training (Colab/GPU)
│   ├── grpo_trainer.py              # GRPO training loop with TRL
│   ├── rollout_generator.py         # Generate rollouts from environment
│   ├── reward_wrapper.py            # Wrap reward function for TRL interface
│   └── config.py                    # All hyperparameters
│
├── notebooks/
│   ├── 01_test_environment.ipynb    # Local testing with Ollama 0.5B
│   ├── 02_train_grpo.ipynb          # Colab notebook for GRPO training
│   └── 03_evaluation.ipynb          # Before/after comparison + plots
│
└── tests/
    ├── test_validator.py            # Chemical validation unit tests
    ├── test_tools.py                # Tool execution tests
    ├── test_rewards.py              # Reward calculation tests
    ├── test_environment.py          # Full environment step/reset tests
    └── test_agents.py               # Sub-agent logic tests
```

---

## Phase 1 — Project Setup & Data Preparation

### 1.1 Initialize Project

#### [NEW] `pyproject.toml`
```toml
[project]
name = "drug-discovery-env"
dependencies = [
  "openenv-core",
  "rdkit",
  "scikit-learn>=1.3",
  "joblib",
  "numpy",
  "pandas",
  "PyTDC",
  "pydantic>=2.0",
  "fastapi",
  "uvicorn",
  "requests",
]

[project.optional-dependencies]
training = ["trl", "unsloth", "peft", "bitsandbytes", "torch", "transformers"]
testing = ["ollama", "pytest"]
```

### 1.2 Data Download Script

#### [NEW] `scripts/download_data.py`
- Download ZINC20 drug-like subset (5,000 SMILES) via HTTP
- Download TDC datasets: hERG, Caco2_Wang, Solubility_AqSolDB, CYP3A4_Substrate_CarbonMangels, Tox21
- Query ChEMBL API for 50 disease-target pairs with association scores
- Curate 200 PubMed abstracts for literature search (local JSON)
- Download approved drug list from DrugBank (open subset) or use RDKit's built-in

### 1.3 Seed Scenario Generation

#### [NEW] `scripts/prepare_seed_scenarios.py`
- Create 50 scenarios with varying difficulty:
  - **Easy (15)**: Well-known targets (EGFR, BRAF, ALK) with many known actives
  - **Medium (20)**: Moderately studied targets, fewer known ligands
  - **Hard (15)**: Novel/understudied targets, low druggability scores
- Each scenario: `{disease, mondo_id, target_gene, target_chembl_id, difficulty, known_actives_count}`

---

## Phase 2 — Chemistry Layer (`chemistry/`)

### 2.1 SMILES Validator (4 layers)

#### [NEW] `drug_discovery_env/chemistry/validator.py`

```python
def validate_smiles(smiles: str, allow_covalent_warhead: bool = False) -> ValidationResult:
    """4-layer validation: syntax → sanitization → synthesizability → reactive groups"""
    # Layer 1: RDKit parse (MolFromSmiles)
    # Layer 2: SanitizeMol (valence, aromaticity, kekulization)
    # Layer 3: SA Score check (reject if > 8)
    # Layer 4: SMARTS reactive group scan (alkyl halides, Michael acceptors, etc.)
    # Returns: ValidationResult(valid, mol, sa_score, warnings, rejected_reason)
```

### 2.2 Fingerprints & Similarity

#### [NEW] `drug_discovery_env/chemistry/fingerprints.py`
- Morgan FP generation (radius=2, 2048 bits)
- Tanimoto similarity between two molecules
- Batch similarity against a reference set (for novelty, selectivity)
- Murcko scaffold extraction for diversity calculation

### 2.3 Reaction Catalog

#### [NEW] `drug_discovery_env/chemistry/reactions.py`
- 5 named reactions as SMARTS: amide_coupling, reductive_amination, Suzuki_coupling, Buchwald_Hartwig, SNAr
- `execute_reaction(rxn_name, smiles_a, smiles_b)` → product SMILES or error
- Stochastic failure: `p_fail = 0.15` for novel combinations

### 2.4 Descriptors & SA Score

#### [NEW] `drug_discovery_env/chemistry/descriptors.py`
- QED, LogP, MW, HBD, HBA, PSA, heavy atom count
- Lipinski Rule-of-Five check

#### [NEW] `drug_discovery_env/chemistry/sa_scorer.py`
- Wrapper around `rdkit.Contrib.SA_Score.sascorer`

---

## Phase 3 — ML Models for ADMET (`models_ml/`)

### 3.1 Training Script

#### [NEW] `scripts/train_admet_models.py`

For each TDC dataset:
1. Load via PyTDC API
2. Compute Morgan FP (radius=2, 2048 bits) for all SMILES
3. Use TDC scaffold split (train/val/test)
4. Train sklearn `GradientBoostingClassifier` or `GradientBoostingRegressor`
5. Evaluate on test set, print metrics
6. Save with `joblib.dump()` to `models_ml/saved/`

| Model | Dataset | Type | Key Metric |
|-------|---------|------|------------|
| hERG classifier | TDC hERG (13k) | Binary classification | AUROC |
| Caco2 regressor | TDC Caco2_Wang (900) | Regression | RMSE, R² |
| Solubility regressor | TDC AqSolDB (10k) | Regression | RMSE, R² |
| CYP3A4 classifier | TDC CYP3A4_Substrate (700) | Binary classification | AUROC |
| Tox21 classifier | DeepChem Tox21 (8k, 12 tasks) | Multi-label classification | Mean AUROC |

### 3.2 Predictor Wrapper

#### [NEW] `drug_discovery_env/models_ml/predictor.py`
- `ADMETPredictor` class: loads all 5 models at init
- `predict(smiles)` → `ADMETResult(herg_prob, caco2_nms, logS, cyp3a4_substrate, tox21_alerts)`
- Handles fingerprint generation internally

---

## Phase 4 — Environment Core (`core/` + `tools/`)

### 4.1 Game State

#### [NEW] `drug_discovery_env/core/state.py`

```python
@dataclass
class GameState:
    disease_context: str
    target_protein: Optional[TargetInfo]  # name, family, druggability_score
    stage: int  # 1–5
    step: int   # 0–50
    budget_remaining: float
    budget_initial: float
    compound_ledger: Dict[str, CompoundRecord]  # SMILES → measurements
    best_compound: Optional[CompoundRecord]
    action_history: List[ActionRecord]  # last 10
    sub_agent_inbox: Dict[str, List[str]]  # agent_name → messages
    knowledge_state: List[str]  # literature findings
    admet_checked: Dict[str, int]  # SMILES → step_when_checked
    flagged_smiles: Set[str]  # toxicologist flags
```

### 4.2 Stage Manager

#### [NEW] `drug_discovery_env/core/stage_manager.py`
- Auto-advance stages based on milestones:
  - Stage 1→2: target selected
  - Stage 2→3: ≥3 compounds with binding scores
  - Stage 3→4: ≥1 compound modified with improved pIC50
  - Stage 4→5: ≥1 compound with ADMET evaluation
- Agent can also manually trigger stage transitions

### 4.3 The 8 Tools

Each tool implements: `execute(params: dict, state: GameState) → ToolResult`

| Tool File | Credit Cost | Key Logic |
|-----------|-------------|-----------|
| `select_target.py` | 5 | Druggability score = 0.4×clinical + 0.35×structural + 0.25×ligand_count |
| `search_compounds.py` | 10 | Filter compound_library.csv by RDKit property filters, return ≤20 SMILES |
| `predict_affinity.py` | 30 | Tanimoto interpolation from known actives + Gaussian noise (σ=0.3) |
| `evaluate_admet.py` | 10 | Run all 5 sklearn models, return structured results |
| `modify_molecule.py` | 20 | Parse instruction → apply RDKit transformations, p_fail=0.10 |
| `synthesize.py` | 25 | Execute SMARTS reaction, validate product, p_fail=0.15 |
| `validate_compound.py` | 80 | Simulated docking score + 8 off-target selectivity panel |
| `search_literature.py` | 5 | TF-IDF search over 200 local PubMed abstracts, return top 3 |

### 4.4 Action Parser

#### [NEW] `drug_discovery_env/core/action_parser.py`
- Parse LLM output: extract `<reasoning>`, `<tool>`, `<params>` via regex
- Validate tool name against known tools
- Validate params JSON schema per tool
- Return `ParsedAction(reasoning, tool_name, params)` or error

---

## Phase 5 — Sub-Agents, Rewards & OpenEnv Integration

### 5.1 Rule-Based Sub-Agents (`agents/`)

**Toxicologist** — runs after every new SMILES:
- PAINS check (3 catalogs via `rdkit.Chem.FilterCatalog`)
- Tox21 model prediction
- hERG classifier
- Reactive group SMARTS scan
- Output: structured warning messages

**Chemist** — runs every 5 steps:
- Identify weakest QED sub-component
- Check scaffold diversity of tested set
- Detect if agent is stuck on one scaffold
- Output: targeted improvement suggestion

**Budget Manager** — runs every step:
- Alert at 50%/70%/85% spend thresholds
- Compute credits-per-improvement efficiency
- Output: remaining action plan recommendation

**Oversight** — sklearn LogisticRegression:
- Features: (tool_called_id, n_flagged, budget_alert_active, stage, step)
- Trained on ~300 hand-labeled examples (generated during testing)
- Flags: ignoring_warning or budget_mismanagement

### 5.2 Reward Function (`rewards/`)

#### Terminal Reward (R_terminal) — `rewards/terminal.py`
- Hard floors: hERG prob > 0.5 → 0.0; PAINS ≥ 2 → 0.05
- 7 sub-components: efficiency(√(LE×LLE)), QED, ADMET composite, selectivity, novelty, synthesizability
- Weighted sum: 0.25×eff + 0.20×QED + 0.20×admet + 0.15×sel + 0.10×novelty + 0.10×synth
- Pareto bonus: 1.15× when ALL dimensions excellent

#### Process Reward (R_process) — `rewards/process.py`
- 7 per-step signals: admet_order, diversity, warning_heed, budget_stage, redundancy, premature_dock, synth_check

#### Reasoning Reward (R_reasoning) — `rewards/reasoning.py`
- Pattern matching for: hypothesis, SAR references, tradeoff language, uncertainty acknowledgment, team message integration
- Exponential decay: 0.97^t

#### Strategy Reward (R_strategy) — `rewards/strategy.py`
- all_stages_visited, progressive_improvement, exploration_convergence

#### Master Aggregation — `rewards/aggregator.py`
```
R(τ) = 0.60 × R_terminal + (1/T) × Σ[0.20 × R_process] + (1/T) × Σ[0.15 × R_reasoning] + 0.05 × R_strategy
```

### 5.3 OpenEnv Integration

#### [NEW] `drug_discovery_env/models.py`
```python
class DrugDiscoveryAction(Action):
    raw_text: str  # Full LLM output with <reasoning>/<tool>/<params>

class DrugDiscoveryObservation(Observation):
    state_text: str         # Serialized state as natural language prompt
    tool_result: Optional[str]
    sub_agent_messages: Dict[str, List[str]]
    reward: float
    done: bool

class DrugDiscoveryState(State):
    stage: int
    step: int
    budget_remaining: float
    compounds_tested: int
    best_score: float
```

#### [NEW] `drug_discovery_env/server/environment.py`
- Subclass OpenEnv `Environment`
- `reset()`: sample seed scenario, init GameState, return initial observation
- `step(action)`: parse action → execute tool → run sub-agents → compute rewards → serialize state → return StepResult
- `state()`: return current metadata

#### [NEW] `drug_discovery_env/client.py`
- Subclass OpenEnv `EnvClient`
- Type-safe wrappers for step/reset

---

## Phase 6 — Local Testing with Ollama (0.5B)

### 6.1 Test Runner

#### [NEW] `notebooks/01_test_environment.ipynb`
1. Start environment server locally (`uvicorn`)
2. Connect Ollama `qwen2.5:0.5b` as the agent
3. Run single 50-step rollout
4. Print: final reward, compounds found, stages reached, budget spent
5. Validate all reward components produce sensible values

### 6.2 Integration Test Script

#### [NEW] `scripts/run_test_rollout.py`
```python
# 1. Load env, reset with seed scenario
# 2. For each step 0–50:
#    a. Serialize state → prompt
#    b. Call ollama.generate(model="qwen2.5:0.5b", prompt=prompt)
#    c. Parse response → action
#    d. env.step(action) → observation, reward, done
#    e. Log everything
# 3. Print summary metrics
```

---

## Phase 7 — GRPO Training (GPU)

### 7.1 Training Config

#### [NEW] `training/config.py`
```python
GRPO_CONFIG = {
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "G": 8,                          # group size
    "max_completion_length": 512,
    "learning_rate": 5e-6,
    "warmup_ratio": 0.05,
    "kl_penalty_beta": 0.04,
    "clip_epsilon": 0.20,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
    "load_in_4bit": True,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "bf16": True,
    "num_iterations": 500,
}
```

### 7.2 GRPO Training Loop

#### [NEW] `training/grpo_trainer.py`
- Uses TRL `GRPOTrainer` with custom reward function
- Reward function wraps `rewards/aggregator.py`
- Each "prompt" = serialized GameState at step t
- Each "completion" = agent's `<reasoning>/<tool>/<params>` response
- Rollout loop: for each iteration, sample seed, run 50-step campaign, collect (prompt, completion, reward) tuples

#### [NEW] `notebooks/02_train_grpo.ipynb`
- Colab notebook: install unsloth + trl, load 4-bit model, run training
- Save LoRA adapter weights to HuggingFace Hub

---

## Phase 8 — Evaluation & Deployment

### 8.1 Before/After Evaluation

#### [NEW] `scripts/run_evaluation.py`
- Run 10 rollouts with untrained base model
- Run 10 rollouts with GRPO-trained model
- Report 8 metrics table:

| Metric | Untrained | Trained |
|--------|-----------|---------|
| Mean R(τ) | 0.12–0.18 | 0.52–0.65 |
| Compound quality | 0.15–0.22 | 0.55–0.70 |
| ADMET pass rate | ~20% | ~65% |
| Campaign completion | 2/10 | 8/10 |
| Budget remaining | ~0% | 28–40% |
| Tox warnings ignored | ~80% | ~15% |
| Scaffold diversity | 0.12–0.20 | 0.42–0.58 |
| hERG floor triggered | ~60% | ~12% |

### 8.2 Docker + HuggingFace Deployment
- Build Docker image with all dependencies
- Deploy to HuggingFace Spaces as OpenEnv environment
- `openenv deploy` CLI command

---

## Verification Plan

### Automated Tests
```bash
# Unit tests
pytest tests/test_validator.py      # SMILES validation (valid/invalid/edge cases)
pytest tests/test_tools.py          # Each tool returns expected types
pytest tests/test_rewards.py        # Reward ranges, hard floors, known inputs
pytest tests/test_environment.py    # Full step/reset lifecycle
pytest tests/test_agents.py         # Sub-agent message generation

# Integration test
python scripts/run_test_rollout.py  # Full 50-step rollout with Ollama 0.5B

# ADMET model quality
python scripts/train_admet_models.py --evaluate  # Print AUROC/RMSE per model
```

### Manual Verification
- Run OpenEnv web interface (`http://localhost:8000/web`) and manually step through a campaign
- Verify sub-agent messages appear at correct intervals
- Confirm budget decrements match tool costs
- Check reward components sum correctly

---

## Execution Order & Time Estimates

| Phase | Description | Est. Time | Dependencies |
|-------|-------------|-----------|--------------|
| 1 | Project setup + data download | 2 hours | None |
| 2 | Chemistry layer (validator, FP, reactions) | 3 hours | Phase 1 |
| 3 | Train ADMET sklearn models | 2 hours | Phase 1 |
| 4 | Environment core (state, tools, parser) | 5 hours | Phases 2, 3 |
| 5 | Sub-agents + rewards + OpenEnv integration | 4 hours | Phase 4 |
| 6 | Local testing with Ollama 0.5B | 2 hours | Phase 5 |
| 7 | GRPO training on Colab | 4 hours | Phase 6 |
| 8 | Evaluation + deployment | 2 hours | Phase 7 |

**Total: ~24 hours** (fits within 48-hour hackathon with margin)

> [!WARNING]
> **Critical path**: Phases 1→2→4→5 must be sequential. Phases 3 and 2 can run in parallel. Phase 6 is the first "demo-ready" checkpoint.

---

## Key Design Decisions

1. **hERG hard floor = zero reward** (not a penalty). This is non-negotiable per the spec — ensures the agent learns cardiac safety is absolute.

2. **Geometric mean for efficiency** (`√(LE×LLE)`) prevents one metric rescuing the other.

3. **ADMET = 0.10 if never measured** — forces the agent to always call `evaluate_admet` before validation.

4. **Stochastic transitions** — noise on affinity (σ=0.3), reaction failures (10–15%) — prevent the agent from memorizing deterministic paths.

5. **Sub-agents are rule-based, not LLM** — keeps complexity manageable, only Project Lead is trained.

-----------------------------------------------------------------------------------

## Perfected Implementation Plan: Advanced Drug Discovery RL Environment (v1)

### Summary
- Build a production-grade OpenEnv-compatible RL environment with three runtime dataset modes: `live_only`, `local_only`, and `hybrid`.
- Keep the 8-tool structure, but upgrade each tool with realistic biological/chemical constraints, pathway-topology effects, uncertainty, and experiment economics.
- Replace pure TF-IDF-only literature scoring with a hybrid retriever (`dense + BM25 + reranker`) while preserving lexical fallback for robustness and speed.
- Centralize all constants/configuration in one source of truth to eliminate hardcoded values.
- Deliver in two tracks: `demo-ready local` first, then `GRPO training/eval`.

### Implementation Changes (Decision-Complete)
- **Configuration and constants (single source of truth)**  
  - Add `drug_discovery_env/config/settings.py` with typed config (`pydantic-settings`) and enums.  
  - Add `config/defaults.yaml` and optional `config/overrides/*.yaml`.  
  - All tool costs, thresholds, reward weights, stochastic noise, stage gates, dataset endpoints/paths, retrieval weights, and model params must be loaded from settings only.  
  - Enforce `no magic numbers` in CI via a lightweight lint rule and config access wrappers.
- **Dataset source strategy (required 3-way selection)**  
  - Implement `DataSourceMode = {LIVE_ONLY, LOCAL_ONLY, HYBRID}`.  
  - Add `data_provider/` abstraction with adapters for Open Targets/ChEMBL/PubMed(TBD endpoint) and local snapshot loaders.  
  - In `HYBRID`, attempt live fetch with timeout/retry/circuit-breaker, then fallback to local snapshot; include provenance in observations (`source=live|local`, timestamp, confidence).  
  - Add offline snapshot preparation/versioning scripts with schema checks and checksum manifests.
- **Advanced state/action/environment design (real-world topology included)**  
  - Extend state to include: target class, pathway graph context, disease-mechanism nodes, off-target risk profile, assay confidence, uncertainty estimates, and experiment queue.  
  - Keep 8 actions but enrich params and outcomes: assay selection, evidence weighting, mechanism-aware hit triage, synthesis route feasibility, and selective validation panel design.  
  - Add topology-aware transitions: action impact propagates through pathway neighborhood (primary target + related nodes + compensatory pathways).  
  - Add realistic non-determinism: assay noise by assay type, batch effect multipliers, route-specific synthesis failure, and evidence-confidence decay.  
  - Add budget model with fixed + variable costs, opportunity cost, and late-stage penalty for redundant low-information experiments.
- **Literature retrieval and reasoning quality**  
  - Implement hybrid retrieval pipeline: dense embeddings index + BM25 lexical retrieval + cross-encoder/light reranker.  
  - Keep TF-IDF/BM25 fallback path for low-resource/offline mode; auto-select via config flag and resource checks.  
  - Add evidence-grounding output contract: each literature-derived claim must map to retrieved snippet ids/confidence.  
  - Use retrieval outputs to influence action priors and reward for evidence-consistent decisions.
- **Reward system (advanced and aligned with real constraints)**  
  - Maintain 4-part reward decomposition but upgrade signals:  
  - Terminal reward: potency/selectivity/safety/synthesizability/novelty/developability with hard clinical safety floors.  
  - Process reward: information gain per credit, stage-appropriate sequencing, uncertainty reduction, and avoiding confirmation bias loops.  
  - Reasoning reward: evidence-grounded rationale, explicit tradeoff handling, and uncertainty-aware justification.  
  - Strategy reward: campaign coherence, topology-aware exploration, and recovery after failed experiments.  
  - Normalize all components and enforce calibrated weight ranges via config; include reward diagnostics in every episode summary.
- **Public interfaces/types (explicit contracts)**  
  - `DrugDiscoveryAction` includes `tool`, `params`, `reasoning`, and optional cited evidence ids.  
  - `DrugDiscoveryObservation` includes state summary, tool result, provenance, uncertainty, sub-agent messages, and reward breakdown.  
  - `GameState` includes topology, evidence ledger, assay history, budget ledger, and best-candidate frontier.  
  - `SearchLiteratureResult` includes ranked docs, method used, and grounding metadata.
- **Sub-agents and oversight upgrades**  
  - Toxicologist: mechanism-linked alerts (hERG + pathway-mediated risk cues).  
  - Chemist: SAR + route-feasibility + scaffold exploration pressure.  
  - Budget manager: ROI and experiment-value forecasting.  
  - Oversight: detect repeated low-information loops and unsafe acceleration to validation.
- **Delivery sequence (implementation order)**  
  - Step 1: config backbone + constants migration.  
  - Step 2: data provider + 3-mode sourcing + local snapshots.  
  - Step 3: core state/stage/action upgrades with topology hooks.  
  - Step 4: literature hybrid retriever + grounding.  
  - Step 5: reward upgrades + diagnostics.  
  - Step 6: sub-agent upgrades + oversight.  
  - Step 7: OpenEnv integration hardening + local rollout runner.  
  - Step 8: GRPO training/evaluation scripts and benchmark report.

### Test Plan
- Unit tests for config loading, constant coverage, and mode switching (`live_only/local_only/hybrid`).
- Contract tests for every tool input/output schema and provenance fields.
- Deterministic-seed simulation tests for stage progression, budget accounting, and stochastic bounds.
- Literature tests comparing retrieval quality across TF-IDF fallback vs hybrid retriever on labeled query-doc pairs.
- Reward tests for hard floors, normalization, and component attribution correctness.
- End-to-end rollouts (50-step) in all three data modes with pass/fail gates on completion, safety floor compliance, and budget efficiency.
- Regression benchmark suite: untrained vs trained model metrics with confidence intervals.

### Assumptions and Defaults
- Runtime must support offline execution; `local_only` is always available if snapshots exist.
- Default mode is `hybrid`, with per-run override via env var/CLI/config.
- Default literature mode is `hybrid dense+BM25+rerank`; fallback to lexical when embedding stack unavailable.
- Existing 8-tool action surface is retained for hackathon compatibility; sophistication is added through richer parameters and transition logic, not by changing tool names.
- All numerical values and thresholds are configurable from the centralized settings system only.
