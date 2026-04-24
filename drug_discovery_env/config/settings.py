from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSourceMode(str, Enum):
    LIVE_ONLY = "live_only"
    LOCAL_ONLY = "local_only"
    HYBRID = "hybrid"


class RetrievalMode(str, Enum):
    HYBRID = "hybrid"
    LEXICAL_ONLY = "lexical_only"
    DENSE_ONLY = "dense_only"


class EndpointConfig(BaseModel):
    open_targets_url: str
    chembl_url: str
    pubmed_esearch_url: str
    pubmed_efetch_url: str


class AppConfig(BaseModel):
    env_name: str
    max_steps: int
    random_seed: int


class DataConfig(BaseModel):
    mode: DataSourceMode
    local_data_dir: str
    request_timeout_seconds: float
    retries: int
    circuit_breaker_failures: int
    cache_ttl_seconds: int
    user_agent: str
    pubmed_tool: str
    pubmed_email: str
    endpoints: EndpointConfig


class ToolConfig(BaseModel):
    costs: dict[str, float]
    assay_noise: dict[str, float]
    synthesis_failure_base: float
    batch_effect_multiplier: float
    evidence_confidence_decay: float


class TransitionConfig(BaseModel):
    stage_gates: dict[str, int]
    topology: dict[str, float | int]


class BudgetConfig(BaseModel):
    initial_credits: float
    opportunity_cost_factor: float
    late_stage_redundancy_penalty: float
    variable_cost_multipliers: dict[str, float]


class RetrievalConfig(BaseModel):
    mode: RetrievalMode
    top_k: int
    bm25_weight: float
    dense_weight: float
    rerank_weight: float
    fallback_min_docs: int
    dense_model_name: str
    reranker_model_name: str
    use_transformer_models: bool


class RewardConfig(BaseModel):
    weights: dict[str, float]
    floors: dict[str, float | int]
    terminal_components: dict[str, float]


class AgentConfig(BaseModel):
    toxicologist_alert_threshold: float
    oversight_loop_window: int
    oversight_low_info_threshold: float


class TrainingConfig(BaseModel):
    model_name: str
    learning_rate: float
    group_size: int
    lora_rank: int
    iterations: int
    max_completion_length: int
    warmup_ratio: float
    kl_penalty_beta: float
    clip_epsilon: float


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DD_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: AppConfig
    data: DataConfig
    tools: ToolConfig
    transitions: TransitionConfig
    budget: BudgetConfig
    retrieval: RetrievalConfig
    reward: RewardConfig
    agents: AgentConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls(**payload)


@lru_cache(maxsize=1)
def get_settings(config_path: str | Path | None = None) -> Settings:
    root = Path(__file__).resolve().parents[2]
    default_path = root / "config" / "defaults.yaml"
    path = Path(config_path) if config_path else default_path
    return Settings.from_yaml(path)


def get_constant(settings: Settings, section: str, key: str, default: Any | None = None) -> Any:
    value = getattr(settings, section)
    if isinstance(value, BaseModel):
        return value.model_dump().get(key, default)
    return default
