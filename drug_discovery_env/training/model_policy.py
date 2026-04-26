from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

from drug_discovery_env.core.models import DrugDiscoveryAction, DrugDiscoveryObservation

SYSTEM_PROMPT = """You are the Project Lead in a simulated drug discovery campaign.

Goal: take a disease through 5 stages (target_selection -> hit_id ->
hit_to_lead -> admet -> lead_validation) and nominate the best lead compound
before your budget or step cap runs out. You have a team of sub-agents
(chemist, toxicologist, budget, oversight) advising you each step. Listen to
toxicology BLOCK warnings. Conserve budget; validate_compound is the most
expensive tool - reserve it for verified candidates.

You MUST reply with a single JSON object on one line, with these keys:
  "tool":  one of select_target, search_compounds, predict_affinity,
           evaluate_admet, modify_molecule, synthesize, validate_compound,
           search_literature, advance_stage, abandon_compound,
           pause_and_review_all, delegate_to_subagent, request_subagent_summary
  "params": object of tool-specific arguments (e.g. {"disease": "Type 2 Diabetes"} or
           {"smiles": "...", "instruction": "add fluorine"})
  "target_compound_id": the C### id of the active compound this acts on, or null
  "reasoning": one or two sentences justifying the decision in scientific terms
               (mention binding, ADMET, PAINS, Lipinski, scaffold, selectivity,
               novelty, budget, hypothesis, uncertainty, tradeoff)

Do not output anything outside the JSON object."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def render_observation(obs_dict: Dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        f"Stage: {obs_dict.get('stage')}  Step: {obs_dict.get('step_index')}/{obs_dict.get('max_steps')}"
    )
    lines.append(
        f"Disease: {obs_dict.get('disease')}  Target: {obs_dict.get('selected_target')}"
    )
    lines.append(
        f"Budget: {obs_dict.get('budget_remaining', 0):.0f}/{obs_dict.get('budget_total', 0):.0f}"
        f"  (last cost {obs_dict.get('last_tool_cost', 0):.0f})"
    )
    if obs_dict.get("last_tool"):
        lines.append(
            f"Last tool: {obs_dict['last_tool']} -> "
            f"{json.dumps(obs_dict.get('last_result', {}))[:240]}"
        )
    msgs = obs_dict.get("subagent_messages") or []
    if msgs:
        lines.append("Sub-agent messages:")
        for msg in msgs[:8]:
            lines.append(f"  [{msg['agent']}/{msg['severity']}] {msg['message']}")
    actives = obs_dict.get("active_compounds") or []
    if actives:
        lines.append(f"Active compounds ({len(actives)}):")
        for compound in actives[:8]:
            admet = compound.get("admet") or {}
            smiles = (compound.get("smiles") or "")[:40]
            lines.append(
                f"  {compound.get('id')}  smiles={smiles}"
                f"  aff={compound.get('binding_affinity_nM')}"
                f"  dock={compound.get('docking_score')}"
                f"  ro5={admet.get('ro5_pass')} pains={admet.get('pains')} tox={admet.get('tox_score')}"
            )
    lines.append(f"Status: {obs_dict.get('message', '')}")
    return "\n".join(lines)


def prompt_text_from_observation(observation: DrugDiscoveryObservation | Dict[str, Any]) -> str:
    obs_dict = observation if isinstance(observation, dict) else observation.model_dump()
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Observation:\n{render_observation(obs_dict)}\n\n"
        "Return one JSON object only."
    )


def parse_action_text(text: str) -> Dict[str, Any]:
    match = _JSON_RE.search(text or "")
    if not match:
        return {"tool": "pause_and_review_all", "params": {}, "reasoning": text or ""}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {"tool": "pause_and_review_all", "params": {}, "reasoning": text or ""}


def action_from_text(text: str) -> DrugDiscoveryAction:
    payload = parse_action_text(text)
    return DrugDiscoveryAction(
        tool=str(payload.get("tool", "pause_and_review_all")),
        params=payload.get("params", {}) or {},
        target_compound_id=payload.get("target_compound_id"),
        reasoning=str(payload.get("reasoning", "")),
    )


def resolve_torch_device(requested: str) -> str:
    import torch

    mode = requested.lower()
    if mode != "auto":
        return mode
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_loadable_model_dir(path: Path) -> bool:
    if (path / "adapter_config.json").exists():
        return True
    config_path = path / "config.json"
    if not config_path.exists():
        return False
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return "model_type" in payload


def resolve_model_artifact_path(model_name_or_path: str) -> Path | None:
    path = Path(model_name_or_path)
    if not path.exists():
        return None
    if path.is_file():
        return path.parent
    if _is_loadable_model_dir(path):
        return path

    checkpoints = [child for child in path.iterdir() if child.is_dir() and child.name.startswith("checkpoint-")]
    if not checkpoints:
        return None

    def checkpoint_key(candidate: Path) -> tuple[int, float]:
        try:
            suffix = int(candidate.name.split("-")[-1])
        except Exception:
            suffix = -1
        return (suffix, candidate.stat().st_mtime)

    for candidate in sorted(checkpoints, key=checkpoint_key, reverse=True):
        if _is_loadable_model_dir(candidate):
            return candidate
    return None


class TransformersToolPolicy:
    def __init__(self, model_name_or_path: str, *, device: str = "auto", max_new_tokens: int = 256) -> None:
        import torch
        from transformers import AutoConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.device = resolve_torch_device(device)
        self.max_new_tokens = max_new_tokens

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        resolved_ref = resolve_model_artifact_path(model_name_or_path)
        model_ref = resolved_ref or Path(model_name_or_path)
        model_load_ref = str(model_ref) if resolved_ref else model_name_or_path

        tokenizer_ref = model_load_ref
        if model_ref.exists() and (model_ref / "adapter_config.json").exists():
            from peft import AutoPeftModelForCausalLM

            adapter_config = json.loads((model_ref / "adapter_config.json").read_text(encoding="utf-8"))
            tokenizer_ref = str(adapter_config.get("base_model_name_or_path") or model_load_ref)
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_load_ref,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_load_ref,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
            )

        if self.device != "cuda" and getattr(self.model, "device", None) is None:
            self.model = self.model.to(self.device)

        if model_ref.exists():
            has_local_tokenizer = any(
                (model_ref / name).exists()
                for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
            )
            if not has_local_tokenizer and (model_ref / "config.json").exists():
                try:
                    config = AutoConfig.from_pretrained(model_load_ref)
                    tokenizer_ref = getattr(config, "_name_or_path", tokenizer_ref) or tokenizer_ref
                except Exception:
                    pass

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prompt_for(self, observation: DrugDiscoveryObservation) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_observation(observation.model_dump())},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{SYSTEM_PROMPT}\n\n{messages[-1]['content']}\n"

    def next_action(self, observation: DrugDiscoveryObservation) -> DrugDiscoveryAction:
        import torch

        prompt = self._prompt_for(observation)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "cuda":
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[-1]
        completion = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return action_from_text(completion)
