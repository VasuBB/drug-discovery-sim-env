"""Transformers-backed Project-Lead policy used for evaluation / inference.

Loads either a base HuggingFace checkpoint or a PEFT adapter directory saved
by the GRPO trainer. The same prompting contract (`prompting.py`) is used end
to end so train / eval / infer paths are identical.
"""

from __future__ import annotations

import json
from pathlib import Path

from drug_discovery_env.core.models import DrugDiscoveryAction, DrugDiscoveryObservation
from drug_discovery_env.training.prompting import (
    SYSTEM_PROMPT,
    action_from_text,
    render_observation,
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


class TransformersToolPolicy:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "auto",
        max_new_tokens: int = 384,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.device = resolve_torch_device(device)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        ref = Path(model_name_or_path)
        tokenizer_ref = model_name_or_path
        if ref.exists() and (ref / "adapter_config.json").exists():
            from peft import AutoPeftModelForCausalLM

            adapter_cfg = json.loads((ref / "adapter_config.json").read_text(encoding="utf-8"))
            tokenizer_ref = str(adapter_cfg.get("base_model_name_or_path") or model_name_or_path)
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
            )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prompt_for(self, observation: DrugDiscoveryObservation) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_observation(observation)},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{SYSTEM_PROMPT}\n\n{messages[-1]['content']}\n"

    def generate_text(self, observation: DrugDiscoveryObservation) -> str:
        import torch

        prompt = self._prompt_for(observation)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            target_device = "cuda"
        else:
            model_device = getattr(self.model, "device", None)
            target_device = str(model_device) if model_device is not None else self.device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(0.01, self.temperature),
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        in_len = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(outputs[0][in_len:], skip_special_tokens=True)

    def next_action(self, observation: DrugDiscoveryObservation) -> tuple[DrugDiscoveryAction, str]:
        text = self.generate_text(observation)
        return action_from_text(text), text
