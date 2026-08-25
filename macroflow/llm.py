"""VLM client — runs Qwen3-VL (or compatible) in-process via transformers.

Self-contained (no dependency on the legacy ``openlayout`` package). Loads the
model once and reuses it; supports multimodal messages (text + layout images)
and tool-use through Qwen3's ``<tool_call>`` chat-template format. Multi-GPU via
``device_map="auto"``.

Note on images in tool results: Qwen tool-role messages are text-only, so a tool
that wants the model to *see* an image cannot return it directly. The agent loop
(:mod:`macroflow.agent`) instead appends the image as a follow-up ``user``
message, which :func:`_extract_images` then feeds to the processor. This keeps
the "which image to view" decision explicit and the context prunable.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw: Any = None
    prompt_tokens: int = 0          # input tokens (incl. image placeholders)
    completion_tokens: int = 0      # generated tokens
    gen_seconds: float = 0.0        # generation wall time


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Extract ``<tool_call>{...}</tool_call>`` blocks; return (clean_text, calls)."""
    calls: list[ToolCall] = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:8]}",
                                  name=obj["name"], input=obj.get("arguments", {})))
        except (json.JSONDecodeError, KeyError):
            continue
    clean = _TOOL_CALL_RE.sub("", text).strip()
    return clean, calls


def _extract_images(messages: list[dict]):
    """Collect PIL images from any message content block of type 'image'."""
    from PIL import Image as PILImage
    images = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            src = item.get("image", "")
            if src.startswith("file://"):
                src = src[len("file://"):]
            if os.path.isfile(src):
                images.append(PILImage.open(src).convert("RGB"))
    return images or None


def _convert_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Anthropic tool defs -> OpenAI function-calling format (Qwen template)."""
    out = []
    for tool in anthropic_tools:
        out.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return out


def read_provider(config_path: str | Path) -> str:
    """Return the ``provider:`` field of a model config ('local' if absent)."""
    try:
        raw = yaml.safe_load(Path(config_path).read_text()) or {}
    except (OSError, ValueError):
        return "local"
    return str(raw.get("provider", "local")).strip().lower() or "local"


def build_vlm(config_path: str | Path):
    """Construct the model client selected by the config's ``provider:`` field.

    ``local`` (default) -> in-process Qwen3-VL via transformers.
    ``openai``          -> OpenAI Responses-API client (see
                           :mod:`macroflow.llm_openai`), which exposes the same
                           ``chat``/transcript interface.
    ``vllm``            -> local vLLM server over OpenAI chat-completions (see
                           :mod:`macroflow.llm_vllm`). Used for models too large
                           to load once per candidate worker.
    """
    provider = read_provider(config_path)
    if provider == "openai":
        from .llm_openai import OpenAIVLM
        return OpenAIVLM.from_config(config_path)
    if provider == "vllm":
        from .llm_vllm import VLLMClient
        return VLLMClient.from_config(config_path)
    if provider != "local":
        raise ValueError(
            f"unsupported provider '{provider}' in {config_path} "
            "(expected 'local', 'openai' or 'vllm')")
    return VLM.from_config(config_path)


def read_model_config(config_path: str | Path) -> tuple[str, str, str | None]:
    """Return (model_path, model_name, lora_path) from a local-provider YAML config.

    ``lora_path`` (optional ``local.lora_path``) points at a PEFT LoRA adapter dir
    whose weights are merged into the base model at load time (see
    :meth:`VLM._merge_lora`). ``None`` when no adapter is configured.
    """
    raw = yaml.safe_load(Path(config_path).read_text())
    local = raw.get("local", {})
    model_path = local.get("model_path", "")
    model_name = local.get("model", os.path.basename(model_path.rstrip("/")))
    lora_path = local.get("lora_path") or None
    return model_path, model_name, lora_path


# ---------------------------------------------------------------------------
# VLM
# ---------------------------------------------------------------------------
class VLM:
    """In-process Qwen3-VL inference client with tool-use + vision."""

    def __init__(self, model_path: str, model_name: str | None = None,
                 lora_path: str | None = None):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self._torch = torch
        print(f"[VLM] Loading model from {model_path} ...", flush=True)
        print(f"[VLM] Available GPUs: {torch.cuda.device_count()}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # transformers 5.x broke multi-GPU sharding for Qwen3-VL: the rotary
        # embedding does a cross-device matmul (inv_freq on cuda:0, position_ids
        # on a later shard) → CUBLAS_STATUS_EXECUTION_FAILED at generate time.
        # The 8B model fits on a single 46GB A6000, and the flow already pins one
        # GPU per parallel worker, so load the whole model on one device rather
        # than sharding. (Set MACROFLOW_DEVICE_MAP=auto to force the old sharding.)
        # Pin the whole model to one device whenever CUDA is available, for ANY
        # visible-GPU count. Sharding is broken (see above), and "auto" leaves
        # placement to accelerate's free-memory probe — which this host cannot do
        # reliably while NVML is version-mismatched with the kernel module
        # ("Can't initialize NVML"), so "auto" can silently fall back to CPU. CPU
        # inference is ~100x slower and turns a 20-minute run into days, so it
        # must never be entered by accident.
        device_map: Any = "auto"
        if os.environ.get("MACROFLOW_DEVICE_MAP", "").strip().lower() != "auto":
            if torch.cuda.is_available():
                device_map = {"": 0}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device_map,
            trust_remote_code=True,
        )
        if lora_path:
            self._merge_lora(lora_path)
        self.model.eval()
        self.model_name = model_name or os.path.basename(str(model_path).rstrip("/"))
        devices = {str(p.device) for p in self.model.parameters()}
        on_cpu = sorted(d for d in devices if d.startswith("cpu"))
        if on_cpu and torch.cuda.is_available():
            print(f"[VLM] WARNING: {len(on_cpu)} parameter device(s) on CPU "
                  f"({devices}) despite CUDA being available — generation will be "
                  "far too slow for the flow. Check accelerate/NVML.", flush=True)
        print(f"[VLM] Loaded: {self.model_name} | devices: {sorted(devices)} | "
              f"device map: {getattr(self.model, 'hf_device_map', 'single')}", flush=True)

    # ------------------------------------------------------------------
    def _merge_lora(self, lora_path: str | Path) -> None:
        """Merge a PEFT LoRA adapter into the base weights in place.

        Done manually (no ``peft`` import) because peft 0.17 is incompatible with
        transformers 5.x (it imports the removed ``HybridCache``). Only standard
        LoRA is supported — no DoRA, no bias, no ``modules_to_save`` — which is
        exactly what this adapter is (``adapter_config.json``: r=64, alpha=128,
        use_dora=false, use_rslora=false, bias=none). For each target Linear the
        merged weight is ``W += (alpha/r) * (B @ A)``.
        """
        import torch
        from safetensors.torch import load_file

        lp = Path(lora_path)
        acfg = json.loads((lp / "adapter_config.json").read_text())
        if acfg.get("use_dora"):
            raise RuntimeError("DoRA adapters are not supported by _merge_lora")
        if acfg.get("modules_to_save"):
            raise RuntimeError("modules_to_save adapters are not supported by _merge_lora")
        r = int(acfg["r"])
        alpha = float(acfg["lora_alpha"])
        scaling = alpha / ((r ** 0.5) if acfg.get("use_rslora") else r)
        print(f"[VLM] Merging LoRA {lp} (r={r} alpha={alpha:g} "
              f"scaling={scaling:g}) ...", flush=True)

        sd = load_file(str(lp / "adapter_model.safetensors"))
        pairs: dict[str, dict[str, Any]] = {}
        for k, v in sd.items():
            if k.endswith(".lora_A.weight"):
                pairs.setdefault(k[: -len(".lora_A.weight")], {})["A"] = v
            elif k.endswith(".lora_B.weight"):
                pairs.setdefault(k[: -len(".lora_B.weight")], {})["B"] = v

        named = dict(self.model.named_modules())
        merged = 0
        missing: list[str] = []
        for base, ab in pairs.items():
            if "A" not in ab or "B" not in ab:
                missing.append(base)
                continue
            # peft wraps the base model as ``base_model.model.<real module path>``.
            mod_name = base
            if mod_name.startswith("base_model.model."):
                mod_name = mod_name[len("base_model.model."):]
            module = named.get(mod_name)
            if module is None or not hasattr(module, "weight"):
                missing.append(mod_name)
                continue
            W = module.weight
            A = ab["A"].to(device=W.device, dtype=torch.float32)
            B = ab["B"].to(device=W.device, dtype=torch.float32)
            delta = (B @ A) * scaling  # (out, in)
            if tuple(delta.shape) != tuple(W.shape):
                raise RuntimeError(
                    f"LoRA delta shape {tuple(delta.shape)} != weight "
                    f"{tuple(W.shape)} for {mod_name}")
            with torch.no_grad():
                W.add_(delta.to(dtype=W.dtype))
            merged += 1
        print(f"[VLM] LoRA merged into {merged} modules "
              f"({len(missing)} unmatched)", flush=True)
        if merged == 0:
            raise RuntimeError(f"LoRA merge matched 0 modules from {lp}")
        if missing:
            print(f"[VLM] WARNING: unmatched LoRA modules: {missing[:8]}"
                  f"{' ...' if len(missing) > 8 else ''}", flush=True)

    @classmethod
    def from_config(cls, config_path: str | Path) -> "VLM":
        model_path, model_name, lora_path = read_model_config(config_path)
        return cls(model_path, model_name, lora_path=lora_path)

    # ------------------------------------------------------------------
    def chat(self, messages: list[dict], system: str, tools: list[dict] | None = None,
             max_tokens: int = 4096) -> LLMResponse:
        """Generate one assistant turn, parsing ``<tool_call>`` blocks."""
        torch = self._torch
        full = [{"role": "system", "content": system}] + messages
        openai_tools = _convert_tools(tools) if tools else None

        text_input = self.processor.apply_chat_template(
            full, tools=openai_tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        images = _extract_images(full)
        proc_kwargs: dict[str, Any] = {"text": [text_input], "return_tensors": "pt",
                                       "padding": True}
        if images:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs).to(self.model.device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                temperature=None, top_p=None,
            )
        gen_seconds = time.time() - t0
        input_len = inputs["input_ids"].shape[1]
        completion_len = int(output_ids.shape[1] - input_len)
        gen = self.processor.decode(output_ids[0, input_len:], skip_special_tokens=True)

        clean_text, tool_calls = _parse_tool_calls(gen)
        stop = "tool_use" if tool_calls else "end_turn"
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": clean_text or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.input)}}
                for tc in tool_calls
            ]
        return LLMResponse(text=clean_text, tool_calls=tool_calls,
                           stop_reason=stop, raw=assistant_msg,
                           prompt_tokens=int(input_len),
                           completion_tokens=completion_len, gen_seconds=gen_seconds)

    # ------------------------------------------------------------------
    @staticmethod
    def build_assistant_message(raw) -> dict:
        return raw

    @staticmethod
    def format_tool_result(tool_call_id: str, content) -> dict:
        """Text-only tool-role message (see module docstring re: images)."""
        if isinstance(content, list):
            parts = []
            for b in content:
                if isinstance(b, dict):
                    if b.get("type") == "text":
                        parts.append(b["text"])
                    elif b.get("type") == "image":
                        parts.append("[image attached in a follow-up message]")
                else:
                    parts.append(str(b))
            content_str = "\n".join(parts)
        else:
            content_str = str(content)
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content_str}

    @staticmethod
    def build_tool_results_message(tool_results: list[dict]) -> list[dict]:
        return tool_results
