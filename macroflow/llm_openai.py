"""OpenAI-backed VLM client — same interface as :class:`macroflow.llm.VLM`.

Drop-in replacement so the whole macro-placement flow (Planner / Placer /
Checker / Optimizer agents, Gate A + Gate B loops, parallel candidate workers)
can be driven by an OpenAI reasoning model instead of the in-process Qwen3-VL.

Interface contract (see ``macroflow.agent.Agent``):

  * ``chat(messages, system, tools, max_tokens) -> LLMResponse``
  * ``build_assistant_message(raw)`` / ``format_tool_result(id, content)`` /
    ``build_tool_results_message(results)``
  * ``model_name`` attribute

Uses the **Responses API** (``/v1/responses``), not chat-completions, because
that is the only endpoint where gpt-5.x supports *reasoning together with
function tools*. Sending ``reasoning_effort`` alongside tools to
``/v1/chat/completions`` returns 400 ("Function tools with reasoning_effort are
not supported for gpt-5.4 ... use /v1/responses"), and every agent in this flow
is a tool-calling agent.

Two hard rules of the Responses tool loop, both learned from 400s and enforced
below in :meth:`_reusable`:

  1. Output items fed back on the next turn must have their ``status`` field
     stripped ("Unknown parameter: input[1].status").
  2. ``reasoning`` items must be kept and stay immediately before the
     ``function_call`` they produced ("Item 'fc_...' of type 'function_call' was
     provided without its required 'reasoning' item: 'rs_...'"). So the whole
     output item list is carried forward verbatim, in order.

macroflow's internal message shape (content is a string, or a list of
``{"type": "text"}`` / ``{"type": "image", "image": "file://..."}`` blocks) is
translated here; images are inlined as base64 ``data:`` URIs.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from .llm import LLMResponse, ToolCall


# Reasoning models bill internal reasoning against the output budget. The agent
# loop asks for ~1536 tokens, which reasoning alone can exhaust — leaving an
# "incomplete" response with no tool call. Give the model a real floor.
_MIN_OUTPUT_BUDGET = 16384

_MAX_ATTEMPTS = 5
_BACKOFF_BASE = 2.0
_VALID_EFFORTS = {"minimal", "low", "medium", "high"}


def read_openai_config(config_path: str | Path) -> dict:
    """Return the ``openai:`` block of a provider-openai YAML config.

    Recognized keys: ``model``, ``api_key``, ``api_key_file``, ``base_url``,
    ``reasoning_effort`` (minimal|low|medium|high|none), ``max_output_tokens``,
    ``max_attempts``, ``timeout``.
    """
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    cfg = raw.get("openai", {}) or {}
    return cfg if isinstance(cfg, dict) else {}


def resolve_api_key(cfg: dict, config_path: str | Path | None = None) -> str:
    """Resolve the API key from config or environment.

    Order: explicit ``openai.api_key`` (``${VAR}`` is expanded) -> ``$OPENAI_API_KEY``
    -> ``openai.api_key_file`` (relative paths resolve against the config file).
    The key file is the robust option for parallel candidate workers, which are
    spawned as separate processes that may not inherit a shell-exported variable.
    """
    key = str(cfg.get("api_key") or "").strip()
    if key.startswith("${") and key.endswith("}"):
        key = os.environ.get(key[2:-1], "").strip()
    if key:
        return key
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    kf = cfg.get("api_key_file")
    if kf:
        p = Path(kf)
        if not p.is_absolute() and config_path is not None:
            p = Path(config_path).resolve().parent / p
        if p.exists():
            return p.read_text().strip()
    raise RuntimeError(
        "no OpenAI API key: set openai.api_key / openai.api_key_file in the "
        "model config, or export OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# Message / tool conversion
# ---------------------------------------------------------------------------
_B64_CACHE: dict[str, str] = {}

# Marker key for an assistant turn carried as raw Responses output items.
_ITEMS_KEY = "_openai_items"


def _data_uri(path: str) -> str | None:
    """Read an image file into a base64 ``data:`` URI (cached per path+mtime)."""
    if path.startswith("file://"):
        path = path[len("file://"):]
    if not os.path.isfile(path):
        return None
    try:
        key = f"{path}:{os.path.getmtime(path)}"
    except OSError:
        return None
    hit = _B64_CACHE.get(key)
    if hit is not None:
        return hit
    mime = mimetypes.guess_type(path)[0] or "image/png"
    try:
        uri = f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"
    except OSError:
        return None
    _B64_CACHE[key] = uri
    return uri


def _user_content(content) -> list[dict]:
    """macroflow content (str or blocks) -> Responses input_text/input_image parts."""
    if not isinstance(content, list):
        return [{"type": "input_text", "text": str(content)}]
    parts: list[dict] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append({"type": "input_text", "text": str(item)})
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append({"type": "input_text", "text": item.get("text", "")})
        elif kind == "image":
            uri = _data_uri(item.get("image", ""))
            parts.append({"type": "input_image", "image_url": uri} if uri else
                         {"type": "input_text",
                          "text": f"[image unavailable: {item.get('image', '')}]"})
    return parts or [{"type": "input_text", "text": ""}]


def _convert_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Anthropic tool defs -> Responses function-tool format.

    ``strict`` is left off (non-strict): macroflow's schemas do not all set
    ``additionalProperties: false``, which strict mode requires.
    """
    return [{
        "type": "function",
        "name": t["name"],
        "description": t.get("description", ""),
        "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
        "strict": False,
    } for t in anthropic_tools]


def _build_input(messages: list[dict]) -> list[dict]:
    """Convert a macroflow transcript into a Responses ``input`` list."""
    out: list[dict] = []
    for msg in messages:
        if _ITEMS_KEY in msg:                      # prior assistant turn
            out.extend(msg[_ITEMS_KEY])
        elif msg.get("type") == "function_call_output":
            out.append(msg)
        elif msg.get("role") == "user":
            out.append({"role": "user", "content": _user_content(msg.get("content"))})
        elif msg.get("role") == "assistant":       # plain text assistant turn
            out.append({"role": "assistant",
                        "content": [{"type": "output_text",
                                     "text": str(msg.get("content", ""))}]})
    return out


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class OpenAIVLM:
    """OpenAI Responses-API client with reasoning, vision and tool use.

    Mirrors :class:`macroflow.llm.VLM` so it can be swapped in without touching
    the agent loop or the orchestrator.
    """

    def __init__(self, model: str, api_key: str, base_url: str | None = None,
                 reasoning_effort: str | None = "medium",
                 max_output_tokens: int = _MIN_OUTPUT_BUDGET,
                 max_attempts: int = _MAX_ATTEMPTS, timeout: float = 600.0):
        from openai import OpenAI

        self.model_name = model
        self.max_attempts = max(1, int(max_attempts))
        self.max_output_tokens = max(256, int(max_output_tokens))
        eff = (reasoning_effort or "").strip().lower()
        self.reasoning_effort = eff if eff in _VALID_EFFORTS else None
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        print(f"[VLM] OpenAI Responses client: {self.model_name} | reasoning="
              f"{self.reasoning_effort or 'off'} | max_output_tokens="
              f"{self.max_output_tokens}", flush=True)

    @classmethod
    def from_config(cls, config_path: str | Path) -> "OpenAIVLM":
        cfg = read_openai_config(config_path)
        return cls(
            model=str(cfg.get("model") or "gpt-5.4").strip(),
            api_key=resolve_api_key(cfg, config_path),
            base_url=cfg.get("base_url") or None,
            reasoning_effort=cfg.get("reasoning_effort", "medium"),
            max_output_tokens=int(cfg.get("max_output_tokens", _MIN_OUTPUT_BUDGET)),
            max_attempts=int(cfg.get("max_attempts", _MAX_ATTEMPTS)),
            timeout=float(cfg.get("timeout", 600.0)),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _reusable(item: Any) -> dict:
        """Output item -> a dict safe to feed back as input on the next turn.

        Strips ``status``, which the API rejects on input items. Everything else
        (including ``reasoning`` items and their ids) is preserved verbatim so
        each ``function_call`` keeps the ``reasoning`` item it requires.
        """
        d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        d.pop("status", None)
        return d

    def _create(self, payload: dict):
        """One API call with retry/backoff on transient errors."""
        from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                            InternalServerError, RateLimitError)

        transient = (RateLimitError, APIConnectionError, APITimeoutError,
                     InternalServerError)
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                return self.client.responses.create(**payload)
            except transient as e:
                last = e
                if attempt == self.max_attempts - 1:
                    break
                delay = (_BACKOFF_BASE ** attempt) + random.random()
                print(f"[VLM] {type(e).__name__} — retry "
                      f"{attempt + 1}/{self.max_attempts - 1} in {delay:.1f}s",
                      flush=True)
                time.sleep(delay)
            except APIStatusError:
                raise           # 4xx other than 429: not retryable
        raise last if last else RuntimeError("responses.create failed")

    def chat(self, messages: list[dict], system: str,
             tools: list[dict] | None = None,
             max_tokens: int = 4096) -> LLMResponse:
        """Generate one assistant turn (text + optional tool calls)."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "instructions": system,
            "input": _build_input(messages),
            "max_output_tokens": max(int(max_tokens), self.max_output_tokens),
        }
        if tools:
            payload["tools"] = _convert_tools(tools)
            payload["tool_choice"] = "auto"
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        t0 = time.time()
        resp = self._create(payload)
        # Ran out of budget mid-reasoning with nothing to show: one retry with a
        # doubled cap, so a thinking-heavy turn does not silently no-op.
        if (getattr(resp, "status", "") == "incomplete"
                and not any(o.type in ("function_call", "message") for o in resp.output)):
            payload["max_output_tokens"] = min(payload["max_output_tokens"] * 2, 100_000)
            print("[VLM] incomplete (reasoning exhausted budget) — retrying with "
                  f"max_output_tokens={payload['max_output_tokens']}", flush=True)
            resp = self._create(payload)
        gen_seconds = time.time() - t0

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for o in resp.output:
            if o.type == "message":
                for part in (o.content or []):
                    if getattr(part, "type", "") == "output_text":
                        text_parts.append(part.text)
            elif o.type == "function_call":
                try:
                    args = json.loads(o.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                if not isinstance(args, dict):
                    args = {"value": args}
                tool_calls.append(ToolCall(
                    id=o.call_id or f"call_{uuid.uuid4().hex[:8]}",
                    name=o.name, input=args))
        text = "\n".join(p for p in text_parts if p).strip()

        # Carry the raw output items forward verbatim (reasoning included).
        raw = {_ITEMS_KEY: [self._reusable(o) for o in resp.output]}

        usage = resp.usage
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            raw=raw,
            prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            gen_seconds=gen_seconds,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def build_assistant_message(raw) -> dict:
        return raw

    @staticmethod
    def format_tool_result(tool_call_id: str, content) -> dict:
        """Responses ``function_call_output`` item (text only).

        Images a tool wants the model to see are appended by the agent loop as a
        follow-up user message, exactly as on the Qwen path.
        """
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
        return {"type": "function_call_output", "call_id": tool_call_id,
                "output": content_str}

    @staticmethod
    def build_tool_results_message(tool_results: list[dict]) -> list[dict]:
        return tool_results
