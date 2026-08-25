"""vLLM (OpenAI-compatible chat-completions) client for macroflow.

Same interface as :class:`macroflow.llm.VLM` — ``chat(messages, system, tools,
max_tokens) -> LLMResponse`` — so the Planner/Placer/Checker agents and the
parallel candidate workers are unchanged.

**Why a server instead of in-process transformers.** The in-process path loads
one full copy of the model per candidate worker. That is fine for the 8B
(5 x 17GB on 5 GPUs) but impossible for Qwen3-VL-30B-A3B (58GB bf16 vs 44.4GB
usable per A6000). A single vLLM server holds one copy, serves all 5 workers
concurrently, and uses fused MoE kernels instead of transformers' eager
128-expert routing.

Distinct from :mod:`macroflow.llm_openai`, which targets the OpenAI **Responses**
API for reasoning models (gpt-5.4). vLLM speaks chat-completions, and tool calls
come back as native ``tool_calls`` (served with ``--enable-auto-tool-choice
--tool-call-parser hermes``), so this module parses those rather than scraping
``<tool_call>`` blocks out of the text — though it falls back to the text form,
because a Qwen template can still emit them inline.

Images are sent as base64 ``data:`` URIs: the agents pass local paths as
``{"type": "image", "image": "file:///..."}``, which no HTTP server can read.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from .llm import LLMResponse, ToolCall, _convert_tools, _parse_tool_calls

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_TIMEOUT = 1800.0
DEFAULT_MAX_ATTEMPTS = 4


def _image_data_uri(path: str) -> str | None:
    """Read a local image into a base64 data URI, or None if unreadable."""
    if path.startswith("file://"):
        path = path[len("file://"):]
    if not os.path.isfile(path):
        return None
    mime = mimetypes.guess_type(path)[0] or "image/png"
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _to_openai_messages(messages: list[dict], system: str) -> list[dict]:
    """Translate macroflow's Anthropic-style messages to chat-completions form.

    Text blocks are concatenated; image blocks become ``image_url`` parts with an
    inlined data URI. Any block whose image cannot be read is dropped rather than
    failing the turn — a missing snapshot must not abort a candidate.
    """
    out: list[dict] = [{"role": "system", "content": system}]
    for msg in messages:
        content = msg.get("content")
        role = msg.get("role", "user")
        if not isinstance(content, list):
            plain: dict[str, Any] = {"role": role,
                                     "content": content if content else ""}
            # An assistant turn that called tools and the tool messages that
            # answer it must keep their linkage: the server rejects a
            # tool_call_id that no preceding assistant tool_call declared.
            if msg.get("tool_calls"):
                plain["tool_calls"] = msg["tool_calls"]
            if msg.get("tool_call_id"):
                plain["tool_call_id"] = msg["tool_call_id"]
            out.append(plain)
            continue
        parts: list[dict] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append({"type": "text", "text": str(item)})
                continue
            kind = item.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": item.get("text", "")})
            elif kind == "image":
                uri = _image_data_uri(str(item.get("image", "")))
                if uri:
                    parts.append({"type": "image_url", "image_url": {"url": uri}})
            elif kind == "tool_result":
                # The agent loop feeds tool output back as a user turn.
                body = item.get("content")
                parts.append({"type": "text", "text": body if isinstance(body, str)
                              else json.dumps(body, default=str)})
        out.append({"role": role, "content": parts or ""})
    return out


class VLLMClient:
    """OpenAI-compatible chat-completions client pointed at a local vLLM server."""

    def __init__(self, model: str, base_url: str = DEFAULT_BASE_URL,
                 api_key: str = "EMPTY", timeout: float = DEFAULT_TIMEOUT,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                 temperature: float = 0.7, top_p: float = 0.8):
        from openai import OpenAI

        self.model_name = model
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.top_p = top_p
        # vLLM ignores the key but the SDK requires a non-empty one.
        self.client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url,
                             timeout=timeout)
        print(f"[VLM] vLLM client -> {base_url} (model {model})", flush=True)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config_path: str | Path) -> "VLLMClient":
        raw = yaml.safe_load(Path(config_path).read_text()) or {}
        cfg = raw.get("vllm") or {}
        return cls(
            model=str(cfg.get("model") or "Qwen3-VL-30B-A3B-Instruct").strip(),
            base_url=str(cfg.get("base_url") or DEFAULT_BASE_URL).strip(),
            api_key=str(cfg.get("api_key") or "EMPTY"),
            timeout=float(cfg.get("timeout") or DEFAULT_TIMEOUT),
            max_attempts=int(cfg.get("max_attempts") or DEFAULT_MAX_ATTEMPTS),
            temperature=float(cfg.get("temperature", 0.7)),
            top_p=float(cfg.get("top_p", 0.8)),
        )

    # ------------------------------------------------------------------
    def chat(self, messages: list[dict], system: str,
             tools: list[dict] | None = None,
             max_tokens: int = 4096) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": _to_openai_messages(messages, system),
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if tools:
            payload["tools"] = _convert_tools(tools)
            payload["tool_choice"] = "auto"

        t0 = time.time()
        resp = self._create_with_retry(payload)
        gen_seconds = time.time() - t0

        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        calls: list[ToolCall] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            args = fn.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (TypeError, ValueError):
                    args = {}
            calls.append(ToolCall(id=getattr(tc, "id", None) or
                                  f"call_{uuid.uuid4().hex[:8]}",
                                  name=fn.name, input=args or {}))
        if not calls and text:
            # Fallback: some templates emit <tool_call>{...}</tool_call> inline
            # instead of populating tool_calls.
            text, calls = _parse_tool_calls(text)

        # Normalize `raw` into a chat-completions assistant message, exactly as
        # macroflow.llm.VLM does, so build_assistant_message is a passthrough.
        # It is rebuilt from `calls` rather than from msg.tool_calls so the
        # inline <tool_call> fallback above also produces the tool_calls the
        # follow-up tool messages refer to.
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": text or ""}
        if calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.input)}}
                for tc in calls
            ]

        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            tool_calls=calls,
            stop_reason=("tool_use" if calls else
                         (choice.finish_reason or "end_turn")),
            raw=assistant_msg,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            gen_seconds=gen_seconds,
        )

    # ------------------------------------------------------------------
    # Message-shaping helpers required by macroflow.agent.Agent.run — same
    # contract as macroflow.llm.VLM (chat-completions tool protocol).
    @staticmethod
    def build_assistant_message(raw) -> dict:
        return raw

    @staticmethod
    def format_tool_result(tool_call_id: str, content) -> dict:
        """Text-only ``tool``-role message. Images a tool wants the model to see
        are appended by the agent loop as a follow-up user turn."""
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
        return {"role": "tool", "tool_call_id": tool_call_id,
                "content": content_str}

    @staticmethod
    def build_tool_results_message(tool_results: list[dict]) -> list[dict]:
        return tool_results

    # ------------------------------------------------------------------
    def _create_with_retry(self, payload: dict):
        """Retry transient server errors; a candidate worker must not die on one."""
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.client.chat.completions.create(**payload)
            except Exception as exc:            # noqa: BLE001 - surfaced below
                last = exc
                if attempt == self.max_attempts:
                    break
                wait = min(30.0, 2.0 ** attempt)
                print(f"[VLM] vLLM request failed ({type(exc).__name__}: "
                      f"{str(exc)[:160]}); retry {attempt}/{self.max_attempts - 1} "
                      f"in {wait:.0f}s", flush=True)
                time.sleep(wait)
        raise last if last else RuntimeError("vLLM chat.completions.create failed")
