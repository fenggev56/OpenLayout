"""LLM client — OpenAI-compatible chat API with token/runtime accounting.

The local code collects design data, attaches skills, sends everything to the
API (or a local vLLM server via ``base_url``), and parses the model's JSON
action back.  Every call is metered (prompt/completion tokens, wall time) into
the shared :class:`~openlayout.telemetry.Telemetry` and the full dialog is
persisted per stage for later inspection.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg, telemetry=None):
        self.cfg = cfg
        self.telemetry = telemetry
        self.provider = cfg.llm.get("provider", "openai")
        self.model = cfg.llm.get("model", "gpt-5-mini")
        self.max_output_tokens = int(cfg.llm.get("max_output_tokens", 8192))
        self.max_attempts = int(cfg.llm.get("max_attempts", 3))
        self.timeout = float(cfg.llm.get("timeout", 600))
        base_url = cfg.llm.get("base_url")

        from openai import OpenAI
        kwargs: dict[str, Any] = {"timeout": self.timeout}
        key = cfg.resolve_api_key()
        if key:
            kwargs["api_key"] = key
        if base_url:  # local vLLM / any OpenAI-compatible endpoint
            kwargs["base_url"] = base_url
            kwargs.setdefault("api_key", "EMPTY")
        self.client = OpenAI(**kwargs)

    # ------------------------------------------------------------------
    def chat_json(self, system: str, user: str, *, stage: str = "",
                  agent: str = "", dialog_dir: Path | None = None) -> dict:
        """One-shot chat expecting a JSON object answer. Returns parsed dict."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        last_err: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.monotonic()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=self.max_output_tokens,
                    response_format={"type": "json_object"},
                )
                dt = time.monotonic() - t0
                text = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                if self.telemetry is not None:
                    self.telemetry.add_llm_call(stage=stage, agent=agent,
                                                prompt_tokens=pt,
                                                completion_tokens=ct,
                                                seconds=dt)
                if dialog_dir is not None:
                    self._save_dialog(dialog_dir, stage, agent, system, user,
                                      text, pt, ct, dt)
                return self._parse_json(text)
            except Exception as exc:  # noqa: BLE001 — retry then surface
                last_err = exc
                if self.telemetry is not None:
                    self.telemetry.add_llm_call(stage=stage, agent=agent,
                                                prompt_tokens=0,
                                                completion_tokens=0,
                                                seconds=time.monotonic() - t0,
                                                error=str(exc))
                time.sleep(min(10.0, 2.0 * attempt))
        raise LLMError(f"LLM call failed after {self.max_attempts} attempts: "
                       f"{last_err}")

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            raise

    @staticmethod
    def _save_dialog(dialog_dir: Path, stage: str, agent: str, system: str,
                     user: str, answer: str, pt: int, ct: int,
                     dt: float) -> None:
        d = dialog_dir / (stage or "flow")
        d.mkdir(parents=True, exist_ok=True)
        n = len(list(d.glob("*.md")))
        p = d / f"{n:03d}_{agent or 'agent'}.md"
        p.write_text(
            f"# stage={stage} agent={agent}\n"
            f"tokens: prompt={pt} completion={ct}  model_time={dt:.1f}s\n\n"
            f"## system\n\n{system}\n\n## user\n\n{user}\n\n"
            f"## response\n\n{answer}\n")
