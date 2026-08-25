"""Agent — a generic, bounded tool-calling loop over the VLM.

One ``Agent`` is one role (Planner / Placer / Checker) with its own system
prompt (built from the stage's SKILL knowledge) and a curated tool subset. The
loop:

  1. ask the VLM for the next turn (text + optional tool calls),
  2. dispatch each tool call through :class:`macroflow.tools.ToolDispatcher`,
  3. feed text results back as tool-role messages, and **inject any image** a
     tool wanted to show as a follow-up user message (Qwen tool messages are
     text-only), then
  4. prune all but the most recent images so context stays small even when many
     snapshots have been viewed.

It stops when a terminal tool fires (``finish`` / ``submit_*``) or the turn
budget is exhausted. The returned :class:`AgentRun` carries the structured
result, the tools actually used, and the final text — everything the
orchestrator needs, without leaking the whole transcript into the next agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tools import ToolDispatcher, ToolResult, schemas_for


@dataclass
class AgentRun:
    final_text: str = ""
    result: dict | None = None         # data from the terminal tool (submit_*/finish)
    tools_used: list[str] = field(default_factory=list)
    turns: int = 0


def _prune_images(messages: list[dict], keep: int) -> None:
    """Keep only the last ``keep`` image blocks; replace older ones with text so
    the image count handed to the processor stays bounded (each image is ~1k+
    vision tokens)."""
    locs = []
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for ii, item in enumerate(content):
                if isinstance(item, dict) and item.get("type") == "image":
                    locs.append((mi, ii))
    drop = locs if keep <= 0 else locs[:-keep]
    for mi, ii in drop:
        messages[mi]["content"][ii] = {
            "type": "text", "text": "[earlier layout image omitted to save context]"}


class Agent:
    def __init__(self, name: str, vlm, dispatcher: ToolDispatcher,
                 tool_names: list[str], system_prompt: str,
                 max_turns: int = 8, keep_images: int = 2, max_tokens: int = 1536,
                 verbose: bool = True, telemetry=None):
        self.name = name
        self.vlm = vlm
        self.dispatcher = dispatcher
        self.telemetry = telemetry
        self.tools = schemas_for(tool_names)
        self._tool_names = {t["name"] for t in self.tools}
        self.system = system_prompt
        self.max_turns = max_turns
        self.keep_images = keep_images
        self.max_tokens = max_tokens
        self.verbose = verbose

    def _say(self, msg: str):
        if self.verbose:
            print(f"  [{self.name}] {msg}", flush=True)

    def run(self, user_content) -> AgentRun:
        """Run the loop on an initial user message (str or content-block list)."""
        messages = [{"role": "user", "content": user_content}]
        run = AgentRun()
        nudges = 0

        for turn in range(self.max_turns):
            run.turns = turn + 1
            try:
                resp = self.vlm.chat(messages, self.system, self.tools,
                                     max_tokens=self.max_tokens)
            except Exception as e:                       # model error — bail gracefully
                self._say(f"VLM error: {e}")
                break

            if self.telemetry is not None:
                self.telemetry.record_llm(self.name, resp.prompt_tokens,
                                          resp.completion_tokens, resp.gen_seconds)
            messages.append(self.vlm.build_assistant_message(resp.raw))
            if resp.text:
                run.final_text = resp.text
                self._say(resp.text[:240])

            if not resp.tool_calls:
                # No action this turn — nudge once, then give up (caller falls back).
                if nudges >= 1:
                    self._say("no tool call after nudge — ending")
                    break
                nudges += 1
                messages.append({"role": "user", "content":
                    "Respond ONLY by calling one of your tools (use a submit_*/finish "
                    "tool when you have your answer)."})
                continue

            tool_results = []
            pending_images: list[tuple[str, str]] = []
            done = False
            for tc in resp.tool_calls:
                self._say(f"-> {tc.name}({tc.input})")
                if tc.name not in self._tool_names:
                    # Prevent cross-agent tool leakage (e.g. a Planner calling
                    # finish, whose text would clobber its strategy).
                    res = ToolResult(text=f"[ERROR] tool '{tc.name}' is not available "
                                          f"to you. Use one of {sorted(self._tool_names)}.")
                else:
                    res = self.dispatcher.dispatch(tc.name, tc.input)
                run.tools_used.append(tc.name)
                tool_results.append(self.vlm.format_tool_result(tc.id, res.text))
                if res.image:
                    pending_images.append((res.image, res.image_caption or ""))
                if res.done:
                    done = True
                    if res.data is not None:
                        run.result = res.data
                    run.final_text = res.text or run.final_text
                    break

            messages.extend(self.vlm.build_tool_results_message(tool_results))
            if pending_images:
                content = []
                for path, cap in pending_images:
                    content.append({"type": "image", "image": f"file://{path}"})
                    content.append({"type": "text", "text": cap})
                messages.append({"role": "user", "content": content})
            _prune_images(messages, self.keep_images)

            if done:
                break

        return run
