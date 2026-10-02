"""
GenericTextAgent - text-observation variant of GenericAgentV2.

Identical to GenericAgentV2 (same system prompt, <THINK>/<ANSWER> format, action space,
multi-turn history, model args) except that the model receives a text view of the screen
(``--obs view``, see env/text_view.py) instead of a screenshot. Elements can be addressed
by ref ("e3"); the agent converts a ref into the 0-1000 point the environment already
understands, so the environment, judge and metrics are identical to the screenshot setting.
"""
from __future__ import annotations

from typing import ClassVar

from bench_env.agent.generic_v2 import GenericAgentV2
from bench_env.env.base import Action, ActionType, Observation
from bench_env.env.text_view import format_view, refs_to_points

_V2 = GenericAgentV2.SYSTEM_PROMPT


def _swap(s: str, old: str, new: str) -> str:
    assert old in s, f"generic_v2 prompt changed, update generic_text: {old!r}"
    return s.replace(old, new, 1)


_PROMPT = _V2
_PROMPT = _swap(_PROMPT, "phone screenshots, and operation history",
                "a text description of the phone screen, and operation history")
_PROMPT = _swap(_PROMPT, "1. Click: {\"action\": \"CLICK\", \"point\": [x, y]}",
                "1. Click: {\"action\": \"CLICK\", \"point\": [x, y]} // or {\"action\": \"CLICK\", \"ref\": \"e3\"} to click listed element e3")
_PROMPT = _swap(_PROMPT, "3. Long press: {\"action\": \"LONGPRESS\", \"point\": [x, y]}",
                "3. Long press: {\"action\": \"LONGPRESS\", \"point\": [x, y]} // or with \"ref\": \"e3\"")
_PROMPT = _swap(_PROMPT, "- Observe the screenshot carefully and make decisions based on visual information.",
                "- Read the screen description carefully and make decisions based on it.")
_PROMPT += """
Screen description format (one visible element per line):
- text "..." : plain text, not clickable
- [e3] button "..." box=[x1, y1, x2, y2] : interactive element with id e3 and its bounding box (same 0-1000 coordinates)
Only elements currently visible on screen are listed; SWIPE to reveal more content.
"""


class GenericTextAgent(GenericAgentV2):
    SYSTEM_PROMPT: ClassVar[str] = _PROMPT

    @property
    def name(self) -> str:
        return "GenericTextAgent"

    def reset(self, task: str) -> None:
        super().reset(task)
        self._refs: dict[str, list[int]] = {}

    def parse_response(self, response_text: str) -> Action:
        action = super().parse_response(response_text)
        if action.action_type in (ActionType.CLICK, ActionType.LONG_PRESS) and not action.data.get("point"):
            _, parsed = self._parse_llm_output(response_text)
            ref = str(parsed.get("ref") or "").strip()
            pt = getattr(self, "_refs", {}).get(ref)
            if pt:
                action.data["point"] = pt
            elif ref:
                # unknown ref: abort cleanly rather than clicking the screen center
                return Action(action_type=ActionType.ABORT,
                              data={"value": f"unknown_ref:{ref}"}, raw_response=response_text)
        return action

    def build_messages(self, obs: Observation) -> list[dict]:
        if not obs.text_view:
            raise RuntimeError("GenericTextAgent needs BENCH_OBS=view (run with --obs view)")
        self._refs = refs_to_points(obs.text_view)

        messages: list[dict] = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        for i, record in enumerate(self._history):
            messages.append({"role": "user",
                             "content": [{"type": "text",
                                          "text": f"[Task]\n{self._task}" if i == 0 else f"[Step {i + 1}]"}]})
            messages.append({"role": "assistant", "content": record.llm_response})

        step_num = len(self._history) + 1
        header = f"[Task]\n{self._task}" if step_num == 1 else f"[Step {step_num}]"
        messages.append({"role": "user",
                         "content": [{"type": "text",
                                      "text": f"{header}\n\n[Screen]\n{format_view(obs.text_view)}"}]})
        return messages