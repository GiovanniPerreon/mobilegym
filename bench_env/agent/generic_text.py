"""
GenericTextAgent - text-observation variant of GenericAgentV2 (generative action mode).

Identical to GenericAgentV2 (same system prompt, <THINK>/<ANSWER> format, action space,
multi-turn history, model args) except for what the model sees of the screen:
  --obs html | a11y | json        text only
  --obs html|a11y|json --obs-image  hybrid: screenshot + text
Elements are addressed by ref ("e3" / "t2" / "f1"). The agent converts a ref into the
0-1000 coordinates the environment already understands (center of the element's box for
CLICK / LONGPRESS / DOUBLE_TAP / TYPE, a path inside the box for SWIPE), so the
environment, judge and metrics are identical in every condition.
"""
from __future__ import annotations

from typing import ClassVar

from bench_env.agent.generic_v2 import GenericAgentV2
from bench_env.env.base import Action, ActionType, Observation
from bench_env.env.text_view import obs_with_image, ref_point, swipe_points

_V2 = GenericAgentV2.SYSTEM_PROMPT


def _swap(s: str, old: str, new: str) -> str:
    assert old in s, f"generic_v2 prompt changed, update generic_text: {old!r}"
    return s.replace(old, new, 1)


_FORMAT_HELP = {
    "html": """The screen is described as the HTML of the visible elements. Interactive elements carry a data-ref="eN" attribute; use that id as ref.""",
    "a11y": """The screen is described as the accessibility tree of the visible elements (role, name, state). Interactive elements carry [ref=eN]; use that id as ref.""",
    "json": """The screen is described as a JSON object: "app", "screen" (route path), "visible_texts", "transitions" (the actions available on this screen) and "input_fields". Each transition (ref t1, t2, ... with its "name", "params" and "text") and each input field (ref f1, f2, ...) can be used as ref.""",
}

_REF_SYNTAX = """
You can address an element by its ref instead of coordinates (the ref is mapped to the element's position):
- {"action": "CLICK", "ref": "e3"}   (same for DOUBLE_TAP and LONGPRESS)
- {"action": "TYPE", "ref": "e3", "value": "text"}   (taps the element first, then types)
- {"action": "SWIPE", "ref": "e5", "direction": "up|down|left|right"}   (finger moves in that direction inside the element; to see content further down a page, swipe up)
Coordinates ("point", "point1", "point2") are still accepted. Only visible elements are listed; SWIPE to reveal more content.
"""


def build_system_prompt(fmt: str, with_image: bool) -> str:
    p = _swap(_V2, "phone screenshots, and operation history",
              ("phone screenshots with a text description of the screen, and operation history" if with_image
               else "a text description of the phone screen, and operation history"))
    p = _swap(p, "- Observe the screenshot carefully and make decisions based on visual information.",
              "- Observe the screenshot and the screen description carefully and make decisions based on both." if with_image
              else "- Read the screen description carefully and make decisions based on it.")
    return p + "\n" + _FORMAT_HELP[fmt] + "\n" + _REF_SYNTAX


class GenericTextAgent(GenericAgentV2):
    SYSTEM_PROMPT: ClassVar[str] = build_system_prompt("a11y", False)   # replaced per format in __init__

    def __init__(self, llm, config=None):
        super().__init__(llm, config)
        self._refs: dict[str, list[int]] = {}
        self._fmt = ""

    @property
    def name(self) -> str:
        return "GenericTextAgent"

    def reset(self, task: str) -> None:
        super().reset(task)
        self._refs = {}

    def _system_prompt(self, fmt: str) -> str:
        return build_system_prompt(fmt, obs_with_image())

    def parse_response(self, response_text: str) -> Action:
        action = super().parse_response(response_text)
        _, parsed = self._parse_llm_output(response_text)
        ref = str(parsed.get("ref") or "").strip() if isinstance(parsed, dict) else ""
        if not ref:
            return action
        box = self._refs.get(ref)
        if box is None:
            return Action(action_type=ActionType.ABORT, data={"value": f"unknown_ref:{ref}"},
                          raw_response=response_text)
        t = action.action_type
        if t in (ActionType.CLICK, ActionType.LONG_PRESS, ActionType.DOUBLE_TAP, ActionType.TYPE) \
                and not action.data.get("point"):
            action.data["point"] = ref_point(box)
        elif t == ActionType.SWIPE and not action.data.get("point1"):
            pts = swipe_points(box, str(parsed.get("direction") or ""))
            if pts is None:
                return Action(action_type=ActionType.ABORT, data={"value": "bad_swipe_direction"},
                              raw_response=response_text)
            action.data["point1"], action.data["point2"] = pts
        return action

    def build_messages(self, obs: Observation) -> list[dict]:
        tv = obs.text_view
        if not tv:
            raise RuntimeError("GenericTextAgent needs a text format: run with --obs html|a11y|json")
        self._fmt = tv["format"]
        self._refs = tv["refs"]

        messages: list[dict] = [{"role": "system", "content": self._system_prompt(self._fmt)}]
        for i, record in enumerate(self._history):
            messages.append({"role": "user",
                             "content": [{"type": "text",
                                          "text": f"[Task]\n{self._task}" if i == 0 else f"[Step {i + 1}]"}]})
            messages.append({"role": "assistant", "content": record.llm_response})

        step_num = len(self._history) + 1
        header = f"[Task]\n{self._task}" if step_num == 1 else f"[Step {step_num}]"
        content: list[dict] = []
        if obs_with_image():
            content.append({"type": "image_url", "image_url": {"url": obs.image_data_url}})
        content.append({"type": "text", "text": f"{header}\n\n[Screen: {self._fmt}]\n{tv['text']}"})
        messages.append({"role": "user", "content": content})
        return messages
