"""
GenericChoiceAgent - "choice" action mode (Jev style).

Instead of writing the action as text (generative mode), the model receives the numbered list of
the actions that are possible on the current screen (env/candidates.py, built the same way for every
observation format) and answers with the number of one of them. The chosen entry is executed as the
normal MobileGym action, so environment, judge and metrics are identical in every condition.

Observation formats (same switches as generic_text):
  (no --obs) / --obs screenshot   screenshot + candidate list
  --obs html|a11y|json            text description + candidate list (no image, for text-only models)
  --obs html|a11y|json --obs-image  hybrid: screenshot + text + candidate list

How the choice is read
  One short call with temperature 0. By default the answer is constrained to the valid numbers with a
  llama.cpp GBNF grammar (BENCH_CHOICE_CONSTRAINT=grammar; "guided_choice" for vLLM, "none" to disable).
  The log-probability of the generated tokens is requested and stored in Action.explain as
  "choice=<id> n=<list length> p=<probability of the chosen entry> calls=<LLM calls in this step>".
  p is the probability of the token sequence that spells the chosen number, without the decision to stop
  generating; it is None when the server returns no logprobs.

Entries that need text (TYPE, ANSWER)
  A decider cannot write free text, so after choosing such an entry a second generative call asks for the
  text (same conversation, one more user turn). ``calls`` counts it.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, ClassVar, Optional

from bench_env.agent.base import AgentConfig, AgentStepRecord, BaseAgent
from bench_env.env.base import Action, ActionType, Observation
from bench_env.env.candidates import format_candidates
from bench_env.env.text_view import obs_with_image
from bench_env.llm import LLMClient

_OBSERVED = {
    "screenshot": "the phone screenshot",
    "text": "a text description of the phone screen",
    "hybrid": "the phone screenshot with a text description of the screen",
}

_FORMAT_HELP = {
    "html": "The screen description is the HTML of the visible elements.",
    "a11y": "The screen description is the accessibility tree of the visible elements (role, name, state).",
    "json": 'The screen description is a JSON object: "app", "screen" (route path), "visible_texts", '
            '"transitions" (the actions available on this screen) and "input_fields".',
}

_PROMPT = """You are a phone GUI-Agent operation expert. Based on the user's task, {observed} and operation history, choose the single next action to complete the task.

At every step you receive the numbered list of the actions that are possible on the current screen. Reply with the number of the action to perform and nothing else.

Requirements:
- Choose only from the list.
- To type into a field choose its "type in: ..." entry; the text to type is asked in the next message.
- When you need to answer a question, you MUST choose "answer the question"; the text is asked in the next message.
- Choose "task is complete" only to end the task, after you have performed the necessary actions.
- Choose "give up" only if the task cannot be completed.
- To see content that is not on the screen, choose a "swipe" entry. Swiping up shows content further down.
"""

# Short answer: the digits of the largest id plus a little slack.
_MIN_CHOICE_TOKENS = 4
_MAX_TEXT_TOKENS = 512
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_NO_PARAMS = {"BACK", "HOME", "RECENT", "ENTER"}


def build_system_prompt(fmt: str, with_image: bool) -> str:
    kind = "hybrid" if (fmt and with_image) else ("text" if fmt else "screenshot")
    p = _PROMPT.format(observed=_OBSERVED[kind])
    if fmt:
        p += "\n" + _FORMAT_HELP[fmt] + " Ref ids such as e3, t7 or f1 in the description are not needed: choose from the numbered list.\n"
    return p


def build_grammar(n: int) -> str:
    """GBNF grammar that only accepts the numbers 1..n."""
    return "root ::= " + " | ".join(f'"{i}"' for i in range(1, n + 1))


def parse_choice(text: str, n: int) -> Optional[int]:
    """First integer in the answer if it is a valid id (1..n), else None."""
    m = re.search(r"\d+", text or "")
    if not m:
        return None
    i = int(m.group())
    return i if 1 <= i <= n else None


def choice_probability(raw: Any) -> Optional[float]:
    """Probability of the generated token sequence from the server logprobs (stream or not)."""
    items: Any = None
    if isinstance(raw, dict):
        if raw.get("logprobs"):
            items = raw["logprobs"]
        else:
            try:
                items = raw["choices"][0]["logprobs"]["content"]
            except (KeyError, IndexError, TypeError):
                items = None
    if not items:
        return None
    try:
        return math.exp(sum(float(it["logprob"]) for it in items))
    except (KeyError, TypeError, ValueError):
        return None


def candidate_to_action(c: dict[str, Any], text: Optional[str] = None) -> Action:
    """Turn one candidate-list entry into the MobileGym action it stands for."""
    kind = str(c.get("action", ""))
    try:
        t = ActionType(kind)
    except ValueError:
        return Action(action_type=ActionType.ABORT, data={"value": f"unknown_candidate_action:{kind}"})
    if kind in ("CLICK", "LONG_PRESS", "DOUBLE_TAP"):
        data: dict[str, Any] = {"point": c["point"]}
    elif kind == "SWIPE":
        data = {"point1": c["point1"], "point2": c["point2"]}
    elif kind == "TYPE":
        data = {"value": text or "", "point": c["point"], "clear": False}
    elif kind == "ANSWER":
        data = {"value": text or ""}
    elif kind == "AWAKE":
        data = {"value": c["value"]}
    elif kind == "WAIT":
        data = {"value": float(c.get("value", 2))}
    elif kind == "COMPLETE":
        data = {"return": ""}
    elif kind == "ABORT":
        data = {"value": "chosen_from_list"}
    elif kind in _NO_PARAMS:
        data = {}
    else:
        return Action(action_type=ActionType.ABORT, data={"value": f"unsupported_candidate_action:{kind}"})
    return Action(action_type=t, data=data)


def _clean_text(s: str) -> str:
    s = _THINK_RE.sub("", s or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
        s = s[1:-1].strip()
    return s


class GenericChoiceAgent(BaseAgent):
    SYSTEM_PROMPT: ClassVar[str] = build_system_prompt("", False)   # replaced per observation format

    def __init__(self, llm: LLMClient, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.llm = llm

    @property
    def name(self) -> str:
        return "GenericChoiceAgent"

    def reset(self, task: str) -> None:
        self._task = task
        self._history = []

    # BaseAgent requires parse_response; the choice is parsed in act() because it needs the list.
    def parse_response(self, response_text: str) -> Action:
        return Action(action_type=ActionType.ABORT, data={"value": "use act()"}, raw_response=response_text)

    # ------------------------------------------------------------------ messages
    def build_messages(self, obs: Observation) -> list[dict]:
        tv = obs.text_view or {}
        fmt = tv.get("format", "")
        with_image = (not fmt) or obs_with_image()
        messages: list[dict] = [{"role": "system", "content": build_system_prompt(fmt, with_image and bool(fmt))}]
        for i, record in enumerate(self._history):
            messages.append({"role": "user",
                             "content": [{"type": "text",
                                          "text": f"[Task]\n{self._task}" if i == 0 else f"[Step {i + 1}]"}]})
            messages.append({"role": "assistant", "content": record.llm_response})

        step_num = len(self._history) + 1
        header = f"[Task]\n{self._task}" if step_num == 1 else f"[Step {step_num}]"
        parts = [header]
        if fmt:
            parts.append(f"[Screen: {fmt}]\n{tv['text']}")
        parts.append("[Available actions]\n" + format_candidates(obs.candidates))
        parts.append("Reply with the number of the action.")
        content: list[dict] = []
        if with_image:
            content.append({"type": "image_url", "image_url": {"url": obs.image_data_url}})
        content.append({"type": "text", "text": "\n\n".join(parts)})
        messages.append({"role": "user", "content": content})
        return messages

    # ------------------------------------------------------------------ LLM calls
    def _choice_args(self, n: int) -> dict[str, Any]:
        margs = dict(self.config.model_args or {})
        extra = dict(margs.pop("extra_body", None) or {})
        for k in ("max_tokens", "temperature", "top_p"):
            margs.pop(k, None)
        mode = os.environ.get("BENCH_CHOICE_CONSTRAINT", "grammar").strip().lower()
        if mode == "grammar":
            extra["grammar"] = build_grammar(n)
        elif mode == "guided_choice":
            extra["guided_choice"] = [str(i) for i in range(1, n + 1)]
        args: dict[str, Any] = {
            **margs, "temperature": 0.0, "top_p": 1.0,
            "max_tokens": max(_MIN_CHOICE_TOKENS, len(str(n)) + 2),
            "logprobs": True, "top_logprobs": 5,
            "stream": self.config.stream, "stream_print": False,
        }
        if extra:
            args["extra_body"] = extra
        return args

    def _text_args(self) -> dict[str, Any]:
        margs = dict(self.config.model_args or {})
        margs["max_tokens"] = min(int(margs.get("max_tokens") or _MAX_TEXT_TOKENS), _MAX_TEXT_TOKENS)
        return {**margs, "stream": self.config.stream, "stream_print": False}

    def _ask_text(self, messages: list[dict], cand: dict[str, Any], chosen: str) -> str:
        what = ("Write the answer to the task as plain text." if cand["action"] == "ANSWER"
                else f'Write the exact text to type ({cand["label"]}).')
        follow = messages + [
            {"role": "assistant", "content": chosen},
            {"role": "user", "content": [{"type": "text", "text": what + " Reply with the text only."}]},
        ]
        return _clean_text(self.llm.chat(messages=follow, args=self._text_args()).content)

    # ------------------------------------------------------------------ core
    def act(self, obs: Observation) -> Action:
        cands = obs.candidates
        n = len(cands)
        messages: list[dict] = []
        label = ""
        calls = 0
        prob: Optional[float] = None
        cid: Optional[int] = None
        raw_text = ""
        if n == 0:
            action = Action(action_type=ActionType.ABORT, data={"value": "no_candidates"})
        else:
            messages = self.build_messages(obs)
            resp = self.llm.chat(messages=messages, args=self._choice_args(n))
            calls = 1
            raw_text = resp.content or ""
            prob = choice_probability(resp.raw)
            cid = parse_choice(raw_text, n)
            if cid is None:
                action = Action(action_type=ActionType.ABORT,
                                data={"value": f"invalid_choice:{raw_text.strip()[:20]}"})
                label = raw_text.strip()[:20] or "(empty)"
            else:
                cand = cands[cid - 1]
                label = f'{cid}. {cand["label"]}'
                text: Optional[str] = None
                if cand.get("needs_text"):
                    text = self._ask_text(messages, cand, label)
                    calls = 2
                    if not text:
                        action = Action(action_type=ActionType.ABORT, data={"value": "empty_text"})
                    else:
                        label += f': "{text}"'
                        action = candidate_to_action(cand, text)
                else:
                    action = candidate_to_action(cand)
        p_txt = "None" if prob is None else f"{prob:.4f}"
        action.explain = f"choice={cid} n={n} p={p_txt} calls={calls}"
        action.raw_response = raw_text

        if self.config.verbose:
            print(f"[GenericChoiceAgent] {action.explain} -> {action.action_type} {action.data}")

        self._history.append(AgentStepRecord(
            step_idx=obs.step_idx, observation=obs, action=action,
            llm_response=label or "(no action)", llm_prompt=messages,
        ))
        self._evict_old_records(keep_recent=2)
        return action
