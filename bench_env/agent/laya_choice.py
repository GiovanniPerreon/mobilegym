from __future__ import annotations

import os
import re
from typing import Any, Optional

from bench_env.agent.base import AgentConfig
from bench_env.agent.generic_choice import GenericChoiceAgent
from bench_env.env.base import Observation
from bench_env.llm import LLMClient

_CHARS_PER_TOKEN = 3.0
_MAX_LABEL_CHARS = 60
_HISTORY_STEPS = 5
_INSTRUCTIONS = "Which action should the phone agent perform next to complete the task?"
_DESCRIPTIONS = {
    "CLICK": "tap this element on the screen",
    "LONG_PRESS": "press and hold this element",
    "DOUBLE_TAP": "double tap this element",
    "SWIPE": "scroll or swipe the screen",
    "TYPE": "type text into this field",
    "AWAKE": "open this app",
    "ANSWER": "write down the final answer",
    "BACK": "press the back button",
    "HOME": "go to the home screen",
    "RECENT": "open the recent apps",
    "ENTER": "press the enter key",
    "WAIT": "wait for the screen to update",
    "COMPLETE": "finish now, everything is done",
    "ABORT": "give up, it cannot be done",
}
_NUM_PREFIX = re.compile(r"^\d+\.\s*")
_PROB_KEYS = ("probs", "probabilities", "scores", "distribution")
_CONF_KEYS = ("confidence", "probability", "prob", "score")


def build_options(cands: list[dict[str, Any]], drop: int = 0, desc: str = "template"
                  ) -> tuple[dict[str, str], dict[str, int]]:
    click_ids = [c["id"] for c in cands if c.get("action") == "CLICK"]
    dropped = set(click_ids[len(click_ids) - drop:]) if drop > 0 else set()
    criteria: dict[str, str] = {}
    by_name: dict[str, int] = {}
    for c in cands:
        if c["id"] in dropped:
            continue
        name = str(c["label"]).strip()[:_MAX_LABEL_CHARS] or str(c.get("action", "action")).lower()
        key, k = name, 2
        while key in criteria:
            key, k = f"{name} ({k})", k + 1
        criteria[key] = key if desc == "label" else _DESCRIPTIONS.get(str(c.get("action", "")), "perform this action")
        by_name[key] = int(c["id"])
    return criteria, by_name


def _options_chars(criteria: dict[str, str]) -> int:
    return sum(len(k) + len(v) + 4 for k, v in criteria.items())


def fit_budget(head: str, screen: str, cands: list[dict[str, Any]], ctx_tokens: int, desc: str = "template"
               ) -> tuple[str, dict[str, str], dict[str, int], int]:
    budget = int(ctx_tokens * _CHARS_PER_TOKEN) - len(_INSTRUCTIONS) - 64
    n_taps = sum(1 for c in cands if c.get("action") == "CLICK")
    drop = 0
    while True:
        criteria, by_name = build_options(cands, drop, desc)
        left = budget - len(head) - _options_chars(criteria)
        if left >= budget // 4 or drop >= n_taps:
            break
        drop += 1
    left = max(left, 0)
    if len(screen) > left:
        screen = screen[: max(left - 12, 0)] + " [truncated]" if left > 12 else ""
    return f"{head}\n\n[Screen]\n{screen}", criteria, by_name, drop


def _probability(answer: dict[str, Any], chosen: str) -> Optional[float]:
    for k in _PROB_KEYS:
        d = answer.get(k)
        if isinstance(d, dict) and chosen in d:
            try:
                return float(d[chosen])
            except (TypeError, ValueError):
                return None
    for k in _CONF_KEYS:
        v = answer.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


class _LayaBackend:

    def __init__(self) -> None:
        self._model: Any = None

    def _load(self) -> Any:
        try:
            import laya
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("laya_choice needs the 'laya' package: pip install laya") from e
        repo = os.environ.get("BENCH_LAYA_MODEL", "").strip()
        sub = os.environ.get("BENCH_LAYA_SUBFOLDER", "").strip()
        if repo or sub:
            kw = {"subfolder": sub} if sub else {}
            return laya.load(repo or "convaiinnovations/laya", **kw)
        return laya.Router()

    def predict(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        if self._model is None:
            self._model = self._load()
        return self._model.predict(state, questions)


class LayaChoiceAgent(GenericChoiceAgent):
    def __init__(self, llm: LLMClient, config: Optional[AgentConfig] = None,
                 backend: Optional[Any] = None):
        super().__init__(llm, config)
        self.backend = backend or _LayaBackend()
        self.ctx_tokens = int(os.environ.get("BENCH_LAYA_CTX", "1024"))
        self.history_steps = max(0, int(os.environ.get("BENCH_LAYA_HISTORY", str(_HISTORY_STEPS))))
        self.desc_mode = "label" if os.environ.get("BENCH_LAYA_DESC", "").strip().lower() == "label" else "template"

    @property
    def name(self) -> str:
        return "LayaChoiceAgent"

    def _head(self) -> str:
        lines = [f"[Task]\n{self._task}"]
        recent = self._history[-self.history_steps:] if self.history_steps else []
        if recent:
            lines.append("[Previous actions]\n" + "\n".join(
                f"{len(self._history) - len(recent) + i + 1}. {_NUM_PREFIX.sub('', r.llm_response)}"
                for i, r in enumerate(recent)))
        return "\n\n".join(lines)

    def _decide(self, obs: Observation, messages: list[dict], n: int
                ) -> tuple[Optional[int], Optional[float], str, int]:
        tv = obs.text_view or {}
        if tv.get("format") not in ("a11y", "json"):
            raise ValueError("laya_choice needs --obs a11y or --obs json (Laya has no image input and a "
                             "context too short for html)")
        state, criteria, by_name, dropped = fit_budget(self._head(), str(tv.get("text", "")),
                                                       obs.candidates, self.ctx_tokens, self.desc_mode)
        questions = {"action": {"type": "choice", "instructions": _INSTRUCTIONS, "criteria": criteria}}
        result = self.backend.predict(state, questions)
        try:
            answer = result["answers"]["action"]
            chosen = str(answer["choice"])
        except (KeyError, TypeError):
            return None, None, f"bad_laya_result:{str(result)[:60]}", 0
        note = f"{chosen} [dropped={dropped}]" if dropped else chosen
        return by_name.get(chosen), _probability(answer, chosen), note, 0
