"""
D1ChoiceAgent - "choice" action mode with Liquid's d1 decision models (d1-omni-600M, d1-3B) as the decider.

Same candidate list, same mapping to MobileGym actions and same text step as generic_choice and laya_choice
(it is a subclass of laya_choice: only the decision backend and the image input change). The difference from Laya:

  laya_choice   text only (--obs a11y|json), Laya runs in-process through the `laya` package
  d1_choice     screenshot and/or text; d1 runs in its own llama-server, called on its /v1/systemone endpoint
                (one forward pass, calibrated probabilities over the options, zero generated tokens)

TYPE / ANSWER entries still need free text, which d1 cannot write: the usual second call goes to the generative
model served by llama-server (--model-base-url / --model-name), exactly as in generic_choice (calls=2).

Observation
  no --obs                 screenshot only: the image is sent, the state is task + recent actions
  --obs a11y|json          text only
  --obs a11y|json --obs-image   screenshot + text
With an image, d1-omni-600M keeps only 896 tokens of state and question text (model card), so the same character
budget as laya_choice is applied (screen text truncated first, then tap entries dropped from the end of the list).

Environment variables
  BENCH_D1_URL      systemone endpoint of the d1 server (default http://127.0.0.1:8081/v1/systemone)
  BENCH_D1_CTX      token budget for state + options (default 896, the d1-omni-600M limit with images)
  BENCH_D1_TIMEOUT  seconds per request (default 600)
  BENCH_LAYA_HISTORY, BENCH_LAYA_DESC   as in laya_choice (history length, option descriptions)

Request and response follow the d1 model cards (state, questions with "type": "choice", "instructions",
"criteria" {name: description}, optional "images" as data URIs; answers.<name>.choice / confidence /
probabilities). They have not been run against a real d1 server yet: _D1Backend is the only place to adapt.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Optional

from bench_env.agent.base import AgentConfig
from bench_env.agent.laya_choice import _INSTRUCTIONS, LayaChoiceAgent, _probability, fit_budget
from bench_env.env.base import Observation
from bench_env.env.text_view import obs_with_image
from bench_env.llm import LLMClient

_DEFAULT_URL = "http://127.0.0.1:8081/v1/systemone"
_DEFAULT_CTX = 896


class _D1Backend:
    """POSTs one decision request to a d1 llama-server."""

    def __init__(self, url: Optional[str] = None, timeout: Optional[float] = None) -> None:
        self.url = url or os.environ.get("BENCH_D1_URL", _DEFAULT_URL)
        self.timeout = float(timeout or os.environ.get("BENCH_D1_TIMEOUT", "600"))

    def predict(self, state: Optional[str], questions: dict[str, Any],
                images: Optional[list[str]] = None) -> dict[str, Any]:
        body: dict[str, Any] = {"state": state, "questions": questions}
        if images:
            body["images"] = images
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))


class D1ChoiceAgent(LayaChoiceAgent):
    def __init__(self, llm: LLMClient, config: Optional[AgentConfig] = None,
                 backend: Optional[Any] = None):
        super().__init__(llm, config, backend=backend or _D1Backend())
        self.ctx_tokens = int(os.environ.get("BENCH_D1_CTX", str(_DEFAULT_CTX)))

    @property
    def name(self) -> str:
        return "D1ChoiceAgent"

    def _decide(self, obs: Observation, messages: list[dict], n: int
                ) -> tuple[Optional[int], Optional[float], str, int]:
        tv = obs.text_view or {}
        fmt = tv.get("format", "")
        if fmt == "html":
            raise ValueError("d1_choice supports the screenshot and --obs a11y|json (html does not fit the "
                             "896-token budget)")
        with_image = (not fmt) or obs_with_image()
        screen = str(tv.get("text", "")) if fmt else ""
        state, criteria, by_name, dropped = fit_budget(self._head(), screen, obs.candidates,
                                                       self.ctx_tokens, self.desc_mode)
        if not fmt:
            state = self._head()        # screenshot only: no [Screen] section
        questions = {"action": {"type": "choice", "instructions": _INSTRUCTIONS, "criteria": criteria}}
        images = [obs.image_data_url] if with_image and obs.image_data_url else None
        try:
            result = self.backend.predict(state, questions, images)
            answer = result["answers"]["action"]
            chosen = str(answer["choice"])
        except (KeyError, TypeError) as e:
            return None, None, f"bad_d1_result:{type(e).__name__}", 0
        note = f"{chosen} [dropped={dropped}]" if dropped else chosen
        return by_name.get(chosen), _probability(answer, chosen), note, 0
