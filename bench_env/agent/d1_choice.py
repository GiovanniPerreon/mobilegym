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
            state = self._head()
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
