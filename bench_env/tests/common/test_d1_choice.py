"""Offline tests for the d1 decider (agent/d1_choice.py): no d1 server, no model server."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest.mock import patch

import pytest

from bench_env.agent.base import AgentConfig
from bench_env.agent.d1_choice import D1ChoiceAgent, _D1Backend
from bench_env.env.base import ActionType, Observation
from bench_env.llm import ChatResult

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
TV = {"format": "a11y", "text": "button 'Mamma' [e1]\ntextbox 'Message' [e2]", "refs": {}}


def _cands() -> list[dict[str, Any]]:
    return [
        {"id": 1, "action": "CLICK", "label": "tap: Mamma", "point": [100, 200]},
        {"id": 2, "action": "TYPE", "label": "type in: Message", "point": [500, 900], "needs_text": True},
        {"id": 3, "action": "BACK", "label": "go back"},
    ]


class FakeD1:
    def __init__(self, *answers: Any):
        self.answers = list(answers)
        self.calls: list[tuple[Any, dict, Any]] = []

    def predict(self, state: Any, questions: dict, images: Any = None) -> dict:
        self.calls.append((state, questions, images))
        a = self.answers.pop(0)
        return a if isinstance(a, dict) and "answers" in a else {"answers": {"action": a}}


class FakeLLM:
    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, *, messages: list[dict], args: dict | None = None) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(content=self.replies.pop(0))


def _agent(d1: FakeD1, *llm_replies: str) -> tuple[D1ChoiceAgent, FakeLLM]:
    llm = FakeLLM(*llm_replies)
    a = D1ChoiceAgent(llm, AgentConfig(verbose=False, stream=False, model_args={}), backend=d1)
    a.reset("Send a message to Mamma")
    return a, llm


def test_screenshot_only_sends_the_image_and_no_screen_text() -> None:
    agent, llm = _agent(FakeD1({"choice": "tap: Mamma", "probabilities": {"tap: Mamma": 0.7}}))
    a = agent.act(Observation(step_idx=1, candidates=_cands(), screenshot_bytes=JPEG))
    assert a.action_type == ActionType.CLICK and a.data == {"point": [100, 200]}
    assert a.explain == "choice=1 n=3 p=0.7000 calls=0" and llm.calls == []
    state, questions, images = agent.backend.calls[0]
    assert state == "[Task]\nSend a message to Mamma" and "[Screen]" not in state
    assert images and images[0].startswith("data:image/jpeg;base64,")
    assert questions["action"]["type"] == "choice" and "go back" in questions["action"]["criteria"]


def test_text_only_sends_no_image() -> None:
    agent, _ = _agent(FakeD1({"choice": "go back"}))
    a = agent.act(Observation(step_idx=1, candidates=_cands(), text_view=TV, screenshot_bytes=JPEG))
    assert a.action_type == ActionType.BACK
    state, _, images = agent.backend.calls[0]
    assert "textbox 'Message'" in state and images is None


def test_hybrid_sends_text_and_image() -> None:
    with patch.dict(os.environ, {"BENCH_OBS_IMAGE": "1"}):
        agent, _ = _agent(FakeD1({"choice": "go back"}))
        agent.act(Observation(step_idx=1, candidates=_cands(), text_view=TV, screenshot_bytes=JPEG))
    state, _, images = agent.backend.calls[0]
    assert "textbox 'Message'" in state and images


def test_type_choice_uses_the_helper_model_for_the_text() -> None:
    agent, llm = _agent(FakeD1({"choice": "type in: Message"}), "Ciao Mamma")
    a = agent.act(Observation(step_idx=1, candidates=_cands(), screenshot_bytes=JPEG))
    assert a.action_type == ActionType.TYPE and a.data["value"] == "Ciao Mamma"
    assert a.explain == "choice=2 n=3 p=None calls=1" and len(llm.calls) == 1


def test_html_and_malformed_answers() -> None:
    agent, _ = _agent(FakeD1())
    with pytest.raises(ValueError):
        agent.act(Observation(step_idx=1, candidates=_cands(), text_view={"format": "html", "text": "<b>"}))
    agent, _ = _agent(FakeD1({"answers": {}}))
    assert agent.act(Observation(step_idx=1, candidates=_cands(), screenshot_bytes=JPEG)).action_type == ActionType.ABORT


def test_default_budget_is_the_d1_image_limit() -> None:
    agent, _ = _agent(FakeD1())
    assert agent.ctx_tokens == 896
    with patch.dict(os.environ, {"BENCH_D1_CTX": "2048"}):
        agent, _ = _agent(FakeD1())
        assert agent.ctx_tokens == 2048


def test_backend_posts_the_documented_body() -> None:
    seen: dict[str, Any] = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"answers": {"action": {"choice": "go back", "confidence": 0.9}}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a: Any) -> None:
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    try:
        b = _D1Backend(url=f"http://127.0.0.1:{srv.server_port}/v1/systemone", timeout=10)
        r = b.predict("s", {"action": {"type": "choice"}}, ["data:image/png;base64,AA"])
    finally:
        srv.server_close()
    assert seen["path"] == "/v1/systemone"
    assert seen["body"] == {"state": "s", "questions": {"action": {"type": "choice"}},
                            "images": ["data:image/png;base64,AA"]}
    assert r["answers"]["action"]["choice"] == "go back"
