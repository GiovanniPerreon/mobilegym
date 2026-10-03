"""Offline tests for the Laya decider (agent/laya_choice.py): no `laya` package, no model server."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from bench_env.agent.base import AgentConfig
from bench_env.agent.laya_choice import LayaChoiceAgent, _probability, build_options, fit_budget
from bench_env.env.base import ActionType, Observation
from bench_env.llm import ChatResult

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
TV = {"format": "a11y", "text": "button 'Mamma' [e1]\ntextbox 'Message' [e2]", "refs": {}}


def _cands() -> list[dict[str, Any]]:
    return [
        {"id": 1, "action": "CLICK", "label": "tap: Mamma", "point": [100, 200]},
        {"id": 2, "action": "CLICK", "label": "tap: Mamma", "point": [100, 300]},     # duplicate label
        {"id": 3, "action": "TYPE", "label": "type in: Message", "point": [500, 900], "needs_text": True},
        {"id": 4, "action": "BACK", "label": "go back"},
        {"id": 5, "action": "ANSWER", "label": "answer the question", "needs_text": True},
    ]


def _obs(**kw: Any) -> Observation:
    kw.setdefault("text_view", TV)
    return Observation(step_idx=1, candidates=_cands(), **kw)


class FakeLaya:
    def __init__(self, *answers: Any):
        self.answers = list(answers)
        self.calls: list[tuple[str, dict]] = []

    def predict(self, state: str, questions: dict) -> dict:
        self.calls.append((state, questions))
        a = self.answers.pop(0)
        return a if isinstance(a, dict) and "answers" in a else {"answers": {"action": a}}


class FakeLLM:
    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, *, messages: list[dict], args: dict | None = None) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(content=self.replies.pop(0))


def _agent(laya: FakeLaya, *llm_replies: str) -> tuple[LayaChoiceAgent, FakeLLM]:
    llm = FakeLLM(*llm_replies)
    a = LayaChoiceAgent(llm, AgentConfig(verbose=False, stream=False, model_args={}), backend=laya)
    a.reset("Send a message to Mamma")
    return a, llm


def test_options_are_unique_and_map_back_to_candidate_ids() -> None:
    criteria, by_name = build_options(_cands())
    assert list(criteria) == ["tap: Mamma", "tap: Mamma (2)", "type in: Message", "go back", "answer the question"]
    assert by_name["tap: Mamma (2)"] == 2 and by_name["go back"] == 4
    assert criteria["go back"] == "press the back button"


def test_drop_removes_last_taps_only() -> None:
    criteria, by_name = build_options(_cands(), drop=1)
    assert "tap: Mamma (2)" not in criteria and "tap: Mamma" in criteria
    assert "go back" in criteria and len(by_name) == 4


def test_fit_budget_truncates_screen_then_drops_taps() -> None:
    cands = [{"id": i + 1, "action": "CLICK", "label": f"tap: item number {i}"} for i in range(40)]
    cands.append({"id": 41, "action": "BACK", "label": "go back"})
    state, criteria, by_name, dropped = fit_budget("[Task]\nx", "S" * 20000, cands, ctx_tokens=512)
    assert len(state) < 512 * 3 and "[truncated]" in state
    assert "go back" in criteria                                  # fixed entries are never dropped
    assert dropped == sum(1 for c in cands if c["action"] == "CLICK") - sum(
        1 for k, v in by_name.items() if v <= 40)
    # a short screen fits untouched
    state, _, _, dropped = fit_budget("[Task]\nx", "short screen", _cands(), ctx_tokens=1024)
    assert state.endswith("short screen") and dropped == 0


def test_probability_extraction() -> None:
    assert _probability({"choice": "a", "probs": {"a": 0.8, "b": 0.2}}, "a") == 0.8
    assert _probability({"choice": "a", "confidence": 0.7}, "a") == 0.7
    assert _probability({"choice": "a"}, "a") is None


def test_click_choice_needs_no_llm_call() -> None:
    agent, llm = _agent(FakeLaya({"choice": "tap: Mamma (2)", "probs": {"tap: Mamma (2)": 0.9}}))
    a = agent.act(_obs())
    assert a.action_type == ActionType.CLICK and a.data == {"point": [100, 300]}   # duplicate label -> id 2
    assert a.explain == "choice=2 n=5 p=0.9000 calls=0"
    assert llm.calls == []
    state, questions = agent.backend.calls[0]
    assert "[Task]\nSend a message to Mamma" in state and "textbox 'Message'" in state
    q = questions["action"]
    assert q["type"] == "choice" and "tap: Mamma (2)" in q["criteria"]


def test_type_choice_uses_the_helper_model_for_the_text() -> None:
    agent, llm = _agent(FakeLaya({"choice": "type in: Message"}), "Ciao Mamma")
    a = agent.act(_obs())
    assert a.action_type == ActionType.TYPE and a.data["value"] == "Ciao Mamma"
    assert a.explain == "choice=3 n=5 p=None calls=1"                             # only the text call
    assert len(llm.calls) == 1


def test_history_goes_into_the_next_state() -> None:
    agent, _ = _agent(FakeLaya({"choice": "go back"}, {"choice": "tap: Mamma"}))
    agent.act(_obs())
    agent.act(Observation(step_idx=2, candidates=_cands(), text_view=TV))
    state, _ = agent.backend.calls[1]
    assert "[Previous actions]\n1. 4. go back" in state


def test_unknown_or_malformed_answer_aborts() -> None:
    agent, _ = _agent(FakeLaya({"choice": "dance"}))
    assert agent.act(_obs()).data["value"].startswith("invalid_choice")
    agent, _ = _agent(FakeLaya({"answers": {}}))
    assert agent.act(_obs()).action_type == ActionType.ABORT


def test_requires_a11y_or_json() -> None:
    agent, _ = _agent(FakeLaya())
    for tv in ({}, {"format": "html", "text": "<b>"}):
        with pytest.raises(ValueError):
            agent.act(_obs(text_view=tv))


def test_ctx_env_variable() -> None:
    with patch.dict(os.environ, {"BENCH_LAYA_CTX": "512"}):
        agent, _ = _agent(FakeLaya())
        assert agent.ctx_tokens == 512
