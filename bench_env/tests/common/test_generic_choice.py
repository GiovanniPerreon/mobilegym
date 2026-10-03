"""Offline tests for the "choice" action mode (agent/generic_choice.py): no simulator and no model server."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

from bench_env.agent.base import AgentConfig
from bench_env.agent.generic_choice import (
    GenericChoiceAgent, build_grammar, candidate_to_action, choice_probability, parse_choice,
)
from bench_env.env.base import ActionType, Observation
from bench_env.env.candidates import choice_mode
from bench_env.llm import ChatResult

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


def _cands() -> list[dict[str, Any]]:
    return [
        {"id": 1, "action": "CLICK", "label": "tap: Mamma", "point": [100, 200], "box": [0, 0, 200, 400]},
        {"id": 2, "action": "TYPE", "label": "type in: Message", "point": [500, 900], "needs_text": True},
        {"id": 3, "action": "SWIPE", "label": "swipe up: list", "point1": [500, 700], "point2": [500, 300]},
        {"id": 4, "action": "AWAKE", "label": "open app: wechat", "value": "wechat"},
        {"id": 5, "action": "BACK", "label": "go back"},
        {"id": 6, "action": "WAIT", "label": "wait 2 seconds", "value": 2},
        {"id": 7, "action": "ANSWER", "label": "answer the question", "needs_text": True},
        {"id": 8, "action": "COMPLETE", "label": "task is complete"},
    ]


def _obs(**kw: Any) -> Observation:
    return Observation(screenshot_bytes=JPEG, step_idx=1, candidates=_cands(), **kw)


class FakeLLM:
    def __init__(self, *replies: Any):
        self.replies = list(replies)
        self.calls: list[tuple[list[dict], dict]] = []

    def chat(self, *, messages: list[dict], args: dict | None = None) -> ChatResult:
        self.calls.append((messages, dict(args or {})))
        r = self.replies.pop(0)
        return ChatResult(content=r, raw=None) if isinstance(r, str) else ChatResult(content=r[0], raw=r[1])


def _agent(*replies: Any) -> tuple[GenericChoiceAgent, FakeLLM]:
    llm = FakeLLM(*replies)
    agent = GenericChoiceAgent(llm, AgentConfig(verbose=False, stream=False,
                                                model_args={"temperature": 0.1, "top_p": 0.95, "max_tokens": 4096}))
    agent.reset("Send a message to Mamma")
    return agent, llm


# ------------------------------------------------------------------ helpers
def test_parse_choice() -> None:
    assert parse_choice("3", 8) == 3
    assert parse_choice(" 7. answer the question", 8) == 7
    assert parse_choice("0", 8) is None
    assert parse_choice("9", 8) is None
    assert parse_choice("none", 8) is None
    assert parse_choice("", 8) is None


def test_build_grammar_only_valid_ids() -> None:
    assert build_grammar(3) == 'root ::= "1" | "2" | "3"'


def test_choice_probability_stream_and_non_stream() -> None:
    items = [{"token": "1", "logprob": -0.1}, {"token": "2", "logprob": -0.2}]
    p = choice_probability({"stream": True, "logprobs": items})
    assert p is not None and abs(p - 0.7408) < 1e-3
    assert choice_probability({"choices": [{"logprobs": {"content": items}}]}) == p
    assert choice_probability({"stream": True}) is None
    assert choice_probability(None) is None


def test_candidate_to_action_mapping() -> None:
    c = {c["id"]: c for c in _cands()}
    a = candidate_to_action(c[1])
    assert a.action_type == ActionType.CLICK and a.data == {"point": [100, 200]}
    a = candidate_to_action(c[3])
    assert a.action_type == ActionType.SWIPE and a.data == {"point1": [500, 700], "point2": [500, 300]}
    a = candidate_to_action(c[4])
    assert a.action_type == ActionType.AWAKE and a.data == {"value": "wechat"}
    a = candidate_to_action(c[5])
    assert a.action_type == ActionType.BACK and a.data == {}
    a = candidate_to_action(c[6])
    assert a.action_type == ActionType.WAIT and a.data == {"value": 2.0}
    a = candidate_to_action(c[8])
    assert a.action_type == ActionType.COMPLETE and "return" in a.data
    a = candidate_to_action(c[2], "Ciao")
    assert a.action_type == ActionType.TYPE and a.data == {"value": "Ciao", "point": [500, 900], "clear": False}
    a = candidate_to_action(c[7], "42")
    assert a.action_type == ActionType.ANSWER and a.data == {"value": "42"}
    assert candidate_to_action({"action": "NOPE"}).action_type == ActionType.ABORT


def test_observation_candidates_default_empty_and_choice_mode_switch() -> None:
    assert Observation().candidates == []
    with patch.dict(os.environ, {"BENCH_ACTION_MODE": "choice"}):
        assert choice_mode()
    with patch.dict(os.environ, {"BENCH_ACTION_MODE": ""}):
        assert not choice_mode()


# ------------------------------------------------------------------ agent
def test_click_choice_screenshot_mode() -> None:
    agent, llm = _agent(("1", {"stream": True, "logprobs": [{"token": "1", "logprob": -0.05}]}))
    a = agent.act(_obs())
    assert a.action_type == ActionType.CLICK and a.data == {"point": [100, 200]}
    assert a.explain.startswith("choice=1 n=8 p=0.95") and a.explain.endswith("calls=1")
    assert len(llm.calls) == 1
    messages, args = llm.calls[0]
    assert args["temperature"] == 0.0 and args["logprobs"] is True
    assert args["extra_body"]["grammar"] == build_grammar(8)
    user = messages[-1]["content"]
    assert user[0]["type"] == "image_url"                       # screenshot condition sends the image
    assert "[Available actions]\n1. tap: Mamma" in user[-1]["text"]
    assert "[Screen:" not in user[-1]["text"]
    assert agent.history[-1].llm_response == "1. tap: Mamma"


def test_type_choice_makes_second_call_for_the_text() -> None:
    agent, llm = _agent("2", '"Ciao Mamma"')
    a = agent.act(_obs())
    assert a.action_type == ActionType.TYPE
    assert a.data == {"value": "Ciao Mamma", "point": [500, 900], "clear": False}
    assert a.explain.endswith("calls=2")
    follow, args = llm.calls[1]
    assert follow[-2] == {"role": "assistant", "content": "2. type in: Message"}
    assert "Write the exact text to type" in follow[-1]["content"][0]["text"]
    assert "logprobs" not in args and "extra_body" not in args   # free text: no grammar
    assert agent.history[-1].llm_response == '2. type in: Message: "Ciao Mamma"'


def test_answer_choice_and_think_block_stripped() -> None:
    agent, _ = _agent("7", "<think>hmm</think>forty two")
    a = agent.act(_obs())
    assert a.action_type == ActionType.ANSWER and a.data == {"value": "forty two"}


def test_invalid_choice_aborts() -> None:
    agent, _ = _agent("99")
    a = agent.act(_obs())
    assert a.action_type == ActionType.ABORT and a.data["value"].startswith("invalid_choice")
    assert "choice=None" in a.explain


def test_empty_text_aborts() -> None:
    agent, _ = _agent("2", "   ")
    a = agent.act(_obs())
    assert a.action_type == ActionType.ABORT and a.data["value"] == "empty_text"


def test_no_candidates_aborts_without_calling_the_model() -> None:
    agent, llm = _agent()
    a = agent.act(Observation(screenshot_bytes=JPEG, step_idx=1))
    assert a.action_type == ActionType.ABORT and a.data["value"] == "no_candidates"
    assert llm.calls == []


def test_text_format_without_image_and_hybrid_with_image() -> None:
    tv = {"format": "html", "text": "<button data-ref=\"e1\">Mamma</button>", "refs": {}}
    agent, llm = _agent("5", "5")
    agent.act(_obs(text_view=tv))
    user = llm.calls[0][0][-1]["content"]
    assert [p["type"] for p in user] == ["text"]                 # text-only model: no image
    assert "[Screen: html]\n<button" in user[0]["text"]
    assert "HTML of the visible elements" in llm.calls[0][0][0]["content"]
    agent.reset("t")
    with patch.dict(os.environ, {"BENCH_OBS_IMAGE": "1"}):
        agent.act(_obs(text_view=tv))
    user = llm.calls[1][0][-1]["content"]
    assert [p["type"] for p in user] == ["image_url", "text"]    # hybrid
    assert "screenshot with a text description" in llm.calls[1][0][0]["content"]


def test_history_keeps_previous_choices_not_previous_screens() -> None:
    agent, llm = _agent("5", "1")
    agent.act(_obs())
    agent.act(Observation(screenshot_bytes=JPEG, step_idx=2, candidates=_cands()))
    messages = llm.calls[1][0]
    assert messages[1]["content"][0]["text"] == "[Task]\nSend a message to Mamma"
    assert messages[2] == {"role": "assistant", "content": "5. go back"}
    assert messages[3]["content"][-1]["text"].startswith("[Step 2]")
    assert sum(1 for m in messages if isinstance(m["content"], list)
               and any(p["type"] == "image_url" for p in m["content"])) == 1   # only the current screenshot


def test_constraint_switch() -> None:
    with patch.dict(os.environ, {"BENCH_CHOICE_CONSTRAINT": "none"}):
        agent, _ = _agent("1")
        assert "extra_body" not in agent._choice_args(8)
    with patch.dict(os.environ, {"BENCH_CHOICE_CONSTRAINT": "guided_choice"}):
        agent, _ = _agent("1")
        assert agent._choice_args(3)["extra_body"]["guided_choice"] == ["1", "2", "3"]
