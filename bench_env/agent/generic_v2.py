"""
GenericAgentV2 - pure vision GUI agent using <think></think> <answer></answer> format.

Features:
- Output format: <think>reasoning</think> <answer>JSON action</answer>
- No current route information (simulates a real pure‑vision mobile GUI agent)
- Supports the full action space of the environment (except NOOP)
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar, Optional

from bench_env.agent.base import BaseAgent, AgentConfig, ActionMapping, AgentStepRecord
from bench_env.env.base import Action, ActionType, Observation
from bench_env.llm import LLMClient


class GenericAgentV2(BaseAgent):
    """
    Pure‑vision GUI agent using think/answer format.
    
    Features:
    - Output format: <THINK>reasoning</THINK> <ANSWER>JSON action</ANSWER>
    - Suitable for evaluating a model's pure‑vision GUI operation ability
    """

    SYSTEM_PROMPT: ClassVar[str] = """You are a phone GUI‑Agent operation expert. Based on the user's task, phone screenshots, and operation history, analyze the current interface and output a single action to interact with the phone and complete the task.

Coordinate system: Origin is the top‑left corner, x increases to the right, y increases downward. Both x and y range from 0 to 1000 (normalized coordinates).

Available actions (JSON format):

1. Click: {"action": "CLICK", "point": [x, y]}
2. Double‑tap: {"action": "DOUBLE_TAP", "point": [x, y]}
3. Long press: {"action": "LONGPRESS", "point": [x, y]}
4. Type: {"action": "TYPE", "value": "text content"} // Optional: "point": [x, y] to specify input location; optional "clear": true to clear the input field first (default is to append to existing text)
5. Swipe: {"action": "SWIPE", "point1": [x1, y1], "point2": [x2, y2]}
6. Drag: {"action": "DRAG", "point1": [x1, y1], "point2": [x2, y2]} // press and drag from start to end
7. Back: {"action": "BACK"}
8. Go to home: {"action": "HOME"}
9. Open recent tasks: {"action": "RECENT"}
10. Press Enter: {"action": "ENTER"}
11. Wait: {"action": "WAIT", "value": seconds}
12. Open an app: {"action": "AWAKE", "value": "app_name"}
13. Submit answer: {"action": "ANSWER", "value": "plain answer text"}
14. Complete task: {"action": "COMPLETE", "return": "completion explanation"} // Use after all subtasks are done, provide a brief explanation
15. Abort task: {"action": "ABORT", "value": "abort reason"} // Use if the task cannot be completed, state the reason

You must output in the following format:

<THINK>
Describe your understanding, analysis and decision process of the current screen.
Include:
1. What is shown on the current screen?
2. What should be done next to complete the task?
3. Which element should be clicked / operated on?
</THINK>
<ANSWER>
{
  "action": "action_type",
  // Fill in corresponding parameters based on the action type
}
</ANSWER>

Requirements:
- Coordinates must be integers in the range 0-1000.
- JSON must be valid.
- Observe the screenshot carefully and make decisions based on visual information.
- When you need to answer a question, you MUST use the ANSWER action.
- Use COMPLETE only to end the task, after you have performed the necessary actions.
- All your responses, including both the analysis and the JSON, MUST be in English.
"""

    ACTION_MAP: ActionMapping = {
        "CLICK": (ActionType.CLICK, lambda p: {"point": p.get("point")}),
        "TAP": (ActionType.CLICK, lambda p: {"point": p.get("point")}),
        "DOUBLE_TAP": (ActionType.DOUBLE_TAP, lambda p: {"point": p.get("point")}),
        "DOUBLETAP": (ActionType.DOUBLE_TAP, lambda p: {"point": p.get("point")}),
        "LONGPRESS": (ActionType.LONG_PRESS, lambda p: {"point": p.get("point")}),
        "LONG_PRESS": (ActionType.LONG_PRESS, lambda p: {"point": p.get("point")}),
        "TYPE": (ActionType.TYPE, lambda p: {"value": p.get("value", p.get("text", "")), "point": p.get("point"), "clear": p.get("clear", False)}),
        "SLIDE": (ActionType.SWIPE, lambda p: {"point1": p.get("point1", p.get("start")), "point2": p.get("point2", p.get("end"))}),
        "SWIPE": (ActionType.SWIPE, lambda p: {"point1": p.get("point1", p.get("start")), "point2": p.get("point2", p.get("end"))}),
        "DRAG": (ActionType.DRAG, lambda p: {"point1": p.get("point1", p.get("start")), "point2": p.get("point2", p.get("end"))}),
        "BACK": (ActionType.BACK, lambda p: {}),
        "HOME": (ActionType.HOME, lambda p: {}),
        "RECENT": (ActionType.RECENT, lambda p: {}),
        "ENTER": (ActionType.ENTER, lambda p: {}),
        "WAIT": (ActionType.WAIT, lambda p: {"value": float(p.get("value", p.get("duration", 1.0)))}),
        "AWAKE": (ActionType.AWAKE, lambda p: {"value": p.get("value", p.get("app", ""))}),
        "LAUNCH": (ActionType.AWAKE, lambda p: {"value": p.get("value", p.get("app", ""))}),
        "ANSWER": (ActionType.ANSWER, lambda p: {"value": p.get("value", p.get("text", ""))}),
        "COMPLETE": (ActionType.COMPLETE, lambda p: {"return": p.get("return", p.get("message", ""))}),
        "FINISH": (ActionType.COMPLETE, lambda p: {"return": p.get("return", p.get("message", ""))}),
        "ABORT": (ActionType.ABORT, lambda p: {"value": p.get("value", p.get("reason", ""))}),
    }

    DEFAULT_MODEL_ARGS: ClassVar[dict[str, Any]] = {
        "temperature": 0.1,
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "max_tokens": 8192,
        # "reasoning_effort": "none",
        # "extra_body": {
            # "chat_template_kwargs": {"enable_thinking": True},  # Qwen3.x via chat template ✓
            # "enable_thinking": False,           # Legacy vLLM flat parameter, ineffective for Qwen3.6-35B-A3B
            # "reasoning_effort": "none",         # OpenAI o1/o3 series
            # "reasoning": {"effort": "none"},    # OpenAI GPT‑5 series
        # },
    }

    # ==================== Initialization ====================

    def __init__(self, llm: LLMClient, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.llm = llm

        merged_args = dict(self.DEFAULT_MODEL_ARGS)
        merged_args.update(self.config.model_args or {})
        self.config.model_args = merged_args

    @property
    def name(self) -> str:
        return "GenericAgentV2"

    def reset(self, task: str) -> None:
        self._task = task
        self._history = []

    # ==================== Response parsing ====================

    @staticmethod
    def _extract_think_answer(text: str) -> tuple[str, str]:
        """
        Extract <think> and <answer> content from the text.
        
        Returns:
            (think_content, answer_content)
        """
        think_content = ""
        answer_content = ""
        
        # Extract <think>...</think> (case‑insensitive)
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
        if think_match:
            think_content = think_match.group(1).strip()
        
        # Extract <answer>...</answer> (case‑insensitive)
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer_content = answer_match.group(1).strip()
        
        return think_content, answer_content

    @staticmethod
    def _extract_first_json(text: str) -> Optional[str]:
        """Extract the first JSON object from the text."""
        s = text
        start = s.find("{")
        if start < 0:
            return None
        in_str = False
        esc = False
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
        return None

    def _parse_llm_output(self, response_text: str) -> tuple[str, dict[str, Any]]:
        """
        Parse LLM output into (thought, action_dict).
        
        Returns:
            (thought, action_dict)
        """
        raw = str(response_text or "").strip()
        if not raw:
            return "", {"_error": "empty_response"}

        # Extract think and answer parts
        think_content, answer_content = self._extract_think_answer(raw)
        
        # If no <answer> tag, try to parse the whole text
        if not answer_content:
            answer_content = raw
        
        # Parse JSON from answer
        parsed: Any = None
        try:
            parsed = json.loads(answer_content)
        except Exception:
            extracted = self._extract_first_json(answer_content)
            if extracted:
                try:
                    parsed = json.loads(extracted)
                except Exception:
                    pass

        if not isinstance(parsed, dict):
            return think_content, {"_error": "invalid_json", "raw": raw}

        return think_content, parsed

    def parse_response(self, response_text: str) -> Action:
        """Parse LLM response into an Action"""
        thought, parsed = self._parse_llm_output(response_text)
        
        if "_error" in parsed:
            return Action(
                action_type=ActionType.ABORT,
                data={"value": parsed.get("_error", "parse_error")},
                raw_response=response_text,
            )

        action_name = str(parsed.get("action") or parsed.get("action_type") or "").strip().upper()
        explain = str(parsed.get("explain", "") or "")
        
        return self.parse_action(
            action_name,
            parsed,
            thought=thought,
            explain=explain,
            raw_response=response_text,
        )

    # ==================== Message building ====================

    def build_messages(self, obs: Observation) -> list[dict]:
        """
        Build messages to send to the LLM.
        
        Uses full multi‑turn conversation history (like AutoGLMAgent):
        - Historical steps: user (task/step marker) + assistant (model response)
        - Current step: user (screenshot + prompt)
        
        Note: Does not include current route information; acts as a pure‑vision GUI agent.
        """
        messages: list[dict] = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        # Build conversation history
        for i, record in enumerate(self._history):
            # Historical user message (no screenshot)
            if i == 0:
                user_text = f"[Task]\n{self._task}"
            else:
                user_text = f"[Step {i + 1}]"
            
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            })
            
            # Historical assistant response
            messages.append({
                "role": "assistant",
                "content": record.llm_response,
            })

        # Current step (with screenshot)
        step_num = len(self._history) + 1
        if len(self._history) == 0:
            user_text = f"[Task]\n{self._task}"
        else:
            user_text = f"[Step {step_num}]"

        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": obs.image_data_url}},
                {"type": "text", "text": user_text},
            ],
        })

        return messages

    # ==================== Core logic ====================

    def act(self, obs: Observation) -> Action:
        """Generate an action."""
        messages = self.build_messages(obs)

        if self.config.verbose:
            print(f"\n[GenericAgentV2] Step {obs.step_idx}, sending prompt...")

        response = self.llm.chat(
            messages=messages,
            args={
                **self.config.model_args,
                "stream": self.config.stream,
                "stream_print": self.config.stream,
            },
        )

        if self.config.verbose and not self.config.stream:
            print(f"\n[LLM Response]\n{response.content}\n")

        action = self.parse_response(response.content)

        self._history.append(AgentStepRecord(
            step_idx=obs.step_idx,
            observation=obs,
            action=action,
            llm_response=response.content,
            llm_prompt=messages,
        ))

        # Memory optimisation: keep only the last 2 full records as text
        self._evict_old_records(keep_recent=2)

        if self.config.verbose:
            print(f"[GenericAgentV2] Action: {action.action_type}, Data: {action.data}")

        return action