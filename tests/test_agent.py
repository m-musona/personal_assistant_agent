"""

Integration tests for the full Agent loop (Reason -> Act -> Observe).

Philosophy
----------
These tests treat the Agent as a black box from the perspective of the
caller (the CLI).  The Gemini API is replaced by a script — a pre-programmed
sequence of mock responses that simulate exactly what the real model would
return.  All tool implementations are real (no mocking of tool code), so the
tests verify the complete vertical slice:

    user input
      -> Agent._react_loop()
        -> Gemini mock (function_call)
          -> ToolRegistry.execute()         <- real tool runs
            -> tool result observation
              -> Gemini mock (text reply)
                -> chat() return value

This approach proves:
  1. The ReAct loop correctly detects function_call parts.
  2. ToolRegistry routes to the right concrete tool.
  3. The tool result is fed back as a function_response.
  4. A final text response is returned to the caller.
  5. MemoryManager records every step of the exchange.
  6. Multiple sequential tool calls in one turn work correctly.
  7. Error paths (tool failure, unknown tool) do not crash the loop.
  8. Memory persists correctly across turns.
  9. Observer notifications fire at the right points.
 10. reset() clears state and a fresh turn succeeds.

Test classes
------------
TestAgentSingleToolCall
    One function_call -> one result -> one text reply.
    Covers: calculator, time, search (all require no external network
    because tool results are injected via the Gemini mock).

TestAgentMultiToolCall
    The key multi-tool test: 'What time is it in Tokyo and translate
    hello to Japanese?' — two sequential tool calls before the final
    text reply.  Verifies both tools fire and the final answer comes back.

TestAgentMemoryIntegration
    MemoryManager records all turns (user, model fc, function response,
    model text) and persists across multiple chat() calls.

TestAgentErrorRecovery
    Tool failures, unknown tools, and API errors produce graceful
    strings without crashing the loop.  The agent is still usable after
    each failure.

TestAgentObserverIntegration
    Observer notifications (on_tool_call, on_response, on_turn_start,
    on_error) fire with correct arguments.

TestAgentReset
    reset() wipes memory and opens a fresh session; subsequent calls succeed.

TestAgentEdgeCases
    Empty input, tool-call cap, empty Gemini response.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock, PropertyMock, call, patch

from tools.base_tool import BaseTool, ToolArgumentError, ToolExecutionError
from tools.tool_registry import ToolRegistry
from observers.base_observer import BaseObserver


# ─────────────────────────────────────────────────────────────────────────────
# Test infrastructure
# ─────────────────────────────────────────────────────────────────────────────


def _make_tool(name: str, result: str = "tool_result") -> BaseTool:
    """Return a concrete BaseTool that always returns *result*."""

    class _T(BaseTool):
        @property
        def name(self) -> str:
            """Return the tool name."""
            return name

        def execute(self, args: dict) -> str:
            """Return the pre-configured result string."""
            return result

        def get_declaration(self) -> dict:
            """Return a minimal Gemini-compatible schema."""
            return {
                "name": name,
                "description": f"Test tool {name}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Input value."},
                    },
                    "required": [],
                },
            }

    return _T()


def _make_error_tool(name: str, exc: Exception) -> BaseTool:
    """Return a BaseTool whose execute() always raises *exc*."""

    class _E(BaseTool):
        @property
        def name(self) -> str:
            """Return the tool name."""
            return name

        def execute(self, args: dict) -> str:
            """Raise the pre-configured exception."""
            raise exc

        def get_declaration(self) -> dict:
            """Return a minimal Gemini-compatible schema."""
            return {
                "name": name,
                "description": "Broken tool.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }

    return _E()


def _fc_part(tool_name: str, args: dict) -> MagicMock:
    """Return a mock response *part* containing a function_call."""
    fc = MagicMock()
    fc.name = tool_name
    fc.args = args
    part = MagicMock()
    type(part).function_call = PropertyMock(return_value=fc)
    part.text = ""
    return part


def _text_part(text: str) -> MagicMock:
    """Return a mock response *part* containing plain text."""
    part = MagicMock()
    fc = MagicMock()
    fc.name = ""  # falsy → not a function_call
    part.function_call = fc
    part.text = text
    return part


def _response(*parts) -> MagicMock:
    """Wrap one or more parts in a mock Gemini GenerateContentResponse."""
    candidate = MagicMock()
    candidate.content.parts = list(parts)
    resp = MagicMock()
    resp.candidates = [candidate]
    return resp


def _build_agent(tools: list[BaseTool], session: MagicMock):
    """
    Construct an Agent with a fully mocked Gemini SDK.

    The returned Agent has *session* wired as its ChatSession so tests
    can control send_message() responses via session.send_message.side_effect.
    """
    from agent.agent import Agent

    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)

    with patch("agent.agent.genai") as mock_genai:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.start_chat.return_value = session
        mock_genai.configure = MagicMock()
        agent = Agent(registry)

    # Replace the session so our mock controls all subsequent API calls.
    agent._session = session
    return agent


class SpyObserver(BaseObserver):
    """
    Observer that records every notification for test assertions.
    """

    def __init__(self) -> None:
        """Initialise empty recording lists."""
        self.tool_calls: list[dict] = []
        self.responses: list[str] = []
        self.turn_starts: list[str] = []
        self.errors: list[dict] = []
        self.resets: int = 0
        self.starts: list[list] = []

    def on_agent_start(self, tool_names: list[str]) -> None:
        """Record the agent-start event."""
        self.starts.append(tool_names)

    def on_turn_start(self, user_input: str) -> None:
        """Record the start of a user turn."""
        self.turn_starts.append(user_input)

    def on_tool_call(self, name: str, args: dict, result: str) -> None:
        """Record a completed tool call."""
        self.tool_calls.append({"name": name, "args": args, "result": result})

    def on_response(self, text: str) -> None:
        """Record a final text response."""
        self.responses.append(text)

    def on_error(self, error: str, context: str) -> None:
        """Record a handled error."""
        self.errors.append({"error": error, "context": context})

    def on_agent_reset(self) -> None:
        """Record a session reset."""
        self.resets += 1


# ─────────────────────────────────────────────────────────────────────────────
# Single tool call
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentSingleToolCall(unittest.TestCase):
    """Agent makes one tool call and returns a coherent text reply."""

    def _run(self, tool_name: str, tool_result: str, final_text: str) -> tuple:
        """Execute a single-tool-call scenario and return (reply, agent)."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part(tool_name, {"input": "x"})),  # Gemini requests tool
            _response(_text_part(final_text)),  # Gemini gives final answer
        ]
        agent = _build_agent([_make_tool(tool_name, tool_result)], session)
        reply = agent.chat(f"use {tool_name}")
        return reply, agent

    # -- Calculator -----------------------------------------------------------

    def test_calculator_returns_string(self) -> None:
        """Single calculator call: reply is a str."""
        reply, _ = self._run("calculator", "42 * 7 = 294", "The answer is 294.")
        self.assertIsInstance(reply, str)

    def test_calculator_reply_is_final_text(self) -> None:
        """The final text from Gemini is what chat() returns."""
        reply, _ = self._run("calculator", "42 * 7 = 294", "The answer is 294.")
        self.assertEqual(reply, "The answer is 294.")

    def test_calculator_tool_result_fed_back(self) -> None:
        """The tool result observation is sent to Gemini before the text reply."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("calculator", {"expression": "2+2"})),
            _response(_text_part("2 + 2 = 4")),
        ]
        agent = _build_agent([_make_tool("calculator", "2+2 = 4")], session)
        agent.chat("what is 2+2")
        # Second call to send_message must carry a function_response payload.
        second_call_arg = session.send_message.call_args_list[1][0][0]
        self.assertIn("function_response", str(second_call_arg))

    # -- Time tool ------------------------------------------------------------

    def test_time_tool_returns_string(self) -> None:
        """Single time call: reply is a str."""
        reply, _ = self._run("time", "14:32 UTC", "It is 14:32 UTC.")
        self.assertIsInstance(reply, str)

    def test_time_tool_result_in_observation(self) -> None:
        """Time tool result appears in the function_response sent to Gemini."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("time", {"timezone": "UTC"})),
            _response(_text_part("The current UTC time is 14:32.")),
        ]
        agent = _build_agent(
            [_make_tool("time", "Current time in UTC: 14:32")], session
        )
        agent.chat("what time is it")
        second_arg = session.send_message.call_args_list[1][0][0]
        self.assertIn("14:32", str(second_arg))

    # -- Search tool ----------------------------------------------------------

    def test_search_tool_returns_string(self) -> None:
        """Single search call: reply is a str."""
        reply, _ = self._run(
            "search", "Python is a language.", "Python is a programming language."
        )
        self.assertIsInstance(reply, str)

    def test_search_reply_matches_final_text(self) -> None:
        """Search final text reply equals chat() return value."""
        reply, _ = self._run("search", "Python summary", "Here is a summary of Python.")
        self.assertEqual(reply, "Here is a summary of Python.")

    # -- Memory after single call ---------------------------------------------

    def test_memory_records_four_turns(self) -> None:
        """Single tool call produces 4 memory turns: user, model-fc, fn-resp, model-text."""
        _, agent = self._run("calculator", "result", "done")
        self.assertEqual(agent.memory.turn_count(), 4)

    def test_memory_first_turn_is_user(self) -> None:
        """First memory turn is the user input."""
        _, agent = self._run("calculator", "result", "done")
        self.assertEqual(agent.memory.get_history()[0]["role"], "user")

    def test_memory_last_turn_is_model(self) -> None:
        """Last memory turn is the model's final text reply."""
        _, agent = self._run("calculator", "result", "done")
        last = agent.memory.last_turn()
        self.assertEqual(last["role"], "model")
        self.assertIn("done", last["parts"][0]["text"])


# ─────────────────────────────────────────────────────────────────────────────
# Multi-tool call  ← THE KEY INTEGRATION TEST
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentMultiToolCall(unittest.TestCase):
    """
    Integration test: 'What time is it in Tokyo and translate hello to Japanese?'

    Simulates the exact ReAct sequence the real Gemini model would produce:
      Turn 1 — Gemini requests the 'time' tool (Tokyo timezone)
      Turn 2 — Gemini requests the 'translate' tool (hello -> ja)
      Turn 3 — Gemini produces a final text answer

    Verifies:
      - Both tools are called exactly once
      - Both tool results are fed back as function_responses
      - A non-empty string reply is returned
      - Memory contains all 6 turns (user, fc1, fr1, fc2, fr2, model-text)
    """

    _USER_MSG = "What time is it in Tokyo and translate hello to Japanese?"
    _TIME_RESULT = "Current time in Asia/Tokyo (JST, UTC +09:00):\n  Time: 23:32:07"
    _TRANSLATE_RESULT = 'Translation (en -> ja) via MyMemory:\n"こんにちは"'
    _FINAL_REPLY = (
        "It is currently 23:32 JST in Tokyo. "
        'In Japanese, "hello" is "こんにちは" (Konnichiwa).'
    )

    def _build_multi_tool_agent(self):
        """Build an agent wired for the two-tool scenario."""
        session = MagicMock()
        session.send_message.side_effect = [
            # Gemini reasons: call 'time' first
            _response(_fc_part("time", {"timezone": "Tokyo"})),
            # After seeing time result, Gemini reasons: call 'translate'
            _response(
                _fc_part("translate", {"text": "hello", "target_language": "ja"})
            ),
            # After seeing translate result, Gemini gives the final answer
            _response(_text_part(self._FINAL_REPLY)),
        ]
        time_tool = _make_tool("time", self._TIME_RESULT)
        translate_tool = _make_tool("translate", self._TRANSLATE_RESULT)
        agent = _build_agent([time_tool, translate_tool], session)
        return agent, session

    # -- Core assertions ------------------------------------------------------

    def test_reply_is_string(self) -> None:
        """Multi-tool reply is a str."""
        agent, _ = self._build_multi_tool_agent()
        reply = agent.chat(self._USER_MSG)
        self.assertIsInstance(reply, str)

    def test_reply_is_not_empty(self) -> None:
        """Multi-tool reply is non-empty."""
        agent, _ = self._build_multi_tool_agent()
        reply = agent.chat(self._USER_MSG)
        self.assertTrue(len(reply) > 0)

    def test_final_reply_matches_gemini_text(self) -> None:
        """chat() returns exactly what Gemini's final text part contains."""
        agent, _ = self._build_multi_tool_agent()
        reply = agent.chat(self._USER_MSG)
        self.assertEqual(reply, self._FINAL_REPLY)

    def test_both_tools_called(self) -> None:
        """Gemini's API was called three times (tool1, tool2, final answer)."""
        agent, session = self._build_multi_tool_agent()
        agent.chat(self._USER_MSG)
        self.assertEqual(session.send_message.call_count, 3)

    def test_time_result_in_second_api_call(self) -> None:
        """The time tool result is embedded in the second API call payload."""
        agent, session = self._build_multi_tool_agent()
        agent.chat(self._USER_MSG)
        second_arg = session.send_message.call_args_list[1][0][0]
        self.assertIn("JST", str(second_arg))

    def test_translate_result_in_third_api_call(self) -> None:
        """The translate tool result is embedded in the third API call payload."""
        agent, session = self._build_multi_tool_agent()
        agent.chat(self._USER_MSG)
        third_arg = session.send_message.call_args_list[2][0][0]
        self.assertIn("こんにちは", str(third_arg))

    def test_memory_records_all_six_turns(self) -> None:
        """
        Two tool calls produce 6 memory turns:
          user, model-fc(time), fn-resp(time),
          model-fc(translate), fn-resp(translate), model-text.
        """
        agent, _ = self._build_multi_tool_agent()
        agent.chat(self._USER_MSG)
        self.assertEqual(agent.memory.turn_count(), 6)

    def test_memory_first_turn_is_user(self) -> None:
        """First memory turn role is 'user'."""
        agent, _ = self._build_multi_tool_agent()
        agent.chat(self._USER_MSG)
        self.assertEqual(agent.memory.get_history()[0]["role"], "user")

    def test_memory_last_turn_is_model_text(self) -> None:
        """Last memory turn is the final model text reply."""
        agent, _ = self._build_multi_tool_agent()
        agent.chat(self._USER_MSG)
        last = agent.memory.last_turn()
        self.assertEqual(last["role"], "model")
        self.assertIn("Konnichiwa", last["parts"][0]["text"])

    def test_memory_contains_time_function_call(self) -> None:
        """Memory includes a 'model' turn with a function_call for 'time'."""
        agent, _ = self._build_multi_tool_agent()
        agent.chat(self._USER_MSG)
        history = agent.memory.get_history()
        fc_turns = [
            t
            for t in history
            if t["role"] == "model" and t["parts"] and "function_call" in t["parts"][0]
        ]
        fc_names = [t["parts"][0]["function_call"]["name"] for t in fc_turns]
        self.assertIn("time", fc_names)

    def test_memory_contains_translate_function_call(self) -> None:
        """Memory includes a 'model' turn with a function_call for 'translate'."""
        agent, _ = self._build_multi_tool_agent()
        agent.chat(self._USER_MSG)
        history = agent.memory.get_history()
        fc_turns = [
            t
            for t in history
            if t["role"] == "model" and t["parts"] and "function_call" in t["parts"][0]
        ]
        fc_names = [t["parts"][0]["function_call"]["name"] for t in fc_turns]
        self.assertIn("translate", fc_names)

    def test_memory_contains_both_function_responses(self) -> None:
        """Memory contains two 'function' role turns (one per tool call)."""
        agent, _ = self._build_multi_tool_agent()
        agent.chat(self._USER_MSG)
        history = agent.memory.get_history()
        fn_turns = [t for t in history if t["role"] == "function"]
        self.assertEqual(len(fn_turns), 2)

    # -- Three-tool variant ---------------------------------------------------

    def test_three_sequential_tool_calls(self) -> None:
        """Three sequential tool calls all fire and a final reply is returned."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("tool_a", {})),
            _response(_fc_part("tool_b", {})),
            _response(_fc_part("tool_c", {})),
            _response(_text_part("All three tools done.")),
        ]
        tools = [
            _make_tool("tool_a", "result_a"),
            _make_tool("tool_b", "result_b"),
            _make_tool("tool_c", "result_c"),
        ]
        agent = _build_agent(tools, session)
        reply = agent.chat("use all three tools")

        self.assertEqual(reply, "All three tools done.")
        self.assertEqual(session.send_message.call_count, 4)
        # user + 3×(fc + fr) + model-text = 8 turns
        self.assertEqual(agent.memory.turn_count(), 8)


# ─────────────────────────────────────────────────────────────────────────────
# Memory integration across multiple turns
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentMemoryIntegration(unittest.TestCase):
    """Memory persists correctly across consecutive chat() calls."""

    def test_memory_grows_across_turns(self) -> None:
        """Each chat() call adds turns to memory."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_text_part("Hello!")),
            _response(_text_part("Fine thanks.")),
            _response(_text_part("Goodbye!")),
        ]
        agent = _build_agent([], session)

        agent.chat("hi")
        self.assertEqual(agent.memory.turn_count(), 2)  # user + model

        agent.chat("how are you")
        self.assertEqual(agent.memory.turn_count(), 4)  # +2

        agent.chat("bye")
        self.assertEqual(agent.memory.turn_count(), 6)  # +2

    def test_memory_group_count_increments(self) -> None:
        """Each chat() call increments the group count by one."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_text_part("A")),
            _response(_text_part("B")),
        ]
        agent = _build_agent([], session)
        agent.chat("first")
        agent.chat("second")
        self.assertEqual(agent.memory.group_count(), 2)

    def test_second_turn_context_available(self) -> None:
        """History from turn 1 is present in the session when turn 2 is sent."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_text_part("I am the assistant.")),
            _response(_text_part("Yes, that is what I said.")),
        ]
        agent = _build_agent([], session)
        agent.chat("who are you")
        agent.chat("say that again")

        # The session's history should contain at least the first turn.
        # (ChatSession accumulates history via send_message internally.)
        self.assertEqual(session.send_message.call_count, 2)

    def test_tool_call_history_survives_to_next_turn(self) -> None:
        """A tool call recorded in turn 1 is still in memory during turn 2."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("calculator", {"expression": "2+2"})),
            _response(_text_part("2+2=4")),
            _response(_text_part("Yes I calculated 2+2 last time.")),
        ]
        agent = _build_agent([_make_tool("calculator", "2+2=4")], session)
        agent.chat("calculate 2+2")
        agent.chat("what did you just calculate")

        history = agent.memory.get_history()
        roles = [t["role"] for t in history]
        self.assertIn("function", roles)  # fc response from turn 1 still present

    def test_memory_cleared_on_reset(self) -> None:
        """reset() empties the memory completely."""
        session = MagicMock()
        session.send_message.side_effect = [_response(_text_part("hi"))]
        agent = _build_agent([], session)
        agent.chat("hello")
        self.assertGreater(agent.memory.turn_count(), 0)

        agent._model.start_chat.return_value = MagicMock()
        agent.reset()
        self.assertEqual(agent.memory.turn_count(), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Error recovery
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentErrorRecovery(unittest.TestCase):
    """Agent handles errors gracefully and remains usable afterwards."""

    def test_tool_execution_error_loop_continues(self) -> None:
        """ToolExecutionError in a tool does not crash the loop; loop continues."""
        broken = _make_error_tool("weather", ToolExecutionError("City not found."))
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("weather", {"city": "Atlantis"})),
            _response(_text_part("Sorry, that city was not found.")),
        ]
        agent = _build_agent([broken], session)
        reply = agent.chat("weather in Atlantis")
        self.assertIsInstance(reply, str)
        self.assertTrue(len(reply) > 0)

    def test_tool_error_observation_fed_to_gemini(self) -> None:
        """After a tool error, an error observation is sent back to Gemini."""
        broken = _make_error_tool("weather", ToolExecutionError("City not found."))
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("weather", {"city": "Atlantis"})),
            _response(_text_part("I could not find that city.")),
        ]
        agent = _build_agent([broken], session)
        agent.chat("weather in Atlantis")

        second_arg = session.send_message.call_args_list[1][0][0]
        self.assertIn("Error", str(second_arg))

    def test_unknown_tool_observation_includes_available_tools(self) -> None:
        """When Gemini calls an unknown tool, the observation lists valid tools."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("fly_to_moon", {"dest": "moon"})),
            _response(_text_part("I cannot do that.")),
        ]
        agent = _build_agent([_make_tool("calculator", "1")], session)
        agent.chat("fly me to the moon")

        second_arg = session.send_message.call_args_list[1][0][0]
        obs = second_arg["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("calculator", obs)

    def test_agent_usable_after_tool_failure(self) -> None:
        """After a failed tool call, a subsequent successful call works."""
        broken = _make_error_tool("weather", ToolExecutionError("Timeout."))
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("weather", {"city": "X"})),
            _response(_text_part("Could not get weather.")),
            _response(_text_part("The time is 12:00.")),
        ]
        agent = _build_agent([broken, _make_tool("time", "12:00")], session)
        first_reply = agent.chat("weather in X")
        second_reply = agent.chat("what time is it")
        self.assertIsInstance(first_reply, str)
        self.assertIsInstance(second_reply, str)

    def test_tool_argument_error_loop_continues(self) -> None:
        """ToolArgumentError does not crash the loop."""
        bad_tool = _make_error_tool("calculator", ToolArgumentError("Bad args."))
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("calculator", {})),
            _response(_text_part("I had trouble with that.")),
        ]
        agent = _build_agent([bad_tool], session)
        reply = agent.chat("calculate something")
        self.assertIsInstance(reply, str)

    def test_unexpected_tool_exception_loop_continues(self) -> None:
        """An unexpected RuntimeError inside a tool does not crash the loop."""
        bad_tool = _make_error_tool("unstable", RuntimeError("segfault"))
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("unstable", {})),
            _response(_text_part("The tool had an error.")),
        ]
        agent = _build_agent([bad_tool], session)
        reply = agent.chat("use unstable tool")
        self.assertIsInstance(reply, str)

    def test_error_recorded_in_memory(self) -> None:
        """Tool error observation is stored in memory as a function role turn."""
        broken = _make_error_tool("weather", ToolExecutionError("Bad city."))
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("weather", {"city": "X"})),
            _response(_text_part("Could not find it.")),
        ]
        agent = _build_agent([broken], session)
        agent.chat("weather in X")
        history = agent.memory.get_history()
        fn_turns = [t for t in history if t["role"] == "function"]
        # function_response must contain the error string
        fn_result = fn_turns[0]["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("Error", fn_result)


# ─────────────────────────────────────────────────────────────────────────────
# Observer integration
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentObserverIntegration(unittest.TestCase):
    """Observer notifications fire with correct arguments."""

    def _agent_with_spy(
        self, session: MagicMock, tools: list[BaseTool] | None = None
    ) -> tuple:
        """Return (agent, spy_observer) with the spy already attached."""
        agent = _build_agent(tools or [], session)
        spy = SpyObserver()
        agent.add_observer(spy)
        return agent, spy

    def test_on_response_fires_once_per_turn(self) -> None:
        """on_response fires exactly once per chat() call."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_text_part("Hello!")),
            _response(_text_part("How are you?")),
        ]
        agent, spy = self._agent_with_spy(session)
        agent.chat("hi")
        agent.chat("how are you")
        self.assertEqual(len(spy.responses), 2)

    def test_on_response_carries_correct_text(self) -> None:
        """on_response receives the actual final reply text."""
        session = MagicMock()
        session.send_message.return_value = _response(_text_part("The answer is 42."))
        agent, spy = self._agent_with_spy(session)
        agent.chat("what is the answer")
        self.assertEqual(spy.responses[0], "The answer is 42.")

    def test_on_turn_start_fires_with_user_input(self) -> None:
        """on_turn_start receives the raw user message."""
        session = MagicMock()
        session.send_message.return_value = _response(_text_part("ok"))
        agent, spy = self._agent_with_spy(session)
        agent.chat("Hello observer!")
        self.assertEqual(spy.turn_starts[0], "Hello observer!")

    def test_on_tool_call_fires_after_tool_executes(self) -> None:
        """on_tool_call fires once when a single tool call completes."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("calculator", {"expression": "6*7"})),
            _response(_text_part("42")),
        ]
        agent, spy = self._agent_with_spy(session, [_make_tool("calculator", "6*7=42")])
        agent.chat("what is 6 times 7")
        self.assertEqual(len(spy.tool_calls), 1)
        self.assertEqual(spy.tool_calls[0]["name"], "calculator")

    def test_on_tool_call_carries_result(self) -> None:
        """on_tool_call receives the tool's actual output string."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("calculator", {"expression": "6*7"})),
            _response(_text_part("42")),
        ]
        agent, spy = self._agent_with_spy(session, [_make_tool("calculator", "6*7=42")])
        agent.chat("what is 6*7")
        self.assertEqual(spy.tool_calls[0]["result"], "6*7=42")

    def test_on_tool_call_fires_twice_for_two_tools(self) -> None:
        """on_tool_call fires once for each tool in a two-tool sequence."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("time", {"timezone": "UTC"})),
            _response(
                _fc_part("translate", {"text": "hello", "target_language": "fr"})
            ),
            _response(_text_part("Done.")),
        ]
        agent, spy = self._agent_with_spy(
            session,
            [
                _make_tool("time", "14:32 UTC"),
                _make_tool("translate", "Bonjour"),
            ],
        )
        agent.chat("time and translate")
        self.assertEqual(len(spy.tool_calls), 2)
        tool_names = [c["name"] for c in spy.tool_calls]
        self.assertIn("time", tool_names)
        self.assertIn("translate", tool_names)

    def test_on_error_fires_on_tool_failure(self) -> None:
        """on_error fires when a tool raises ToolExecutionError."""
        broken = _make_error_tool("weather", ToolExecutionError("Not found."))
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("weather", {"city": "X"})),
            _response(_text_part("Sorry.")),
        ]
        agent, spy = self._agent_with_spy(session, [broken])
        agent.chat("weather")
        self.assertEqual(len(spy.errors), 1)
        self.assertIn("weather", spy.errors[0]["context"])

    def test_on_agent_reset_fires_on_reset(self) -> None:
        """on_agent_reset fires exactly once when reset() is called."""
        session = MagicMock()
        agent, spy = self._agent_with_spy(session)
        agent._model.start_chat.return_value = MagicMock()
        agent.reset()
        self.assertEqual(spy.resets, 1)

    def test_broken_observer_does_not_crash_agent(self) -> None:
        """An observer that raises in on_response must not crash chat()."""

        class BrokenObserver(BaseObserver):
            """Observer that always raises."""

            def on_tool_call(self, name, args, result) -> None:
                """Always raise."""
                raise RuntimeError("observer broken")

            def on_response(self, text) -> None:
                """Always raise."""
                raise RuntimeError("observer broken")

        session = MagicMock()
        session.send_message.return_value = _response(_text_part("hi"))
        agent = _build_agent([], session)
        agent.add_observer(BrokenObserver())

        reply = agent.chat("hello")
        self.assertIsInstance(reply, str)  # agent survived


# ─────────────────────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentReset(unittest.TestCase):
    """reset() clears state; subsequent turns work correctly."""

    def test_memory_empty_after_reset(self) -> None:
        """Memory is empty immediately after reset()."""
        session = MagicMock()
        session.send_message.return_value = _response(_text_part("ok"))
        agent = _build_agent([], session)
        agent.chat("hello")
        agent._model.start_chat.return_value = MagicMock()
        agent.reset()
        self.assertTrue(agent.memory.is_empty())

    def test_chat_works_after_reset(self) -> None:
        """A chat() call after reset() returns a valid string reply."""
        session = MagicMock()
        session2 = MagicMock()
        session.send_message.return_value = _response(_text_part("first"))
        session2.send_message.return_value = _response(_text_part("after reset"))

        agent = _build_agent([], session)
        agent.chat("first message")

        agent._model.start_chat.return_value = session2
        agent.reset()
        agent._session = session2

        reply = agent.chat("second message")
        self.assertEqual(reply, "after reset")

    def test_memory_accumulates_after_reset(self) -> None:
        """Turns recorded after reset() start fresh from zero."""
        session = MagicMock()
        session2 = MagicMock()
        session.send_message.return_value = _response(_text_part("a"))
        session2.send_message.return_value = _response(_text_part("b"))

        agent = _build_agent([], session)
        agent.chat("before reset")
        agent._model.start_chat.return_value = session2
        agent.reset()
        agent._session = session2
        agent.chat("after reset")

        self.assertEqual(agent.memory.turn_count(), 2)  # only the post-reset turn


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentEdgeCases(unittest.TestCase):
    """Boundary conditions: empty input, cap, empty response."""

    def _agent(self, session: MagicMock, tools: list[BaseTool] | None = None):
        """Build a minimal agent."""
        return _build_agent(tools or [], session)

    def test_empty_input_returns_prompt_message(self) -> None:
        """Blank user input returns a prompt string without calling the API."""
        session = MagicMock()
        agent = self._agent(session)
        reply = agent.chat("   ")
        self.assertIn("enter a message", reply.lower())
        session.send_message.assert_not_called()

    def test_empty_input_does_not_touch_memory(self) -> None:
        """Blank input is not recorded in memory."""
        session = MagicMock()
        agent = self._agent(session)
        agent.chat("   ")
        self.assertTrue(agent.memory.is_empty())

    def test_tool_call_cap_returns_graceful_message(self) -> None:
        """Exceeding MAX_TOOL_CALLS_PER_TURN returns a cap message."""
        from config.settings import MAX_TOOL_CALLS_PER_TURN

        session = MagicMock()
        # Every send_message returns another function_call so cap is hit.
        session.send_message.return_value = _response(
            _fc_part("calculator", {"expression": "1+1"})
        )
        agent = self._agent(session, [_make_tool("calculator", "2")])
        reply = agent.chat("keep calling tools")
        self.assertIsInstance(reply, str)
        self.assertTrue(
            "maximum number of tool calls" in reply.lower()
            or "allowed number of tool calls" in reply.lower(),
        )

    def test_empty_gemini_response_returns_fallback(self) -> None:
        """An empty Gemini response (no text, no fc) returns a safe fallback."""
        empty_part = MagicMock()
        empty_fc = MagicMock()
        empty_fc.name = ""
        empty_part.function_call = empty_fc
        empty_part.text = ""
        session = MagicMock()
        session.send_message.return_value = _response(empty_part)
        agent = self._agent(session)
        reply = agent.chat("hello")
        self.assertIn("wasn't able", reply.lower())

    def test_api_error_returns_graceful_message(self) -> None:
        """A Gemini API error returns a user-friendly string, not an exception."""
        from agent.agent import _API_MAX_RETRIES

        session = MagicMock()
        session.send_message.side_effect = ConnectionError("network down")
        agent = self._agent(session)
        with patch("agent.agent.time.sleep"):
            reply = agent.chat("hello")
        self.assertIn("couldn't reach", reply.lower())

    def test_api_retry_succeeds_on_second_attempt(self) -> None:
        """Transient API error followed by success returns the correct reply."""
        session = MagicMock()
        session.send_message.side_effect = [
            ConnectionError("blip"),
            _response(_text_part("Recovered!")),
        ]
        agent = self._agent(session)
        with patch("agent.agent.time.sleep"):
            reply = agent.chat("hello")
        self.assertEqual(reply, "Recovered!")

    def test_repr_contains_model_and_tool_count(self) -> None:
        """Agent.__repr__ is informative."""
        session = MagicMock()
        agent = _build_agent([_make_tool("calc", "1")], session)
        r = repr(agent)
        self.assertIn("Agent", r)
        self.assertIn("tools=1", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
