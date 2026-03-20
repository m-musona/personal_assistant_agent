"""

Tests for the Agent's three-tier error handling strategy.

Test coverage
-------------
Tier 1 -- Fatal API errors:
    - Single API failure returns graceful user-facing message.
    - Retries fire (up to _API_MAX_RETRIES) before GeminiAPIError is raised.
    - chat() catches GeminiAPIError and returns a graceful string (no raise).

Tier 2 -- Recoverable tool errors:
    - ToolNotFoundError: observation includes available tool list.
    - ToolArgumentError: observation includes required parameter list.
    - ToolExecutionError: observation relays the tool's error message.
    All three: loop continues (agent does not crash), memory records error.

Tier 3 -- Unexpected tool bugs:
    - Bare Exception from tool code: observation returned, loop survives.

Additional:
    - Empty Gemini response: safe fallback message returned.
    - Tool call cap exceeded: graceful cap message returned.
    - Agent survives a full sequence: tool error -> second valid response.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from tools.base_tool import BaseTool, ToolArgumentError, ToolExecutionError
from tools.tool_registry import ToolNotFoundError, ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str, result: str = "ok") -> BaseTool:
    """Return a minimal concrete BaseTool that always returns `result`."""

    class _Tool(BaseTool):
        @property
        def name(self) -> str:
            return name

        def execute(self, args: dict) -> str:
            return result

        def get_declaration(self) -> dict:
            return {
                "name": name,
                "description": f"Test tool: {name}",
                "parameters": {
                    # "type": "object",
                    "type": "OBJECT",
                    "properties": {
                        "input": {
                            # "type": "string",
                            "type": "STRING",
                            "description": "Input.",
                        }
                    },
                    "required": ["input"],
                },
            }

    return _Tool()


def _make_error_tool(name: str, exc: Exception) -> BaseTool:
    """Return a BaseTool whose execute() always raises `exc`."""

    class _ErrorTool(BaseTool):
        @property
        def name(self) -> str:
            return name

        def execute(self, args: dict) -> str:
            raise exc

        def get_declaration(self) -> dict:
            return {
                "name": name,
                "description": "Broken tool.",
                "parameters": {
                    # "type": "object",
                    "type": "OBJECT",
                    "properties": {},
                    "required": [],
                },
            }

    return _ErrorTool()


def _make_text_response(text: str) -> MagicMock:
    """Build a mock Gemini response containing a plain text part."""
    part = MagicMock()
    part.function_call = MagicMock()
    part.function_call.name = ""  # falsy name => not a function call
    part.text = text
    candidate = MagicMock()
    candidate.content.parts = [part]
    response = MagicMock()
    response.candidates = [candidate]
    return response


def _make_function_call_response(tool_name: str, args: dict) -> MagicMock:
    """Build a mock Gemini response containing a function_call part."""
    fc = MagicMock()
    fc.name = tool_name
    fc.args = args

    part = MagicMock()
    # hasattr(part, "function_call") is True, and .name is truthy
    type(part).function_call = PropertyMock(return_value=fc)
    part.text = ""

    candidate = MagicMock()
    candidate.content.parts = [part]
    response = MagicMock()
    response.candidates = [candidate]
    return response


def _make_empty_response() -> MagicMock:
    """Build a mock Gemini response with no text and no function_call."""
    part = MagicMock()
    part.function_call = MagicMock()
    part.function_call.name = ""
    part.text = ""
    candidate = MagicMock()
    candidate.content.parts = [part]
    response = MagicMock()
    response.candidates = [candidate]
    return response


def _build_agent(registry: ToolRegistry, session_mock: MagicMock):
    """Construct an Agent with the Gemini SDK fully mocked."""
    from agent.agent import Agent

    with patch("agent.agent.genai") as mock_genai:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.start_chat.return_value = session_mock
        mock_genai.configure = MagicMock()
        agent = Agent(registry)

    # Wire in the session mock directly so send_message is under test control.
    agent._session = session_mock
    return agent


# ---------------------------------------------------------------------------
# Tier 1 -- Fatal API errors
# ---------------------------------------------------------------------------


class TestTier1APIErrors(unittest.TestCase):
    """Gemini API is unreachable or returns an error."""

    def _agent(self, session):
        r = ToolRegistry()
        r.register(_make_tool("calc", "42"))
        return _build_agent(r, session)

    def test_permanent_failure_returns_graceful_message(self):
        """All retries fail -> chat() returns a user-friendly string, not an exception."""
        from agent.agent import _API_MAX_RETRIES

        session = MagicMock()
        session.send_message.side_effect = ConnectionError("network down")

        agent = self._agent(session)
        with patch("agent.agent.time.sleep"):
            reply = agent.chat("hello")

        self.assertIn("couldn't reach", reply.lower())
        self.assertEqual(session.send_message.call_count, _API_MAX_RETRIES)

    def test_retry_succeeds_on_second_attempt(self):
        """First call fails, second succeeds -> normal reply returned."""
        session = MagicMock()
        session.send_message.side_effect = [
            ConnectionError("blip"),
            _make_text_response("Hello!"),
        ]

        agent = self._agent(session)
        with patch("agent.agent.time.sleep"):
            reply = agent.chat("hello")

        self.assertEqual(reply, "Hello!")
        self.assertEqual(session.send_message.call_count, 2)

    def test_agent_usable_after_api_failure(self):
        """After one failed turn, the next turn must succeed normally."""
        from agent.agent import _API_MAX_RETRIES

        session = MagicMock()
        session.send_message.side_effect = [
            ConnectionError("down")
        ] * _API_MAX_RETRIES + [_make_text_response("Recovered!")]

        agent = self._agent(session)
        with patch("agent.agent.time.sleep"):
            first = agent.chat("fail")
            second = agent.chat("hello")

        self.assertIn("couldn't reach", first.lower())
        self.assertEqual(second, "Recovered!")

    def test_memory_records_error_reply_on_failure(self):
        """Even on API failure the error message is persisted in memory."""
        from agent.agent import _API_MAX_RETRIES

        session = MagicMock()
        session.send_message.side_effect = RuntimeError("quota exceeded")

        agent = self._agent(session)
        with patch("agent.agent.time.sleep"):
            agent.chat("hello")

        self.assertEqual(agent.memory.turn_count(), 2)
        last = agent.memory.last_turn()
        self.assertEqual(last["role"], "model")
        self.assertIn("couldn't reach", last["parts"][0]["text"].lower())


# ---------------------------------------------------------------------------
# Tier 2 -- ToolNotFoundError
# ---------------------------------------------------------------------------


class TestTier2ToolNotFound(unittest.TestCase):
    """Gemini hallucinated a tool name that is not registered."""

    def test_observation_lists_valid_tools(self):
        """Error observation sent to Gemini must name the missing tool and list valid ones."""
        r = ToolRegistry()
        r.register(_make_tool("calculator", "9"))
        session = MagicMock()
        session.send_message.side_effect = [
            _make_function_call_response("fly_to_moon", {"dest": "moon"}),
            _make_text_response("Cannot do that, but I can calculate."),
        ]

        agent = _build_agent(r, session)
        agent.chat("fly me to the moon")

        second_arg = session.send_message.call_args_list[1][0][0]
        obs = second_arg["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("fly_to_moon", obs)
        self.assertIn("calculator", obs)

    def test_agent_does_not_crash(self):
        """Loop must continue after ToolNotFoundError and return a string."""
        r = ToolRegistry()
        r.register(_make_tool("calculator", "9"))
        session = MagicMock()
        session.send_message.side_effect = [
            _make_function_call_response("ghost", {}),
            _make_text_response("No such tool."),
        ]

        agent = _build_agent(r, session)
        reply = agent.chat("use ghost")
        self.assertIsInstance(reply, str)

    def test_memory_records_full_exchange(self):
        """history: user, model-fc, function-response, model-text = 4 turns."""
        r = ToolRegistry()
        r.register(_make_tool("calculator", "9"))
        session = MagicMock()
        session.send_message.side_effect = [
            _make_function_call_response("ghost_tool", {}),
            _make_text_response("Handled."),
        ]

        agent = _build_agent(r, session)
        agent.chat("use ghost tool")
        self.assertEqual(agent.memory.turn_count(), 4)


# ---------------------------------------------------------------------------
# Tier 2 -- ToolArgumentError
# ---------------------------------------------------------------------------


class TestTier2ToolArgumentError(unittest.TestCase):
    """Gemini generated malformed or missing arguments."""

    def test_observation_mentions_required_params(self):
        """Observation must be informative enough for Gemini to self-correct."""
        bad = _make_error_tool(
            "calculator",
            ToolArgumentError("Missing required argument: 'expression'."),
        )
        r = ToolRegistry()
        r.register(bad)
        session = MagicMock()
        session.send_message.side_effect = [
            _make_function_call_response("calculator", {}),
            _make_text_response("Let me fix that."),
        ]

        agent = _build_agent(r, session)
        agent.chat("calculate")

        obs_arg = session.send_message.call_args_list[1][0][0]
        obs = obs_arg["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("invalid arguments", obs.lower())

    def test_agent_survives_argument_error(self):
        """chat() must return a non-empty string, not raise."""
        bad = _make_error_tool("calculator", ToolArgumentError("Bad."))
        r = ToolRegistry()
        r.register(bad)
        session = MagicMock()
        session.send_message.side_effect = [
            _make_function_call_response("calculator", {}),
            _make_text_response("Handled."),
        ]

        agent = _build_agent(r, session)
        reply = agent.chat("broken calc")
        self.assertGreater(len(reply), 0)


# ---------------------------------------------------------------------------
# Tier 2 -- ToolExecutionError
# ---------------------------------------------------------------------------


class TestTier2ToolExecutionError(unittest.TestCase):
    """Tool ran but hit a recoverable runtime failure."""

    def test_error_message_relayed_in_observation(self):
        """Gemini must see the tool's specific error text so it can respond."""
        broken = _make_error_tool(
            "weather",
            ToolExecutionError("City 'Atlantis' not found."),
        )
        r = ToolRegistry()
        r.register(broken)
        session = MagicMock()
        session.send_message.side_effect = [
            _make_function_call_response("weather", {"city": "Atlantis"}),
            _make_text_response("That city doesn't exist."),
        ]

        agent = _build_agent(r, session)
        agent.chat("weather in Atlantis")

        obs_arg = session.send_message.call_args_list[1][0][0]
        obs = obs_arg["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("Atlantis", obs)

    def test_loop_continues_after_execution_error(self):
        """Agent must return the recovery text, not abort with an exception."""
        broken = _make_error_tool("weather", ToolExecutionError("Timeout."))
        r = ToolRegistry()
        r.register(broken)
        session = MagicMock()
        session.send_message.side_effect = [
            _make_function_call_response("weather", {"city": "X"}),
            _make_text_response("Could not get weather."),
        ]

        agent = _build_agent(r, session)
        reply = agent.chat("weather in X")
        self.assertEqual(reply, "Could not get weather.")


# ---------------------------------------------------------------------------
# Tier 3 -- Unexpected exceptions
# ---------------------------------------------------------------------------


class TestTier3UnexpectedErrors(unittest.TestCase):
    """Unexpected exceptions from tool code (bugs)."""

    def test_bare_exception_does_not_crash_agent(self):
        """A RuntimeError inside a tool must be absorbed; loop continues."""
        broken = _make_error_tool("unstable", RuntimeError("segfault"))
        r = ToolRegistry()
        r.register(broken)
        session = MagicMock()
        session.send_message.side_effect = [
            _make_function_call_response("unstable", {}),
            _make_text_response("Tool had an unexpected error."),
        ]

        agent = _build_agent(r, session)
        reply = agent.chat("use unstable")
        self.assertIsInstance(reply, str)

    def test_observation_mentions_tool_name(self):
        """Even for unexpected errors the observation must name the tool."""
        broken = _make_error_tool("boom", ValueError("kaboom"))
        r = ToolRegistry()
        r.register(broken)
        session = MagicMock()
        session.send_message.side_effect = [
            _make_function_call_response("boom", {}),
            _make_text_response("Something went wrong."),
        ]

        agent = _build_agent(r, session)
        agent.chat("boom")

        obs_arg = session.send_message.call_args_list[1][0][0]
        obs = obs_arg["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("boom", obs)
        self.assertIn("unexpected error", obs.lower())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    """Empty responses, tool cap, input validation, reset."""

    def _agent(self, tools, session):
        r = ToolRegistry()
        for t in tools:
            r.register(t)
        return _build_agent(r, session)

    def test_empty_response_returns_fallback(self):
        session = MagicMock()
        session.send_message.return_value = _make_empty_response()
        agent = self._agent([], session)
        reply = agent.chat("hello")
        self.assertIn("wasn't able", reply.lower())

    def test_tool_call_cap_returns_graceful_message(self):
        session = MagicMock()
        # Every send_message returns another function_call -> cap is always hit.
        session.send_message.return_value = _make_function_call_response(
            "calculator", {"expression": "1+1"}
        )
        agent = self._agent([_make_tool("calculator", "2")], session)
        reply = agent.chat("keep calling tools")
        self.assertIn("maximum number of tool calls", reply.lower())

    def test_empty_input_returns_prompt_no_api_call(self):
        session = MagicMock()
        agent = self._agent([], session)
        reply = agent.chat("   ")
        self.assertIn("enter a message", reply.lower())
        session.send_message.assert_not_called()

    def test_full_error_recovery_sequence(self):
        """tool error -> Gemini re-reasons -> correct final reply."""
        broken = _make_error_tool("weather", ToolExecutionError("City not found."))
        session = MagicMock()
        session.send_message.side_effect = [
            _make_function_call_response("weather", {"city": "NoWhere"}),
            _make_text_response("I'm sorry, that city was not found."),
        ]
        agent = self._agent([broken], session)
        reply = agent.chat("weather in NoWhere")

        self.assertEqual(reply, "I'm sorry, that city was not found.")
        # user + model-fc + fn-response + model-text
        self.assertEqual(agent.memory.turn_count(), 4)

    def test_reset_clears_memory_and_replaces_session(self):
        session = MagicMock()
        new_session = MagicMock()
        session.send_message.return_value = _make_text_response("hi")

        agent = self._agent([_make_tool("t", "1")], session)
        agent.chat("hi")
        self.assertGreater(agent.memory.turn_count(), 0)

        agent._model.start_chat.return_value = new_session
        agent.reset()

        self.assertEqual(agent.memory.turn_count(), 0)
        self.assertIs(agent._session, new_session)


if __name__ == "__main__":
    unittest.main(verbosity=2)
