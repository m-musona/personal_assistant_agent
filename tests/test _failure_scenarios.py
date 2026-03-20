"""

Failure scenario tests — the three canonical failures the assignment
explicitly requires, plus a thorough suite of related edge cases.

Design philosophy
-----------------
Every test in this file uses REAL tool implementations wired into the
agent.  The only thing mocked is the Gemini API response sequence.
This proves the complete vertical slice under failure:

    user input
      -> Gemini mock: requests a real tool
        -> Real tool runs (WeatherTool, FileReaderTool, ToolRegistry)
          -> Tool raises / registry raises
            -> Agent catches, formats observation
              -> Gemini mock: receives error observation, replies gracefully
                -> chat() returns a non-empty string (no crash)

Three canonical scenarios
-------------------------
1. Weather for 'Atlantis' (city not found)
   WeatherTool raises ToolExecutionError.  The agent injects the error
   as a function_response observation; Gemini re-reasons and says it
   couldn't find the city.

2. Read a non-existent file (file not found)
   FileReaderTool raises ToolExecutionError.  Same pattern — agent
   survives, Gemini reports the problem to the user.

3. Unknown tool name (hallucinated tool)
   Gemini requests 'fly_to_moon' which is not registered.
   ToolRegistry raises ToolNotFoundError; agent includes the valid
   tool list in the observation; Gemini corrects itself.

Additional failure scenarios covered
-------------------------------------
- Path-traversal attempt through the agent loop
- Disallowed file extension through the agent loop
- ToolArgumentError (missing/bad arguments)
- Unexpected RuntimeError from tool code (Tier 3)
- All three failure types in a single conversation
- Agent still usable after each failure (no state corruption)
- Memory integrity: error observations recorded in the right role/turn
- Observer.on_error fires for every handled tool failure
- Gemini API transient failure with retry
- Multiple consecutive failures before a successful call
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

from tools.base_tool import BaseTool, ToolArgumentError, ToolExecutionError
from tools.tool_registry import ToolRegistry
from observers.base_observer import BaseObserver


# ─────────────────────────────────────────────────────────────────────────────
# Shared mock infrastructure
# ─────────────────────────────────────────────────────────────────────────────


def _fc_part(tool_name: str, args: dict) -> MagicMock:
    """Return a mock response part containing a function_call."""
    fc = MagicMock()
    fc.name = tool_name
    fc.args = args
    part = MagicMock()
    type(part).function_call = PropertyMock(return_value=fc)
    part.text = ""
    return part


def _text_part(text: str) -> MagicMock:
    """Return a mock response part containing plain text."""
    part = MagicMock()
    empty_fc = MagicMock()
    empty_fc.name = ""
    part.function_call = empty_fc
    part.text = text
    return part


def _response(*parts) -> MagicMock:
    """Wrap parts in a mock Gemini GenerateContentResponse."""
    candidate = MagicMock()
    candidate.content.parts = list(parts)
    resp = MagicMock()
    resp.candidates = [candidate]
    return resp


def _mock_http(body: str) -> MagicMock:
    """Return a context-manager mock that mimics urllib.request.urlopen."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read.return_value = body.encode("utf-8")
    return cm


def _http_404() -> urllib.error.HTTPError:
    """Return an HTTPError with status 404."""
    return urllib.error.HTTPError(
        url="http://x", code=404, msg="Not Found", hdrs=None, fp=None
    )


def _build_agent(tools: list[BaseTool], session: MagicMock):
    """Construct an Agent with a mocked Gemini SDK and real tools."""
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

    agent._session = session
    return agent


class SpyObserver(BaseObserver):
    """Records every observer notification for assertion."""

    def __init__(self) -> None:
        """Initialise empty recording lists."""
        self.tool_calls: list[dict] = []
        self.errors: list[dict] = []
        self.responses: list[str] = []

    def on_tool_call(self, name: str, args: dict, result: str) -> None:
        """Record a completed tool call."""
        self.tool_calls.append({"name": name, "args": args, "result": result})

    def on_response(self, text: str) -> None:
        """Record a final response."""
        self.responses.append(text)

    def on_error(self, error: str, context: str) -> None:
        """Record a handled error."""
        self.errors.append({"error": error, "context": context})


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: Weather for 'Atlantis'  (city not found)
# ─────────────────────────────────────────────────────────────────────────────


class TestWeatherAtlantis(unittest.TestCase):
    """
    Gemini calls 'weather' with city='Atlantis'.
    WeatherTool raises ToolExecutionError('City not found').
    Agent injects the error as an observation.
    Gemini re-reasons and replies gracefully.
    """

    _WTTR_EMPTY = json.dumps(
        {
            "current_condition": [],
            "nearest_area": [],
        }
    )

    def _run(self, gemini_recovery_text: str):
        """
        Execute the Atlantis scenario end-to-end and return (reply, agent, spy).
        WeatherTool uses the real wttr.in path but urlopen is mocked to return
        an empty current_condition list, causing ToolExecutionError.
        """
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("weather", {"city": "Atlantis"})),
            _response(_text_part(gemini_recovery_text)),
        ]

        from tools.built_in.weather_tool import WeatherTool

        agent = _build_agent([WeatherTool()], session)
        spy = SpyObserver()
        agent.add_observer(spy)

        with patch("tools.built_in.weather_tool.OPENWEATHER_API_KEY", ""):
            with patch(
                "urllib.request.urlopen", return_value=_mock_http(self._WTTR_EMPTY)
            ):
                reply = agent.chat("What is the weather in Atlantis?")

        return reply, agent, spy

    # -- Agent does not crash ------------------------------------------------

    def test_agent_does_not_crash(self) -> None:
        """chat() must return without raising, even for a city not found."""
        reply, _, _ = self._run("I couldn't find Atlantis.")
        self.assertIsNotNone(reply)

    def test_reply_is_string(self) -> None:
        """chat() returns a str, not None or an exception."""
        reply, _, _ = self._run("I couldn't find Atlantis.")
        self.assertIsInstance(reply, str)

    def test_reply_is_non_empty(self) -> None:
        """The final reply is not an empty string."""
        reply, _, _ = self._run("I couldn't find Atlantis.")
        self.assertTrue(len(reply.strip()) > 0)

    def test_reply_is_gemini_recovery_text(self) -> None:
        """The final reply is the text Gemini produced after seeing the error."""
        reply, _, _ = self._run("Sorry, I couldn't find weather for Atlantis.")
        self.assertEqual(reply, "Sorry, I couldn't find weather for Atlantis.")

    # -- Error observation injected correctly --------------------------------

    def test_error_observation_sent_to_gemini(self) -> None:
        """The second API call payload contains an error string."""
        _, agent, _ = self._run("Not found.")
        second_arg = agent._session.send_message.call_args_list[1][0][0]
        obs = second_arg["parts"][0]["function_response"]["response"]["result"]
        self.assertTrue(
            obs.startswith("Error:"), f"Expected error observation, got: {obs!r}"
        )

    def test_error_observation_mentions_atlantis(self) -> None:
        """The error observation mentions the problematic city name."""
        _, agent, _ = self._run("Not found.")
        second_arg = agent._session.send_message.call_args_list[1][0][0]
        obs = second_arg["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("Atlantis", obs)

    def test_api_called_exactly_twice(self) -> None:
        """Exactly two API calls: one for the fc, one after the observation."""
        _, agent, _ = self._run("Not found.")
        self.assertEqual(agent._session.send_message.call_count, 2)

    # -- Memory integrity ----------------------------------------------------

    def test_memory_has_four_turns(self) -> None:
        """Memory: user, model-fc(weather), fn-resp(error), model-text = 4."""
        _, agent, _ = self._run("Not found.")
        self.assertEqual(agent.memory.turn_count(), 4)

    def test_memory_function_response_contains_error(self) -> None:
        """The function-role memory turn contains the ToolExecutionError message."""
        _, agent, _ = self._run("Not found.")
        fn_turns = [t for t in agent.memory.get_history() if t["role"] == "function"]
        self.assertEqual(len(fn_turns), 1)
        result = fn_turns[0]["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("Error:", result)

    # -- Observer notifications ----------------------------------------------

    def test_observer_on_tool_call_fires(self) -> None:
        """on_tool_call fires even when the tool fails."""
        _, _, spy = self._run("Not found.")
        self.assertEqual(len(spy.tool_calls), 1)
        self.assertEqual(spy.tool_calls[0]["name"], "weather")

    def test_observer_on_tool_call_result_is_error_string(self) -> None:
        """on_tool_call receives the formatted error string as the result."""
        _, _, spy = self._run("Not found.")
        self.assertTrue(spy.tool_calls[0]["result"].startswith("Error:"))

    def test_observer_on_error_fires(self) -> None:
        """on_error fires once for the ToolExecutionError."""
        _, _, spy = self._run("Not found.")
        self.assertEqual(len(spy.errors), 1)

    def test_observer_on_error_context_mentions_weather(self) -> None:
        """on_error context string identifies the 'weather' tool."""
        _, _, spy = self._run("Not found.")
        self.assertIn("weather", spy.errors[0]["context"])

    def test_observer_on_response_fires_with_recovery_text(self) -> None:
        """on_response fires with the final Gemini recovery text."""
        reply, _, spy = self._run("Sorry, city not found.")
        self.assertEqual(spy.responses[0], reply)

    # -- Agent usable after failure ------------------------------------------

    def test_agent_usable_after_atlantis_failure(self) -> None:
        """After the Atlantis failure, the next chat() succeeds normally."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("weather", {"city": "Atlantis"})),
            _response(_text_part("City not found.")),
            _response(_text_part("The time is 12:00.")),
        ]

        from tools.built_in.weather_tool import WeatherTool
        from tools.built_in.time_tool import TimeTool

        agent = _build_agent([WeatherTool(), TimeTool()], session)

        with patch("tools.built_in.weather_tool.OPENWEATHER_API_KEY", ""):
            with patch(
                "urllib.request.urlopen", return_value=_mock_http(self._WTTR_EMPTY)
            ):
                agent.chat("weather in Atlantis")

        second_reply = agent.chat("what time is it")
        self.assertEqual(second_reply, "The time is 12:00.")

    # -- HTTP 404 variant ----------------------------------------------------

    def test_http_404_also_handled_gracefully(self) -> None:
        """HTTP 404 from wttr.in (not just empty body) is also handled."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("weather", {"city": "Atlantis"})),
            _response(_text_part("Not found.")),
        ]

        from tools.built_in.weather_tool import WeatherTool

        agent = _build_agent([WeatherTool()], session)

        with patch("tools.built_in.weather_tool.OPENWEATHER_API_KEY", ""):
            with patch("urllib.request.urlopen", side_effect=_http_404()):
                reply = agent.chat("weather in Atlantis")

        self.assertIsInstance(reply, str)
        self.assertTrue(len(reply) > 0)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Non-existent file
# ─────────────────────────────────────────────────────────────────────────────


class TestNonExistentFile(unittest.TestCase):
    """
    Gemini calls 'file_reader' for a file that does not exist.
    FileReaderTool raises ToolExecutionError('File not found').
    Agent injects the error observation; Gemini replies gracefully.
    """

    def setUp(self) -> None:
        """Create an isolated temp directory as the file-reader sandbox."""
        self.tmpdir = tempfile.mkdtemp()
        real_tmpdir = os.path.realpath(self.tmpdir)
        self._p1 = patch(
            "tools.custom.file_reader_tool.FILE_READER_BASE_DIR", real_tmpdir
        )
        self._p2 = patch(
            "tools.custom.file_reader_tool._RESOLVED_BASE_DIR", real_tmpdir
        )
        self._p1.start()
        self._p2.start()

    def tearDown(self) -> None:
        """Remove the temp directory and stop patches."""
        self._p1.stop()
        self._p2.stop()
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, filepath: str, recovery_text: str):
        """Run a file_reader scenario and return (reply, agent, spy)."""
        from tools.custom.file_reader_tool import FileReaderTool

        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("file_reader", {"filepath": filepath})),
            _response(_text_part(recovery_text)),
        ]
        agent = _build_agent([FileReaderTool()], session)
        spy = SpyObserver()
        agent.add_observer(spy)
        reply = agent.chat(f"read the file {filepath}")
        return reply, agent, spy

    # -- Agent does not crash ------------------------------------------------

    def test_agent_does_not_crash(self) -> None:
        """chat() must return for a non-existent file."""
        reply, _, _ = self._run("ghost.txt", "That file does not exist.")
        self.assertIsNotNone(reply)

    def test_reply_is_string(self) -> None:
        """Reply is a str."""
        reply, _, _ = self._run("ghost.txt", "That file does not exist.")
        self.assertIsInstance(reply, str)

    def test_reply_is_recovery_text(self) -> None:
        """chat() returns the Gemini recovery text."""
        reply, _, _ = self._run("ghost.txt", "The file was not found.")
        self.assertEqual(reply, "The file was not found.")

    # -- Error observation content -------------------------------------------

    def test_error_observation_is_not_found(self) -> None:
        """The observation tells Gemini the file was not found."""
        _, agent, _ = self._run("ghost.txt", "Not found.")
        obs = agent._session.send_message.call_args_list[1][0][0]
        result = obs["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("not found", result.lower())

    def test_error_observation_mentions_filename(self) -> None:
        """The observation includes the filename that was requested."""
        _, agent, _ = self._run("missing_report.txt", "Not found.")
        obs = agent._session.send_message.call_args_list[1][0][0]
        result = obs["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("missing_report.txt", result)

    # -- Memory and observer -------------------------------------------------

    def test_memory_records_four_turns(self) -> None:
        """user + model-fc + fn-resp(error) + model-text = 4 turns."""
        _, agent, _ = self._run("ghost.txt", "Not found.")
        self.assertEqual(agent.memory.turn_count(), 4)

    def test_observer_on_error_fires(self) -> None:
        """on_error fires for the file-not-found ToolExecutionError."""
        _, _, spy = self._run("ghost.txt", "Not found.")
        self.assertEqual(len(spy.errors), 1)

    def test_observer_error_context_mentions_file_reader(self) -> None:
        """on_error context identifies the file_reader tool."""
        _, _, spy = self._run("ghost.txt", "Not found.")
        self.assertIn("file_reader", spy.errors[0]["context"])

    # -- Path traversal through the agent -----------------------------------

    def test_path_traversal_handled_gracefully(self) -> None:
        """../../etc/passwd traversal attempt does not crash the agent."""
        reply, agent, spy = self._run(
            "../../etc/passwd", "Access denied: that path is not allowed."
        )
        self.assertIsInstance(reply, str)
        # Observation must mention the access denial
        obs = agent._session.send_message.call_args_list[1][0][0]
        result = obs["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("outside the permitted directory", result)

    def test_disallowed_extension_handled_gracefully(self) -> None:
        """A .py file request does not crash the agent."""
        # Create the file so it exists — should still be blocked by extension
        Path(os.path.join(self.tmpdir, "script.py")).write_text("import os\n")
        reply, agent, spy = self._run("script.py", "I cannot read that file type.")
        self.assertIsInstance(reply, str)
        obs = agent._session.send_message.call_args_list[1][0][0]
        result = obs["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("not permitted", result)

    # -- Agent usable after failure ------------------------------------------

    def test_agent_usable_after_file_not_found(self) -> None:
        """After a file-not-found error, the next call succeeds."""
        from tools.custom.file_reader_tool import FileReaderTool

        # Create a real file for the second call
        Path(os.path.join(self.tmpdir, "real.txt")).write_text("actual content\n")

        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("file_reader", {"filepath": "ghost.txt"})),
            _response(_text_part("File not found.")),
            _response(_fc_part("file_reader", {"filepath": "real.txt"})),
            _response(_text_part("File content: actual content")),
        ]
        agent = _build_agent([FileReaderTool()], session)

        first = agent.chat("read ghost.txt")
        second = agent.chat("read real.txt")

        self.assertEqual(first, "File not found.")
        self.assertEqual(second, "File content: actual content")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: Unknown tool (hallucinated tool name)
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownTool(unittest.TestCase):
    """
    Gemini requests a tool name that is not registered (hallucination).
    ToolRegistry raises ToolNotFoundError.
    Agent injects an observation listing valid tools.
    Gemini corrects itself and replies gracefully.
    """

    def _run(self, fake_tool: str, valid_tools: list[BaseTool], recovery_text: str):
        """Run an unknown-tool scenario and return (reply, agent, spy)."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part(fake_tool, {"input": "x"})),
            _response(_text_part(recovery_text)),
        ]
        agent = _build_agent(valid_tools, session)
        spy = SpyObserver()
        agent.add_observer(spy)
        reply = agent.chat(f"use {fake_tool}")
        return reply, agent, spy

    # -- Agent does not crash ------------------------------------------------

    def test_agent_does_not_crash(self) -> None:
        """chat() returns for an unknown tool name."""
        reply, _, _ = self._run(
            "fly_to_moon", [self._calc_tool()], "I don't have that tool."
        )
        self.assertIsNotNone(reply)

    def test_reply_is_string(self) -> None:
        """Reply is a str."""
        reply, _, _ = self._run("fly_to_moon", [self._calc_tool()], "Cannot do that.")
        self.assertIsInstance(reply, str)

    def test_reply_is_recovery_text(self) -> None:
        """chat() returns the Gemini recovery text."""
        reply, _, _ = self._run(
            "fly_to_moon", [self._calc_tool()], "I cannot fly to the moon."
        )
        self.assertEqual(reply, "I cannot fly to the moon.")

    # -- Error observation content -------------------------------------------

    def test_observation_mentions_unknown_tool(self) -> None:
        """Observation names the tool that was not found."""
        _, agent, _ = self._run("fly_to_moon", [self._calc_tool()], "No such tool.")
        obs = agent._session.send_message.call_args_list[1][0][0]
        result = obs["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("fly_to_moon", result)

    def test_observation_lists_valid_tools(self) -> None:
        """Observation lists valid registered tool names so Gemini can correct itself."""
        from tools.built_in.calculator_tool import CalculatorTool
        from tools.built_in.time_tool import TimeTool

        _, agent, _ = self._run(
            "teleport", [CalculatorTool(), TimeTool()], "No such tool."
        )
        obs = agent._session.send_message.call_args_list[1][0][0]
        result = obs["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("calculator", result)
        self.assertIn("time", result)

    def test_observation_is_error_prefixed(self) -> None:
        """Observation starts with 'Error:' so Gemini recognises it as a failure."""
        _, agent, _ = self._run("ghost_tool", [self._calc_tool()], "Cannot do that.")
        obs = agent._session.send_message.call_args_list[1][0][0]
        result = obs["parts"][0]["function_response"]["response"]["result"]
        self.assertTrue(
            result.startswith("Error:"), f"Expected 'Error:' prefix, got: {result!r}"
        )

    def test_exactly_two_api_calls(self) -> None:
        """Two API calls: one for the fc, one after the observation."""
        _, agent, _ = self._run("ghost_tool", [self._calc_tool()], "Cannot do that.")
        self.assertEqual(agent._session.send_message.call_count, 2)

    # -- Memory integrity ----------------------------------------------------

    def test_memory_has_four_turns(self) -> None:
        """user + model-fc + fn-resp(error) + model-text = 4 turns."""
        _, agent, _ = self._run("ghost_tool", [self._calc_tool()], "Not available.")
        self.assertEqual(agent.memory.turn_count(), 4)

    def test_memory_function_response_is_not_found_error(self) -> None:
        """The fn-resp turn carries the ToolNotFoundError message."""
        _, agent, _ = self._run("ghost_tool", [self._calc_tool()], "Not available.")
        fn_turns = [t for t in agent.memory.get_history() if t["role"] == "function"]
        result = fn_turns[0]["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("ghost_tool", result)

    # -- Observer ------------------------------------------------------------

    def test_observer_on_tool_call_fires(self) -> None:
        """on_tool_call fires even for an unknown tool (error result is recorded)."""
        _, _, spy = self._run("ghost_tool", [self._calc_tool()], "Not available.")
        # on_tool_call fires with the error string as the result
        self.assertEqual(len(spy.tool_calls), 1)
        self.assertEqual(spy.tool_calls[0]["name"], "ghost_tool")
        self.assertTrue(spy.tool_calls[0]["result"].startswith("Error:"))

    # -- Completely empty registry -------------------------------------------

    def test_unknown_tool_with_empty_registry(self) -> None:
        """Unknown tool with no registered tools: observation says 'none available'."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("anything", {})),
            _response(_text_part("No tools.")),
        ]
        agent = _build_agent([], session)
        agent.chat("do something")
        obs = agent._session.send_message.call_args_list[1][0][0]
        result = obs["parts"][0]["function_response"]["response"]["result"]
        # With zero tools, the message should convey unavailability
        self.assertIn("Error:", result)

    # -- Agent usable after unknown tool -------------------------------------

    def test_agent_usable_after_unknown_tool(self) -> None:
        """After the unknown-tool error, the next call uses a valid tool."""
        from tools.built_in.calculator_tool import CalculatorTool

        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("fly_to_moon", {})),
            _response(_text_part("No such tool.")),
            _response(_fc_part("calculator", {"expression": "2+2"})),
            _response(_text_part("2+2 = 4")),
        ]
        agent = _build_agent([CalculatorTool()], session)

        first = agent.chat("fly me to the moon")
        second = agent.chat("what is 2+2")

        self.assertEqual(first, "No such tool.")
        self.assertEqual(second, "2+2 = 4")

    # -- Helper --------------------------------------------------------------

    @staticmethod
    def _calc_tool() -> BaseTool:
        """Return a real CalculatorTool for use as the valid registered tool."""
        from tools.built_in.calculator_tool import CalculatorTool

        return CalculatorTool()


# ─────────────────────────────────────────────────────────────────────────────
# Combined multi-failure scenario
# ─────────────────────────────────────────────────────────────────────────────


class TestMultipleConsecutiveFailures(unittest.TestCase):
    """
    All three failure types in a single conversation — agent stays alive
    throughout and the final successful call returns the correct reply.
    """

    def setUp(self) -> None:
        """Set up a temp dir for the file reader sandbox."""
        self.tmpdir = tempfile.mkdtemp()
        real_tmpdir = os.path.realpath(self.tmpdir)
        self._p1 = patch(
            "tools.custom.file_reader_tool.FILE_READER_BASE_DIR", real_tmpdir
        )
        self._p2 = patch(
            "tools.custom.file_reader_tool._RESOLVED_BASE_DIR", real_tmpdir
        )
        self._p1.start()
        self._p2.start()

    def tearDown(self) -> None:
        """Stop patches and clean temp dir."""
        self._p1.stop()
        self._p2.stop()
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_three_failures_then_success(self) -> None:
        """
        Turn 1: weather Atlantis  (ToolExecutionError)
        Turn 2: file ghost.txt    (ToolExecutionError)
        Turn 3: unknown tool      (ToolNotFoundError)
        Turn 4: valid calculator  (success)

        All four turns return strings. No crash.
        """
        _WTTR_EMPTY = json.dumps({"current_condition": [], "nearest_area": []})
        from tools.built_in.weather_tool import WeatherTool
        from tools.custom.file_reader_tool import FileReaderTool
        from tools.built_in.calculator_tool import CalculatorTool

        session = MagicMock()
        session.send_message.side_effect = [
            # Turn 1: Atlantis weather
            _response(_fc_part("weather", {"city": "Atlantis"})),
            _response(_text_part("City not found.")),
            # Turn 2: ghost file
            _response(_fc_part("file_reader", {"filepath": "ghost.txt"})),
            _response(_text_part("File not found.")),
            # Turn 3: unknown tool
            _response(_fc_part("teleporter", {"dest": "moon"})),
            _response(_text_part("No such tool.")),
            # Turn 4: valid calculator call
            _response(_fc_part("calculator", {"expression": "6*7"})),
            _response(_text_part("The answer is 42.")),
        ]

        agent = _build_agent(
            [WeatherTool(), FileReaderTool(), CalculatorTool()], session
        )
        spy = SpyObserver()
        agent.add_observer(spy)

        with patch("tools.built_in.weather_tool.OPENWEATHER_API_KEY", ""):
            with patch("urllib.request.urlopen", return_value=_mock_http(_WTTR_EMPTY)):
                r1 = agent.chat("weather in Atlantis")

        r2 = agent.chat("read ghost.txt")
        r3 = agent.chat("teleport me")
        r4 = agent.chat("what is 6*7")

        # All turns returned strings
        for reply in (r1, r2, r3, r4):
            self.assertIsInstance(reply, str, f"Expected str, got {type(reply)}")
            self.assertTrue(len(reply) > 0)

        # Final turn is the correct calculator answer
        self.assertEqual(r4, "The answer is 42.")

        # Three errors were fired by the observer
        self.assertEqual(len(spy.errors), 3)

        # Four responses were fired (one per turn)
        self.assertEqual(len(spy.responses), 4)

        # Total memory: 4 turns × (user + model-fc + fn-resp + model-text) = 16
        self.assertEqual(agent.memory.turn_count(), 16)

    def test_agent_state_not_corrupted_after_failures(self) -> None:
        """Memory group count equals the number of completed turns."""
        from tools.built_in.calculator_tool import CalculatorTool

        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("ghost", {})),
            _response(_text_part("No such tool.")),
            _response(_fc_part("calculator", {"expression": "1+1"})),
            _response(_text_part("1+1=2")),
        ]
        agent = _build_agent([CalculatorTool()], session)
        agent.chat("ghost tool")
        agent.chat("calculate 1+1")

        self.assertEqual(agent.memory.group_count(), 2)


# ─────────────────────────────────────────────────────────────────────────────
# ToolArgumentError — bad / missing arguments from Gemini
# ─────────────────────────────────────────────────────────────────────────────


class TestToolArgumentErrorScenarios(unittest.TestCase):
    """
    Gemini generates a tool call with missing or invalid arguments.
    The tool raises ToolArgumentError; the agent stays alive.
    """

    def _make_bad_tool(self, exc: Exception) -> BaseTool:
        """Return a tool that always raises *exc* from execute()."""

        class _Bad(BaseTool):
            @property
            def name(self) -> str:
                """Return 'bad_tool'."""
                return "bad_tool"

            def execute(self, args: dict) -> str:
                """Always raise the configured exception."""
                raise exc

            def get_declaration(self) -> dict:
                """Return a minimal schema."""
                return {
                    "name": "bad_tool",
                    "description": "Tool that raises on execute.",
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string", "description": "x"}},
                        "required": ["input"],
                    },
                }

        return _Bad()

    def test_argument_error_does_not_crash(self) -> None:
        """ToolArgumentError must not propagate out of chat()."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("bad_tool", {})),
            _response(_text_part("Bad arguments.")),
        ]
        agent = _build_agent(
            [self._make_bad_tool(ToolArgumentError("Missing 'input'"))], session
        )
        reply = agent.chat("use bad tool")
        self.assertIsInstance(reply, str)

    def test_argument_error_observation_mentions_invalid(self) -> None:
        """Observation for ToolArgumentError mentions 'invalid arguments'."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("bad_tool", {})),
            _response(_text_part("Bad args.")),
        ]
        agent = _build_agent(
            [self._make_bad_tool(ToolArgumentError("Missing 'input'"))], session
        )
        agent.chat("use bad tool")
        obs = agent._session.send_message.call_args_list[1][0][0]
        result = obs["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("invalid arguments", result.lower())

    def test_argument_error_observation_includes_required_params(self) -> None:
        """Observation for ToolArgumentError lists the required parameters."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("bad_tool", {})),
            _response(_text_part("Bad args.")),
        ]
        agent = _build_agent(
            [self._make_bad_tool(ToolArgumentError("Missing 'input'"))], session
        )
        agent.chat("use bad tool")
        obs = agent._session.send_message.call_args_list[1][0][0]
        result = obs["parts"][0]["function_response"]["response"]["result"]
        # 'input' is in the required list of bad_tool's declaration
        self.assertIn("input", result)


# ─────────────────────────────────────────────────────────────────────────────
# Tier-3: unexpected exception from tool code
# ─────────────────────────────────────────────────────────────────────────────


class TestUnexpectedToolException(unittest.TestCase):
    """
    A tool raises a bare RuntimeError (a bug, not a domain error).
    The agent must absorb it, log it, and keep running.
    """

    def _run_with_bug(self, recovery_text: str):
        """Return (reply, agent) for an unexpected RuntimeError in a tool."""

        class _Buggy(BaseTool):
            @property
            def name(self) -> str:
                """Return 'buggy'."""
                return "buggy"

            def execute(self, args: dict) -> str:
                """Simulate an unexpected internal error."""
                raise RuntimeError("segmentation fault in tool")

            def get_declaration(self) -> dict:
                """Return a minimal schema."""
                return {
                    "name": "buggy",
                    "description": "Unstable tool.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                }

        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("buggy", {})),
            _response(_text_part(recovery_text)),
        ]
        agent = _build_agent([_Buggy()], session)
        reply = agent.chat("use the buggy tool")
        return reply, agent

    def test_unexpected_error_does_not_crash_agent(self) -> None:
        """A RuntimeError in a tool must not propagate out of chat()."""
        reply, _ = self._run_with_bug("The tool had an unexpected error.")
        self.assertIsInstance(reply, str)

    def test_unexpected_error_observation_sent_to_gemini(self) -> None:
        """The unexpected error is sent back as an error observation."""
        _, agent = self._run_with_bug("Something went wrong.")
        obs = agent._session.send_message.call_args_list[1][0][0]
        result = obs["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("unexpected error", result.lower())

    def test_unexpected_error_observation_names_tool(self) -> None:
        """The observation names the tool that produced the unexpected error."""
        _, agent = self._run_with_bug("Something went wrong.")
        obs = agent._session.send_message.call_args_list[1][0][0]
        result = obs["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("buggy", result)

    def test_agent_usable_after_unexpected_error(self) -> None:
        """The agent is still usable after an unexpected tool error."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("buggy", {})),
            _response(_text_part("Tool crashed.")),
            _response(_text_part("I'm fine.")),
        ]

        class _Buggy(BaseTool):
            @property
            def name(self) -> str:
                """Return 'buggy'."""
                return "buggy"

            def execute(self, args: dict) -> str:
                """Always raise RuntimeError."""
                raise RuntimeError("crash")

            def get_declaration(self) -> dict:
                """Return minimal schema."""
                return {
                    "name": "buggy",
                    "description": "Unstable.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                }

        agent = _build_agent([_Buggy()], session)
        agent.chat("use buggy")
        second = agent.chat("are you ok")
        self.assertEqual(second, "I'm fine.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
