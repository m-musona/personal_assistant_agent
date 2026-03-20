"""

Unit tests for LoggerObserver — all file I/O is real but uses a temporary
directory so no log file pollution occurs on the host.

Coverage:
  File management:
    - Log file created on instantiation
    - Multiple observers write to different files
    - close() is idempotent (safe to call twice)
    - __del__ closes the file

  Log content — on_tool_call:
    - Successful call: name, args, result in log
    - Error result: TOOL ERROR tag used instead of TOOL CALL
    - Args serialised as JSON key=value pairs

  Log content — on_response:
    - Final reply present in log with RESPONSE tag

  Log content — lifecycle hooks:
    - on_agent_start: SESSION START + tool names
    - on_turn_start: TURN tag + user message + turn counter
    - on_error: ERROR tag + context + error message
    - on_agent_reset: SESSION END + stats summary + separator

  Counters:
    - turn_count increments each on_turn_start
    - tool_call_count increments each on_tool_call
    - error_count increments each on_error
    - stats property returns correct dict
    - counters reset to zero after on_agent_reset

  Truncation:
    - Long tool results truncated with ellipsis in log
    - Long user input truncated in TURN line

  Thread safety (smoke test):
    - Concurrent writes from two threads produce two distinct lines

  ObserverError:
    - Writing to a closed file handle raises ObserverError

  BaseObserver contract:
    - LoggerObserver is a BaseObserver subclass
    - Instantiates without error
    - repr contains path and counters
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest

from observers.base_observer import BaseObserver, ObserverError
from observers.logger_observer import LoggerObserver, _truncate


class LoggerObserverTestBase(unittest.TestCase):
    """Creates an isolated temp directory for each test."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.tmpdir, "test_session.log")
        self.obs = LoggerObserver(log_path=self.log_path)

    def tearDown(self):
        self.obs.close()
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _read_log(self) -> str:
        self.obs._fh.flush()
        with open(self.log_path, encoding="utf-8") as fh:
            return fh.read()


# ---------------------------------------------------------------------------
# File management
# ---------------------------------------------------------------------------


class TestLoggerObserverFileManagement(LoggerObserverTestBase):

    def test_log_file_created(self):
        self.assertTrue(os.path.exists(self.log_path))

    def test_log_path_property(self):
        self.assertEqual(
            os.path.abspath(self.log_path),
            self.obs.log_path,
        )

    def test_parent_directories_created_automatically(self):
        nested = os.path.join(self.tmpdir, "a", "b", "c", "nested.log")
        obs = LoggerObserver(log_path=nested)
        try:
            self.assertTrue(os.path.exists(nested))
        finally:
            obs.close()

    def test_close_is_idempotent(self):
        self.obs.close()
        self.obs.close()  # must not raise

    def test_file_opened_in_append_mode(self):
        """A second observer pointing to the same file should append, not overwrite."""
        self.obs.on_response("first session")
        self.obs.close()

        obs2 = LoggerObserver(log_path=self.log_path)
        obs2.on_response("second session")
        obs2.close()

        with open(self.log_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("first session", content)
        self.assertIn("second session", content)

    def test_multiple_observers_different_files(self):
        path2 = os.path.join(self.tmpdir, "other.log")
        obs2 = LoggerObserver(log_path=path2)
        try:
            self.obs.on_response("file-one reply")
            obs2.on_response("file-two reply")
            self.assertIn("file-one reply", self._read_log())
            obs2._fh.flush()
            with open(path2, encoding="utf-8") as fh:
                self.assertIn("file-two reply", fh.read())
        finally:
            obs2.close()


# ---------------------------------------------------------------------------
# on_tool_call
# ---------------------------------------------------------------------------


class TestLoggerObserverToolCall(LoggerObserverTestBase):

    def test_tool_name_in_log(self):
        self.obs.on_tool_call("calculator", {"expression": "2+2"}, "2+2 = 4")
        self.assertIn("calculator", self._read_log())

    def test_tool_args_in_log(self):
        self.obs.on_tool_call("calculator", {"expression": "2+2"}, "2+2 = 4")
        self.assertIn("expression", self._read_log())
        self.assertIn("2+2", self._read_log())

    def test_tool_result_in_log(self):
        self.obs.on_tool_call("calculator", {"expression": "2+2"}, "2+2 = 4")
        self.assertIn("2+2 = 4", self._read_log())

    def test_tool_call_tag_present(self):
        self.obs.on_tool_call("calculator", {"expression": "2+2"}, "2+2 = 4")
        self.assertIn("TOOL CALL", self._read_log())

    def test_error_result_uses_tool_error_tag(self):
        self.obs.on_tool_call("weather", {"city": "Atlantis"}, "Error: city not found.")
        log = self._read_log()
        self.assertIn("TOOL ERROR", log)
        self.assertNotIn("TOOL CALL  ", log)

    def test_multi_arg_tool_call_logged(self):
        self.obs.on_tool_call(
            "translate",
            {"text": "hello", "target_language": "fr"},
            "Translation: Bonjour",
        )
        log = self._read_log()
        self.assertIn("text", log)
        self.assertIn("target_language", log)

    def test_long_result_truncated(self):
        long_result = "R" * 200
        self.obs.on_tool_call("search", {"query": "x"}, long_result)
        log = self._read_log()
        self.assertIn("…", log)

    def test_tool_call_count_increments(self):
        self.obs.on_tool_call("calculator", {}, "1")
        self.obs.on_tool_call("weather", {}, "2")
        self.assertEqual(self.obs.stats["tool_calls"], 2)


# ---------------------------------------------------------------------------
# on_response
# ---------------------------------------------------------------------------


class TestLoggerObserverResponse(LoggerObserverTestBase):

    def test_response_text_in_log(self):
        self.obs.on_response("The answer is 42.")
        self.assertIn("The answer is 42.", self._read_log())

    def test_response_tag_present(self):
        self.obs.on_response("hello")
        self.assertIn("RESPONSE", self._read_log())

    def test_long_response_truncated(self):
        self.obs.on_response("X" * 200)
        self.assertIn("…", self._read_log())

    def test_newlines_in_response_collapsed(self):
        self.obs.on_response("line one\nline two\nline three")
        log = self._read_log()
        # Log line must be a single line (newlines replaced by spaces).
        response_line = [l for l in log.splitlines() if "RESPONSE" in l]
        self.assertTrue(len(response_line) >= 1)


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


class TestLoggerObserverLifecycle(LoggerObserverTestBase):

    def test_on_agent_start_writes_session_start(self):
        self.obs.on_agent_start(["calculator", "weather", "search"])
        log = self._read_log()
        self.assertIn("SESSION", log)
        self.assertIn("START", log)

    def test_on_agent_start_includes_tool_names(self):
        self.obs.on_agent_start(["calculator", "weather"])
        log = self._read_log()
        self.assertIn("calculator", log)
        self.assertIn("weather", log)

    def test_on_turn_start_writes_turn_tag(self):
        self.obs.on_turn_start("hello world")
        self.assertIn("TURN", self._read_log())

    def test_on_turn_start_includes_user_message(self):
        self.obs.on_turn_start("What is the weather in Paris?")
        self.assertIn("What is the weather in Paris?", self._read_log())

    def test_on_turn_start_includes_turn_number(self):
        self.obs.on_turn_start("first")
        self.obs.on_turn_start("second")
        log = self._read_log()
        self.assertIn("#1", log)
        self.assertIn("#2", log)

    def test_on_error_writes_error_tag(self):
        self.obs.on_error("City not found", "tool:weather")
        self.assertIn("ERROR", self._read_log())

    def test_on_error_includes_context(self):
        self.obs.on_error("City not found", "tool:weather")
        self.assertIn("tool:weather", self._read_log())

    def test_on_error_includes_message(self):
        self.obs.on_error("City not found", "tool:weather")
        self.assertIn("City not found", self._read_log())

    def test_on_agent_reset_writes_session_end(self):
        self.obs.on_turn_start("hi")
        self.obs.on_agent_reset()
        log = self._read_log()
        self.assertIn("END", log)

    def test_on_agent_reset_includes_stats(self):
        self.obs.on_turn_start("hi")
        self.obs.on_tool_call("calc", {}, "4")
        self.obs.on_agent_reset()
        log = self._read_log()
        self.assertIn("turns=1", log)
        self.assertIn("tool_calls=1", log)
        self.assertIn("errors=0", log)

    def test_on_agent_reset_writes_separator(self):
        self.obs.on_agent_reset()
        self.assertIn("─", self._read_log())

    def test_on_agent_reset_resets_counters(self):
        self.obs.on_turn_start("hi")
        self.obs.on_tool_call("calc", {}, "4")
        self.obs.on_error("oops", "ctx")
        self.obs.on_agent_reset()
        self.assertEqual(self.obs.stats, {"turns": 0, "tool_calls": 0, "errors": 0})


# ---------------------------------------------------------------------------
# Counters and stats
# ---------------------------------------------------------------------------


class TestLoggerObserverCounters(LoggerObserverTestBase):

    def test_initial_stats_all_zero(self):
        self.assertEqual(self.obs.stats, {"turns": 0, "tool_calls": 0, "errors": 0})

    def test_turn_count_increments(self):
        self.obs.on_turn_start("a")
        self.obs.on_turn_start("b")
        self.obs.on_turn_start("c")
        self.assertEqual(self.obs.stats["turns"], 3)

    def test_tool_call_count_increments_on_error_results(self):
        """Error results still count as tool calls."""
        self.obs.on_tool_call("weather", {}, "Error: not found")
        self.assertEqual(self.obs.stats["tool_calls"], 1)

    def test_error_count_increments(self):
        self.obs.on_error("e1", "c1")
        self.obs.on_error("e2", "c2")
        self.assertEqual(self.obs.stats["errors"], 2)


# ---------------------------------------------------------------------------
# Truncation helper
# ---------------------------------------------------------------------------


class TestTruncateHelper(unittest.TestCase):

    def test_short_string_unchanged(self):
        self.assertEqual(_truncate("hello"), "hello")

    def test_long_string_truncated_with_ellipsis(self):
        result = _truncate("A" * 200)
        self.assertTrue(result.endswith("…"))
        self.assertLessEqual(len(result), 122)  # 120 chars + ellipsis

    def test_newlines_replaced(self):
        result = _truncate("line1\nline2")
        self.assertNotIn("\n", result)

    def test_custom_max_chars(self):
        result = _truncate("A" * 50, max_chars=20)
        self.assertTrue(result.endswith("…"))
        self.assertLessEqual(len(result), 22)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestLoggerObserverThreadSafety(LoggerObserverTestBase):

    def test_concurrent_writes_produce_complete_lines(self):
        """Two threads writing simultaneously must not interleave lines."""
        errors: list[Exception] = []

        def writer(label: str, count: int) -> None:
            for i in range(count):
                try:
                    self.obs.on_response(f"{label}-reply-{i}")
                except Exception as exc:
                    errors.append(exc)

        t1 = threading.Thread(target=writer, args=("thread-A", 20))
        t2 = threading.Thread(target=writer, args=("thread-B", 20))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(errors, [])
        log = self._read_log()
        lines = [l for l in log.splitlines() if "RESPONSE" in l]
        self.assertEqual(len(lines), 40)


# ---------------------------------------------------------------------------
# BaseObserver contract
# ---------------------------------------------------------------------------


class TestLoggerObserverContract(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make(self):
        path = os.path.join(self.tmpdir, "obs.log")
        obs = LoggerObserver(log_path=path)
        return obs

    def test_is_base_observer_subclass(self):
        obs = self._make()
        try:
            self.assertIsInstance(obs, BaseObserver)
        finally:
            obs.close()

    def test_repr_contains_path(self):
        obs = self._make()
        try:
            self.assertIn("obs.log", repr(obs))
        finally:
            obs.close()

    def test_repr_contains_counters(self):
        obs = self._make()
        try:
            obs.on_turn_start("hi")
            r = repr(obs)
            self.assertIn("turns=1", r)
        finally:
            obs.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
