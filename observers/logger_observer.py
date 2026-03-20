"""

LoggerObserver - a concrete BaseObserver that writes a structured,
timestamped session log to logs/session.log (and optionally to the
Python logging system).

Log format
----------
Every entry is a single UTF-8 line with a fixed-width timestamp followed
by a type tag and payload:

    2025-06-20 14:32:01 [SESSION START] tools=calculator,weather,search,time,translate,file_reader
    2025-06-20 14:32:05 [TURN      ] user="What is 42 * 7?"
    2025-06-20 14:32:06 [TOOL CALL ] calculator(expression='42 * 7') -> "42 * 7 = 294"
    2025-06-20 14:32:06 [RESPONSE  ] "The answer is 294."
    2025-06-20 14:32:10 [TURN      ] user="Translate hello to Japanese"
    2025-06-20 14:32:11 [TOOL CALL ] translate(text='hello', target_language='ja') -> "Translation (en -> ja)..."
    2025-06-20 14:32:11 [RESPONSE  ] "In Japanese, "hello" is "JAPANESE_SYMBOLS" (Konnichiwa)."
    2025-06-20 14:32:15 [ERROR     ] context=tool:weather  error="City 'Atlantis' not found."
    2025-06-20 14:32:20 [RESET     ] session cleared
    2025-06-20 14:35:00 [SESSION END] turns=4  tool_calls=3  errors=1

Design notes
------------
SRP
    LoggerObserver has one job: record events to a file. It does not analyse,
    filter, or forward events. Rotation, shipping to a log aggregator, or
    pretty-printing to stdout are responsibilities of separate observers.

OCP / DIP
    The Agent never imports LoggerObserver. It depends only on BaseObserver.
    Attaching this observer in main.py is the sole wiring point.

Error isolation
---------------
All file I/O is wrapped in try/except. A full disk or permission error
raises ObserverError, which the Agent catches and logs without propagating.
A broken observer must never crash the assistant.

Thread safety
-------------
A threading.Lock guards every write so the observer is safe if the agent
is ever run in a multi-threaded context.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import TextIO

from config.settings import LOG_DIR
from observers.base_observer import BaseObserver, ObserverError

logger = logging.getLogger(__name__)

# Width of the tag field in log lines, for alignment.
_TAG_WIDTH = 13


def _now() -> str:
    """Return the current UTC time as a compact ISO-8601 string."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _truncate(text: str, max_chars: int = 120) -> str:
    """Shorten a string for inline display, appending '…' if truncated."""
    text = text.replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


class LoggerObserver(BaseObserver):
    """
    Writes a human-readable, timestamped event log for every agent session.

    One log file is opened per LoggerObserver instance. Multiple sessions
    (resets) are separated by SESSION START / SESSION END markers within
    the same file so the complete history of a process run is preserved.

    Usage
    -----
    from observers.logger_observer import LoggerObserver

    observer = LoggerObserver()          # defaults to logs/session.log
    agent.add_observer(observer)

    # Or with a custom path:
    observer = LoggerObserver(log_path="logs/debug.log")
    agent.add_observer(observer)
    """

    def __init__(self, log_path: str | None = None) -> None:
        """
        Open (or create) the log file and prepare per-session counters.

        Parameters
        ----------
        log_path : str | None
            Full path to the log file. Defaults to LOG_DIR/session.log.
            Parent directories are created automatically.
        """
        if log_path is None:
            os.makedirs(LOG_DIR, exist_ok=True)
            log_path = os.path.join(LOG_DIR, "session.log")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

        self._log_path: str = log_path
        self._lock: threading.Lock = threading.Lock()
        self._fh: TextIO | None = None
        self._turn_count: int = 0
        self._tool_call_count: int = 0
        self._error_count: int = 0

        # Open in append mode so multiple sessions accumulate in one file.
        self._open_file()
        logger.debug("LoggerObserver writing to: %s", self._log_path)

    # ------------------------------------------------------------------
    # Required BaseObserver methods
    # ------------------------------------------------------------------

    def on_tool_call(self, name: str, args: dict, result: str) -> None:
        """
        Log a completed tool call with name, arguments, and result preview.

        Format:
            2025-06-20 14:32:06 [TOOL CALL ] calculator(expression='42 * 7') -> "42 * 7 = 294"
        """
        self._tool_call_count += 1

        # Format args as a compact key=value string.
        args_str = ", ".join(
            f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items()
        )
        result_preview = _truncate(result)

        is_error = result.startswith("Error:")
        tag = "TOOL ERROR " if is_error else "TOOL CALL  "

        self._write(tag, f'{name}({args_str}) -> "{result_preview}"')

    def on_response(self, text: str) -> None:
        """
        Log the agent's final reply for the current turn.

        Format:
            2025-06-20 14:32:06 [RESPONSE  ] "The answer is 294."
        """
        self._write("RESPONSE   ", f'"{_truncate(text)}"')

    # ------------------------------------------------------------------
    # Optional lifecycle hooks (all overridden for rich logging)
    # ------------------------------------------------------------------

    def on_agent_start(self, tool_names: list[str]) -> None:
        """
        Write a session-start marker listing registered tools.

        Format:
            2025-06-20 14:32:01 [SESSION    ] START  tools=calculator,weather,...
        """
        tools_str = ",".join(tool_names)
        self._write("SESSION    ", f"START  tools={tools_str}")

    def on_turn_start(self, user_input: str) -> None:
        """
        Log the start of a new user turn with the raw input.

        Format:
            2025-06-20 14:32:05 [TURN       ] #1  user="What is 42 * 7?"
        """
        self._turn_count += 1
        preview = _truncate(user_input)
        self._write("TURN       ", f'#{self._turn_count}  user="{preview}"')

    def on_error(self, error: str, context: str) -> None:
        """
        Log a handled error with context.

        Format:
            2025-06-20 14:32:15 [ERROR      ] context=tool:weather  error="City not found."
        """
        self._error_count += 1
        error_preview = _truncate(error)
        self._write("ERROR      ", f'context={context}  error="{error_preview}"')

    def on_agent_reset(self) -> None:
        """
        Write a session-end summary then reset per-session counters.

        Format:
            2025-06-20 14:32:20 [SESSION    ] END  turns=4  tool_calls=3  errors=1
            ────────────────────────────────────────
        """
        self._write(
            "SESSION    ",
            f"END  turns={self._turn_count}  "
            f"tool_calls={self._tool_call_count}  "
            f"errors={self._error_count}",
        )
        self._write_raw("─" * 56 + "\n")
        # Reset counters for the new session.
        self._turn_count = 0
        self._tool_call_count = 0
        self._error_count = 0

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Flush and close the log file.

        Write a final session-end marker if the session was not explicitly
        reset. Safe to call multiple times (idempotent).
        """
        if self._fh is not None:
            with self._lock:
                try:
                    # Write a graceful end marker if counters show activity.
                    if self._turn_count > 0:
                        ts = _now()
                        tag = "SESSION    "
                        line = (
                            f"{ts} [{tag}] "
                            f"END  turns={self._turn_count}  "
                            f"tool_calls={self._tool_call_count}  "
                            f"errors={self._error_count}\n"
                        )
                        self._fh.write(line)
                    self._fh.flush()
                    self._fh.close()
                except OSError as exc:
                    logger.warning("LoggerObserver could not close file: %s", exc)
                finally:
                    self._fh = None

    def __del__(self) -> None:
        """Ensure the file is closed when the observer is garbage-collected."""
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        """Open the log file in append+line-buffered mode."""
        try:
            self._fh = open(  # noqa: WPS515
                self._log_path,
                mode="a",
                encoding="utf-8",
                buffering=1,  # line-buffered: each write is flushed immediately
            )
        except OSError as exc:
            raise ObserverError(
                f"LoggerObserver could not open {self._log_path!r}: {exc}"
            ) from exc

    def _write(self, tag: str, payload: str) -> None:
        """
        Write a single timestamped log entry in a thread-safe manner.

        Format:
            <timestamp> [<tag>] <payload>\\n

        Raises ObserverError on I/O failure so the Agent can log and
        continue without propagating the exception further.
        """
        if self._fh is None:
            return

        ts = _now()
        line = f"{ts} [{tag}] {payload}\n"

        with self._lock:
            try:
                self._fh.write(line)
            except OSError as exc:
                raise ObserverError(f"LoggerObserver write failed: {exc}") from exc

    def _write_raw(self, text: str) -> None:
        """Write raw text (e.g. a separator line) without a timestamp."""
        if self._fh is None:
            return
        with self._lock:
            try:
                self._fh.write(text)
            except OSError:
                pass  # Best-effort separator - don't raise for decorative lines.

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def log_path(self) -> str:
        """Absolute path of the log file being written."""
        return os.path.abspath(self._log_path)

    @property
    def stats(self) -> dict[str, int]:
        """Current session counters: turns, tool_calls, errors."""
        return {
            "turns": self._turn_count,
            "tool_calls": self._tool_call_count,
            "errors": self._error_count,
        }

    def __repr__(self) -> str:
        return (
            f"<LoggerObserver path={self._log_path!r} "
            f"turns={self._turn_count} "
            f"tool_calls={self._tool_call_count} "
            f"errors={self._error_count}>"
        )
