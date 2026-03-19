"""
agent/memory_manager.py

Implements MemoryManager -- the agent's short-term conversational memory.

Design principles applied
--------------------------
SRP (Single Responsibility Principle)
    This class has exactly one job: store, retrieve, and manage the ordered
    list of conversation turns.  It knows nothing about the LLM, tools,
    or prompts.  The agent delegates all memory concerns here and never
    manipulates the history list directly.

Encapsulation
    The internal history list is private (_history).  All access goes
    through the public API so invariants (valid roles, turn cap) are always
    enforced in one place.

Extended for ReAct memory integration
    add_turn()      -- plain text turns  (user input, final model reply)
    add_raw_turn()  -- pre-built dicts   (function_call, function_response)
    Both methods enforce the cap and keep the history Gemini-compatible.
"""

from __future__ import annotations

import logging
from typing import Literal

from config.settings import MAX_HISTORY_TURNS

logger = logging.getLogger(__name__)

# Roles accepted by the Gemini chat API.
Role = Literal["user", "model", "function"]

# A single conversation turn as expected by the Gemini messages format.
Turn = dict


class MemoryManager:
    """
    Manages the agent's conversation history for a single session.

    Stores two kinds of turns:

    1. Plain-text turns (via add_turn):
           {"role": "user",  "parts": [{"text": "What is 2+2?"}]}
           {"role": "model", "parts": [{"text": "The answer is 4."}]}

    2. Structured turns (via add_raw_turn) for the ReAct loop:
       -- Model intends a tool call (function_call):
           {"role": "model", "parts": [{"function_call": {"name": ..., "args": ...}}]}
       -- Tool result sent back (function_response):
           {"role": "function", "parts": [{"function_response": {"name": ..., "response": ...}}]}

    The history is capped at MAX_HISTORY_TURNS *exchange groups*.
    One exchange group = the full set of turns for a single user request:
        user input -> [model function_call -> function_response]* -> model text reply
    When the cap is reached the oldest group is evicted atomically.

    Usage
    -----
    memory = MemoryManager()

    # Plain text turns
    memory.add_turn("user", "What's the weather in Paris?")

    # Structured turns from the ReAct loop
    memory.add_raw_turn({
        "role": "model",
        "parts": [{"function_call": {"name": "weather", "args": {"city": "Paris"}}}]
    })
    memory.add_raw_turn({
        "role": "function",
        "parts": [{"function_response": {"name": "weather", "response": {"result": "18 C"}}}]
    })
    memory.add_turn("model", "The weather in Paris is 18 C.")

    history = memory.get_history()   # pass to Gemini ChatSession
    memory.clear()                   # reset for a new session
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS) -> None:
        """
        Parameters
        ----------
        max_turns : int
            Maximum number of user/model exchange *groups* to retain.
            Defaults to MAX_HISTORY_TURNS from settings.
            Set to 0 for unlimited history (not recommended in production).
        """
        if max_turns < 0:
            raise ValueError(f"max_turns must be >= 0, got {max_turns}.")

        self._max_turns: int = max_turns
        self._history: list[Turn] = []

        # Tracks where each exchange group starts in _history.
        # Each entry is the index of the "user" turn that opened that group.
        # Used by _enforce_cap() to evict whole groups atomically.
        self._group_starts: list[int] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_turn(self, role: Role, content: str) -> None:
        """
        Append a plain-text turn to the conversation history.

        Use this for:
          - User input:          add_turn("user", "Hello!")
          - Final model reply:   add_turn("model", "Hi! How can I help?")

        For structured turns (function_call / function_response parts)
        produced inside the ReAct loop, use add_raw_turn() instead.

        Parameters
        ----------
        role : Role
            "user" or "model" for plain-text turns.
        content : str
            The text content of this turn. Must not be empty.

        Raises
        ------
        ValueError
            If role is invalid or content is empty.
        """
        self._validate_role(role)

        if not content or not content.strip():
            raise ValueError(f"Content for role {role!r} must not be empty.")

        # A new "user" turn opens a new exchange group.
        if role == "user":
            self._group_starts.append(len(self._history))

        turn: Turn = {"role": role, "parts": [{"text": content}]}
        self._history.append(turn)
        logger.debug("add_turn role=%r chars=%d", role, len(content))

        self._enforce_cap()

    def add_raw_turn(self, turn: Turn) -> None:
        """
        Append a pre-built Gemini-format turn dict to the conversation history.

        Use this inside the ReAct loop for structured parts that cannot be
        expressed as plain text:

        Model intending a tool call --
            {
                "role": "model",
                "parts": [{"function_call": {"name": "weather", "args": {...}}}]
            }

        Tool result returned --
            {
                "role": "function",
                "parts": [{"function_response": {"name": "weather",
                                                 "response": {"result": "18 C"}}}]
            }

        The turn is appended as-is after basic structure validation.
        These intermediate turns belong to the *current* exchange group
        (opened by the preceding add_turn("user", ...)) so they are
        evicted together when that group ages out.

        Parameters
        ----------
        turn : Turn
            A dict with at minimum a "role" key and a non-empty "parts" list.

        Raises
        ------
        ValueError
            If the turn dict is missing required keys or has an invalid role.
        """
        if not isinstance(turn, dict):
            raise ValueError(f"turn must be a dict, got {type(turn).__name__!r}.")
        if "role" not in turn or "parts" not in turn:
            raise ValueError(
                "turn dict must contain 'role' and 'parts' keys. "
                f"Got keys: {list(turn.keys())}"
            )
        if not turn["parts"]:
            raise ValueError("turn['parts'] must not be empty.")

        self._validate_role(turn["role"])

        self._history.append(turn)
        logger.debug(
            "add_raw_turn role=%r parts=%d",
            turn["role"],
            len(turn["parts"]),
        )
        # Raw turns (function_call / function_response) are mid-group;
        # no new group_start is recorded -- they belong to the open group.

    def get_history(self) -> list[Turn]:
        """
        Return a shallow copy of the full conversation history.

        The copy ensures callers cannot accidentally mutate internal state.
        Pass this directly to a Gemini ChatSession's history parameter or
        use it to reseed a fresh session after reset().

        Returns
        -------
        list[Turn]
            Ordered list of turn dicts, oldest first. Each dict is
            Gemini-compatible and can be passed verbatim to the SDK.
        """
        return list(self._history)

    def clear(self) -> None:
        """
        Erase all conversation history and group bookmarks.

        Call this to start a fresh session without constructing a new
        MemoryManager instance (e.g. when the user types 'reset').
        """
        count = len(self._history)
        self._history.clear()
        self._group_starts.clear()
        logger.info("Memory cleared -- removed %d turns.", count)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def turn_count(self) -> int:
        """Total number of individual turns stored (text + structured)."""
        return len(self._history)

    def group_count(self) -> int:
        """Number of complete exchange groups (one per user request)."""
        return len(self._group_starts)

    def is_empty(self) -> bool:
        """Return True if no turns have been recorded yet."""
        return len(self._history) == 0

    def last_turn(self) -> Turn | None:
        """Return the most recent turn dict, or None if history is empty."""
        return self._history[-1] if self._history else None

    def get_last_n_turns(self, n: int) -> list[Turn]:
        """
        Return the most recent ``n`` individual turns.

        Parameters
        ----------
        n : int
            Number of turns to retrieve (must be > 0).

        Returns
        -------
        list[Turn]
            A copy of the last n turns, or all turns if n > len(history).
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}.")
        return list(self._history[-n:])

    def summary(self) -> str:
        """
        Return a human-readable one-liner for logging and debugging.

        Example: "MemoryManager: 3 groups, 8 turns, cap=20"
        """
        return (
            f"MemoryManager: {self.group_count()} groups, "
            f"{self.turn_count()} turns, "
            f"cap={self._max_turns}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_cap(self) -> None:
        """
        Evict the oldest exchange group when the group cap is exceeded.

        A group is the cluster of turns for one user request:
            user turn -> model function_call(s) + function_response(s) -> model text
        Evicting whole groups keeps the history Gemini-compatible (must
        start with a "user" turn) and avoids orphaned function_response
        turns that have no matching function_call.

        A max_turns of 0 disables the cap entirely.
        """
        if self._max_turns == 0:
            return

        while len(self._group_starts) > self._max_turns:
            # Find where the oldest group ends (= where the next group starts).
            oldest_start = self._group_starts.pop(0)

            if self._group_starts:
                next_start = self._group_starts[0]
                evict_count = next_start - oldest_start
            else:
                # Only one group left; evict everything.
                evict_count = len(self._history)

            for _ in range(evict_count):
                evicted = self._history.pop(0)
                logger.debug(
                    "Evicted turn (cap=%d groups) -- role: %r",
                    self._max_turns,
                    evicted.get("role"),
                )

            # After eviction the remaining group_starts are offset by
            # evict_count positions -- adjust them.
            self._group_starts = [s - evict_count for s in self._group_starts]

    @staticmethod
    def _validate_role(role: str) -> None:
        """Raise ValueError if role is not an accepted Gemini role."""
        valid = {"user", "model", "function"}
        if role not in valid:
            raise ValueError(
                f"Invalid role {role!r}. "
                f"Must be one of: {', '.join(sorted(valid))}."
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of individual turns stored."""
        return len(self._history)

    def __repr__(self) -> str:
        return (
            f"<MemoryManager "
            f"groups={self.group_count()} "
            f"turns={self.turn_count()} "
            f"max_turns={self._max_turns}>"
        )
