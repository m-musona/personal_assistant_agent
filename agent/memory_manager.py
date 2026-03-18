"""

Implements MemoryManager — the agent's short-term conversational memory.

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
"""

from __future__ import annotations

import logging
from typing import Literal

from config.settings import MAX_HISTORY_TURNS

logger = logging.getLogger(__name__)

# Roles accepted by the Gemini chat API.
Role = Literal["user", "model", "function"]

# A single conversation turn as expected by the Gemini messages format.
Turn = dict  # {"role": Role, "parts": [{"text": str}]}


class MemoryManager:
    """
    Manages the agent's conversation history for a single session.

    Each turn is stored as a Gemini-compatible dict:

        {"role": "user",  "parts": [{"text": "What is 2 + 2?"}]}
        {"role": "model", "parts": [{"text": "The answer is 4."}]}

    The history is capped at MAX_HISTORY_TURNS pairs (user + model).
    When the cap is reached the oldest *pair* is evicted so the list never
    exceeds 2 × MAX_HISTORY_TURNS entries. This keeps token usage bounded
    while preserving the alternating user/model structure Gemini requires.

    Usage
    -----
    memory = MemoryManager()
    memory.add_turn("user", "Hello!")
    memory.add_turn("model", "Hi! How can I help?")

    history = memory.get_history()   # pass to Gemini chat session
    memory.clear()                   # reset for a new session
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS) -> None:
        """
        Parameters
        ----------
        max_turns : int
            Maximum number of user/model *pairs* to retain.
            Defaults to MAX_HISTORY_TURNS from settings.
            Set to 0 for unlimited history (not recommended for production).
        """
        if max_turns < 0:
            raise ValueError(f"max_turns must be >= 0, got {max_turns}.")

        self._max_turns: int = max_turns
        self._history: list[Turn] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_turn(self, role: Role, content: str) -> None:
        """
        Append a single turn to the conversation history.

        Parameters
        ----------
        role : Role
            One of "user", "model", or "function".
            "user"     — human input or tool result fed back to the model.
            "model"    — assistant response (text or function call intent).
            "function" — raw tool execution result (Gemini-specific role).
        content : str
            The text content of this turn.

        Raises
        ------
        ValueError
            If role is not one of the accepted values, or content is empty.
        """
        self._validate_role(role)

        if not content or not content.strip():
            raise ValueError(f"Content for role {role!r} must not be empty.")

        turn: Turn = {"role": role, "parts": [{"text": content}]}
        self._history.append(turn)
        logger.debug("Added turn — role: %r, chars: %d", role, len(content))

        self._enforce_cap()

    def get_history(self) -> list[Turn]:
        """
        Return a shallow copy of the full conversation history.

        Returns a copy so callers cannot accidentally mutate internal state.
        Pass this directly to a Gemini ChatSession or to genai.send_message().

        Returns
        -------
        list[Turn]
            Ordered list of turn dicts, oldest first.
        """
        return list(self._history)

    def clear(self) -> None:
        """
        Erase all conversation history.

        Call this to start a fresh session without constructing a new
        MemoryManager instance (e.g. when the user types 'reset' in the CLI).
        """
        count = len(self._history)
        self._history.clear()
        logger.info("Memory cleared — removed %d turns.", count)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def turn_count(self) -> int:
        """
        Return the total number of individual turns stored.

        A turn is a single entry (user OR model), not a pair.
        So one user message + one model reply = 2 turns.
        """
        return len(self._history)

    def pair_count(self) -> int:
        """
        Return the approximate number of user/model exchange pairs stored.

        Useful for capacity checks:  pair_count() == max_turns means the
        next add_turn() will evict the oldest pair.
        """
        return len(self._history) // 2

    def is_empty(self) -> bool:
        """Return True if no turns have been recorded yet."""
        return len(self._history) == 0

    def last_turn(self) -> Turn | None:
        """
        Return the most recent turn, or None if history is empty.

        Useful for the agent loop to inspect the last model response
        without copying the whole list.
        """
        return self._history[-1] if self._history else None

    def get_last_n_turns(self, n: int) -> list[Turn]:
        """
        Return the most recent ``n`` turns.

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_cap(self) -> None:
        """
        Evict the oldest user/model pair when the cap is exceeded.

        Gemini requires the history to begin with a "user" turn.
        Evicting one pair at a time preserves the alternating structure
        and keeps the list bounded at 2 × max_turns entries.
        A max_turns of 0 disables the cap entirely.
        """
        if self._max_turns == 0:
            return

        max_entries = self._max_turns * 2

        while len(self._history) > max_entries:
            evicted = self._history.pop(0)
            logger.debug(
                "Evicted oldest turn (cap=%d pairs) — role: %r",
                self._max_turns,
                evicted.get("role"),
            )

    @staticmethod
    def _validate_role(role: str) -> None:
        """Raise ValueError if role is not an accepted Gemini role."""
        valid_roles = {"user", "model", "function"}
        if role not in valid_roles:
            raise ValueError(
                f"Invalid role {role!r}. Must be one of: "
                + ", ".join(sorted(valid_roles))
                + "."
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of individual turns stored (same as turn_count)."""
        return len(self._history)

    def __repr__(self) -> str:
        return (
            f"<MemoryManager turns={self.turn_count()} "
            f"pairs={self.pair_count()} "
            f"max_turns={self._max_turns}>"
        )
