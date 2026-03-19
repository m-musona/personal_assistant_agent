"""

Implements the Agent -- the orchestrator that wires together the Gemini
client, MemoryManager, ToolRegistry, PromptBuilder, and the full
Reason -> Act -> Observe (ReAct) loop.

Error handling strategy
-----------------------
Errors are classified into three tiers:

  TIER 1 -- Fatal (API / network):
      The Gemini API call itself fails (timeout, auth, quota).
      Action: log at ERROR, raise GeminiAPIError, caught in chat() which
      returns a graceful user-facing message. The loop is aborted; no
      partial tool state is left dangling.

  TIER 2 -- Recoverable (tool layer):
      ToolNotFoundError  -- Gemini hallucinated a tool name.
      ToolArgumentError  -- Gemini generated malformed arguments.
      ToolExecutionError -- Tool ran but failed (city not found, etc.).
      Action: log at WARNING, inject a structured error observation, let
      Gemini re-reason. The loop continues; Gemini sees the error as a
      function_response and can apologise, retry with different args, or
      answer from memory.

  TIER 3 -- Unexpected (bugs):
      Any other exception from tool code.
      Action: log at ERROR with full traceback, inject a generic error
      observation, loop continues (same as Tier 2) so the agent stays alive.

Retry logic
-----------
Tier-1 API errors support a configurable number of retries with
exponential back-off before the GeminiAPIError is raised.

ReAct loop overview
--------------------
1. Receive user input.
2. Send to Gemini via ChatSession.
3. Inspect the response:
   a. function_call present -> dispatch to ToolRegistry (Act),
      send tool result back as function_response (Observe), go to 3.
   b. Plain text present    -> return to caller. Done.
4. Safety valve: if MAX_TOOL_CALLS_PER_TURN is reached before a text
   response, force-stop and return a graceful fallback.

Memory integration
------------------
Every step of the ReAct loop is recorded in MemoryManager so the full
conversation -- including intermediate tool calls and their results --
is preserved across turns and survives a session reset:

    User turn         -> add_turn("user", text)
    Model function_call -> add_raw_turn({"role": "model", "parts": [{"function_call": ...}]})
    Tool result       -> add_raw_turn({"role": "function", "parts": [{"function_response": ...}]})
    Model final reply -> add_turn("model", text)

On reset(), a fresh ChatSession is seeded with the full history from
MemoryManager so Gemini retains context across session restarts.


Design principles applied
--------------------------
SRP -- Agent orchestrates; it does not implement tools, memory, or prompts.
DIP -- Agent depends on abstractions (BaseTool via ToolRegistry), never on
       concrete tool classes.
OCP -- New tools, observers, or memory strategies plug in without touching
       this class.
"""

from __future__ import annotations

import logging
import time

import google.generativeai as genai
from google.generativeai import GenerativeModel
from google.generativeai.types import GenerationConfig

from agent.memory_manager import MemoryManager
from agent.prompt_builder import PromptBuilder
from config.settings import (
    GEMINI_API_KEY,
    MAX_OUTPUT_TOKENS,
    MAX_TOOL_CALLS_PER_TURN,
    MODEL_NAME,
    TEMPERATURE,
)
from tools.base_tool import ToolArgumentError, ToolExecutionError
from tools.tool_registry import ToolNotFoundError, ToolRegistry

logger = logging.getLogger(__name__)


# Retry settings for Tier-1 API failures.
_API_MAX_RETRIES: int = 3
_API_RETRY_BASE_DELAY: float = 1.0  # seconds; doubles each retry


# ---------------------------------------------------------------------------
# Agent-level exceptions
# ---------------------------------------------------------------------------


class GeminiAPIError(Exception):
    """
    Raised when the Gemini API fails after all retries are exhausted.

    Caught by chat() which converts it into a user-facing message so the
    CLI never shows a raw traceback to the user.
    """


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """
    Personal Assistant agent backed by the Google Gemini API.

    Responsibilities
    ----------------
    - Configure and hold the Gemini GenerativeModel.
    - Run the Reason -> Act -> Observe loop on every user turn.
    - Record every loop step (including tool calls) in MemoryManager.
    - Classify errors by tier and handle each appropriately (see module
      docstring for the full error-handling strategy).

    Usage
    -----
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    agent = Agent(registry)
    print(agent.chat("What is 42 * 7?"))
    print(agent.chat("Now multiply that by 2."))   # remembers context
    """

    def __init__(self, registry: ToolRegistry) -> None:
        """
        Initialise the Gemini client and all supporting components.

        Parameters
        ----------
        registry : ToolRegistry
            Fully populated registry of available tools.
        """
        if not isinstance(registry, ToolRegistry):
            raise TypeError(
                f"Expected a ToolRegistry, got {type(registry).__name__!r}."
            )

        self._registry = registry
        self._memory = MemoryManager()
        self._prompt_builder = PromptBuilder(registry)
        self._observers: list = []

        # 1. Configure Gemini SDK globally (process-wide credential store).
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini SDK configured -- model: %r", MODEL_NAME)

        # 2. Build system prompt from PromptBuilder.
        system_prompt = self._prompt_builder.build_system_prompt()
        logger.debug(
            "System prompt ready -- %d chars, %d tool(s).",
            len(system_prompt),
            len(registry),
        )

        # 3. Create GenerativeModel with schemas, prompt, and generation config.
        self._model: GenerativeModel = genai.GenerativeModel(
            model_name=MODEL_NAME,
            tools=registry.get_declarations(),
            system_instruction=system_prompt,
            generation_config=GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        logger.info(
            "GenerativeModel ready -- %d tool declaration(s) attached.",
            len(registry),
        )

        # 4. Open ChatSession with automatic function calling DISABLED.
        self._session = self._model.start_chat(
            enable_automatic_function_calling=False,
            history=[],
        )
        logger.info("ChatSession opened -- agent ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> str:
        """
        Process a user message and return the assistant's final response.

        All errors are caught here and converted to graceful user-facing
        strings -- the caller (CLI) never receives a raw exception.

        Parameters
        ----------
        user_input : str
            Raw text from the CLI or user.

        Returns
        -------
        str
            The assistant's final plain-text reply, or a graceful error
            message if the request could not be completed.
        """
        if not user_input or not user_input.strip():
            return "Please enter a message."

        logger.info("User input -- %d chars.", len(user_input))
        self._memory.add_turn("user", user_input)

        try:
            reply = self._react_loop(user_input)

        except GeminiAPIError as exc:
            # Tier 1: Gemini is unreachable or refused the request.
            logger.error("Gemini API unavailable: %s", exc)
            reply = (
                "I'm sorry, I couldn't reach the AI service right now. "
                "Please check your connection or API key and try again. "
                f"(Details: {exc})"
            )

        except Exception as exc:
            # Safety net -- any unclassified error from the loop.
            logger.exception("Unhandled error in ReAct loop.")
            reply = (
                "I encountered an unexpected error and couldn't complete "
                f"your request. Details: {exc}"
            )

        self._memory.add_turn("model", reply)
        self._notify_observers_response(reply)
        logger.info("Turn complete. %s", self._memory.summary())
        return reply

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------

    def _react_loop(self, user_input: str) -> str:
        """
        Core Reason -> Act -> Observe loop with full error handling.

        Error handling per iteration
        ----------------------------
        - Gemini API failure  -> _call_gemini_with_retry() raises
          GeminiAPIError after exhausting retries. Propagated to chat()
          which returns a graceful Tier-1 message. Loop aborted cleanly.

        - Tool error (any tier) -> _dispatch_tool() always returns a
          string (never raises). The string is injected back as a
          function_response observation; Gemini sees it and re-reasons.
          Loop continues.

        - Empty response -> logged as WARNING; safe fallback returned.

        Parameters
        ----------
        user_input : str
            Validated, non-empty user message to start the loop.

        Returns
        -------
        str
            Final plain-text answer from the model.

        Raises
        ------
        GeminiAPIError
            Propagated from _call_gemini_with_retry() on fatal API failure.
        """
        message = user_input
        tool_calls_made = 0

        while True:
            # ---- REASON ------------------------------------------------
            logger.debug("Sending to Gemini (tool_calls_so_far=%d).", tool_calls_made)
            # Raises GeminiAPIError on fatal failure -- propagates to chat().
            response = self._call_gemini_with_retry(message)

            candidate = response.candidates[0]
            text_parts: list[str] = []
            function_call_part = None

            for part in candidate.content.parts:
                if (
                    hasattr(part, "function_call")
                    and part.function_call
                    and part.function_call.name
                ):
                    function_call_part = part.function_call
                    break
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

            # ---- ACT ---------------------------------------------------
            if function_call_part is not None:
                tool_calls_made += 1

                if tool_calls_made > MAX_TOOL_CALLS_PER_TURN:
                    logger.warning(
                        "Tool call cap (%d) exceeded -- forcing stop.",
                        MAX_TOOL_CALLS_PER_TURN,
                    )
                    partial = " ".join(text_parts).strip()
                    if partial:
                        return (
                            "I reached the maximum number of tool calls for "
                            "this request. Here is what I found so far: " + partial
                        )
                    return (
                        "I was unable to complete the request within the "
                        "allowed number of tool calls. "
                        "Please try a simpler query."
                    )

                tool_name = function_call_part.name
                tool_args = dict(function_call_part.args)

                logger.info(
                    "Tool call %d/%d -- name=%r args=%s",
                    tool_calls_made,
                    MAX_TOOL_CALLS_PER_TURN,
                    tool_name,
                    tool_args,
                )

                # Record model's intent in memory.
                self._memory.add_raw_turn(
                    {
                        "role": "model",
                        "parts": [
                            {"function_call": {"name": tool_name, "args": tool_args}}
                        ],
                    }
                )

                # ---- OBSERVE -------------------------------------------
                # _dispatch_tool() never raises -- errors become strings.
                tool_result = self._dispatch_tool(tool_name, tool_args)
                self._notify_observers_tool(tool_name, tool_args, tool_result)

                function_response = self._build_function_response(
                    tool_name, tool_result
                )
                self._memory.add_raw_turn(function_response)

                logger.debug("Tool exchange recorded. %s", self._memory.summary())

                # Feed the observation (success or error string) back to
                # Gemini so it can re-reason on the next iteration.
                message = function_response
                continue

            # ---- Final text response ------------------------------------
            final_text = " ".join(text_parts).strip()

            if not final_text:
                logger.warning("Gemini returned an empty response.")
                return "I wasn't able to generate a response. Please try again."

            return final_text

    # ------------------------------------------------------------------
    # Tier-1: Gemini API call with retry
    # ------------------------------------------------------------------

    def _call_gemini_with_retry(self, message) -> object:
        """
        Send a message to the Gemini ChatSession with exponential back-off.

        Retries up to _API_MAX_RETRIES times on any exception from the SDK.
        After all retries are exhausted, raises GeminiAPIError so the caller
        (chat()) can return a graceful user-facing Tier-1 message.

        Parameters
        ----------
        message : str | dict
            Either the plain user-input string (first iteration) or a
            function_response dict (subsequent iterations).

        Returns
        -------
        google.generativeai.types.GenerateContentResponse
            The raw Gemini response object.

        Raises
        ------
        GeminiAPIError
            When all retry attempts have failed.
        """
        last_exc: Exception | None = None

        for attempt in range(1, _API_MAX_RETRIES + 1):
            try:
                return self._session.send_message(message)

            except Exception as exc:
                last_exc = exc
                if attempt < _API_MAX_RETRIES:
                    delay = _API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Gemini API error (attempt %d/%d) -- retrying in %.1fs. "
                        "Error: %s",
                        attempt,
                        _API_MAX_RETRIES,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Gemini API failed after %d attempts: %s",
                        _API_MAX_RETRIES,
                        exc,
                    )

        raise GeminiAPIError(
            f"Gemini API unavailable after {_API_MAX_RETRIES} attempts. "
            f"Last error: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Tier-2/3: Tool dispatch -- always returns a string, never raises
    # ------------------------------------------------------------------

    def _dispatch_tool(self, name: str, args: dict) -> str:
        """
        Execute a tool via ToolRegistry, classifying and absorbing all errors.

        This method NEVER raises. Every error path returns a descriptive
        string that becomes the function_response observation fed back to
        Gemini, allowing it to re-reason rather than crashing the loop.

        Error classification
        --------------------
        ToolNotFoundError  (Tier 2) -- Gemini hallucinated a tool name.
            Observation includes the list of valid tool names so Gemini
            can immediately correct itself on the next reasoning step.

        ToolArgumentError  (Tier 2) -- Gemini generated malformed args.
            Observation explains which argument was wrong. Gemini can
            retry the call with corrected arguments.

        ToolExecutionError (Tier 2) -- Tool ran but hit a runtime failure.
            Observation relays the tool's own error message. Gemini can
            inform the user or try an alternative.

        Exception          (Tier 3) -- Unexpected bug in tool code.
            Full traceback logged at ERROR; generic observation returned.
            Loop stays alive.

        Parameters
        ----------
        name : str
            Tool name exactly as returned by Gemini's function_call.
        args : dict
            Argument payload from Gemini's function_call.

        Returns
        -------
        str
            Tool output on success, or a structured error observation on
            failure. The string is always safe to inject back into Gemini.
        """
        try:
            result = self._registry.execute(name, args)
            logger.debug("Tool %r succeeded: %s", name, result)
            return result

        except ToolNotFoundError:
            # Tier 2: Gemini hallucinated a tool that doesn't exist.
            available = ", ".join(self._registry.tool_names()) or "none"
            logger.warning(
                "ToolNotFoundError -- name=%r available=[%s]", name, available
            )
            return (
                f"Error: no tool named '{name}' exists. "
                f"Available tools are: {available}. "
                "Please use one of the available tool names exactly."
            )

        except ToolArgumentError as exc:
            # Tier 2: Gemini's generated args don't match the schema.
            logger.warning(
                "ToolArgumentError -- tool=%r args=%s error=%s",
                name,
                args,
                exc,
            )
            declaration = self._registry.get_tool(name).get_declaration()
            required = declaration.get("parameters", {}).get("required", [])
            return (
                f"Error: invalid arguments for tool '{name}'. "
                f"Details: {exc}. "
                f"Required parameters are: {required}. "
                "Please retry with correct arguments."
            )

        except ToolExecutionError as exc:
            # Tier 2: Tool ran but encountered a recoverable runtime failure.
            logger.warning("ToolExecutionError -- tool=%r error=%s", name, exc)
            return (
                f"Error: the tool '{name}' encountered a problem. "
                f"Details: {exc}. "
                "You may want to try different inputs or let the user know."
            )

        except Exception as exc:
            # Tier 3: Unexpected bug in tool code -- log full traceback.
            logger.exception("Unexpected error in tool %r with args %s", name, args)
            return (
                f"Error: an unexpected error occurred while running '{name}'. "
                f"Details: {exc}. "
                "Please inform the user that this tool is temporarily unavailable."
            )

    # ------------------------------------------------------------------
    # Gemini message helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_function_response(tool_name: str, tool_result: str) -> dict:
        """
        Build the function_response payload Gemini requires after a tool call.

        The same dict is sent to the ChatSession AND stored in MemoryManager,
        so there is exactly one source of truth for the tool result structure.

        Parameters
        ----------
        tool_name : str
            Must match the function_call name exactly.
        tool_result : str
            Tool output or structured error string from _dispatch_tool().

        Returns
        -------
        dict
            Gemini-compatible Content dict with role "function".
        """
        return {
            "role": "function",
            "parts": [
                {
                    "function_response": {
                        "name": tool_name,
                        "response": {"result": tool_result},
                    }
                }
            ],
        }

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear conversation history and open a fresh ChatSession."""
        self._memory.clear()
        self._session = self._model.start_chat(
            enable_automatic_function_calling=False,
            history=[],
        )
        logger.info("Session reset -- history cleared, new ChatSession opened.")

    def restore_session(self) -> None:
        """Reseed a fresh ChatSession from current MemoryManager state."""
        history = self._memory.get_history()
        self._session = self._model.start_chat(
            enable_automatic_function_calling=False,
            history=history,
        )
        logger.info("Session restored from memory -- %d turns loaded.", len(history))

    # ------------------------------------------------------------------
    # Observer support (Phase 6 hooks -- safe no-ops until wired)
    # ------------------------------------------------------------------

    def add_observer(self, observer: object) -> None:
        """Register an observer for on_tool_call / on_response events."""
        self._observers.append(observer)
        logger.debug("Observer registered: %r", observer)

    def _notify_observers_tool(self, name: str, args: dict, result: str) -> None:
        for observer in self._observers:
            try:
                observer.on_tool_call(name, args, result)
            except Exception:
                logger.exception("Observer error in on_tool_call.")

    def _notify_observers_response(self, text: str) -> None:
        for observer in self._observers:
            try:
                observer.on_response(text)
            except Exception:
                logger.exception("Observer error in on_response.")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def memory(self) -> MemoryManager:
        """Expose MemoryManager for inspection (read-only intent)."""
        return self._memory

    @property
    def registry(self) -> ToolRegistry:
        """Expose ToolRegistry for inspection (read-only intent)."""
        return self._registry

    def __repr__(self) -> str:
        return (
            f"<Agent model={MODEL_NAME!r} "
            f"tools={len(self._registry)} "
            f"turns={self._memory.turn_count()}>"
        )
