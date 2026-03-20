"""

Defines the BaseObserver interface — the contract every observer must satisfy.

Observer Pattern
----------------
The Observer pattern decouples the subject (Agent) from the objects that
react to its state changes (loggers, UI updaters, token counters, dashboards).

The Agent maintains a list of BaseObserver instances. At each significant
lifecycle event it calls the appropriate notification method on every
registered observer. Observers react however they like — write to a log,
update a counter, push to a web socket — without the Agent knowing or caring
which concrete observers are attached.

Benefits for this project
--------------------------
OCP (Open/Closed Principle)
    Adding a new observer (e.g. a token-usage tracker, a test spy) requires
    zero changes to Agent, ToolRegistry, or any tool. Just subclass
    BaseObserver, implement the two methods, and call agent.add_observer().

SRP (Single Responsibility Principle)
    The Agent is responsible for orchestrating the ReAct loop.
    Observers are responsible for reacting to loop events.
    Neither knows about the other's internals.

LSP (Liskov Substitution Principle)
    Any BaseObserver subclass can be registered and called through the same
    interface. The Agent never needs to know which concrete observer it is
    notifying.

DIP (Dependency Inversion Principle)
    The Agent depends on BaseObserver (the abstraction), never on
    LoggerObserver, TokenCounter, or any concrete class.

Lifecycle events
----------------
on_agent_start(tool_names)      Called once when the Agent is fully initialised.
on_turn_start(user_input)       Called at the start of each user turn.
on_tool_call(name, args, result) Called after each tool is executed.
on_turn_end(reply)              Called when a final reply is ready (alias: on_response).
on_error(error, context)        Called when a handled error occurs in the loop.
on_agent_reset()                Called when the Agent session is reset.

Minimal implementation contract
--------------------------------
Subclasses MUST implement on_tool_call() and on_response().
All other methods have safe default no-op implementations so minimal
observers only need two methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ObserverError(Exception):
    """
    Raised by an observer when it cannot complete its own task
    (e.g. file system full, network write failed).

    The Agent catches ObserverError from each notifier call and logs it
    without propagating, so a broken observer can never crash the main loop.
    """


class BaseObserver(ABC):
    """
    Abstract base class for all Agent observers.

    Subclass this and implement at minimum on_tool_call() and on_response()
    to create a new observer.  Register the instance with agent.add_observer().

    All notification methods receive only serialisable, plain-Python values
    (str, dict) — never live objects — so observers can safely serialise,
    log, or transmit event data without coupling to internal Agent types.

    Minimal example
    ---------------
    class PrintObserver(BaseObserver):
        def on_tool_call(self, name: str, args: dict, result: str) -> None:
            print(f"[tool] {name}({args}) -> {result[:60]}")

        def on_response(self, text: str) -> None:
            print(f"[reply] {text[:80]}")

    agent.add_observer(PrintObserver())
    """

    # ------------------------------------------------------------------
    # Required methods — subclasses MUST implement these two
    # ------------------------------------------------------------------

    @abstractmethod
    def on_tool_call(self, name: str, args: dict, result: str) -> None:
        """
        Called immediately after a tool finishes executing.

        Invoked for every tool call in the ReAct loop, including those that
        returned an error string (the error is the result — the observer
        receives it as-is so it can distinguish failures from successes by
        inspecting result.startswith("Error:")).

        Parameters
        ----------
        name : str
            The tool name exactly as registered in ToolRegistry.
        args : dict
            The argument payload that was passed to the tool.
            A shallow copy is provided — mutating it has no effect.
        result : str
            The tool's output string, or a structured error message if the
            tool raised ToolExecutionError / ToolArgumentError.

        Raises
        ------
        ObserverError
            If the observer encounters a non-recoverable internal error.
            The Agent catches this and logs it without re-raising.
        """

    @abstractmethod
    def on_response(self, text: str) -> None:
        """
        Called once the Agent has a final plain-text reply ready to return.

        This is the last event in a turn — it fires after all tool calls
        for the turn are complete and the model has produced its closing
        text. It is NOT called for intermediate reasoning steps.

        Parameters
        ----------
        text : str
            The final reply that will be returned to the caller of chat().

        Raises
        ------
        ObserverError
            If the observer encounters a non-recoverable internal error.
            The Agent catches this and logs it without re-raising.
        """

    # ------------------------------------------------------------------
    # Optional lifecycle hooks — default to safe no-ops
    # ------------------------------------------------------------------

    def on_agent_start(self, tool_names: list[str]) -> None:
        """
        Called once immediately after the Agent is fully initialised.

        Use this to record which tools are available, open a log file,
        write a session header, or initialise counters.

        Parameters
        ----------
        tool_names : list[str]
            Sorted list of tool names registered at startup.
        """

    def on_turn_start(self, user_input: str) -> None:
        """
        Called at the beginning of each user turn, before any Gemini API call.

        Use this to record the user message, timestamp the start of the turn,
        or reset per-turn accumulators.

        Parameters
        ----------
        user_input : str
            The raw, stripped user message passed to agent.chat().
        """

    def on_error(self, error: str, context: str) -> None:
        """
        Called whenever the Agent handles a non-fatal error.

        Covers Tier-2 tool errors (ToolNotFoundError, ToolExecutionError,
        ToolArgumentError) and Tier-3 unexpected exceptions — i.e. every
        error that was caught and converted to an observation string rather
        than propagating.  Fatal Tier-1 API errors are NOT reported here
        because they abort the turn before observers are notified.

        Parameters
        ----------
        error : str
            Short description of the error (the exception message).
        context : str
            Where the error occurred, e.g. "tool:weather" or "api_call".
        """

    def on_agent_reset(self) -> None:
        """
        Called when agent.reset() is invoked.

        Use this to flush buffers, write a session-end summary, close file
        handles, or reset per-session counters before the new session begins.
        """

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"
