"""
tools/tool_registry.py

Implements the ToolRegistry — a Factory/Registry that manages all tools
available to the agent.

Design principles applied
--------------------------
Factory Pattern
    ToolRegistry acts as the single point of instantiation and lookup for
    tools.  Callers never construct or reference concrete tool classes
    directly; they ask the registry by name.

SRP (Single Responsibility Principle)
    This class has exactly one job: map tool names to implementations and
    dispatch execution.  It knows nothing about the LLM, memory, or prompts.

OCP (Open/Closed Principle)
    Registering a new tool is one call — registry.register(MyTool()).
    No existing code changes.  The agent loop, the prompt builder, and every
    other class remain untouched.

DIP (Dependency Inversion Principle)
    ToolRegistry depends only on BaseTool (the abstraction), never on any
    concrete tool class.
"""

from __future__ import annotations

import logging
from typing import Iterator

from tools.base_tool import BaseTool, ToolExecutionError, ToolArgumentError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolNotFoundError(Exception):
    """
    Raised when the LLM requests a tool name that is not registered.

    The agent loop catches this and feeds an informative observation back to
    the LLM so it can recover instead of crashing.

    Example
    -------
    "Tool 'fly_to_moon' is not available. Available tools: calculator, weather."
    """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """
    Central registry and dispatcher for all agent tools.

    Usage
    -----
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WeatherTool())

    # Agent loop calls this after receiving a function_call from the LLM:
    result = registry.execute("calculator", {"expression": "12 * 7"})

    # PromptBuilder calls this to attach schemas to the Gemini model:
    declarations = registry.get_declarations()
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        # Internal store: tool name → BaseTool instance
        self._tools: dict[str, BaseTool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """
        Add a tool to the registry.

        Parameters
        ----------
        tool : BaseTool
            A fully instantiated concrete tool.  Its ``name`` property is
            used as the lookup key and must be unique across all registered
            tools.

        Raises
        ------
        TypeError
            If ``tool`` is not a BaseTool subclass instance.
        ValueError
            If a tool with the same name is already registered.
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"Expected a BaseTool instance, got {type(tool).__name__!r}."
            )

        if tool.name in self._tools:
            raise ValueError(
                f"A tool named {tool.name!r} is already registered. "
                "Use a unique name or call unregister() first."
            )

        self._tools[tool.name] = tool
        logger.debug("Registered tool: %r", tool.name)

    def unregister(self, name: str) -> None:
        """
        Remove a tool by name.  Silently does nothing if the tool is absent.

        Useful in tests when you need to swap out a tool mid-session.
        """
        removed = self._tools.pop(name, None)
        if removed:
            logger.debug("Unregistered tool: %r", name)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def execute(self, name: str, args: dict) -> str:
        """
        Look up a tool by name and call its execute() method.

        Parameters
        ----------
        name : str
            The tool name exactly as returned by the LLM's function_call.
        args : dict
            The argument payload from the LLM's function_call.

        Returns
        -------
        str
            The tool's string result, ready to be sent back to the LLM as a
            function_response observation.

        Raises
        ------
        ToolNotFoundError
            If ``name`` does not match any registered tool.
        ToolArgumentError
            Re-raised from the tool if required arguments are missing or
            have invalid types.  The agent loop should log this and let
            the LLM try again.
        ToolExecutionError
            Re-raised from the tool if a recoverable runtime error occurred
            (API timeout, city not found, etc.).  The agent loop feeds the
            message back to the LLM as an observation.
        """
        tool = self._tools.get(name)

        if tool is None:
            available = ", ".join(self._tools.keys()) or "none"
            raise ToolNotFoundError(
                f"Tool {name!r} is not registered. " f"Available tools: {available}."
            )

        logger.info("Executing tool %r with args: %s", name, args)

        try:
            result = tool.execute(args)
            logger.debug("Tool %r returned: %s", name, result)
            return result

        except (ToolExecutionError, ToolArgumentError):
            # Let the agent loop handle these with proper context.
            raise

        except Exception as exc:
            # Unexpected errors from a tool must not crash the agent loop.
            # Wrap and re-raise as ToolExecutionError so the loop handles it.
            logger.exception("Unexpected error in tool %r", name)
            raise ToolExecutionError(
                f"Tool {name!r} raised an unexpected error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_declarations(self) -> list[dict]:
        """
        Return a list of all tool schemas in Gemini function-calling format.

        Pass the result directly to the GenerativeModel tools parameter:

            model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                tools=registry.get_declarations(),
            )
        """
        return [tool.get_declaration() for tool in self._tools.values()]

    def get_tool(self, name: str) -> BaseTool:
        """
        Return a registered tool instance by name.

        Raises
        ------
        ToolNotFoundError
            If the name is not found.
        """
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools.keys()) or "none"
            raise ToolNotFoundError(
                f"Tool {name!r} not found. Available tools: {available}."
            )
        return tool

    def has_tool(self, name: str) -> bool:
        """Return True if a tool with this name is registered."""
        return name in self._tools

    def tool_names(self) -> list[str]:
        """Return a sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    def __iter__(self) -> Iterator[BaseTool]:
        """Iterate over all registered tool instances."""
        return iter(self._tools.values())

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __repr__(self) -> str:
        names = ", ".join(self._tools.keys()) or "empty"
        return f"<ToolRegistry [{names}]>"
