"""

Defines the BaseTool interface - the single contract every tool must satisfy.

Design principles applied
--------------------------
DIP (Dependency Inversion Principle)
    The Agent and ToolRegistry depend on this abstraction, never on concrete
    tool classes. Adding or swapping tools requires zero changes to either.

OCP (Open/Closed Principle)
    The system is open for extension (add a new tool by subclassing BaseTool)
    but closed for modification (no existing class needs to change).

LSP (Liskov Substitution Principle)
    Any BaseTool subclass can be registered and called through the same
    interface without the caller knowing or caring which concrete class it is.
"""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Subclass this and implement both abstract methods to create a new tool.
    Register the instance with ToolRegistry - the agent loop will discover
    and invoke it automatically with no changes to any other class.

    Example
    -------
    class MyTool(BaseTool):
        @property
        def name(self) -> str:
            return "my_tool"

        def execute(self, args: dict) -> str:
            return f"result: {args.get('input')}"

        def get_declaration(self) -> dict:
            return {
                "name": "my_tool",
                "description": "Does something useful.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The input value.",
                        }
                    },
                    "required": ["input"],
                },
            }
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique tool identifier.

        Must exactly match the "name" field returned by get_declaration().
        ToolRegistry uses this to route function calls from the LLM.
        """

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, args: dict) -> str:
        """
        Run the tool with the arguments provided by the LLM.

        Parameters
        ----------
        args : dict
            Key/value pairs extracted from the LLM's function_call payload.
            Keys correspond to the parameter names declared in get_declaration().

        Returns
        -------
        str
            The tool's result as a plain string.  This string is sent back to
            the LLM as a function_response so it can reason about the outcome.
            Always return a string - even on error (raise ToolExecutionError
            for errors that the agent should handle, see below).

        Raises
        ------
        ToolExecutionError
            Raised when the tool encounters a recoverable problem (bad city
            name, file not found, API timeout, etc.).  The agent loop catches
            this and feeds the message back to the LLM as an observation so
            it can try a different approach or inform the user gracefully.
        """

    @abstractmethod
    def get_declaration(self) -> dict:
        """
        Return the Gemini function-calling schema for this tool.

        The dict must conform to the OpenAPI subset that Gemini accepts:

            {
                "name": str,          # unique snake_case identifier
                "description": str,   # what the tool does (seen by the LLM)
                "parameters": {
                    "type": "object",
                    "properties": {
                        "<param>": {
                            "type": "string" | "number" | "boolean" | "array",
                            "description": str,
                            "enum": [...],   # optional, for fixed choices
                        },
                        ...
                    },
                    "required": [str, ...],   # list of mandatory param names
                },
            }

        The description field is the most important - the LLM reads it to
        decide whether and how to invoke the tool.  Be specific and concise.
        """

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r}>"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolExecutionError(Exception):
    """
    Raised by a tool when it encounters a recoverable runtime error.

    The agent loop catches ToolExecutionError, formats the message as a
    function_response observation, and lets the LLM decide how to proceed.
    Use this instead of returning error strings so the caller can distinguish
    a failed tool from a tool that legitimately returned empty output.

    Example
    -------
    if city not found:
        raise ToolExecutionError(f"City '{city}' not found. Try a different name.")
    """


class ToolArgumentError(Exception):
    """
    Raised when required arguments are missing or have invalid types/values.

    Typically thrown inside execute() after validating the args dict.
    Distinct from ToolExecutionError so callers can tell the difference
    between a bad LLM-generated call and a genuine runtime failure.

    Example
    -------
    if "expression" not in args:
        raise ToolArgumentError("Missing required argument: 'expression'.")
    """
