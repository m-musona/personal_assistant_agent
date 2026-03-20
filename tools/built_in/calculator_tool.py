"""

A safe arithmetic calculator tool for the agent.

Security model
--------------
Raw eval() on arbitrary user/LLM input is a code-execution vulnerability.
This tool uses a two-layer approach instead:

  Layer 1 — ast.parse() with a strict AST whitelist.
    The expression is compiled to an AST. Every node type is checked against
    _ALLOWED_NODE_TYPES. Any node not on the whitelist (function calls,
    attribute access, imports, comprehensions, etc.) raises ToolArgumentError
    before a single byte of the expression is evaluated.

  Layer 2 — eval() inside a locked-down globals/locals dict.
    Even after the AST is verified, eval() runs with:
      globals={"__builtins__": {}}   — no builtins at all
      locals=_SAFE_NAMES             — only math constants and functions

    This means even if a malicious expression somehow survived the AST
    check, it would hit a NameError rather than executing arbitrary code.

Supported operations
--------------------
  Arithmetic     : + - * / // % **
  Comparisons    : == != < > <= >=
  Bit ops        : & | ^ ~ << >>
  Unary          : - + ~
  Math functions : sqrt, abs, ceil, floor, round, log, log2, log10,
                   sin, cos, tan, asin, acos, atan, atan2, degrees, radians,
                   factorial, gcd, pow, exp, hypot, trunc, inf, nan
  Constants      : pi, e, tau
  Literals       : integers, floats, complex numbers
"""

from __future__ import annotations

import ast
import math
import logging
from typing import Any

from tools.base_tool import BaseTool, ToolArgumentError, ToolExecutionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AST node whitelist
# Only these node types may appear in a valid expression.
# ---------------------------------------------------------------------------
_ALLOWED_NODE_TYPES: frozenset[type] = frozenset(
    {
        # Expression wrappers
        ast.Expression,
        ast.Expr,
        # Literals
        ast.Constant,
        # Collections (for atan2(y, x) tuple-like calls — excluded; listed for clarity)
        # Operators
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        # Operator tokens
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.Invert,
        ast.LShift,
        ast.RShift,
        ast.UAdd,
        ast.USub,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.And,
        ast.Or,
        ast.Not,
        # Names (resolved against _SAFE_NAMES only)
        ast.Name,
        ast.Load,
        # Function calls (whitelisted names enforced separately in _validate_ast)
        ast.Call,
        # Needed for multi-arg calls like atan2(y, x)
        ast.Tuple,
    }
)

# ---------------------------------------------------------------------------
# Safe names available inside the expression evaluator
# ---------------------------------------------------------------------------
_SAFE_NAMES: dict[str, Any] = {
    # Constants
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
    # One-argument math functions
    "sqrt": math.sqrt,
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "pow": math.pow,
    "exp": math.exp,
    "hypot": math.hypot,
    "trunc": math.trunc,
}

# Names allowed as callable functions (subset of _SAFE_NAMES).
_ALLOWED_CALL_NAMES: frozenset[str] = frozenset(
    {k for k, v in _SAFE_NAMES.items() if callable(v)}
)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class CalculatorTool(BaseTool):
    """
    Evaluates arithmetic and mathematical expressions safely.

    The LLM passes a plain expression string such as "2 ** 10" or
    "sqrt(2) * pi". The tool validates the AST, evaluates in a sandboxed
    namespace, and returns a clean numeric result string.

    Raises ToolArgumentError for syntactically invalid or unsafe expressions.
    Raises ToolExecutionError for mathematically invalid operations
    (division by zero, domain errors, overflow, etc.).
    """

    @property
    def name(self) -> str:
        """Return the unique tool identifier used by ToolRegistry."""
        return "calculator"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def execute(self, args: dict) -> str:
        """
        Evaluate the expression in args["expression"] and return the result.

        Parameters
        ----------
        args : dict
            Must contain "expression" (str): the math expression to evaluate.

        Returns
        -------
        str
            Human-readable result, e.g. "2 ** 10 = 1024" or
            "sqrt(2) = 1.4142135623730951".

        Raises
        ------
        ToolArgumentError
            If "expression" is missing, empty, contains disallowed syntax,
            or references names outside the safe whitelist.
        ToolExecutionError
            If the expression is mathematically invalid (ZeroDivisionError,
            ValueError from math domain, OverflowError, etc.).
        """
        expression = self._extract_expression(args)
        tree = self._parse_expression(expression)
        self._validate_ast(tree)
        result = self._evaluate(expression, expression)
        return self._format_result(expression, result)

    def get_declaration(self) -> dict:
        """Return the Gemini function-calling schema for this tool."""
        return {
            "name": "calculator",
            "description": (
                "Evaluates a mathematical or arithmetic expression and returns "
                "the numeric result. Supports standard operators (+, -, *, /, "
                "//, %, **), comparison operators, bitwise operators, and common "
                "math functions (sqrt, sin, cos, log, factorial, etc.) and "
                "constants (pi, e, tau). "
                "Use this tool for any numerical calculation rather than "
                "computing mentally."
            ),
            "parameters": {
                # "type": "object",
                "type": "OBJECT",
                "properties": {
                    "expression": {
                        # "type": "string",
                        "type": "STRING",
                        "description": (
                            "A valid Python arithmetic expression to evaluate. "
                            "Examples: '2 + 2', '(100 / 4) * 3', '2 ** 10', "
                            "'sqrt(144)', 'factorial(10)', 'sin(pi / 2)', "
                            "'log(1000, 10)'. "
                            "Do not include assignment operators (=), "
                            "print(), or any non-math code."
                        ),
                    }
                },
                "required": ["expression"],
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_expression(args: dict) -> str:
        """Pull and validate the 'expression' argument from the args dict."""
        expression = args.get("expression")
        if expression is None:
            raise ToolArgumentError(
                "Missing required argument: 'expression'. "
                "Provide a math expression string, e.g. '2 + 2'."
            )
        if not isinstance(expression, str):
            raise ToolArgumentError(
                f"'expression' must be a string, got {type(expression).__name__!r}."
            )
        expression = expression.strip()
        if not expression:
            raise ToolArgumentError("'expression' must not be empty.")
        return expression

    @staticmethod
    def _parse_expression(expression: str) -> ast.AST:
        """
        Parse the expression string into an AST.

        Raises ToolArgumentError on any syntax error so the agent loop
        can feed a descriptive observation back to the LLM.
        """
        try:
            return ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolArgumentError(
                f"Invalid expression syntax: {exc.msg!r} in {expression!r}. "
                "Ensure the expression is valid Python arithmetic."
            ) from exc

    @staticmethod
    def _validate_ast(tree: ast.AST) -> None:
        """
        Walk the AST and reject any node type not in _ALLOWED_NODE_TYPES,
        and any Name/Call that references something outside _SAFE_NAMES.

        This is the primary security gate. Anything that could call
        arbitrary Python (builtins, attribute access, imports, lambdas,
        comprehensions, subscripts) is blocked here.
        """
        for node in ast.walk(tree):
            node_type = type(node)

            if node_type not in _ALLOWED_NODE_TYPES:
                raise ToolArgumentError(
                    f"Disallowed expression construct: {node_type.__name__!r}. "
                    "Only arithmetic operators and whitelisted math functions "
                    "are permitted. "
                    f"Expression contained: {ast.dump(node)!r}."
                )

            # Extra check: Name nodes must reference a whitelisted identifier.
            if isinstance(node, ast.Name):
                if node.id not in _SAFE_NAMES:
                    raise ToolArgumentError(
                        f"Unknown name {node.id!r} in expression. "
                        f"Allowed names are: {', '.join(sorted(_SAFE_NAMES))}."
                    )

            # Extra check: Call nodes must call a whitelisted function.
            if isinstance(node, ast.Call):
                func = node.func
                if not isinstance(func, ast.Name):
                    raise ToolArgumentError(
                        "Only direct function calls (e.g. sqrt(x)) are allowed. "
                        "Method calls and attribute access are not permitted."
                    )
                if func.id not in _ALLOWED_CALL_NAMES:
                    raise ToolArgumentError(
                        f"Function {func.id!r} is not allowed. "
                        f"Allowed functions: {', '.join(sorted(_ALLOWED_CALL_NAMES))}."
                    )

    @staticmethod
    def _evaluate(expression: str, original: str) -> Any:
        """
        Evaluate the pre-validated expression in a locked-down namespace.

        The empty builtins dict ensures no built-in function is accessible
        even if one somehow slipped through the AST validation.

        Raises ToolExecutionError for math-domain errors.
        """
        try:
            result = eval(  # noqa: S307 — intentional; AST validated above
                expression,
                {"__builtins__": {}},
                _SAFE_NAMES,
            )
            return result
        except ZeroDivisionError as exc:
            raise ToolExecutionError(
                f"Division by zero in expression {original!r}."
            ) from exc
        except ValueError as exc:
            raise ToolExecutionError(
                f"Math domain error in expression {original!r}: {exc}."
            ) from exc
        except OverflowError as exc:
            raise ToolExecutionError(
                f"Arithmetic overflow in expression {original!r}: {exc}."
            ) from exc
        except Exception as exc:
            raise ToolExecutionError(
                f"Could not evaluate {original!r}: {exc}."
            ) from exc

    @staticmethod
    def _format_result(expression: str, result: Any) -> str:
        """
        Format the numeric result into a clean, human-readable string.

        - Integers are shown without a decimal point.
        - Floats are shown with up to 10 significant digits, trailing zeros
          stripped, to keep the output concise.
        - Complex numbers use the standard Python repr.
        - Boolean results (from comparison expressions) shown as True/False.
        """
        if isinstance(result, bool):
            return f"{expression} = {result}"

        if isinstance(result, int):
            return f"{expression} = {result:,}"

        if isinstance(result, float):
            if math.isnan(result):
                return f"{expression} = NaN"
            if math.isinf(result):
                sign = "" if result > 0 else "-"
                return f"{expression} = {sign}Infinity"
            # Up to 10 significant figures, strip trailing zeros.
            formatted = f"{result:.10g}"
            return f"{expression} = {formatted}"

        if isinstance(result, complex):
            return f"{expression} = {result}"

        # Fallback for any other numeric type.
        return f"{expression} = {result}"
