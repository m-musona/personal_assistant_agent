"""

Unit tests for CalculatorTool covering:
  - Valid arithmetic and math-function expressions
  - Edge cases: zero, negatives, large numbers, floats, complex
  - Security: blocked builtins, attribute access, imports, lambdas
  - Error paths: ToolArgumentError (bad args / unsafe), ToolExecutionError (math)
"""

from __future__ import annotations

import math
import unittest

from tools.base_tool import ToolArgumentError, ToolExecutionError
from tools.built_in.calculator_tool import CalculatorTool


class TestCalculatorToolArithmetic(unittest.TestCase):
    """Basic arithmetic operations."""

    def setUp(self) -> None:
        self.calc = CalculatorTool()

    def _result(self, expr: str) -> str:
        return self.calc.execute({"expression": expr})

    def test_addition(self):
        self.assertIn("= 5", self._result("2 + 3"))

    def test_subtraction(self):
        self.assertIn("= 1", self._result("4 - 3"))

    def test_multiplication(self):
        self.assertIn("= 294", self._result("42 * 7"))

    def test_integer_division(self):
        self.assertIn("= 3", self._result("10 // 3"))

    def test_float_division(self):
        out = self._result("10 / 4")
        self.assertIn("2.5", out)

    def test_modulo(self):
        self.assertIn("= 1", self._result("10 % 3"))

    def test_exponentiation(self):
        self.assertIn("= 1,024", self._result("2 ** 10"))

    def test_negative_result(self):
        self.assertIn("= -3", self._result("5 - 8"))

    def test_unary_minus(self):
        self.assertIn("= -7", self._result("-7"))

    def test_chained_operations(self):
        self.assertIn("= 14", self._result("2 + 3 * 4"))

    def test_parentheses_respected(self):
        self.assertIn("= 20", self._result("(2 + 3) * 4"))

    def test_float_arithmetic(self):
        out = self._result("0.1 + 0.2")
        # Result should be approximately 0.3
        self.assertIn("0.3", out)

    def test_large_integer(self):
        out = self._result("2 ** 32")
        self.assertIn("4,294,967,296", out)


class TestCalculatorToolMathFunctions(unittest.TestCase):
    """Whitelisted math functions and constants."""

    def setUp(self) -> None:
        self.calc = CalculatorTool()

    def _result(self, expr: str) -> str:
        return self.calc.execute({"expression": expr})

    def test_sqrt(self):
        self.assertIn("= 12", self._result("sqrt(144)"))

    def test_abs_negative(self):
        self.assertIn("= 5", self._result("abs(-5)"))

    def test_pi_constant(self):
        out = self._result("pi")
        self.assertIn("3.14159", out)

    def test_e_constant(self):
        out = self._result("e")
        self.assertIn("2.71828", out)

    def test_sin_of_pi_over_2(self):
        out = self._result("sin(pi / 2)")
        self.assertIn("= 1", out)

    def test_cos_of_zero(self):
        out = self._result("cos(0)")
        self.assertIn("= 1", out)

    def test_log_base_10(self):
        out = self._result("log10(1000)")
        self.assertIn("= 3", out)

    def test_factorial(self):
        self.assertIn("= 120", self._result("factorial(5)"))

    def test_ceil(self):
        self.assertIn("= 3", self._result("ceil(2.1)"))

    def test_floor(self):
        self.assertIn("= 2", self._result("floor(2.9)"))

    def test_round(self):
        self.assertIn("= 3", self._result("round(2.6)"))

    def test_exp(self):
        out = self._result("exp(1)")
        self.assertIn("2.71828", out)

    def test_tau(self):
        out = self._result("tau")
        self.assertIn("6.28318", out)

    def test_pow_function(self):
        self.assertIn("= 8", self._result("pow(2, 3)"))

    def test_gcd(self):
        self.assertIn("= 6", self._result("gcd(12, 18)"))


class TestCalculatorToolEdgeCases(unittest.TestCase):
    """Boundary and edge-case values."""

    def setUp(self) -> None:
        self.calc = CalculatorTool()

    def _result(self, expr: str) -> str:
        return self.calc.execute({"expression": expr})

    def test_zero(self):
        self.assertIn("= 0", self._result("0"))

    def test_negative_zero_float(self):
        out = self._result("-0.0")
        self.assertIn("0", out)

    def test_boolean_comparison_true(self):
        self.assertIn("True", self._result("3 > 2"))

    def test_boolean_comparison_false(self):
        self.assertIn("False", self._result("1 == 2"))

    def test_infinity_result(self):
        out = self._result("inf")
        self.assertIn("Infinity", out)

    def test_nan_result(self):
        out = self._result("nan")
        self.assertIn("NaN", out)

    def test_very_small_float(self):
        out = self._result("1e-10")
        self.assertIn("1e-10", out)

    def test_bitwise_and(self):
        self.assertIn("= 4", self._result("12 & 6"))

    def test_bitwise_or(self):
        self.assertIn("= 14", self._result("12 | 6"))

    def test_left_shift(self):
        self.assertIn("= 8", self._result("1 << 3"))


class TestCalculatorToolArgumentErrors(unittest.TestCase):
    """ToolArgumentError paths: bad input, missing args, unsafe expressions."""

    def setUp(self) -> None:
        self.calc = CalculatorTool()

    def test_missing_expression_key(self):
        with self.assertRaises(ToolArgumentError) as ctx:
            self.calc.execute({})
        self.assertIn("expression", str(ctx.exception).lower())

    def test_none_expression(self):
        with self.assertRaises(ToolArgumentError):
            self.calc.execute({"expression": None})

    def test_empty_string_expression(self):
        with self.assertRaises(ToolArgumentError):
            self.calc.execute({"expression": "   "})

    def test_wrong_type_expression(self):
        with self.assertRaises(ToolArgumentError):
            self.calc.execute({"expression": 42})

    def test_syntax_error(self):
        with self.assertRaises(ToolArgumentError) as ctx:
            self.calc.execute({"expression": "2 +"})
        self.assertIn("syntax", str(ctx.exception).lower())

    def test_blocked_builtin_open(self):
        with self.assertRaises(ToolArgumentError):
            self.calc.execute({"expression": "open('/etc/passwd')"})

    def test_blocked_builtin_exec(self):
        with self.assertRaises(ToolArgumentError):
            self.calc.execute({"expression": "exec('import os')"})

    def test_blocked_import(self):
        with self.assertRaises(ToolArgumentError):
            self.calc.execute({"expression": "__import__('os')"})

    def test_blocked_attribute_access(self):
        with self.assertRaises(ToolArgumentError):
            self.calc.execute({"expression": "(1).__class__"})

    def test_blocked_lambda(self):
        with self.assertRaises(ToolArgumentError):
            self.calc.execute({"expression": "(lambda: 42)()"})

    def test_blocked_list_comprehension(self):
        with self.assertRaises(ToolArgumentError):
            self.calc.execute({"expression": "[x for x in range(10)]"})

    def test_blocked_unknown_name(self):
        # os.getcwd() is caught by the attribute-access check first.
        # Use a bare unknown name to test the Name-whitelist path.
        with self.assertRaises(ToolArgumentError) as ctx:
            self.calc.execute({"expression": "secret_var + 1"})
        self.assertIn("secret_var", str(ctx.exception))

    def test_blocked_assignment(self):
        with self.assertRaises(ToolArgumentError):
            self.calc.execute({"expression": "x = 5"})

    def test_blocked_dunder_builtins(self):
        with self.assertRaises(ToolArgumentError):
            self.calc.execute({"expression": "__builtins__"})


class TestCalculatorToolExecutionErrors(unittest.TestCase):
    """ToolExecutionError paths: mathematically invalid operations."""

    def setUp(self) -> None:
        self.calc = CalculatorTool()

    def test_division_by_zero_integer(self):
        with self.assertRaises(ToolExecutionError) as ctx:
            self.calc.execute({"expression": "1 / 0"})
        self.assertIn("zero", str(ctx.exception).lower())

    def test_division_by_zero_floor(self):
        with self.assertRaises(ToolExecutionError):
            self.calc.execute({"expression": "5 // 0"})

    def test_sqrt_negative_number(self):
        with self.assertRaises(ToolExecutionError) as ctx:
            self.calc.execute({"expression": "sqrt(-1)"})
        self.assertIn("domain", str(ctx.exception).lower())

    def test_log_of_zero(self):
        with self.assertRaises(ToolExecutionError):
            self.calc.execute({"expression": "log(0)"})

    def test_log_of_negative(self):
        with self.assertRaises(ToolExecutionError):
            self.calc.execute({"expression": "log(-5)"})

    def test_asin_out_of_domain(self):
        with self.assertRaises(ToolExecutionError):
            self.calc.execute({"expression": "asin(2)"})

    def test_factorial_negative(self):
        with self.assertRaises(ToolExecutionError):
            self.calc.execute({"expression": "factorial(-1)"})


class TestCalculatorToolDeclaration(unittest.TestCase):
    """Schema structure validation."""

    def setUp(self) -> None:
        self.calc = CalculatorTool()

    def test_name_matches_property(self):
        self.assertEqual(self.calc.get_declaration()["name"], self.calc.name)

    def test_has_description(self):
        d = self.calc.get_declaration()
        self.assertIn("description", d)
        self.assertGreater(len(d["description"]), 10)

    def test_expression_parameter_present(self):
        props = self.calc.get_declaration()["parameters"]["properties"]
        self.assertIn("expression", props)

    def test_expression_is_required(self):
        required = self.calc.get_declaration()["parameters"]["required"]
        self.assertIn("expression", required)

    def test_expression_type_is_string(self):
        props = self.calc.get_declaration()["parameters"]["properties"]
        self.assertIn(props["expression"]["type"].lower(), ("string",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
