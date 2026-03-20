"""

Unified tool test suite — the single file requested by the assignment.

Covers all six tools with:
  - At least one valid-input test that proves execute() returns a str
  - The canonical failure scenario for each tool
    (bad city name, invalid filepath, unknown language, etc.)
  - Argument-validation tests (missing / empty / wrong-type arguments)
  - A declaration-schema check (name, description, required params)

All external HTTP calls are replaced with unittest.mock so the tests
run offline with no API key and complete in milliseconds.

Test classes
------------
TestCalculatorTool   — arithmetic, math functions, invalid expressions,
                       blocked builtins, division by zero
TestWeatherTool      — valid city, city not found (HTTP 404 + empty body),
                       network error, missing argument
TestSearchTool       — Wikipedia hit, DuckDuckGo fallback, no results,
                       network error, missing argument
TestTimeTool         — local time, IANA name, abbreviation, city name,
                       unknown timezone, wrong-type argument
TestTranslateTool    — valid translation, MyMemory fallback to LibreTranslate,
                       unknown language, missing text / target arguments
TestFileReaderTool   — valid file read, file not found, path traversal blocked,
                       disallowed extension, missing argument

Running
-------
    pytest tests/test_tools.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.error
from unittest.mock import MagicMock, PropertyMock, patch

from tools.base_tool import ToolArgumentError, ToolExecutionError

# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP mock helpers
# ─────────────────────────────────────────────────────────────────────────────


def _mock_http(body: str, status: int = 200):
    """Return a context-manager mock that mimics urllib.request.urlopen."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read.return_value = body.encode("utf-8")
    cm.status = status
    return cm


def _http_error(code: int) -> urllib.error.HTTPError:
    """Return an HTTPError with the given status code."""
    return urllib.error.HTTPError(
        url="http://x", code=code, msg="err", hdrs=None, fp=None
    )


def _url_error() -> urllib.error.URLError:
    """Return a URLError simulating a network failure."""
    return urllib.error.URLError("connection refused")


# ─────────────────────────────────────────────────────────────────────────────
# CalculatorTool
# ─────────────────────────────────────────────────────────────────────────────


class TestCalculatorTool(unittest.TestCase):
    """Tests for the safe arithmetic calculator tool."""

    def setUp(self) -> None:
        from tools.built_in.calculator_tool import CalculatorTool

        self.tool = CalculatorTool()

    # -- Valid inputs --------------------------------------------------------

    def test_returns_string(self) -> None:
        """execute() must always return a str."""
        result = self.tool.execute({"expression": "1 + 1"})
        self.assertIsInstance(result, str)

    def test_basic_addition(self) -> None:
        """Simple addition produces the correct numeric result."""
        self.assertIn("2", self.tool.execute({"expression": "1 + 1"}))

    def test_multiplication(self) -> None:
        """Multiplication result is included in the output."""
        self.assertIn("294", self.tool.execute({"expression": "42 * 7"}))

    def test_exponentiation(self) -> None:
        """Power operator works correctly."""
        self.assertIn("1,024", self.tool.execute({"expression": "2 ** 10"}))

    def test_sqrt_function(self) -> None:
        """Whitelisted math function sqrt() is supported."""
        self.assertIn("12", self.tool.execute({"expression": "sqrt(144)"}))

    def test_pi_constant(self) -> None:
        """The pi constant is accessible."""
        self.assertIn("3.14159", self.tool.execute({"expression": "pi"}))

    def test_factorial(self) -> None:
        """factorial() produces the correct result."""
        self.assertIn("120", self.tool.execute({"expression": "factorial(5)"}))

    def test_float_division(self) -> None:
        """Float division returns a decimal result."""
        self.assertIn("2.5", self.tool.execute({"expression": "10 / 4"}))

    def test_comparison_returns_boolean(self) -> None:
        """Comparison expressions return True or False."""
        result = self.tool.execute({"expression": "3 > 2"})
        self.assertIn("True", result)

    # -- Invalid inputs (ToolArgumentError) ---------------------------------

    def test_missing_expression_raises_argument_error(self) -> None:
        """Missing 'expression' key raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({})

    def test_empty_expression_raises_argument_error(self) -> None:
        """Empty string expression raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"expression": "   "})

    def test_wrong_type_raises_argument_error(self) -> None:
        """Non-string expression raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"expression": 42})

    def test_syntax_error_raises_argument_error(self) -> None:
        """Syntactically invalid expression raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"expression": "2 +"})

    # -- Security (ToolArgumentError) ----------------------------------------

    def test_blocked_builtin_open(self) -> None:
        """Calling open() in an expression is blocked by the AST whitelist."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"expression": "open('/etc/passwd')"})

    def test_blocked_import(self) -> None:
        """__import__ is blocked by the AST whitelist."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"expression": "__import__('os')"})

    def test_blocked_attribute_access(self) -> None:
        """Attribute access is blocked by the AST whitelist."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"expression": "(1).__class__"})

    def test_blocked_lambda(self) -> None:
        """Lambda expressions are blocked by the AST whitelist."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"expression": "(lambda: 42)()"})

    # -- Runtime errors (ToolExecutionError) ---------------------------------

    def test_division_by_zero_raises_execution_error(self) -> None:
        """Division by zero raises ToolExecutionError."""
        with self.assertRaises(ToolExecutionError):
            self.tool.execute({"expression": "1 / 0"})

    def test_sqrt_negative_raises_execution_error(self) -> None:
        """sqrt of a negative number raises ToolExecutionError."""
        with self.assertRaises(ToolExecutionError):
            self.tool.execute({"expression": "sqrt(-1)"})

    def test_log_zero_raises_execution_error(self) -> None:
        """log(0) raises ToolExecutionError."""
        with self.assertRaises(ToolExecutionError):
            self.tool.execute({"expression": "log(0)"})

    # -- Declaration schema --------------------------------------------------

    def test_declaration_name(self) -> None:
        """Declaration name matches the tool name property."""
        self.assertEqual(self.tool.get_declaration()["name"], self.tool.name)

    def test_declaration_expression_required(self) -> None:
        """'expression' is listed as a required parameter."""
        self.assertIn(
            "expression", self.tool.get_declaration()["parameters"]["required"]
        )


# ─────────────────────────────────────────────────────────────────────────────
# WeatherTool
# ─────────────────────────────────────────────────────────────────────────────

_WTTR_VALID = json.dumps(
    {
        "current_condition": [
            {
                "temp_C": "18",
                "FeelsLikeC": "16",
                "weatherDesc": [{"value": "Partly cloudy"}],
                "humidity": "72",
                "windspeedKmph": "14",
            }
        ],
        "nearest_area": [
            {
                "areaName": [{"value": "London"}],
                "country": [{"value": "United Kingdom"}],
            }
        ],
    }
)

_WTTR_EMPTY = json.dumps(
    {
        "current_condition": [],
        "nearest_area": [],
    }
)


class TestWeatherTool(unittest.TestCase):
    """Tests for WeatherTool with all HTTP calls mocked."""

    def setUp(self) -> None:
        from tools.built_in.weather_tool import WeatherTool

        self.tool = WeatherTool()
        # Force wttr.in backend by clearing any OWM key in the module.
        patcher = patch("tools.built_in.weather_tool.OPENWEATHER_API_KEY", "")
        self.addCleanup(patcher.stop)
        patcher.start()

    # -- Valid city ----------------------------------------------------------

    def test_returns_string(self) -> None:
        """execute() must return a str for a valid city."""
        with patch("urllib.request.urlopen", return_value=_mock_http(_WTTR_VALID)):
            result = self.tool.execute({"city": "London"})
        self.assertIsInstance(result, str)

    def test_contains_temperature(self) -> None:
        """Output contains the temperature value."""
        with patch("urllib.request.urlopen", return_value=_mock_http(_WTTR_VALID)):
            result = self.tool.execute({"city": "London"})
        self.assertIn("18", result)

    def test_contains_condition(self) -> None:
        """Output contains the weather description."""
        with patch("urllib.request.urlopen", return_value=_mock_http(_WTTR_VALID)):
            result = self.tool.execute({"city": "London"})
        self.assertIn("Partly cloudy", result)

    def test_contains_humidity(self) -> None:
        """Output contains the humidity percentage."""
        with patch("urllib.request.urlopen", return_value=_mock_http(_WTTR_VALID)):
            result = self.tool.execute({"city": "London"})
        self.assertIn("72%", result)

    def test_city_name_in_output(self) -> None:
        """Output includes the resolved city name."""
        with patch("urllib.request.urlopen", return_value=_mock_http(_WTTR_VALID)):
            result = self.tool.execute({"city": "London"})
        self.assertIn("London", result)

    # -- Bad city name -------------------------------------------------------

    def test_city_not_found_http_404(self) -> None:
        """HTTP 404 from wttr.in raises ToolExecutionError with 'not found'."""
        with patch("urllib.request.urlopen", side_effect=_http_error(404)):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"city": "Atlantis"})
        self.assertIn("not found", str(ctx.exception).lower())

    def test_city_not_found_empty_condition(self) -> None:
        """Empty current_condition list raises ToolExecutionError."""
        with patch("urllib.request.urlopen", return_value=_mock_http(_WTTR_EMPTY)):
            with self.assertRaises(ToolExecutionError):
                self.tool.execute({"city": "Xyzzy"})

    def test_nonexistent_city_error_mentions_city(self) -> None:
        """ToolExecutionError message references the city name."""
        with patch("urllib.request.urlopen", side_effect=_http_error(404)):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"city": "Atlantis"})
        self.assertIn("Atlantis", str(ctx.exception))

    # -- Network errors ------------------------------------------------------

    def test_network_error_raises_execution_error(self) -> None:
        """URLError raises ToolExecutionError."""
        with patch("urllib.request.urlopen", side_effect=_url_error()):
            with self.assertRaises(ToolExecutionError):
                self.tool.execute({"city": "London"})

    def test_timeout_raises_execution_error(self) -> None:
        """TimeoutError raises ToolExecutionError."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            with self.assertRaises(ToolExecutionError):
                self.tool.execute({"city": "London"})

    # -- Argument validation -------------------------------------------------

    def test_missing_city_raises_argument_error(self) -> None:
        """Missing 'city' key raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({})

    def test_empty_city_raises_argument_error(self) -> None:
        """Empty city string raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"city": "   "})

    def test_none_city_raises_argument_error(self) -> None:
        """None city raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"city": None})

    # -- Declaration schema --------------------------------------------------

    def test_declaration_name(self) -> None:
        """Declaration name matches the tool name property."""
        self.assertEqual(self.tool.get_declaration()["name"], self.tool.name)

    def test_city_is_required(self) -> None:
        """'city' is listed as a required parameter."""
        self.assertIn("city", self.tool.get_declaration()["parameters"]["required"])

    def test_country_is_optional(self) -> None:
        """'country' is in properties but NOT in required."""
        decl = self.tool.get_declaration()["parameters"]
        self.assertIn("country", decl["properties"])
        self.assertNotIn("country", decl["required"])


# ─────────────────────────────────────────────────────────────────────────────
# SearchTool
# ─────────────────────────────────────────────────────────────────────────────

_WIKI_VALID = json.dumps(
    {
        "type": "standard",
        "title": "Python (programming language)",
        "description": "High-level programming language",
        "extract": "Python is a high-level, general-purpose programming language.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Python"}},
    }
)

_WIKI_DISAMBIGUATION = json.dumps(
    {
        "type": "disambiguation",
        "title": "Python",
    }
)

_DDG_VALID = json.dumps(
    {
        "Heading": "Python",
        "AbstractText": "Python is a widely-used programming language.",
        "AbstractSource": "Wikipedia",
        "AbstractURL": "https://en.wikipedia.org/wiki/Python",
    }
)

_DDG_EMPTY = json.dumps(
    {
        "AbstractText": "",
        "Heading": "",
        "AbstractSource": "",
    }
)


class TestSearchTool(unittest.TestCase):
    """Tests for SearchTool with all HTTP calls mocked."""

    def setUp(self) -> None:
        from tools.built_in.search_tool import SearchTool

        self.tool = SearchTool()

    # -- Valid query ----------------------------------------------------------

    def test_returns_string(self) -> None:
        """execute() must return a str for a valid query."""
        with patch("urllib.request.urlopen", return_value=_mock_http(_WIKI_VALID)):
            result = self.tool.execute({"query": "Python programming"})
        self.assertIsInstance(result, str)

    def test_contains_title(self) -> None:
        """Wikipedia article title appears in the output."""
        with patch("urllib.request.urlopen", return_value=_mock_http(_WIKI_VALID)):
            result = self.tool.execute({"query": "Python programming"})
        self.assertIn("Python", result)

    def test_contains_extract(self) -> None:
        """Wikipedia extract appears in the output."""
        with patch("urllib.request.urlopen", return_value=_mock_http(_WIKI_VALID)):
            result = self.tool.execute({"query": "Python programming"})
        self.assertIn("high-level", result)

    def test_source_url_in_output(self) -> None:
        """Source URL is appended to the result."""
        with patch("urllib.request.urlopen", return_value=_mock_http(_WIKI_VALID)):
            result = self.tool.execute({"query": "Python programming"})
        self.assertIn("https://en.wikipedia.org", result)

    # -- DuckDuckGo fallback -------------------------------------------------

    def test_duckduckgo_fallback_when_wiki_404(self) -> None:
        """Wikipedia 404 triggers DuckDuckGo fallback."""

        def side_effect(request, timeout=10):
            if "wikipedia.org" in request.full_url:
                raise _http_error(404)
            return _mock_http(_DDG_VALID)

        with patch("urllib.request.urlopen", side_effect=side_effect):
            result = self.tool.execute({"query": "Python"})
        self.assertIn("Python", result)
        self.assertIsInstance(result, str)

    # -- No results ----------------------------------------------------------

    def test_no_results_raises_execution_error(self) -> None:
        """Both backends empty raises ToolExecutionError."""

        def side_effect(request, timeout=10):
            if "wikipedia.org" in request.full_url:
                raise _http_error(404)
            return _mock_http(_DDG_EMPTY)

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"query": "xyzzy_nonexistent_topic_abc"})
        self.assertIn("No results", str(ctx.exception))

    # -- Network errors ------------------------------------------------------

    def test_network_error_raises_execution_error(self) -> None:
        """Network failure raises ToolExecutionError."""
        with patch("urllib.request.urlopen", side_effect=_url_error()):
            with self.assertRaises(ToolExecutionError):
                self.tool.execute({"query": "anything"})

    # -- Argument validation -------------------------------------------------

    def test_missing_query_raises_argument_error(self) -> None:
        """Missing 'query' key raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({})

    def test_empty_query_raises_argument_error(self) -> None:
        """Empty query string raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"query": "   "})

    def test_wrong_type_raises_argument_error(self) -> None:
        """Non-string query raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"query": 123})

    # -- Declaration schema --------------------------------------------------

    def test_declaration_name(self) -> None:
        """Declaration name matches the tool name property."""
        self.assertEqual(self.tool.get_declaration()["name"], self.tool.name)

    def test_query_is_required(self) -> None:
        """'query' is listed as a required parameter."""
        self.assertIn("query", self.tool.get_declaration()["parameters"]["required"])

    def test_language_is_optional(self) -> None:
        """'language' is in properties but NOT in required."""
        decl = self.tool.get_declaration()["parameters"]
        self.assertIn("language", decl["properties"])
        self.assertNotIn("language", decl["required"])


# ─────────────────────────────────────────────────────────────────────────────
# TimeTool
# ─────────────────────────────────────────────────────────────────────────────

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-reuse-def]

from datetime import datetime, timezone

_FIXED_UTC = datetime(2025, 6, 20, 14, 32, 7, tzinfo=timezone.utc)


def _fake_current_time(tz=None):
    """Return a deterministic datetime adjusted to the requested timezone."""
    if tz is None:
        return _FIXED_UTC.astimezone(ZoneInfo("Europe/Paris"))
    return _FIXED_UTC.astimezone(tz)


class TestTimeTool(unittest.TestCase):
    """Tests for TimeTool — clock is frozen for deterministic assertions."""

    def setUp(self) -> None:
        from tools.built_in.time_tool import TimeTool

        self.tool = TimeTool()

    def _run(self, args: dict) -> str:
        """Execute the tool with a frozen clock."""
        with patch.object(
            type(self.tool), "_current_time", staticmethod(_fake_current_time)
        ):
            return self.tool.execute(args)

    # -- Valid inputs --------------------------------------------------------

    def test_returns_string(self) -> None:
        """execute() must return a str."""
        self.assertIsInstance(self._run({}), str)

    def test_local_time_no_arg(self) -> None:
        """No argument returns local time mentioning 'local'."""
        self.assertIn("local", self._run({}).lower())

    def test_utc_timezone(self) -> None:
        """UTC timezone returns the correct UTC time."""
        result = self._run({"timezone": "UTC"})
        self.assertIn("14:32:07", result)
        self.assertIn("+00:00", result)

    def test_iana_timezone_asia_tokyo(self) -> None:
        """Asia/Tokyo (UTC+9) offsets the time correctly."""
        result = self._run({"timezone": "Asia/Tokyo"})
        # 14:32:07 UTC -> 23:32:07 JST
        self.assertIn("23:32:07", result)
        self.assertIn("+09:00", result)

    def test_iana_timezone_america_new_york(self) -> None:
        """America/New_York (EDT, UTC-4 in June) offsets correctly."""
        result = self._run({"timezone": "America/New_York"})
        # 14:32:07 UTC -> 10:32:07 EDT
        self.assertIn("10:32:07", result)

    def test_abbreviation_jst(self) -> None:
        """JST abbreviation resolves to Asia/Tokyo."""
        result = self._run({"timezone": "JST"})
        self.assertIn("23:32:07", result)

    def test_city_name_tokyo(self) -> None:
        """City name 'Tokyo' resolves correctly."""
        result = self._run({"timezone": "Tokyo"})
        self.assertIn("23:32:07", result)

    def test_output_contains_date_label(self) -> None:
        """Output includes a 'Date' label."""
        self.assertIn("Date", self._run({}))

    def test_output_contains_time_label(self) -> None:
        """Output includes a 'Time' label."""
        self.assertIn("Time", self._run({}))

    def test_output_contains_utc_offset(self) -> None:
        """Output includes a UTC offset label."""
        self.assertIn("UTC offset", self._run({}))

    def test_output_contains_year(self) -> None:
        """Output includes the year 2025."""
        self.assertIn("2025", self._run({}))

    # -- Invalid timezone ----------------------------------------------------

    def test_unknown_timezone_raises_execution_error(self) -> None:
        """Unrecognised timezone raises ToolExecutionError."""
        with self.assertRaises(ToolExecutionError) as ctx:
            self._run({"timezone": "Neverland/Pixie_Hollow"})
        self.assertIn("Unknown timezone", str(ctx.exception))

    # -- Argument validation -------------------------------------------------

    def test_wrong_type_raises_argument_error(self) -> None:
        """Non-string timezone raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self._run({"timezone": 42})

    def test_none_timezone_returns_local(self) -> None:
        """None timezone treated as 'use local time' — no exception raised."""
        result = self._run({"timezone": None})
        self.assertIsInstance(result, str)

    # -- Declaration schema --------------------------------------------------

    def test_declaration_name(self) -> None:
        """Declaration name matches the tool name property."""
        self.assertEqual(self.tool.get_declaration()["name"], self.tool.name)

    def test_timezone_is_optional(self) -> None:
        """'timezone' is NOT in the required list."""
        self.assertNotIn(
            "timezone", self.tool.get_declaration()["parameters"]["required"]
        )

    def test_required_list_is_empty(self) -> None:
        """No parameters are required — timezone is always optional."""
        self.assertEqual(self.tool.get_declaration()["parameters"]["required"], [])


# ─────────────────────────────────────────────────────────────────────────────
# TranslateTool
# ─────────────────────────────────────────────────────────────────────────────

_MYMEMORY_OK = json.dumps(
    {
        "responseStatus": 200,
        "responseData": {
            "translatedText": "Bonjour",
            "detectedLanguage": "en",
            "match": 1.0,
        },
    }
)

_MYMEMORY_QUOTA = json.dumps(
    {
        "responseStatus": 403,
        "responseData": {"translatedText": ""},
    }
)

_LIBRETRANSLATE_OK = json.dumps(
    {
        "translatedText": "Hola",
        "detectedLanguage": {"language": "en", "confidence": 99},
    }
)


class TestTranslateTool(unittest.TestCase):
    """Tests for TranslateTool — HTTP mocked for both backends."""

    def setUp(self) -> None:
        from tools.custom.translate_tool import TranslateTool

        self.tool = TranslateTool()

    def _run(self, args: dict, response_body: str = _MYMEMORY_OK) -> str:
        with patch("urllib.request.urlopen", return_value=_mock_http(response_body)):
            return self.tool.execute(args)

    # -- Valid translation ---------------------------------------------------

    def test_returns_string(self) -> None:
        """execute() must return a str."""
        self.assertIsInstance(
            self._run({"text": "Hello", "target_language": "fr"}), str
        )

    def test_translated_text_in_output(self) -> None:
        """The translated text appears in the output."""
        result = self._run({"text": "Hello", "target_language": "fr"})
        self.assertIn("Bonjour", result)

    def test_target_language_in_output(self) -> None:
        """The target language code appears in the output."""
        result = self._run({"text": "Hello", "target_language": "fr"})
        self.assertIn("fr", result)

    def test_backend_label_in_output(self) -> None:
        """Output indicates which backend was used."""
        result = self._run({"text": "Hello", "target_language": "fr"})
        self.assertIn("MyMemory", result)

    def test_full_language_name_accepted(self) -> None:
        """Full language name 'French' is normalised to 'fr'."""
        with patch(
            "urllib.request.urlopen", return_value=_mock_http(_MYMEMORY_OK)
        ) as mock_open:
            self.tool.execute({"text": "Hello", "target_language": "French"})
        url = mock_open.call_args[0][0].full_url
        self.assertIn("fr", url)

    # -- Fallback to LibreTranslate ------------------------------------------

    def test_falls_back_to_libretranslate_on_quota(self) -> None:
        """MyMemory quota exceeded triggers LibreTranslate fallback."""
        call_count = 0

        def side_effect(request, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_http(_MYMEMORY_QUOTA)
            return _mock_http(_LIBRETRANSLATE_OK)

        with patch("urllib.request.urlopen", side_effect=side_effect):
            result = self.tool.execute({"text": "Hello", "target_language": "es"})
        self.assertIn("Hola", result)
        self.assertIn("LibreTranslate", result)

    # -- Invalid language ----------------------------------------------------

    def test_unknown_language_raises_argument_error(self) -> None:
        """Unknown target language raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError) as ctx:
            self.tool.execute({"text": "Hello", "target_language": "Klingon"})
        self.assertIn("Klingon", str(ctx.exception))

    # -- Argument validation -------------------------------------------------

    def test_missing_text_raises_argument_error(self) -> None:
        """Missing 'text' raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"target_language": "fr"})

    def test_missing_target_raises_argument_error(self) -> None:
        """Missing 'target_language' raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"text": "Hello"})

    def test_empty_text_raises_argument_error(self) -> None:
        """Empty 'text' raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"text": "   ", "target_language": "fr"})

    def test_none_text_raises_argument_error(self) -> None:
        """None 'text' raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"text": None, "target_language": "fr"})

    # -- Network errors ------------------------------------------------------

    def test_network_error_raises_execution_error(self) -> None:
        """Complete network failure raises ToolExecutionError."""
        with patch("urllib.request.urlopen", side_effect=_url_error()):
            with self.assertRaises(ToolExecutionError):
                self.tool.execute({"text": "Hello", "target_language": "fr"})

    # -- Declaration schema --------------------------------------------------

    def test_declaration_name(self) -> None:
        """Declaration name matches the tool name property."""
        self.assertEqual(self.tool.get_declaration()["name"], self.tool.name)

    def test_text_and_target_are_required(self) -> None:
        """Both 'text' and 'target_language' are required."""
        required = self.tool.get_declaration()["parameters"]["required"]
        self.assertIn("text", required)
        self.assertIn("target_language", required)

    def test_source_language_is_optional(self) -> None:
        """'source_language' is in properties but NOT required."""
        decl = self.tool.get_declaration()["parameters"]
        self.assertIn("source_language", decl["properties"])
        self.assertNotIn("source_language", decl["required"])


# ─────────────────────────────────────────────────────────────────────────────
# FileReaderTool
# ─────────────────────────────────────────────────────────────────────────────


class TestFileReaderTool(unittest.TestCase):
    """Tests for FileReaderTool — uses an isolated temp directory as sandbox."""

    def setUp(self) -> None:
        from tools.custom.file_reader_tool import FileReaderTool

        self.tmpdir = tempfile.mkdtemp()
        # Patch module-level constants to confine the tool to tmpdir.
        real_tmpdir = os.path.realpath(self.tmpdir)
        patcher_base = patch(
            "tools.custom.file_reader_tool.FILE_READER_BASE_DIR", real_tmpdir
        )
        patcher_resolved = patch(
            "tools.custom.file_reader_tool._RESOLVED_BASE_DIR", real_tmpdir
        )
        patcher_base.start()
        patcher_resolved.start()
        self.addCleanup(patcher_base.stop)
        self.addCleanup(patcher_resolved.stop)
        self.tool = FileReaderTool()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name: str, content: str = "hello\nworld\n") -> str:
        """Write a file inside the sandbox and return its full path."""
        path = os.path.join(self.tmpdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        from pathlib import Path

        Path(path).write_text(content, encoding="utf-8")
        return path

    def _run(self, filepath: str) -> str:
        return self.tool.execute({"filepath": filepath})

    # -- Valid file read -----------------------------------------------------

    def test_returns_string(self) -> None:
        """execute() must return a str for a valid file."""
        self._write("notes.txt", "Hello world\n")
        self.assertIsInstance(self._run("notes.txt"), str)

    def test_content_in_output(self) -> None:
        """File content appears in the output."""
        self._write("data.txt", "Secret recipe\n")
        self.assertIn("Secret recipe", self._run("data.txt"))

    def test_header_contains_filename(self) -> None:
        """Output header includes the filename."""
        self._write("report.txt", "data\n")
        self.assertIn("report.txt", self._run("report.txt"))

    def test_header_contains_line_count(self) -> None:
        """Output header includes the line count."""
        self._write("lines.txt", "a\nb\nc\n")
        self.assertIn("3 lines", self._run("lines.txt"))

    def test_nested_file_allowed(self) -> None:
        """Files in subdirectories within the sandbox are accessible."""
        self._write("sub/deep.txt", "nested\n")
        self.assertIn("nested", self._run("sub/deep.txt"))

    def test_csv_extension_allowed(self) -> None:
        """CSV files are on the extension allow-list."""
        self._write("data.csv", "a,b\n1,2\n")
        self.assertIn("a,b", self._run("data.csv"))

    def test_absolute_path_inside_sandbox_allowed(self) -> None:
        """An absolute path inside the sandbox is permitted."""
        full = self._write("abs.txt", "absolute\n")
        self.assertIn("absolute", self._run(full))

    # -- File not found ------------------------------------------------------

    def test_nonexistent_file_raises_execution_error(self) -> None:
        """Non-existent file raises ToolExecutionError mentioning 'not found'."""
        with self.assertRaises(ToolExecutionError) as ctx:
            self._run("ghost.txt")
        self.assertIn("not found", str(ctx.exception).lower())

    # -- Path traversal prevention -------------------------------------------

    def test_dotdot_traversal_blocked(self) -> None:
        """../../etc/passwd-style traversal raises ToolExecutionError."""
        with self.assertRaises(ToolExecutionError) as ctx:
            self._run("../../etc/passwd")
        self.assertIn("outside the permitted directory", str(ctx.exception))

    def test_absolute_path_outside_sandbox_blocked(self) -> None:
        """Absolute path outside sandbox raises ToolExecutionError."""
        with self.assertRaises(ToolExecutionError) as ctx:
            self._run("/etc/passwd")
        self.assertIn("outside the permitted directory", str(ctx.exception))

    def test_symlink_outside_sandbox_blocked(self) -> None:
        """Symlink pointing outside sandbox raises ToolExecutionError."""
        outside = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
        outside.write("outside")
        outside.close()
        link = os.path.join(self.tmpdir, "link.txt")
        os.symlink(outside.name, link)
        try:
            with self.assertRaises(ToolExecutionError) as ctx:
                self._run("link.txt")
            self.assertIn("outside the permitted directory", str(ctx.exception))
        finally:
            os.unlink(outside.name)

    def test_sibling_prefix_attack_blocked(self) -> None:
        """A sibling directory whose name starts with the sandbox name is blocked."""
        sibling = self.tmpdir + "_evil"
        os.makedirs(sibling, exist_ok=True)
        from pathlib import Path

        Path(os.path.join(sibling, "secret.txt")).write_text("pwned")
        try:
            with self.assertRaises(ToolExecutionError) as ctx:
                self._run(os.path.join(sibling, "secret.txt"))
            self.assertIn("outside the permitted directory", str(ctx.exception))
        finally:
            import shutil

            shutil.rmtree(sibling, ignore_errors=True)

    # -- Invalid filepath argument -------------------------------------------

    def test_missing_filepath_raises_argument_error(self) -> None:
        """Missing 'filepath' key raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({})

    def test_empty_filepath_raises_argument_error(self) -> None:
        """Empty filepath string raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"filepath": "   "})

    def test_none_filepath_raises_argument_error(self) -> None:
        """None filepath raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"filepath": None})

    def test_null_byte_raises_argument_error(self) -> None:
        """Null byte in filepath raises ToolArgumentError."""
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"filepath": "file\x00.txt"})

    # -- Disallowed extension ------------------------------------------------

    def test_python_file_blocked(self) -> None:
        """Python files are not on the extension allow-list."""
        self._write("script.py", "import os\n")
        with self.assertRaises(ToolExecutionError) as ctx:
            self._run("script.py")
        self.assertIn("not permitted", str(ctx.exception))

    def test_env_file_blocked(self) -> None:
        """.env files are not on the extension allow-list."""
        self._write(".env", "SECRET=xyz\n")
        with self.assertRaises(ToolExecutionError):
            self._run(".env")

    def test_no_extension_blocked(self) -> None:
        """Files with no extension are not permitted."""
        self._write("noext", "data")
        with self.assertRaises(ToolExecutionError):
            self._run("noext")

    # -- Declaration schema --------------------------------------------------

    def test_declaration_name(self) -> None:
        """Declaration name matches the tool name property."""
        self.assertEqual(self.tool.get_declaration()["name"], self.tool.name)

    def test_filepath_is_required(self) -> None:
        """'filepath' is listed as a required parameter."""
        self.assertIn("filepath", self.tool.get_declaration()["parameters"]["required"])


# ─────────────────────────────────────────────────────────────────────────────
# Cross-tool checks — each tool returns a str and has a valid declaration
# ─────────────────────────────────────────────────────────────────────────────


class TestAllToolsReturnString(unittest.TestCase):
    """
    Smoke tests confirming every tool's execute() returns a str and
    get_declaration() yields a schema with the required keys.
    """

    def _wttr_side(self, *args, **kwargs):
        """Side-effect for urlopen that returns valid wttr.in JSON."""
        return _mock_http(_WTTR_VALID)

    def _wiki_side(self, *args, **kwargs):
        """Side-effect for urlopen that returns valid Wikipedia JSON."""
        return _mock_http(_WIKI_VALID)

    def _mm_side(self, *args, **kwargs):
        """Side-effect for urlopen that returns valid MyMemory JSON."""
        return _mock_http(_MYMEMORY_OK)

    def test_calculator_returns_str(self) -> None:
        """CalculatorTool.execute() returns str."""
        from tools.built_in.calculator_tool import CalculatorTool

        result = CalculatorTool().execute({"expression": "6 * 7"})
        self.assertIsInstance(result, str)

    def test_weather_returns_str(self) -> None:
        """WeatherTool.execute() returns str."""
        from tools.built_in.weather_tool import WeatherTool

        with patch("tools.built_in.weather_tool.OPENWEATHER_API_KEY", ""):
            with patch("urllib.request.urlopen", side_effect=self._wttr_side):
                result = WeatherTool().execute({"city": "London"})
        self.assertIsInstance(result, str)

    def test_search_returns_str(self) -> None:
        """SearchTool.execute() returns str."""
        from tools.built_in.search_tool import SearchTool

        with patch("urllib.request.urlopen", side_effect=self._wiki_side):
            result = SearchTool().execute({"query": "Python"})
        self.assertIsInstance(result, str)

    def test_time_returns_str(self) -> None:
        """TimeTool.execute() returns str."""
        from tools.built_in.time_tool import TimeTool

        result = TimeTool().execute({})
        self.assertIsInstance(result, str)

    def test_translate_returns_str(self) -> None:
        """TranslateTool.execute() returns str."""
        from tools.custom.translate_tool import TranslateTool

        with patch("urllib.request.urlopen", side_effect=self._mm_side):
            result = TranslateTool().execute({"text": "Hello", "target_language": "fr"})
        self.assertIsInstance(result, str)

    def test_file_reader_returns_str(self) -> None:
        """FileReaderTool.execute() returns str for a valid file."""
        from tools.custom.file_reader_tool import FileReaderTool
        from pathlib import Path

        tmpdir = tempfile.mkdtemp()
        real_tmpdir = os.path.realpath(tmpdir)
        Path(os.path.join(tmpdir, "test.txt")).write_text("hello")
        with patch("tools.custom.file_reader_tool.FILE_READER_BASE_DIR", real_tmpdir):
            with patch("tools.custom.file_reader_tool._RESOLVED_BASE_DIR", real_tmpdir):
                result = FileReaderTool().execute({"filepath": "test.txt"})
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
        self.assertIsInstance(result, str)

    def test_all_declarations_have_name(self) -> None:
        """Every tool's get_declaration() dict includes a 'name' key."""
        from tools.built_in.calculator_tool import CalculatorTool
        from tools.built_in.weather_tool import WeatherTool
        from tools.built_in.search_tool import SearchTool
        from tools.built_in.time_tool import TimeTool
        from tools.custom.translate_tool import TranslateTool
        from tools.custom.file_reader_tool import FileReaderTool

        tools = [
            CalculatorTool(),
            WeatherTool(),
            SearchTool(),
            TimeTool(),
            TranslateTool(),
            FileReaderTool(),
        ]
        for tool in tools:
            with self.subTest(tool=tool.name):
                decl = tool.get_declaration()
                self.assertIn("name", decl, f"{tool.name} declaration missing 'name'")
                self.assertIn(
                    "description",
                    decl,
                    f"{tool.name} declaration missing 'description'",
                )
                self.assertIn(
                    "parameters", decl, f"{tool.name} declaration missing 'parameters'"
                )
                self.assertEqual(
                    decl["name"],
                    tool.name,
                    f"{tool.name} declaration 'name' doesn't match property",
                )

    def test_all_declarations_name_matches_property(self) -> None:
        """declaration['name'] == tool.name for every tool."""
        from tools.built_in.calculator_tool import CalculatorTool
        from tools.built_in.weather_tool import WeatherTool
        from tools.built_in.search_tool import SearchTool
        from tools.built_in.time_tool import TimeTool
        from tools.custom.translate_tool import TranslateTool
        from tools.custom.file_reader_tool import FileReaderTool

        for ToolClass in [
            CalculatorTool,
            WeatherTool,
            SearchTool,
            TimeTool,
            TranslateTool,
            FileReaderTool,
        ]:
            tool = ToolClass()
            with self.subTest(name=tool.name):
                self.assertEqual(tool.get_declaration()["name"], tool.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
