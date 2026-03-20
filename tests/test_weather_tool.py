"""
tests/test_weather_tool.py

Unit tests for WeatherTool — all HTTP calls are mocked so no real
network access is needed.

Coverage:
  - Valid wttr.in response: normalised output, temperature formatting
  - Valid OpenWeatherMap response
  - City not found: HTTP 404, empty current_condition, OWM cod 404
  - Network errors: URLError, TimeoutError, generic HTTPError
  - Malformed JSON from upstream
  - Argument validation: missing key, empty string, wrong type
  - Optional country arg appended to query
  - Declaration schema structure
"""

from __future__ import annotations

import json
import unittest
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

from tools.base_tool import ToolArgumentError, ToolExecutionError
from tools.built_in.weather_tool import WeatherTool

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

_WTTR_PARIS = {
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
            "areaName": [{"value": "Paris"}],
            "country": [{"value": "France"}],
        }
    ],
}

_OWM_LONDON = {
    "cod": 200,
    "name": "London",
    "sys": {"country": "GB"},
    "main": {
        "temp": 15.0,
        "feels_like": 13.5,
        "humidity": 80,
    },
    "weather": [{"description": "light rain"}],
    "wind": {"speed": 5.0},  # m/s -> 18.0 km/h
}

_OWM_NOT_FOUND = {"cod": "404", "message": "city not found"}


def _mock_response(body: str, status: int = 200):
    """Return a context-manager mock that mimics urllib urlopen."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read.return_value = body.encode("utf-8")
    cm.status = status
    return cm


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://x", code=code, msg="err", hdrs=None, fp=None
    )


# ---------------------------------------------------------------------------
# wttr.in backend
# ---------------------------------------------------------------------------


class TestWeatherToolWttr(unittest.TestCase):

    def setUp(self):
        self.tool = WeatherTool()
        # Ensure OWM key is absent so wttr.in path is taken.
        patcher = patch("tools.built_in.weather_tool.OPENWEATHER_API_KEY", "")
        self.addCleanup(patcher.stop)
        patcher.start()

    def _execute(self, city: str, country: str = "") -> str:
        args = {"city": city}
        if country:
            args["country"] = country
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(json.dumps(_WTTR_PARIS)),
        ):
            return self.tool.execute(args)

    def test_returns_string(self):
        result = self._execute("Paris")
        self.assertIsInstance(result, str)

    def test_contains_temperature(self):
        result = self._execute("Paris")
        self.assertIn("18", result)

    def test_contains_feels_like(self):
        result = self._execute("Paris")
        self.assertIn("16", result)

    def test_contains_condition(self):
        result = self._execute("Paris")
        self.assertIn("Partly cloudy", result)

    def test_contains_humidity(self):
        result = self._execute("Paris")
        self.assertIn("72%", result)

    def test_contains_wind(self):
        result = self._execute("Paris")
        self.assertIn("14", result)

    def test_resolved_location_in_output(self):
        result = self._execute("Paris")
        self.assertIn("Paris", result)
        self.assertIn("France", result)

    def test_country_arg_appended_to_query(self):
        """country arg must be included in the URL query string."""
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(json.dumps(_WTTR_PARIS)),
        ) as mock_open:
            self.tool.execute({"city": "Paris", "country": "FR"})
            url_used = mock_open.call_args[0][0].full_url
            self.assertIn("Paris", url_used)
            self.assertIn("FR", url_used)

    def test_city_not_found_empty_current_condition(self):
        """Empty current_condition list must raise ToolExecutionError."""
        bad_payload = {"current_condition": [], "nearest_area": []}
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(json.dumps(bad_payload)),
        ):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"city": "Zzzzz"})
        self.assertIn("Zzzzz", str(ctx.exception))

    def test_http_404_raises_tool_execution_error(self):
        with patch("urllib.request.urlopen", side_effect=_http_error(404)):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"city": "Atlantis"})
        self.assertIn("Atlantis", str(ctx.exception))
        self.assertIn("not found", str(ctx.exception).lower())

    def test_http_500_raises_tool_execution_error(self):
        with patch("urllib.request.urlopen", side_effect=_http_error(500)):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"city": "London"})
        self.assertIn("500", str(ctx.exception))

    def test_network_error_raises_tool_execution_error(self):
        with patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("no route")
        ):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"city": "London"})
        self.assertIn("no route", str(ctx.exception).lower())

    def test_timeout_raises_tool_execution_error(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"city": "London"})
        self.assertIn("timed out", str(ctx.exception).lower())

    def test_malformed_json_raises_tool_execution_error(self):
        with patch("urllib.request.urlopen", return_value=_mock_response("not json")):
            with self.assertRaises(ToolExecutionError):
                self.tool.execute({"city": "London"})


# ---------------------------------------------------------------------------
# OpenWeatherMap backend
# ---------------------------------------------------------------------------


class TestWeatherToolOWM(unittest.TestCase):

    def setUp(self):
        self.tool = WeatherTool()
        patcher = patch("tools.built_in.weather_tool.OPENWEATHER_API_KEY", "fake_key")
        self.addCleanup(patcher.stop)
        patcher.start()

    def _execute(self, city: str) -> str:
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(json.dumps(_OWM_LONDON)),
        ):
            return self.tool.execute({"city": city})

    def test_returns_string(self):
        self.assertIsInstance(self._execute("London"), str)

    def test_contains_temperature(self):
        self.assertIn("15", self._execute("London"))

    def test_contains_condition(self):
        result = self._execute("London")
        self.assertIn("rain", result.lower())

    def test_resolved_location_in_output(self):
        result = self._execute("London")
        self.assertIn("London", result)
        self.assertIn("GB", result)

    def test_wind_converted_to_kmh(self):
        # 5 m/s * 3.6 = 18 km/h
        result = self._execute("London")
        self.assertIn("18", result)

    def test_owm_404_raises_tool_execution_error(self):
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(json.dumps(_OWM_NOT_FOUND)),
        ):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"city": "Atlantis"})
        self.assertIn("not found", str(ctx.exception).lower())

    def test_owm_error_message_included(self):
        error_payload = {"cod": "401", "message": "Invalid API key."}
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(json.dumps(error_payload)),
        ):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"city": "London"})
        self.assertIn("Invalid API key", str(ctx.exception))


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestWeatherToolArgs(unittest.TestCase):

    def setUp(self):
        self.tool = WeatherTool()
        patch("tools.built_in.weather_tool.OPENWEATHER_API_KEY", "").start()
        self.addCleanup(patch.stopall)

    def test_missing_city_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError) as ctx:
            self.tool.execute({})
        self.assertIn("city", str(ctx.exception).lower())

    def test_empty_city_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"city": "   "})

    def test_wrong_type_city_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"city": 42})

    def test_none_city_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"city": None})


# ---------------------------------------------------------------------------
# Declaration schema
# ---------------------------------------------------------------------------


class TestWeatherToolDeclaration(unittest.TestCase):

    def setUp(self):
        self.tool = WeatherTool()

    def test_name_matches_property(self):
        self.assertEqual(self.tool.get_declaration()["name"], self.tool.name)

    def test_has_description(self):
        d = self.tool.get_declaration()
        self.assertIn("description", d)
        self.assertGreater(len(d["description"]), 10)

    def test_city_parameter_present(self):
        props = self.tool.get_declaration()["parameters"]["properties"]
        self.assertIn("city", props)

    def test_city_is_required(self):
        required = self.tool.get_declaration()["parameters"]["required"]
        self.assertIn("city", required)

    def test_country_parameter_optional(self):
        """country must exist in properties but NOT in required."""
        decl = self.tool.get_declaration()["parameters"]
        self.assertIn("country", decl["properties"])
        self.assertNotIn("country", decl["required"])

    def test_city_type_is_string(self):
        props = self.tool.get_declaration()["parameters"]["properties"]
        self.assertIn(props["city"]["type"].lower(), ("string",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
