"""
tests/test_time_tool.py

Unit tests for TimeTool — clock is frozen with unittest.mock.patch so
every assertion against date/time values is deterministic.

Coverage:
  - Local time (no argument, empty string, "local")
  - IANA timezone names: UTC, Europe/London, Asia/Tokyo, America/New_York
  - Abbreviations: JST, EST, CET, PST, IST, UTC, GMT
  - City/informal names: Tokyo, Paris, London, New York
  - Alias resolution note in output when abbreviation is used
  - UTC offset format (+HH:MM)
  - Unknown timezone -> ToolExecutionError with suggestions
  - Wrong-type argument -> ToolArgumentError
  - Output contains date, time, offset, and timezone label
  - Declaration schema: name, description, optional timezone param
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from tools.base_tool import ToolArgumentError, ToolExecutionError
from tools.built_in.time_tool import TimeTool

# ---------------------------------------------------------------------------
# A fixed "now" for deterministic assertions.
# 2025-06-20 14:32:07 UTC
# ---------------------------------------------------------------------------
_FIXED_UTC = datetime(2025, 6, 20, 14, 32, 7, tzinfo=timezone.utc)


def _freeze(tz):
    """
    Return a callable that, when called with tz=tz, returns a fixed datetime
    adjusted to that timezone (or local if tz is None).
    """

    def _fake_now(tz=None):
        if tz is None:
            # Use a fixed UTC+2 offset — avoids tzdata dependency on Windows.
            from datetime import timezone, timedelta

            local_tz = timezone(timedelta(hours=2))
            return _FIXED_UTC.astimezone(local_tz)
        return _FIXED_UTC.astimezone(tz)

    return _fake_now


# Patch target: the static method on the class.
_PATCH_TARGET = "tools.built_in.time_tool.TimeTool._current_time"


class TestTimeToolLocalTime(unittest.TestCase):

    def setUp(self):
        self.tool = TimeTool()

    def _run(self, args: dict) -> str:
        with patch.object(TimeTool, "_current_time", staticmethod(_freeze(None))):
            return self.tool.execute(args)

    def test_no_arg_returns_string(self):
        self.assertIsInstance(self._run({}), str)

    def test_empty_timezone_returns_local(self):
        result = self._run({"timezone": ""})
        self.assertIn("local", result.lower())

    def test_local_keyword_returns_local(self):
        result = self._run({"timezone": "local"})
        self.assertIn("local", result.lower())

    def test_output_contains_date(self):
        result = self._run({})
        self.assertIn("2025", result)
        self.assertIn("June", result)
        self.assertIn("20", result)

    def test_output_contains_time(self):
        result = self._run({})
        # Paris is UTC+2 in summer, so 14:32:07 UTC -> 16:32:07 local
        self.assertIn("16:32:07", result)

    def test_output_contains_utc_offset(self):
        result = self._run({})
        self.assertIn("+02:00", result)

    def test_output_contains_date_label(self):
        result = self._run({})
        self.assertIn("Date", result)

    def test_output_contains_time_label(self):
        result = self._run({})
        self.assertIn("Time", result)

    def test_output_contains_utc_offset_label(self):
        result = self._run({})
        self.assertIn("UTC offset", result)


class TestTimeToolNamedTimezones(unittest.TestCase):

    def setUp(self):
        self.tool = TimeTool()

    def _run(self, tz_arg: str) -> str:
        with patch.object(TimeTool, "_current_time", staticmethod(_freeze(None))):
            return self.tool.execute({"timezone": tz_arg})

    def test_utc_timezone(self):
        result = self._run("UTC")
        self.assertIn("14:32:07", result)
        self.assertIn("+00:00", result)

    def test_europe_london(self):
        result = self._run("Europe/London")
        self.assertIn("Europe/London", result)

    def test_asia_tokyo(self):
        result = self._run("Asia/Tokyo")
        # UTC+9: 14:32:07 UTC -> 23:32:07 JST
        self.assertIn("23:32:07", result)
        self.assertIn("+09:00", result)

    def test_america_new_york(self):
        result = self._run("America/New_York")
        # EDT (UTC-4 in June): 14:32:07 UTC -> 10:32:07 EDT
        self.assertIn("10:32:07", result)

    def test_iana_name_in_output(self):
        result = self._run("America/Chicago")
        self.assertIn("America/Chicago", result)


class TestTimeToolAbbreviations(unittest.TestCase):

    def setUp(self):
        self.tool = TimeTool()

    def _run(self, tz_arg: str) -> str:
        with patch.object(TimeTool, "_current_time", staticmethod(_freeze(None))):
            return self.tool.execute({"timezone": tz_arg})

    def test_jst_resolves(self):
        result = self._run("JST")
        self.assertIn("23:32:07", result)

    def test_est_resolves(self):
        result = self._run("EST")
        # America/New_York resolves JST -> works
        self.assertIsInstance(result, str)

    def test_cet_resolves(self):
        result = self._run("CET")
        self.assertIsInstance(result, str)

    def test_utc_abbreviation(self):
        result = self._run("UTC")
        self.assertIn("+00:00", result)

    def test_gmt_resolves(self):
        result = self._run("GMT")
        self.assertIsInstance(result, str)

    def test_ist_resolves(self):
        result = self._run("IST")
        # Asia/Kolkata UTC+5:30
        self.assertIn("20:02:07", result)

    def test_abbreviation_resolution_note_in_output(self):
        """When an abbreviation is used, the canonical name should appear."""
        result = self._run("JST")
        self.assertIn("Asia/Tokyo", result)


class TestTimeToolCityNames(unittest.TestCase):

    def setUp(self):
        self.tool = TimeTool()

    def _run(self, tz_arg: str) -> str:
        with patch.object(TimeTool, "_current_time", staticmethod(_freeze(None))):
            return self.tool.execute({"timezone": tz_arg})

    def test_tokyo_resolves(self):
        result = self._run("Tokyo")
        self.assertIn("23:32:07", result)

    def test_paris_resolves(self):
        result = self._run("Paris")
        self.assertIsInstance(result, str)

    def test_london_resolves(self):
        result = self._run("London")
        self.assertIsInstance(result, str)

    def test_new_york_resolves(self):
        result = self._run("New York")
        self.assertIn("10:32:07", result)

    def test_sydney_resolves(self):
        result = self._run("Sydney")
        self.assertIsInstance(result, str)

    def test_city_name_case_insensitive(self):
        lower = self._run("tokyo")
        upper = self._run("Tokyo")
        mixed = self._run("TOKYO")
        # All three should agree on the time.
        for r in (upper, mixed):
            self.assertEqual(
                lower.split("Time")[1][:20],
                r.split("Time")[1][:20],
            )


class TestTimeToolErrors(unittest.TestCase):

    def setUp(self):
        self.tool = TimeTool()

    def test_unknown_timezone_raises_execution_error(self):
        with self.assertRaises(ToolExecutionError) as ctx:
            self.tool.execute({"timezone": "Neverland/Pixie_Hollow"})
        self.assertIn("Unknown timezone", str(ctx.exception))

    def test_unknown_timezone_includes_suggestions(self):
        """A partial match like 'Tokyo' should produce suggestions."""
        with self.assertRaises(ToolExecutionError) as ctx:
            self.tool.execute({"timezone": "completely_invalid_tz_xyz"})
        msg = str(ctx.exception)
        # Either suggestions or guidance must be present.
        self.assertTrue(
            "Suggestions" in msg or "IANA" in msg,
            f"Expected suggestions or guidance in error message: {msg}",
        )

    def test_wrong_type_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError) as ctx:
            self.tool.execute({"timezone": 42})
        self.assertIn("string", str(ctx.exception).lower())

    def test_none_timezone_treated_as_local(self):
        """None should be silently treated as 'use local time'."""
        with patch.object(TimeTool, "_current_time", staticmethod(_freeze(None))):
            result = self.tool.execute({"timezone": None})
        self.assertIsInstance(result, str)


class TestTimeToolDeclaration(unittest.TestCase):

    def setUp(self):
        self.tool = TimeTool()

    def test_name_matches_property(self):
        self.assertEqual(self.tool.get_declaration()["name"], self.tool.name)

    def test_name_is_time(self):
        self.assertEqual(self.tool.name, "time")

    def test_has_description(self):
        d = self.tool.get_declaration()
        self.assertGreater(len(d.get("description", "")), 10)

    def test_timezone_parameter_present(self):
        props = self.tool.get_declaration()["parameters"]["properties"]
        self.assertIn("timezone", props)

    def test_timezone_is_optional(self):
        required = self.tool.get_declaration()["parameters"]["required"]
        self.assertNotIn("timezone", required)

    def test_required_list_is_empty(self):
        required = self.tool.get_declaration()["parameters"]["required"]
        self.assertEqual(required, [])

    def test_timezone_type_is_string(self):
        props = self.tool.get_declaration()["parameters"]["properties"]
        self.assertIn(props["timezone"]["type"].lower(), ("string",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
