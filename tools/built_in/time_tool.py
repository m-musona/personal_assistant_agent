"""

TimeTool — returns the current date and time, optionally for a named
timezone.

Why this tool matters beyond its simplicity
--------------------------------------------
The LLM's training data has a fixed knowledge cutoff, so it genuinely
cannot answer "what time is it in Tokyo right now?" without a tool.
TimeTool is therefore a clean, deterministic end-to-end test of the
entire ReAct loop:

  1. User asks for the time.
  2. Gemini recognises it cannot answer from memory and generates a
     function_call for "time".
  3. Agent dispatches to TimeTool via ToolRegistry.
  4. TimeTool calls datetime.now(tz) — no network, no side-effects.
  5. Result flows back as a function_response.
  6. Gemini composes a natural-language answer.

This round-trip is fully testable with no mocking beyond freezing the
clock.

Timezone handling
-----------------
  - No argument / "local" / "UTC"  ->  system local time or UTC.
  - IANA timezone name             ->  e.g. "Europe/London", "Asia/Tokyo".
  - Common abbreviations           ->  UTC, GMT, EST, PST, CET, IST, JST,
                                       AEST are mapped to IANA names.
  - zoneinfo.ZoneInfo is used (stdlib since Python 3.9); for Python 3.8
    the `backports.zoneinfo` package is required.

Error handling
--------------
  - Unknown timezone name  ->  ToolExecutionError listing how to find
                               valid names.
  - Missing/bad argument   ->  ToolArgumentError.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones
except ImportError:  # Python 3.8 fallback
    from backports.zoneinfo import (  # type: ignore[no-reuse-def]
        ZoneInfo,
        ZoneInfoNotFoundError,
        available_timezones,
    )

from tools.base_tool import BaseTool, ToolArgumentError, ToolExecutionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timezone alias map
# Maps common abbreviations and informal names to IANA identifiers.
# ---------------------------------------------------------------------------
_TZ_ALIASES: dict[str, str] = {
    # UTC variants
    "utc": "UTC",
    "gmt": "GMT",
    "z": "UTC",
    # Americas
    "est": "America/New_York",
    "edt": "America/New_York",
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "ast": "America/Halifax",
    "akt": "America/Anchorage",
    "hst": "Pacific/Honolulu",
    "brt": "America/Sao_Paulo",
    "art": "America/Argentina/Buenos_Aires",
    # Europe
    "wet": "Europe/Lisbon",
    "cet": "Europe/Paris",
    "cest": "Europe/Paris",
    "eet": "Europe/Helsinki",
    "eest": "Europe/Helsinki",
    "msk": "Europe/Moscow",
    "trt": "Europe/Istanbul",
    # Africa
    "wat": "Africa/Lagos",
    "cat": "Africa/Harare",
    "eat": "Africa/Nairobi",
    # Asia
    "ist": "Asia/Kolkata",
    "pkt": "Asia/Karachi",
    "bst": "Asia/Dhaka",
    "ict": "Asia/Bangkok",
    "wib": "Asia/Jakarta",
    "cst_china": "Asia/Shanghai",
    "hkt": "Asia/Hong_Kong",
    "jst": "Asia/Tokyo",
    "kst": "Asia/Seoul",
    "sgt": "Asia/Singapore",
    "tst": "Asia/Taipei",
    # Oceania
    "aest": "Australia/Sydney",
    "aedt": "Australia/Sydney",
    "acst": "Australia/Darwin",
    "awst": "Australia/Perth",
    "nzst": "Pacific/Auckland",
    "nzdt": "Pacific/Auckland",
    # Middle East
    "ast_gulf": "Asia/Dubai",
    "irst": "Asia/Tehran",
    "idt": "Asia/Jerusalem",
    # Informal / common names
    "local": "local",
    "here": "local",
    "mine": "local",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "moscow": "Europe/Moscow",
    "dubai": "Asia/Dubai",
    "mumbai": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "beijing": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai",
    "hong kong": "Asia/Hong_Kong",
    "singapore": "Asia/Singapore",
    "tokyo": "Asia/Tokyo",
    "seoul": "Asia/Seoul",
    "sydney": "Australia/Sydney",
    "auckland": "Pacific/Auckland",
    "new york": "America/New_York",
    "los angeles": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "toronto": "America/Toronto",
    "sao paulo": "America/Sao_Paulo",
    "buenos aires": "America/Argentina/Buenos_Aires",
}


class TimeTool(BaseTool):
    """
    Returns the current date and time for the local timezone or any
    named timezone.

    Accepts IANA names ("Europe/London"), common abbreviations ("JST"),
    and city names ("Tokyo").  With no argument it returns local time.
    """

    @property
    def name(self) -> str:
        return "time"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def execute(self, args: dict) -> str:
        """
        Return the current date/time, optionally in a named timezone.

        Parameters
        ----------
        args : dict
            "timezone" (str, optional) — IANA timezone name, abbreviation,
            or city name.  Defaults to local system time when absent or
            empty.

        Returns
        -------
        str
            Formatted date/time string, e.g.:
            "Current time in Asia/Tokyo (JST):
              Date : Friday, 20 June 2025
              Time : 14:32:07
              UTC offset : +09:00"

        Raises
        ------
        ToolArgumentError
            If the timezone argument has an invalid type.
        ToolExecutionError
            If the timezone name is unrecognised.
        """
        tz_arg = self._extract_tz_arg(args)
        tz, tz_label = self._resolve_timezone(tz_arg)
        now = self._current_time(tz)
        return self._format_output(now, tz_label, tz_arg)

    def get_declaration(self) -> dict:
        return {
            "name": "time",
            "description": (
                "Returns the current date and time. "
                "Accepts an optional timezone as an IANA name "
                "(e.g. 'Europe/London', 'America/New_York'), a common "
                "abbreviation (e.g. 'JST', 'EST', 'CET'), or a city name "
                "(e.g. 'Tokyo', 'Paris', 'New York'). "
                "Returns local system time when no timezone is provided. "
                "Use this whenever the user asks what time or date it is."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": (
                            "Optional timezone identifier. "
                            "Examples: 'UTC', 'Europe/Paris', 'JST', 'Tokyo', "
                            "'America/New_York', 'IST'. "
                            "Omit for local system time."
                        ),
                    },
                },
                "required": [],
            },
        }

    # ------------------------------------------------------------------
    # Argument extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tz_arg(args: dict) -> str:
        """
        Extract and lightly validate the optional 'timezone' argument.

        Returns the raw string (possibly empty) for further resolution.
        Raises ToolArgumentError if the type is wrong.
        """
        raw = args.get("timezone", "")
        if raw is None:
            return ""
        if not isinstance(raw, str):
            raise ToolArgumentError(
                f"'timezone' must be a string, got {type(raw).__name__!r}."
            )
        return raw.strip()

    # ------------------------------------------------------------------
    # Timezone resolution
    # ------------------------------------------------------------------

    def _resolve_timezone(self, tz_arg: str) -> tuple[Optional[ZoneInfo], str]:
        """
        Resolve a timezone argument string to a ZoneInfo object.

        Resolution order:
          1. Empty / "local" / "here"  ->  None (system local time).
          2. Alias lookup (case-insensitive).
          3. Direct ZoneInfo construction (IANA name).
          4. Raise ToolExecutionError for anything unrecognised.

        Returns
        -------
        (tz, label)
            tz    : ZoneInfo | None — None means system local time.
            label : str             — human-readable timezone label.
        """
        if not tz_arg or tz_arg.lower() in ("local", "here", "mine"):
            return None, "local"

        # Step 1: alias lookup.
        alias_key = tz_arg.lower()
        if alias_key in _TZ_ALIASES:
            resolved_name = _TZ_ALIASES[alias_key]
            if resolved_name == "local":
                return None, "local"
            try:
                return ZoneInfo(resolved_name), resolved_name
            except ZoneInfoNotFoundError:
                pass  # fall through to direct lookup

        # Step 2: direct IANA name.
        try:
            zi = ZoneInfo(tz_arg)
            return zi, tz_arg
        except (ZoneInfoNotFoundError, KeyError):
            pass

        # Step 3: case-insensitive IANA scan for close matches.
        lower_arg = tz_arg.lower()
        matches = [tz for tz in available_timezones() if lower_arg in tz.lower()]
        suggestions = sorted(matches)[:5]

        hint = (
            f"  Suggestions: {', '.join(suggestions)}"
            if suggestions
            else (
                "  Use a full IANA name such as 'Europe/London' or "
                "'America/New_York', a city name like 'Tokyo', "
                "or an abbreviation like 'JST'."
            )
        )
        raise ToolExecutionError(f"Unknown timezone: {tz_arg!r}.\n{hint}")

    # ------------------------------------------------------------------
    # Time retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _current_time(tz: Optional[ZoneInfo]) -> datetime:
        """
        Return the current datetime in the given timezone.

        If tz is None, returns the local system time with tzinfo populated
        via astimezone() so the UTC offset is always available for display.
        """
        if tz is None:
            # datetime.now().astimezone() fills in the local tzinfo.
            return datetime.now().astimezone()
        return datetime.now(tz=tz)

    # ------------------------------------------------------------------
    # Output formatter
    # ------------------------------------------------------------------

    @staticmethod
    def _format_output(now: datetime, tz_label: str, original_arg: str) -> str:
        """
        Format the datetime into a structured, human-readable string.

        Includes date, time, UTC offset, and a note when the timezone was
        resolved through an alias (so the user can see the canonical name).
        """
        # Date and time strings.
        date_str = now.strftime("%A, %d %B %Y")  # e.g. "Friday, 20 June 2025"
        time_str = now.strftime("%H:%M:%S")  # e.g. "14:32:07"

        # UTC offset, e.g. "+09:00" or "+00:00".
        utc_offset = now.strftime("%z")  # "+0900"
        if utc_offset:
            offset_fmt = f"{utc_offset[:-2]}:{utc_offset[-2:]}"  # "+09:00"
        else:
            offset_fmt = "N/A"

        # Timezone display name.
        tz_name = now.tzname() or tz_label

        if tz_label == "local":
            location_line = f"Current local time ({tz_name}, UTC {offset_fmt}):"
        else:
            location_line = f"Current time in {tz_label} ({tz_name}, UTC {offset_fmt}):"
            # Append alias resolution note when user supplied an abbreviation
            # or city name that was mapped to a canonical IANA name.
            if (
                original_arg
                and original_arg.lower() != tz_label.lower()
                and original_arg not in (tz_label, "local")
            ):
                location_line += f"  ['{original_arg}' -> '{tz_label}']"

        return (
            f"{location_line}\n"
            f"  Date       : {date_str}\n"
            f"  Time       : {time_str}\n"
            f"  UTC offset : {offset_fmt}"
        )
