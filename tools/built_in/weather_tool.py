"""
tools/built_in/weather_tool.py

WeatherTool — fetches current weather for a city using wttr.in (no API key
required) with an optional fallback to OpenWeatherMap when
OPENWEATHER_API_KEY is set in settings.

Backend selection
-----------------
wttr.in (default, always available)
    Free, no key needed. Returns a compact JSON payload with temperature,
    weather description, humidity, wind speed, and "feels like".
    URL: https://wttr.in/{city}?format=j1

OpenWeatherMap (optional, richer data)
    Activated automatically when OPENWEATHER_API_KEY is set.
    Returns the same normalised dict as wttr.in so the rest of the
    system never needs to know which backend was used.
    URL: https://api.openweathermap.org/data/2.5/weather

Error handling
--------------
City not found (404 / empty wttr.in result)
    -> ToolExecutionError with a helpful suggestion to try a different name.
Network / timeout
    -> ToolExecutionError with a retry suggestion.
Malformed response
    -> ToolExecutionError explaining the upstream issue.
Missing argument
    -> ToolArgumentError.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from config.settings import OPENWEATHER_API_KEY, WEATHER_UNITS
from tools.base_tool import BaseTool, ToolArgumentError, ToolExecutionError

logger = logging.getLogger(__name__)

# Seconds to wait for a response before giving up.
_REQUEST_TIMEOUT: int = 10

# wttr.in unit flag per WEATHER_UNITS setting.
_WTTR_UNIT_FLAG: dict[str, str] = {
    "metric": "m",  # Celsius
    "imperial": "u",  # Fahrenheit
    "standard": "m",  # Kelvin not supported by wttr.in; fall back to Celsius
}

# Degree symbol + unit label per WEATHER_UNITS.
_UNIT_LABEL: dict[str, str] = {
    "metric": "°C",
    "imperial": "°F",
    "standard": "K",
}


class WeatherTool(BaseTool):
    """
    Returns current weather conditions for a given city.

    Uses wttr.in by default (no setup required). Automatically switches to
    OpenWeatherMap when OPENWEATHER_API_KEY is present in settings.

    Output format (plain string returned to the agent):
        "Weather in Paris, France:
         Temperature: 18°C (feels like 16°C)
         Condition:   Partly cloudy
         Humidity:    72%
         Wind:        14 km/h"
    """

    @property
    def name(self) -> str:
        """Return the unique tool identifier used by ToolRegistry."""
        return "weather"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def execute(self, args: dict) -> str:
        """
        Fetch weather for the city named in args["city"].

        Parameters
        ----------
        args : dict
            Must contain "city" (str): name of the city to look up.
            Optionally "country" (str): ISO country code to disambiguate
            cities with common names (e.g. "Springfield", "US").

        Returns
        -------
        str
            Formatted weather summary ready to be shown to the user.

        Raises
        ------
        ToolArgumentError
            If "city" is missing or empty.
        ToolExecutionError
            If the city is not found, the network is unreachable, or the
            upstream API returns an unexpected response.
        """
        city, country = self._extract_args(args)
        location = f"{city}, {country}" if country else city

        if OPENWEATHER_API_KEY:
            logger.debug("Using OpenWeatherMap backend for %r", location)
            data = self._fetch_openweathermap(city, country)
        else:
            logger.debug("Using wttr.in backend for %r", location)
            data = self._fetch_wttr(city, country)

        return self._format_weather(location, data)

    def get_declaration(self) -> dict:
        """Return the Gemini function-calling schema for this tool."""
        return {
            "name": "weather",
            "description": (
                "Returns the current weather conditions for a city, including "
                "temperature, feels-like temperature, weather description, "
                "humidity, and wind speed. Use this whenever the user asks "
                "about weather, temperature, or climate in a specific place."
            ),
            "parameters": {
                # "type": "object",
                "type": "OBJECT",
                "properties": {
                    "city": {
                        # "type": "string",
                        "type": "STRING",
                        "description": (
                            "Name of the city to get weather for. "
                            "Use the most common English name "
                            "(e.g. 'London', 'New York', 'Tokyo'). "
                            "Avoid abbreviations."
                        ),
                    },
                    "country": {
                        # "type": "string",
                        "type": "STRING",
                        "description": (
                            "Optional ISO 3166-1 alpha-2 country code to "
                            "disambiguate cities with common names "
                            "(e.g. 'US', 'GB', 'FR'). Leave empty if the "
                            "city name is unambiguous."
                        ),
                    },
                },
                "required": ["city"],
            },
        }

    # ------------------------------------------------------------------
    # Argument extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_args(args: dict) -> tuple[str, str]:
        """Return (city, country), raising ToolArgumentError if city is absent."""
        city = args.get("city")
        if city is None:
            raise ToolArgumentError(
                "Missing required argument: 'city'. "
                "Provide a city name such as 'London' or 'Tokyo'."
            )
        if not isinstance(city, str):
            raise ToolArgumentError(
                f"'city' must be a string, got {type(city).__name__!r}."
            )
        city = city.strip()
        if not city:
            raise ToolArgumentError(
                "'city' must not be empty. "
                "Provide a city name such as 'London' or 'Tokyo'."
            )
        country = str(args.get("country", "")).strip()
        return city, country

    # ------------------------------------------------------------------
    # wttr.in backend
    # ------------------------------------------------------------------

    def _fetch_wttr(self, city: str, country: str) -> dict[str, Any]:
        """
        Fetch weather from wttr.in JSON API (no key required).

        Returns a normalised dict:
            {
                "temp":       float,   # current temperature
                "feels_like": float,   # apparent temperature
                "condition":  str,     # human description
                "humidity":   int,     # percent
                "wind_kmh":   float,   # wind speed in km/h
                "location":   str,     # resolved city name from API
            }
        """
        query = urllib.parse.quote_plus(f"{city},{country}" if country else city)
        unit_flag = _WTTR_UNIT_FLAG.get(WEATHER_UNITS, "m")
        url = f"https://wttr.in/{query}?format=j1&{unit_flag}"

        logger.debug("wttr.in request: %s", url)
        raw = self._http_get(url, city)

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                f"Received an unexpected response from the weather service "
                f"for '{city}'. Please try again later."
            ) from exc

        # wttr.in returns {"nearest_area": [...]} even for unknown locations,
        # but "current_condition" will be empty for genuinely bad queries.
        current = payload.get("current_condition", [])
        if not current:
            raise ToolExecutionError(
                f"No weather data found for '{city}'. "
                "Check the spelling or try adding a country code "
                "(e.g. 'Springfield, US')."
            )

        cond = current[0]

        # Resolved location name from wttr.in.
        nearest = payload.get("nearest_area", [{}])[0]
        area_name = nearest.get("areaName", [{}])[0].get("value", city)
        country_name = nearest.get("country", [{}])[0].get("value", "")
        resolved = f"{area_name}, {country_name}" if country_name else area_name

        # wttr.in expresses temperature in the requested unit.
        temp_key = "temp_C" if unit_flag == "m" else "temp_F"
        feels_key = "FeelsLikeC" if unit_flag == "m" else "FeelsLikeF"

        try:
            return {
                "temp": float(cond.get(temp_key, cond.get("temp_C", 0))),
                "feels_like": float(cond.get(feels_key, cond.get("FeelsLikeC", 0))),
                "condition": cond.get("weatherDesc", [{}])[0].get("value", "Unknown"),
                "humidity": int(cond.get("humidity", 0)),
                "wind_kmh": float(cond.get("windspeedKmph", 0)),
                "location": resolved,
            }
        except (KeyError, IndexError, ValueError) as exc:
            raise ToolExecutionError(
                f"Could not parse weather data for '{city}': {exc}. "
                "Please try again later."
            ) from exc

    # ------------------------------------------------------------------
    # OpenWeatherMap backend
    # ------------------------------------------------------------------

    def _fetch_openweathermap(self, city: str, country: str) -> dict[str, Any]:
        """
        Fetch weather from OpenWeatherMap current-weather endpoint.

        Returns the same normalised dict shape as _fetch_wttr() so the
        formatter and tests are backend-agnostic.
        """
        query = f"{city},{country}" if country else city
        units = WEATHER_UNITS if WEATHER_UNITS in ("metric", "imperial") else "metric"
        params = urllib.parse.urlencode(
            {
                "q": query,
                "appid": OPENWEATHER_API_KEY,
                "units": units,
            }
        )
        url = f"https://api.openweathermap.org/data/2.5/weather?{params}"

        logger.debug("OpenWeatherMap request for %r", query)
        raw = self._http_get(url, city)

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                f"Received an unexpected response from OpenWeatherMap " f"for '{city}'."
            ) from exc

        # OWM returns {"cod": 404, "message": "city not found"} on bad input.
        cod = str(payload.get("cod", "200"))
        if cod == "404":
            raise ToolExecutionError(
                f"City '{city}' was not found. "
                "Check the spelling or try a nearby larger city."
            )
        if cod not in ("200", ""):
            message = payload.get("message", "unknown error")
            raise ToolExecutionError(
                f"OpenWeatherMap returned an error for '{city}': {message}."
            )

        try:
            main = payload["main"]
            wind = payload.get("wind", {})
            weather_list = payload.get("weather", [{}])
            name = payload.get("name", city)
            sys = payload.get("sys", {})
            country_code = sys.get("country", "")
            resolved = f"{name}, {country_code}" if country_code else name

            unit_label = _UNIT_LABEL.get(units, "°C")
            wind_speed_ms = float(wind.get("speed", 0))
            wind_kmh = round(wind_speed_ms * 3.6, 1)

            return {
                "temp": float(main["temp"]),
                "feels_like": float(main.get("feels_like", main["temp"])),
                "condition": weather_list[0].get("description", "Unknown").capitalize(),
                "humidity": int(main.get("humidity", 0)),
                "wind_kmh": wind_kmh,
                "location": resolved,
            }
        except (KeyError, IndexError, ValueError) as exc:
            raise ToolExecutionError(
                f"Could not parse OpenWeatherMap response for '{city}': {exc}."
            ) from exc

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _http_get(self, url: str, city: str) -> str:
        """
        Perform a GET request and return the response body as a string.

        Raises ToolExecutionError for:
          - HTTP 404 (city not found on wttr.in)
          - Other HTTP errors
          - Network / timeout errors
        """
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "PersonalAssistantAgent/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                return resp.read().decode("utf-8")

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ToolExecutionError(
                    f"City '{city}' was not found. "
                    "Check the spelling, use the English name, or add a "
                    "country code (e.g. 'Paris, FR')."
                ) from exc
            raise ToolExecutionError(
                f"The weather service returned HTTP {exc.code} for '{city}'. "
                "Please try again later."
            ) from exc

        except urllib.error.URLError as exc:
            raise ToolExecutionError(
                f"Could not reach the weather service for '{city}': {exc.reason}. "
                "Check your internet connection and try again."
            ) from exc

        except TimeoutError as exc:
            raise ToolExecutionError(
                f"The weather service timed out for '{city}'. "
                "Please try again in a moment."
            ) from exc

    # ------------------------------------------------------------------
    # Formatter
    # ------------------------------------------------------------------

    @staticmethod
    def _format_weather(requested_location: str, data: dict[str, Any]) -> str:
        """
        Render the normalised weather dict into a clean human-readable string.

        Uses the resolved location name from the API when available,
        falling back to the user-supplied name.
        """
        unit = _UNIT_LABEL.get(WEATHER_UNITS, "°C")
        resolved = data.get("location") or requested_location

        temp = data["temp"]
        feels_like = data["feels_like"]
        condition = data["condition"]
        humidity = data["humidity"]
        wind_kmh = data["wind_kmh"]

        # Format temperatures as integers when they are whole numbers.
        def fmt_temp(t: float) -> str:
            """Format a temperature float, omitting the decimal for whole values."""
            return f"{int(t)}{unit}" if t == int(t) else f"{t:.1f}{unit}"

        return (
            f"Weather in {resolved}:\n"
            f"  Temperature : {fmt_temp(temp)} (feels like {fmt_temp(feels_like)})\n"
            f"  Condition   : {condition}\n"
            f"  Humidity    : {humidity}%\n"
            f"  Wind        : {wind_kmh:.0f} km/h"
        )
