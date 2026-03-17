"""

Central configuration for the Personal Assistant Agent.
All constants, API credentials, and tunable parameters live here.
Import from this module instead of hardcoding values anywhere else.
"""

import os
import sys


# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    print(
        "[settings] ERROR: GEMINI_API_KEY environment variable is not set.\n"
        "  Set it with:  export GEMINI_API_KEY='your_key_here'  (Linux/macOS)\n"
        "             or  set GEMINI_API_KEY=your_key_here       (Windows)\n"
        "  Or place it in a .env file and load it with python-dotenv.",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Gemini model identifier.
# Swap to "gemini-1.5-pro" for a more capable (but slower) model.
MODEL_NAME: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Maximum number of output tokens the model may generate per turn.
MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))

# Sampling temperature: 0.0 = deterministic, 1.0 = creative.
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.4"))


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

# Maximum number of conversation turns (user + assistant pairs) kept in
# MemoryManager before the oldest turns are dropped.
# Increase for longer context; decrease to stay within token limits.
MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "20"))


# ---------------------------------------------------------------------------
# Agent behaviour
# ---------------------------------------------------------------------------

# Maximum times the agent may call a tool in a single user request before
# it is forced to return a final answer. Prevents infinite ReAct loops.
MAX_TOOL_CALLS_PER_TURN: int = int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "5"))

# System prompt injected at the start of every conversation.
SYSTEM_PROMPT: str = (
    "You are a helpful and knowledgeable personal assistant. "
    "You have access to a set of tools — use them whenever they would give "
    "a more accurate or up-to-date answer than your own knowledge. "
    "Think step by step, be concise, and always respond in the same "
    "language the user writes in."
)


# ---------------------------------------------------------------------------
# Tool-specific settings
# ---------------------------------------------------------------------------

# Weather tool — OpenWeatherMap API key (optional; falls back to wttr.in).
OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")

# Weather tool — unit system: "metric" (°C), "imperial" (°F), or "standard" (K).
WEATHER_UNITS: str = os.getenv("WEATHER_UNITS", "metric")

# FileReaderTool — restrict file reads to this directory.
# Defaults to the current working directory; set to an absolute path to lock
# the agent to a specific folder.
FILE_READER_BASE_DIR: str = os.getenv("FILE_READER_BASE_DIR", os.getcwd())

# TranslateTool — LibreTranslate endpoint (self-hosted or public mirror).
LIBRETRANSLATE_URL: str = os.getenv(
    "LIBRETRANSLATE_URL", "https://libretranslate.com/translate"
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Directory where LoggerObserver writes session logs.
LOG_DIR: str = os.getenv("LOG_DIR", "logs")

# Logging level for the standard Python logger: DEBUG, INFO, WARNING, ERROR.
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# ---------------------------------------------------------------------------
# Sanity-check helper (used in tests)
# ---------------------------------------------------------------------------


def get_all() -> dict:
    """Return all settings as a plain dict — useful for debugging and tests."""
    return {
        "MODEL_NAME": MODEL_NAME,
        "MAX_OUTPUT_TOKENS": MAX_OUTPUT_TOKENS,
        "TEMPERATURE": TEMPERATURE,
        "MAX_HISTORY_TURNS": MAX_HISTORY_TURNS,
        "MAX_TOOL_CALLS_PER_TURN": MAX_TOOL_CALLS_PER_TURN,
        "WEATHER_UNITS": WEATHER_UNITS,
        "FILE_READER_BASE_DIR": FILE_READER_BASE_DIR,
        "LIBRETRANSLATE_URL": LIBRETRANSLATE_URL,
        "LOG_DIR": LOG_DIR,
        "LOG_LEVEL": LOG_LEVEL,
        "GEMINI_API_KEY_SET": bool(GEMINI_API_KEY),
        "OPENWEATHER_API_KEY_SET": bool(OPENWEATHER_API_KEY),
    }
