"""
main.py

Entry point and CLI for the Personal Assistant Agent.

Responsibilities
----------------
This is the composition root — the only file in the project that:
  - names concrete tool classes and registers them
  - names the concrete observer and attaches it
  - owns the read-eval-print loop

Every other module depends only on abstractions (BaseTool, BaseObserver,
ToolRegistry). Adding a new tool or observer is a one-line change here.

Startup sequence
----------------
  1. Configure Python logging (level from LOG_LEVEL, file + stderr handlers).
  2. Run a pre-flight check (Python version, API key present).
  3. Instantiate and register all tools into ToolRegistry.
  4. Construct the Agent with the populated registry.
  5. Attach LoggerObserver for structured session logging.
  6. Print the welcome banner and enter the REPL.

REPL input handling
-------------------
  /help     — print command reference
  /tools    — list registered tools and their descriptions
  /history  — summarise conversation memory
  /reset    — wipe history, start a new session
  /quit     — exit cleanly (also Ctrl+C or Ctrl+D at any point)
  <blank>   — ignored, re-prompt
  anything else — forwarded to agent.chat(); reply is printed

Interrupt handling
------------------
KeyboardInterrupt is caught in two places:
  1. At the input() call — user pressed Ctrl+C while typing.
  2. Around agent.chat() — user pressed Ctrl+C during a long API call.
Both lead to a clean "Goodbye!" message and graceful shutdown.
EOFError (Ctrl+D, piped input exhausted) is treated identically.
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Settings — imported first; exits immediately if GEMINI_API_KEY is unset.
# ---------------------------------------------------------------------------
from config.settings import FILE_READER_BASE_DIR, LOG_DIR, LOG_LEVEL, MODEL_NAME

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
from agent.agent import Agent
from tools.tool_registry import ToolRegistry

# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------
from tools.built_in.calculator_tool import CalculatorTool
from tools.built_in.search_tool import SearchTool
from tools.built_in.time_tool import TimeTool
from tools.built_in.weather_tool import WeatherTool

# ---------------------------------------------------------------------------
# Custom tools
# ---------------------------------------------------------------------------
from tools.custom.file_reader_tool import FileReaderTool
from tools.custom.translate_tool import TranslateTool

# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------
from observers.logger_observer import LoggerObserver


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_PYTHON = (3, 9)  # zoneinfo requires 3.9+

_DIVIDER = "─" * 56
_WIDE_DIVIDER = "═" * 56

_BANNER = f"""\
{_WIDE_DIVIDER}
  Personal Assistant Agent  ·  Gemini {MODEL_NAME}
  Type /help for commands, or just start chatting.
{_WIDE_DIVIDER}"""

_HELP = f"""\

Commands
{_DIVIDER}
  /help      Show this message
  /tools     List all available tools
  /history   Show conversation memory summary
  /reset     Clear history and start fresh
  /quit      Exit  (also Ctrl+C or Ctrl+D)
{_DIVIDER}
Anything else is sent to the assistant.
"""

# Special command aliases
_CMD_QUIT = {"/quit", "/exit", "/q"}
_CMD_HELP = {"/help", "/?", "/h"}
_CMD_TOOLS = {"/tools", "/t"}
_CMD_HISTORY = {"/history", "/hist"}
_CMD_RESET = {"/reset", "/r", "/clear"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """
    Configure the Python root logger.

    Writes to both stderr and LOG_DIR/agent.log.
    Log level is read from LOG_LEVEL in settings (default INFO).
    Third-party noise is suppressed to WARNING.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "agent.log")

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )

    # Suppress chatty third-party loggers.
    for noisy in ("urllib3", "google", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def _preflight() -> None:
    """
    Abort with a clear message if the runtime environment is not suitable.

    Checks:
      - Python version ≥ _MIN_PYTHON (zoneinfo requires 3.9+)

    Note: GEMINI_API_KEY presence is validated by config/settings.py on
    import — if it is missing the process has already exited.
    """
    if sys.version_info < _MIN_PYTHON:
        print(
            f"ERROR: Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ is required. "
            f"You are running {sys.version}.",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Tool registration  (composition root)
# ---------------------------------------------------------------------------


def build_registry() -> ToolRegistry:
    """
    Instantiate every tool and register it with a fresh ToolRegistry.

    This is the only place in the codebase that names concrete tool classes.
    Agent, PromptBuilder, and ToolRegistry all depend on BaseTool only.

    To add a new tool:
      1. Import its class at the top of this file.
      2. Add  MyNewTool()  to the list below.
      That is all — nothing else in the codebase changes.

    Returns
    -------
    ToolRegistry
        Fully populated registry, ready for Agent().
    """
    registry = ToolRegistry()

    all_tools = [
        # ── Built-in (4 required by assignment) ──────────────────────────
        CalculatorTool(),  # safe arithmetic + math functions
        WeatherTool(),  # current weather via wttr.in / OpenWeatherMap
        SearchTool(),  # Wikipedia + DuckDuckGo instant answers
        TimeTool(),  # current date/time with timezone support
        # ── Custom (2 required by assignment) ────────────────────────────
        TranslateTool(),  # text translation via MyMemory / LibreTranslate
        FileReaderTool(),  # sandboxed local plain-text file reader
    ]

    for tool in all_tools:
        registry.register(tool)
        logger.debug("Registered tool: %r", tool.name)

    logger.info(
        "Registry ready — %d tools: %s",
        len(registry),
        ", ".join(registry.tool_names()),
    )
    return registry


# ---------------------------------------------------------------------------
# CLI display helpers
# ---------------------------------------------------------------------------


def _print_tools(registry: ToolRegistry) -> None:
    """Print every registered tool with its name and description."""
    print(f"\n{len(registry)} tool(s) registered:\n")
    for tool in registry:
        decl = tool.get_declaration()
        name = decl.get("name", tool.name)
        desc = decl.get("description", "(no description)")
        # Wrap to 60 chars with a hanging indent aligned under the desc.
        wrapped = textwrap.fill(desc, width=60, subsequent_indent=" " * 15)
        print(f"  {name:<12}  {wrapped}")
    print()


def _print_history(agent: Agent) -> None:
    """Print a compact summary of the current conversation memory."""
    mem = agent.memory
    print(f"\n{mem.summary()}")

    if not mem.is_empty():
        last = mem.last_turn()
        role = last.get("role", "?")
        parts = last.get("parts", [{}])
        # parts[0] may be a text part or a structured part (function call).
        raw_text = parts[0].get("text", "") if parts else ""
        if raw_text:
            preview = raw_text[:120] + ("…" if len(raw_text) > 120 else "")
            print(f"Last [{role}]: {preview}")
        else:
            print(f"Last [{role}]: (structured turn — function call or response)")
    print()


def _print_reset_confirmation() -> None:
    """Print a bordered confirmation message after a session reset."""
    print(f"\n{_DIVIDER}")
    print("  History cleared. Starting a fresh conversation.")
    print(f"{_DIVIDER}\n")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


def run_repl(agent: Agent, registry: ToolRegistry) -> None:
    """
    Run the interactive command-line loop until the user quits.

    This function never raises — all exceptions from agent.chat() are caught
    and displayed gracefully so the loop can continue. KeyboardInterrupt and
    EOFError are caught at both the input() and the chat() call sites.

    Parameters
    ----------
    agent    : Agent         — fully initialised assistant
    registry : ToolRegistry  — used only for /tools display
    """
    print(_BANNER)
    print(f"\nFile reader sandbox: {FILE_READER_BASE_DIR}")
    print(f"Session log:         {os.path.join(LOG_DIR, 'session.log')}")
    print(f"{_DIVIDER}\n")

    while True:
        # ── Read ────────────────────────────────────────────────────────
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            # Ctrl+C while the cursor is at the input prompt.
            print("\n\nKeyboard interrupt received. Goodbye!")
            break
        except EOFError:
            # Ctrl+D, or stdin was piped and is now exhausted.
            print("\nGoodbye!")
            break

        # Blank line — re-prompt silently.
        if not user_input:
            continue

        cmd = user_input.lower()

        # ── Commands ────────────────────────────────────────────────────
        if cmd in _CMD_QUIT:
            print("Goodbye!")
            break

        if cmd in _CMD_HELP:
            print(_HELP)
            continue

        if cmd in _CMD_TOOLS:
            _print_tools(registry)
            continue

        if cmd in _CMD_HISTORY:
            _print_history(agent)
            continue

        if cmd in _CMD_RESET:
            agent.reset()
            _print_reset_confirmation()
            continue

        # ── Eval + Print ─────────────────────────────────────────────────
        print("\nAssistant:", end=" ", flush=True)
        try:
            reply = agent.chat(user_input)
        except KeyboardInterrupt:
            # Ctrl+C while the agent is waiting for the API response.
            print("\n\n[Request cancelled by user.]\n")
            continue
        except Exception as exc:
            # Unexpected error that escaped agent.chat()'s own handler.
            # Print a friendly message and keep the loop alive.
            logger.exception("Unexpected error escaping agent.chat().")
            print(f"\n[Error: {exc}]\n")
            continue

        print(reply)
        print()  # blank line separates turns visually


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def _shutdown(agent: Agent) -> None:
    """
    Perform graceful cleanup before the process exits.

    Closes the LoggerObserver (flushes and writes a session-end marker) and
    logs the shutdown event. Errors here are swallowed — cleanup must not
    prevent the process from exiting.
    """
    try:
        for obs in agent._observers:  # access private list for cleanup
            if hasattr(obs, "close"):
                obs.close()
    except Exception:
        pass
    logger.info("Agent shut down cleanly.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Full startup sequence:
      preflight → logging → registry → agent → observer → REPL → shutdown.
    """
    _preflight()
    _configure_logging()
    logger.info("Starting Personal Assistant Agent.")

    registry = build_registry()
    agent = Agent(registry)

    # Structured session log — one file per process run, sessions separated
    # by markers.  The observer is attached after Agent.__init__ so
    # on_agent_start fires with the correct tool list.
    observer = LoggerObserver()
    agent.add_observer(observer)
    # Manually fire on_agent_start since it was registered after __init__.
    observer.on_agent_start(registry.tool_names())

    try:
        run_repl(agent, registry)
    finally:
        # Always runs — whether the user typed /quit, Ctrl+C, or the
        # process is killed. Ensures the log is flushed and closed.
        _shutdown(agent)

    logger.info("Session ended.")


if __name__ == "__main__":
    main()
