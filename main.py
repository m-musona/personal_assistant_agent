"""

Entry point for the Personal Assistant Agent CLI.

Responsibilities
----------------
This module has one job: wire everything together and run the read-eval-print
loop. It is the only place in the codebase that knows about concrete tool
classes. Every other component (Agent, ToolRegistry, PromptBuilder) depends
on abstractions — main.py is the composition root where those abstractions
are bound to real implementations.

Composition root pattern
------------------------
All tool instantiation and registration happens here, in one explicit list,
before the Agent is constructed. The Agent receives a fully populated
ToolRegistry and never imports or names a single tool class.

Startup sequence
----------------
  1. Configure logging (respects LOG_LEVEL from settings).
  2. Instantiate each tool and register it in ToolRegistry.
  3. Construct the Agent with the populated registry.
  4. Print the welcome banner and enter the REPL loop.
  5. Route special commands (/help, /tools, /history, /reset, /quit).
  6. Pass all other input to agent.chat() and print the reply.
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Settings must be imported first — it exits early if GEMINI_API_KEY is unset.
# ---------------------------------------------------------------------------
from config.settings import FILE_READER_BASE_DIR, LOG_DIR, LOG_LEVEL

# ---------------------------------------------------------------------------
# Core components
# ---------------------------------------------------------------------------
from agent.agent import Agent
from tools.tool_registry import ToolRegistry

# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------
from tools.built_in.calculator_tool import CalculatorTool
from tools.built_in.weather_tool import WeatherTool
from tools.built_in.search_tool import SearchTool
from tools.built_in.time_tool import TimeTool

# ---------------------------------------------------------------------------
# Custom tools
# ---------------------------------------------------------------------------
from tools.custom.translate_tool import TranslateTool
from tools.custom.file_reader_tool import FileReaderTool

# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------
from observers.logger_observer import LoggerObserver


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """
    Set up the root logger for the session.

    Log level is read from LOG_LEVEL in settings (default INFO).
    A file handler writes to LOG_DIR/session.log alongside the console
    handler so graders can inspect the full tool-call trace.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "session.log")

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def build_registry() -> ToolRegistry:
    """
    Instantiate every tool and register it in a fresh ToolRegistry.

    This is the composition root for tools: it is the ONLY place in the
    entire codebase that names concrete tool classes. The Agent, ToolRegistry,
    and PromptBuilder all depend on the BaseTool abstraction only.

    To add a new tool:
      1. Import its class at the top of this file.
      2. Add one line here: registry.register(MyNewTool())
      Nothing else changes.

    Returns
    -------
    ToolRegistry
        Fully populated registry ready to be passed to Agent().
    """
    registry = ToolRegistry()

    tools = [
        # ── Built-in (4 required) ────────────────────────────────────────
        CalculatorTool(),  # arithmetic & math expressions
        WeatherTool(),  # current weather by city
        SearchTool(),  # Wikipedia + DuckDuckGo search
        TimeTool(),  # current date and time by timezone
        # ── Custom (2 required) ──────────────────────────────────────────
        TranslateTool(),  # text translation (MyMemory + LibreTranslate)
        FileReaderTool(),  # local plain-text file reader (sandboxed)
    ]

    for tool in tools:
        registry.register(tool)
        logger.debug("Registered tool: %r", tool.name)

    logger.info(
        "ToolRegistry ready — %d tool(s): %s",
        len(registry),
        ", ".join(registry.tool_names()),
    )
    return registry


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

_BANNER = """
╔══════════════════════════════════════════════════════╗
║          Personal Assistant Agent  (Gemini)          ║
║  Type your message, or /help for available commands  ║
╚══════════════════════════════════════════════════════╝
""".strip()

_HELP_TEXT = """
Available commands
──────────────────
  /help      Show this help message
  /tools     List all registered tools with their descriptions
  /history   Show the current conversation history summary
  /reset     Clear history and start a fresh conversation
  /quit      Exit the assistant  (also Ctrl+C or Ctrl+D)

Anything else is sent to the assistant as a message.
""".strip()


def _print_tools(registry: ToolRegistry) -> None:
    """Print a formatted table of every registered tool."""
    print(f"\n{len(registry)} tool(s) available:\n")
    for tool in registry:
        decl = tool.get_declaration()
        name = decl.get("name", tool.name)
        desc = decl.get("description", "(no description)")
        # Wrap long descriptions to 60 chars with a hanging indent.
        wrapped = textwrap.fill(desc, width=60, subsequent_indent="           ")
        print(f"  {name:<12} {wrapped}")
    print()


def _print_history(agent: Agent) -> None:
    """Print a brief summary of the current conversation memory."""
    mem = agent.memory
    print(f"\nMemory: {mem.summary()}")
    if not mem.is_empty():
        last = mem.last_turn()
        role = last.get("role", "?")
        parts = last.get("parts", [{}])
        text = parts[0].get("text", "(structured part)") if parts else ""
        preview = text[:120] + ("…" if len(text) > 120 else "")
        print(f"Last turn [{role}]: {preview}")
    print()


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


def run_repl(agent: Agent, registry: ToolRegistry) -> None:
    """
    Run the read-eval-print loop until the user quits.

    Routing:
      /help, /tools, /history, /reset, /quit — handled locally.
      Everything else — forwarded to agent.chat().
    """
    print(_BANNER)
    print(f"\nBase directory for file reading: {FILE_READER_BASE_DIR}\n" "─" * 56)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Special commands ─────────────────────────────────────────────
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break

        if user_input.lower() in ("/help", "/?"):
            print("\n" + _HELP_TEXT + "\n")
            continue

        if user_input.lower() in ("/tools", "/t"):
            _print_tools(registry)
            continue

        if user_input.lower() in ("/history", "/h"):
            _print_history(agent)
            continue

        if user_input.lower() in ("/reset", "/r"):
            agent.reset()
            print("\nConversation history cleared. Starting fresh.\n")
            continue

        # ── Normal message ───────────────────────────────────────────────
        print("\nAssistant: ", end="", flush=True)
        reply = agent.chat(user_input)
        print(reply)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Configure logging, build the registry, construct the agent, start the REPL.
    """
    _configure_logging()
    logger.info("Starting Personal Assistant Agent.")

    registry = build_registry()
    agent = Agent(registry)

    # Attach the LoggerObserver — writes timestamped entries to logs/session.log.
    agent.add_observer(LoggerObserver())

    run_repl(agent, registry)
    logger.info("Session ended.")


if __name__ == "__main__":
    main()
