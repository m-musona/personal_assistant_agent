# Personal Assistant Agent

An adaptive AI assistant built on the **Google Gemini API** that autonomously decides when to answer from its own knowledge and when to call an external tool. The project demonstrates a production-quality software architecture using **SOLID principles** and **Gang of Four design patterns** throughout.

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Project structure](#2-project-structure)
3. [Architecture overview](#3-architecture-overview)
4. [Design patterns](#4-design-patterns)
5. [Available tools](#5-available-tools)
6. [Configuration reference](#6-configuration-reference)
7. [Running the tests](#7-running-the-tests)
8. [CLI commands](#8-cli-commands)
9. [Adding a new tool](#9-adding-a-new-tool)
10. [Assignment criteria mapping](#10-assignment-criteria-mapping)

---

## 1. Quick start

**Prerequisites:** Python 3.9+, a [Google AI Studio](https://aistudio.google.com/) API key.

```bash
# 1. Install dependencies
pip install google-generativeai requests

# 2. Set your API key
export GEMINI_API_KEY="your_key_here"        # Linux / macOS
set GEMINI_API_KEY=your_key_here             # Windows CMD

# 3. Run
python main.py
```

**Optional `.env` file** (copy from `.env.example`):

```
GEMINI_API_KEY=your_key_here
OPENWEATHER_API_KEY=optional_for_richer_weather_data
WEATHER_UNITS=metric
LOG_LEVEL=INFO
```

Load with `python-dotenv`: add `from dotenv import load_dotenv; load_dotenv()` at the top of `main.py`.

---

## 2. Project structure

```
personal_assistant_agent/
│
├── main.py                        <- CLI entry point & composition root
│
├── agent/
│   ├── agent.py                   <- ReAct loop orchestrator
│   ├── memory_manager.py          <- Conversation history (SRP)
│   └── prompt_builder.py          <- System prompt assembly (SRP)
│
├── tools/
│   ├── base_tool.py               <- Abstract BaseTool interface (DIP/OCP)
│   ├── tool_registry.py           <- Factory/Registry pattern
│   ├── built_in/
│   │   ├── calculator_tool.py     <- Safe AST-validated arithmetic
│   │   ├── weather_tool.py        <- wttr.in / OpenWeatherMap
│   │   ├── search_tool.py         <- Wikipedia + DuckDuckGo
│   │   └── time_tool.py           <- Current date/time with timezone
│   └── custom/
│       ├── translate_tool.py      <- MyMemory + LibreTranslate  [custom #1]
│       └── file_reader_tool.py    <- Sandboxed file reader      [custom #2]
│
├── observers/
│   ├── base_observer.py           <- Abstract BaseObserver interface
│   └── logger_observer.py        <- Structured session log writer
│
├── config/
│   └── settings.py                <- All constants, loaded from env vars
│
├── tests/                         <- 311 tests, 0 failures
│   ├── test_calculator_tool.py
│   ├── test_weather_tool.py
│   ├── test_search_tool.py
│   ├── test_time_tool.py
│   ├── test_translate_tool.py
│   ├── test_file_reader_tool.py
│   ├── test_logger_observer.py
│   └── test_error_handling.py
│
├── logs/
│   ├── agent.log                  <- Python logging output
│   └── session.log                <- LoggerObserver structured log
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 3. Architecture overview

```
+------------------------------------------------------------------+
|  main.py  --  Composition Root                                   |
|  Instantiates all concrete classes; wires registry & observer    |
+------------------+-----------------------------------------------+
                   | creates & passes ToolRegistry
                   v
+------------------------------------------------------------------+
|  Agent                                                           |
|  +------------------+  +------------------+  +---------------+  |
|  |  MemoryManager   |  |  PromptBuilder   |  | ToolRegistry  |  |
|  |  (conversation   |  |  (system prompt  |  | (Factory --   |  |
|  |   history)       |  |   assembly)      |  |  dispatch)    |  |
|  +------------------+  +------------------+  +-------+-------+  |
|                                                       |          |
|  +----------------------------------------------------+-------+  |
|  |  ReAct Loop  (Reason -> Act -> Observe)            |       |  |
|  |                                                    |       |  |
|  |  1. send_message(user_input or fn_response)        |       |  |
|  |  2. inspect response parts:                        |       |  |
|  |       function_call? -> ToolRegistry.execute() <---+       |  |
|  |       text?          -> return reply to caller             |  |
|  |  3. record every step in MemoryManager                     |  |
|  |  4. notify all registered observers                        |  |
|  +------------------------------------------------------------+  |
|                                                                  |
|  +------------------------------------------------------------+  |
|  |  Observers  (Observer pattern)                             |  |
|  |  BaseObserver <-- LoggerObserver  (logs/session.log)      |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
                            ^
                            | HTTPS
                     Gemini API (external)
```

### ReAct loop flow

```
User input
    |
    v
[REASON]  session.send_message(message)
    |
    +-- response has function_call?
    |         |
    |         v  YES
    |     [ACT]  ToolRegistry.execute(name, args)
    |         |
    |         v
    |     [OBSERVE]  build function_response dict
    |                store in MemoryManager
    |                notify observers.on_tool_call()
    |                |
    |                +----------> loop back to REASON
    |
    +-- response has text?
              |
              v  YES
          add_turn("model", reply)
          notify observers.on_response()
          return reply to CLI
```

---

## 4. Design patterns

### Strategy + Factory/Registry

Every tool is a concrete `BaseTool` subclass. `ToolRegistry` maps names to instances and dispatches all calls by name. The `Agent` never imports or names a single tool class.

```python
# main.py -- the ONE place that names concrete classes
registry.register(CalculatorTool())
registry.register(WeatherTool())
# Adding a new tool: one line here, nothing else changes.
```

The tool call chain inside the agent loop:

```
Agent._react_loop()
  -> Agent._dispatch_tool(name, args)
    -> ToolRegistry.execute(name, args)   # Factory dispatch
      -> BaseTool.execute(args)           # Strategy execution
```

### Observer

The `Agent` holds a `list[BaseObserver]` and fires six lifecycle notifications. Adding a token counter, UI updater, or test spy requires zero changes to `Agent` -- just subclass `BaseObserver` and call `agent.add_observer()`.

```
Lifecycle events fired by Agent:
  on_agent_start(tool_names)         -- once, after __init__
  on_turn_start(user_input)          -- start of each user turn
  on_tool_call(name, args, result)   -- after every tool execution
  on_response(text)                  -- final reply for each turn
  on_error(error, context)           -- every handled error
  on_agent_reset()                   -- on session reset
```

### SOLID principles

| Principle | Where it is demonstrated                                                                                                                                         |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SRP**   | `MemoryManager` stores history only. `PromptBuilder` builds prompts only. `LoggerObserver` logs only. Each class has exactly one reason to change.               |
| **OCP**   | Adding a new tool touches only `main.py` (one `register()` call). `Agent`, `ToolRegistry`, and `PromptBuilder` are closed for modification.                      |
| **LSP**   | Any `BaseTool` / `BaseObserver` subclass substitutes into the system transparently -- the agent loop never needs to know which concrete type it is working with. |
| **ISP**   | `BaseObserver` separates the two required methods (`on_tool_call`, `on_response`) from four optional lifecycle hooks, each with a safe default no-op.            |
| **DIP**   | `Agent` depends on `BaseTool` and `BaseObserver` (abstractions). Concrete tool and observer classes are only ever named in `main.py`.                            |

---

## 5. Available tools

### Built-in tools

**`calculator`** -- evaluates arithmetic and mathematical expressions.

Accepts any combination of operators (`+`, `-`, `*`, `/`, `//`, `%`, `**`), comparison operators, bitwise operators, and whitelisted math functions (`sqrt`, `sin`, `cos`, `log`, `factorial`, `gcd`, `pow`, and more) plus constants (`pi`, `e`, `tau`).

Security: a two-layer model -- AST node whitelist (blocks builtins, attribute access, lambdas, imports) followed by `eval()` with `__builtins__: {}` -- means no arbitrary Python can execute even if the LLM generates a malicious expression.

---

**`weather`** -- returns current weather for a city.

Reports temperature, feels-like temperature, weather description, humidity, and wind speed. Uses **wttr.in** by default (zero configuration needed). Automatically switches to **OpenWeatherMap** when `OPENWEATHER_API_KEY` is set.

---

**`search`** -- fetches a factual summary from the web.

Queries the **Wikipedia REST API** first (handles disambiguation pages automatically). Falls back to the **DuckDuckGo instant-answer API** when Wikipedia has no article. Returns up to 600 characters with a source URL appended.

---

**`time`** -- returns the current date and time.

Accepts IANA timezone names (`America/New_York`), common abbreviations (`JST`, `CET`, `EST`), and city names (`Tokyo`, `Paris`, `London`). Returns date, time, and UTC offset. Local system time when no timezone is given.

### Custom tools

**`translate`** (custom #1) -- translates text between languages.

Accepts full language names (`French`) or ISO 639-1 codes (`fr`) for both source and target. Source language is auto-detected when omitted. Uses **MyMemory** as the primary backend (500 words/day free, no key) with **LibreTranslate** as a configurable fallback.

---

**`file_reader`** (custom #2) -- reads a local plain-text file.

Enforces a four-layer security model to prevent directory traversal:

1. **Argument sanitisation** -- rejects null bytes, wrong types, empty strings
2. **Canonical path confinement** -- `os.path.realpath()` resolves all `..` sequences and symlinks; the result must start with `FILE_READER_BASE_DIR`
3. **Extension allow-list** -- only `.txt`, `.md`, `.csv`, `.json`, `.yaml`, `.log`, and related text formats are permitted
4. **Existence and type validation** -- confirms the path is a regular file, not a directory or device

Files are truncated to 8,000 characters; files over 100 KB are refused.

---

## 6. Configuration reference

All values live in `config/settings.py` and can be overridden with environment variables.

| Variable                  | Default                                | Description                                                      |
| ------------------------- | -------------------------------------- | ---------------------------------------------------------------- |
| `GEMINI_API_KEY`          | _(required)_                           | Google Gemini API key -- process exits if unset                  |
| `GEMINI_MODEL`            | `gemini-1.5-flash`                     | Model identifier passed to `GenerativeModel`                     |
| `MAX_OUTPUT_TOKENS`       | `2048`                                 | Maximum tokens per model response                                |
| `TEMPERATURE`             | `0.4`                                  | Sampling temperature (0 = deterministic)                         |
| `MAX_HISTORY_TURNS`       | `20`                                   | Conversation turn groups kept in `MemoryManager`                 |
| `MAX_TOOL_CALLS_PER_TURN` | `5`                                    | Safety cap -- prevents runaway ReAct loops                       |
| `OPENWEATHER_API_KEY`     | _(optional)_                           | Activates OpenWeatherMap backend in `WeatherTool`                |
| `WEATHER_UNITS`           | `metric`                               | `metric` (Celsius), `imperial` (Fahrenheit), `standard` (Kelvin) |
| `FILE_READER_BASE_DIR`    | `os.getcwd()`                          | Sandbox root for `FileReaderTool`                                |
| `LIBRETRANSLATE_URL`      | `https://libretranslate.com/translate` | LibreTranslate endpoint for `TranslateTool`                      |
| `LOG_DIR`                 | `logs`                                 | Directory for `agent.log` and `session.log`                      |
| `LOG_LEVEL`               | `INFO`                                 | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)       |

---

## 7. Running the tests

```bash
# Install pytest (once)
pip install pytest

# Run everything
pytest tests/ -v

# Run one module
pytest tests/test_calculator_tool.py -v

# With coverage
pip install pytest-cov
pytest tests/ --cov=. --cov-report=term-missing
```

All 311 tests pass with zero failures. All HTTP calls are mocked -- no network access or API key is needed to run the test suite.

| Test file                  | Count | Focus                                                                                           |
| -------------------------- | ----- | ----------------------------------------------------------------------------------------------- |
| `test_calculator_tool.py`  | 64    | Arithmetic, math functions, AST security (blocked builtins, attribute access, lambdas, imports) |
| `test_weather_tool.py`     | 31    | wttr.in + OWM backends, city-not-found (404 and empty body), network errors                     |
| `test_search_tool.py`      | 30    | Wikipedia + DuckDuckGo cascade, disambiguation, truncation, 5xx propagation                     |
| `test_time_tool.py`        | 38    | IANA names, abbreviations, city names, frozen-clock assertions, unknown timezone                |
| `test_translate_tool.py`   | 45    | MyMemory -> LibreTranslate fallback, language normalisation, quota + rate-limit errors          |
| `test_file_reader_tool.py` | 43+1  | Six path-traversal attack vectors, extension allow-list, size cap, encoding replacement         |
| `test_logger_observer.py`  | 42    | All lifecycle events, counter accuracy, append mode, thread safety (40 concurrent writes)       |
| `test_error_handling.py`   | 28    | Tier-1 API retries with back-off, Tier-2 re-reasoning, Tier-3 unexpected errors, cap fallback   |

---

## 8. CLI commands

```
You: /help       Show available commands
You: /tools      List all registered tools with descriptions
You: /history    Show conversation memory summary and last turn
You: /reset      Clear history and start a fresh session
You: /quit       Exit cleanly  (Ctrl+C and Ctrl+D also work)
```

**Example session:**

```
You: What is sqrt(144) * pi?
Assistant: Using the calculator... sqrt(144) is 12, and 12 * pi = 37.699...

You: What time is it in Tokyo?
Assistant: It is currently 23:45 JST (UTC+09:00) in Tokyo.

You: Translate "good morning" to Japanese
Assistant: "Good morning" in Japanese is "おはようございます" (Ohayou gozaimasu).

You: /history
Memory: MemoryManager: 3 groups, 12 turns, cap=20
Last [model]: "Good morning" in Japanese is "おはようございます"...
```

The session log at `logs/session.log` records every tool call and response:

```
2025-06-20 14:32:01 [SESSION    ] START  tools=calculator,file_reader,search,time,translate,weather
2025-06-20 14:32:05 [TURN       ] #1  user="What is sqrt(144) * pi?"
2025-06-20 14:32:06 [TOOL CALL  ] calculator(expression="sqrt(144) * pi") -> "sqrt(144) * pi = 37.6991..."
2025-06-20 14:32:06 [RESPONSE   ] "sqrt(144) is 12, and 12 * pi = 37.699..."
```

---

## 9. Adding a new tool

The system is designed so that adding a tool touches **only two files**.

**Step 1 -- Create the tool class** in `tools/built_in/` or `tools/custom/`:

```python
# tools/built_in/my_tool.py
from tools.base_tool import BaseTool, ToolArgumentError, ToolExecutionError

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        """Return the unique tool identifier used by ToolRegistry."""
        return "my_tool"

    def execute(self, args: dict) -> str:
        """Execute the tool and return a plain-text result."""
        query = args.get("query")
        if not query:
            raise ToolArgumentError("Missing required argument: 'query'.")
        return f"Result for: {query}"

    def get_declaration(self) -> dict:
        """Return the Gemini function-calling schema for this tool."""
        return {
            "name": "my_tool",
            "description": "Does something useful.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The input."}
                },
                "required": ["query"],
            },
        }
```

**Step 2 -- Register it in `main.py`**:

```python
from tools.built_in.my_tool import MyTool   # add import
# ...
all_tools = [
    CalculatorTool(),
    # ...
    MyTool(),   # add one line
]
```

That is all. The `Agent`, `ToolRegistry`, `PromptBuilder`, and all tests are unaffected.

---

## 10. Assignment criteria mapping

| Criterion                    | Implementation                                                                                                                                                                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **SOLID principles**         | SRP: `MemoryManager`, `PromptBuilder`, `LoggerObserver`. OCP: tool registration in `main.py` only. DIP: `Agent` depends on `BaseTool` / `BaseObserver`. LSP: all subclasses substitutable. ISP: `BaseObserver` separates required from optional hooks. |
| **Strategy Pattern**         | Each tool is a `BaseTool` strategy. `ToolRegistry` selects and executes the correct strategy by name. No `if/elif` chains anywhere in the agent.                                                                                                       |
| **Factory/Registry Pattern** | `ToolRegistry` -- maps tool names to instances, validates types, dispatches execution, and exposes JSON schemas.                                                                                                                                       |
| **Observer Pattern (bonus)** | `BaseObserver` + `LoggerObserver`. Six lifecycle events. Agent holds `list[BaseObserver]`; observers added via `add_observer()`.                                                                                                                       |
| **ReAct loop**               | `Agent._react_loop()` -- full Reason -> Act -> Observe cycle with memory recording, tool dispatch, error recovery, and tool-call cap.                                                                                                                  |
| **Contextual memory**        | `MemoryManager` -- stores all turns (text + structured function_call / function_response parts) with group-aware eviction.                                                                                                                             |
| **Tool integration (4+)**    | `calculator`, `weather`, `search`, `time` (built-in) + `translate`, `file_reader` (custom).                                                                                                                                                            |
| **Custom tools (2)**         | `TranslateTool` (dual-backend translation + language normalisation) and `FileReaderTool` (four-layer security model).                                                                                                                                  |
| **Error handling**           | Three-tier model: Tier-1 fatal (API retry + graceful message), Tier-2 recoverable (error observation injected, loop continues), Tier-3 unexpected (logged, loop continues).                                                                            |
| **Code quality**             | Type hints and docstrings on every method. 311 tests, 0 failures. Audit script confirms zero gaps.                                                                                                                                                     |
