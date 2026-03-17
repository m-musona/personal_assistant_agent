# Personal Assistant Agent — Project Plan

A phased build plan for a Python-based AI agent using Google Gemini, the ReAct loop pattern, SOLID principles, and a pluggable tool system.

---

## Phase 1 — Environment & Scaffold

| #   | Task                                      | Type          | Notes                                                                                                                                 |
| --- | ----------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Install Python 3.10+ and dependencies** | `impl`        | `pip install google-generativeai requests python-dotenv` — add all to `requirements.txt`                                              |
| 2   | **Get Gemini API key**                    | `impl`        | Create a key on Google AI Studio, store it in `.env` using `GEMINI_API_KEY`. Copy `.env.example` as template.                         |
| 3   | **Create the full folder structure**      | `arch`        | Create all directories and empty `__init__.py` files: `agent/`, `tools/built_in/`, `tools/custom/`, `observers/`, `config/`, `tests/` |
| 4   | **Write `config/settings.py`**            | `arch` `impl` | Load API key via `os.getenv()`, define `MODEL_NAME`, `MAX_HISTORY_TURNS`, and other constants in one place.                           |

---

## Phase 2 — Core Abstractions

| #   | Task                                           | Type          | Notes                                                                                                                                                                          |
| --- | ---------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | **Implement `BaseTool`** (DIP + OCP)           | `arch`        | Abstract class with two abstract methods: `execute(args: dict) → str` and `get_declaration() → dict`. No concrete logic — this is the interface every tool must satisfy.       |
| 2   | **Implement `ToolRegistry`** (Factory pattern) | `arch` `impl` | Dict-based registry: `register(tool: BaseTool)` stores tool by name, `execute(name, args)` dispatches to the right tool. Raises a clean `ToolNotFoundError` for unknown names. |
| 3   | **Implement `MemoryManager`** (SRP)            | `arch` `impl` | Stores turns as `list[{role, content}]`. Methods: `add_turn()`, `get_history()`, `clear()`. Optionally cap history length using `MAX_HISTORY_TURNS` from settings.             |
| 4   | **Implement `PromptBuilder`** (SRP)            | `arch` `impl` | Single method `build_system_prompt()` returning the string that tells Gemini its role, available tools, and how to invoke them.                                                |

---

## Phase 3 — Agent Loop

| #   | Task                                        | Type          | Notes                                                                                                                                                                                          |
| --- | ------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Wire up the Gemini client**               | `impl`        | In `agent.py`, initialise `genai.configure()` from settings, create the `GenerativeModel`, pass tool declarations from the registry to the model.                                              |
| 2   | **Implement the ReAct loop**                | `arch` `impl` | Core method: send message → receive response → if response has `function_call`, dispatch to `ToolRegistry` → append tool result → send again → repeat until a final text response is produced. |
| 3   | **Integrate `MemoryManager` into the loop** | `impl`        | Each user turn and assistant response (including tool calls/results) gets appended to history. Each API call includes the full history so Gemini has context.                                  |
| 4   | **Add error handling to the loop**          | `impl` `test` | Wrap API calls and tool dispatch in `try/except`. On API error: log and return a graceful message. On `ToolNotFoundError` or bad args: inject an error observation and let Gemini re-reason.   |

---

## Phase 4 — Built-in Tools

| #   | Task                 | Type   | Notes                                                                                                                                                                   |
| --- | -------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **`CalculatorTool`** | `impl` | `execute()` uses Python's `eval()` (safely) or `ast.literal_eval` to handle math expressions. `get_declaration()` describes an `expression` string parameter.           |
| 2   | **`WeatherTool`**    | `impl` | Calls a free weather API (e.g. `wttr.in` or OpenWeatherMap). `execute()` takes a `city` arg and returns temperature + conditions. Handle 404 city-not-found gracefully. |
| 3   | **`TimeTool`**       | `impl` | Returns current date and time, optionally for a given timezone. Uses Python's `datetime` + `zoneinfo`. Simple but great for testing the loop end-to-end.                |
| 4   | **`SearchTool`**     | `impl` | Wraps a simple web search or Wikipedia API. `execute()` takes a `query` arg and returns a short summary. Use `wikipedia-api` or requests to DuckDuckGo's JSON endpoint. |

---

## Phase 5 — Custom Tools

| #   | Task                                | Type          | Notes                                                                                                                                                                                     |
| --- | ----------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **`TranslateTool`** (custom #1)     | `impl` `arch` | Calls a free translation API (e.g. LibreTranslate or MyMemory). Takes `text` and `target_language`. Demonstrates external API integration beyond weather.                                 |
| 2   | **`FileReaderTool`** (custom #2)    | `impl` `arch` | `execute()` takes a `filepath` arg and reads a local `.txt` file, returning its content. Include path validation to prevent directory traversal. Great for security-aware error handling. |
| 3   | **Register all tools in `main.py`** | `arch` `impl` | Instantiate each tool class, call `registry.register()` for each. The agent receives the populated registry — no tool names hardcoded in `agent.py`.                                      |

---

## Phase 6 — Observer Pattern _(bonus)_

| #   | Task                                   | Type   | Notes                                                                                                                                                     |
| --- | -------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Implement `BaseObserver` interface** | `arch` | Abstract class with `on_tool_call(name, args)` and `on_response(text)` methods. The Agent holds a list of observers and notifies them at key loop points. |
| 2   | **Implement `LoggerObserver`**         | `impl` | Writes a timestamped log entry to `logs/session.log` on each tool call and final response. Attach via `agent.add_observer(LoggerObserver())`.             |

---

## Phase 7 — CLI and Polish

| #   | Task                              | Type   | Notes                                                                                                                                                     |
| --- | --------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Build `main.py` CLI loop**      | `impl` | `while True` input loop. Print a welcome message, read user input, pass to `agent.chat()`, print the response. Handle `KeyboardInterrupt` for clean exit. |
| 2   | **Add type hints and docstrings** | `impl` | Every method in every class should have a return type annotation and a one-line docstring. Covers the Code Quality criterion directly.                    |
| 3   | **Write `README.md`**             | `impl` | Setup instructions, how to run, list of tools, architecture overview with a diagram reference. Graders will read this first.                              |

---

## Phase 8 — Testing and Hardening

| #   | Task                                         | Type          | Notes                                                                                                                                                                         |
| --- | -------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Unit test each tool**                      | `test`        | `test_tools.py`: test valid inputs, invalid inputs (bad city name, invalid filepath), and that each tool returns a string. Mock external API calls with `unittest.mock`.      |
| 2   | **Integration test the agent loop**          | `test`        | `test_agent.py`: ask a multi-tool question (e.g. _"What time is it in Tokyo and translate hello to Japanese?"_). Verify both tools fire and a coherent answer comes back.     |
| 3   | **Test failure scenarios**                   | `test`        | Ask for weather in `'Atlantis'`, request a non-existent file, trigger an unknown tool. Verify the agent continues gracefully and gives a helpful error message — not a crash. |
| 4   | **Final review against evaluation criteria** | `test` `arch` | Check: SOLID principles enforced? No `if/elif` tool chains? ReAct loop complete? Two custom tools with valid schemas? Error handling on all three failure modes?              |

---

## Legend

| Badge  | Meaning                        |
| ------ | ------------------------------ |
| `arch` | Architecture / design decision |
| `impl` | Implementation task            |
| `test` | Testing task                   |
