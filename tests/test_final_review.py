"""

Final automated review against every evaluation criterion.

Criterion 1 (40%): Software Architecture & Patterns
    - SOLID principles enforced (SRP, OCP, LSP, ISP, DIP)
    - No if/elif tool-dispatch chains in agent.py
    - Strategy + Factory/Registry patterns correctly applied
    - Observer pattern wired correctly

Criterion 2 (30%): Agent Functionality
    - Gemini client configures and initialises without error
    - MemoryManager records all turn types (text + structured)
    - ReAct loop: function_call detected, dispatched, result fed back
    - Loop exits cleanly on final text response
    - Tool-call cap prevents infinite loops

Criterion 3 (20%): Custom Tool Implementation
    - Both custom tools (translate, file_reader) instantiate without error
    - Both produce valid Gemini function-calling schemas
    - Required parameters correctly declared
    - Tool executes and returns a string under happy-path conditions

Criterion 4 (10%): Code Quality & Error Handling
    - Every method has a docstring and return-type annotation
    - All three error tiers handled: API (Tier 1), tool (Tier 2), bug (Tier 3)
    - Tier 1: GeminiAPIError raised after retries, chat() returns graceful message
    - Tier 2: ToolExecutionError → error observation → loop continues
    - Tier 3: bare Exception → error observation → loop continues
    - LoggerObserver writes structured log entries
    - ToolRegistry validates types on register()
    - MemoryManager enforces group-aware cap
"""

from __future__ import annotations

import ast
import json
import os
import re
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

from tools.base_tool import BaseTool, ToolArgumentError, ToolExecutionError
from tools.tool_registry import ToolRegistry, ToolNotFoundError
from observers.base_observer import BaseObserver


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers  (identical to test_agent.py so files are self-contained)
# ─────────────────────────────────────────────────────────────────────────────


def _fc_part(name: str, args: dict) -> MagicMock:
    """Return a mock Gemini response part containing a function_call."""
    fc = MagicMock()
    fc.name = name
    fc.args = args
    part = MagicMock()
    type(part).function_call = PropertyMock(return_value=fc)
    part.text = ""
    return part


def _text_part(text: str) -> MagicMock:
    """Return a mock Gemini response part containing plain text."""
    part = MagicMock()
    empty = MagicMock()
    empty.name = ""
    part.function_call = empty
    part.text = text
    return part


def _response(*parts) -> MagicMock:
    """Wrap parts in a mock GenerateContentResponse."""
    candidate = MagicMock()
    candidate.content.parts = list(parts)
    resp = MagicMock()
    resp.candidates = [candidate]
    return resp


def _build_agent(tools: list[BaseTool], session: MagicMock):
    """Construct an Agent with a mocked Gemini SDK."""
    from agent.agent import Agent

    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    with patch("agent.agent.genai") as g:
        m = MagicMock()
        g.GenerativeModel.return_value = m
        m.start_chat.return_value = session
        g.configure = MagicMock()
        agent = Agent(registry)
    agent._session = session
    return agent


def _make_stub(name: str, result: str = "ok") -> BaseTool:
    """Return a minimal BaseTool stub."""

    class _S(BaseTool):
        @property
        def name(self) -> str:
            """Return tool name."""
            return name

        def execute(self, args: dict) -> str:
            """Return configured result."""
            return result

        def get_declaration(self) -> dict:
            """Return minimal schema."""
            return {
                "name": name,
                "description": f"Stub {name}.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }

    return _S()


# ─────────────────────────────────────────────────────────────────────────────
# CRITERION 1a: No if/elif tool-dispatch chains
# ─────────────────────────────────────────────────────────────────────────────


class TestNoIfElifToolChains(unittest.TestCase):
    """Agent must use ToolRegistry dispatch — no hardcoded tool name checks."""

    def _source(self, path: str) -> str:
        """Read and return the source of a project file."""
        with open(path) as f:
            return f.read()

    def test_agent_has_no_tool_name_if_elif(self) -> None:
        """agent.py must contain no if/elif that compares against a tool name."""
        src = self._source("agent/agent.py")
        tool_names = [
            "calculator",
            "weather",
            "search",
            "time",
            "translate",
            "file_reader",
        ]
        for name in tool_names:
            pattern = rf'(if|elif)\s+.*["\'{name}].*==|==[^\n]*["\'{name}]'
            self.assertIsNone(
                re.search(pattern, src),
                f"Found if/elif comparison against tool name {name!r} in agent.py",
            )

    def test_tool_registry_has_no_tool_name_if_elif(self) -> None:
        """tool_registry.py must not dispatch via if/elif name comparisons."""
        src = self._source("tools/tool_registry.py")
        tool_names = [
            "calculator",
            "weather",
            "search",
            "time",
            "translate",
            "file_reader",
        ]
        for name in tool_names:
            self.assertNotIn(
                f'== "{name}"',
                src,
                f"Found hardcoded tool name comparison for {name!r} in tool_registry.py",
            )

    def test_dispatch_uses_dict_lookup(self) -> None:
        """ToolRegistry.execute() dispatches via dict lookup, not branching."""
        registry = ToolRegistry()
        registry.register(_make_stub("alpha", "result_alpha"))
        registry.register(_make_stub("beta", "result_beta"))

        self.assertEqual(registry.execute("alpha", {}), "result_alpha")
        self.assertEqual(registry.execute("beta", {}), "result_beta")

    def test_new_tool_registered_without_modifying_agent(self) -> None:
        """Registering a new tool requires zero changes to Agent source."""
        src = self._source("agent/agent.py")
        # agent.py must not import any concrete tool module
        self.assertNotRegex(
            src,
            r"from tools\.(built_in|custom)\.\w+ import",
            "agent.py should not import concrete tool classes",
        )


# ─────────────────────────────────────────────────────────────────────────────
# CRITERION 1b: SOLID principles
# ─────────────────────────────────────────────────────────────────────────────


class TestSOLIDPrinciples(unittest.TestCase):
    """Structural checks that SOLID principles are enforced in the code."""

    # SRP ─────────────────────────────────────────────────────────────────────

    def test_srp_memory_manager_has_no_genai(self) -> None:
        """MemoryManager must not import or reference the Gemini SDK."""
        with open("agent/memory_manager.py") as f:
            src = f.read()
        self.assertNotIn("genai", src)
        self.assertNotIn("GenerativeModel", src)

    def test_srp_memory_manager_has_no_tool_registry(self) -> None:
        """MemoryManager must not reference ToolRegistry."""
        with open("agent/memory_manager.py") as f:
            src = f.read()
        self.assertNotIn("ToolRegistry", src)

    def test_srp_prompt_builder_has_no_send_message(self) -> None:
        """PromptBuilder must not call the Gemini API directly."""
        with open("agent/prompt_builder.py") as f:
            src = f.read()
        self.assertNotIn("send_message", src)
        self.assertNotIn("genai.configure", src)

    def test_srp_tool_registry_has_no_memory_manager(self) -> None:
        """ToolRegistry must not reference MemoryManager."""
        with open("tools/tool_registry.py") as f:
            src = f.read()
        self.assertNotIn("MemoryManager", src)

    def test_srp_base_tool_has_no_agent_imports(self) -> None:
        """BaseTool must not import Agent, MemoryManager, or ToolRegistry."""
        with open("tools/base_tool.py") as f:
            src = f.read()
        # Parse the AST and check import statements only — not prose in docstrings
        tree = ast.parse(src)
        imported_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imported_names.append(f"{module}.{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.append(alias.name)
        for cls in ("Agent", "MemoryManager", "ToolRegistry"):
            for imp in imported_names:
                self.assertNotIn(
                    cls, imp, f"base_tool.py imports {cls!r} which violates SRP/DIP"
                )

    # OCP ─────────────────────────────────────────────────────────────────────

    def test_ocp_concrete_tool_instantiation_only_in_main(self) -> None:
        """
        Concrete tool classes are instantiated only in main.py.
        Docstring examples are excluded from this check.
        """
        concrete = [
            "CalculatorTool",
            "WeatherTool",
            "SearchTool",
            "TimeTool",
            "TranslateTool",
            "FileReaderTool",
        ]
        violations = []
        for root, _, files in os.walk("."):
            if any(x in root for x in ["__pycache__", "tests", ".git"]):
                continue
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                if os.path.basename(fpath) == "main.py":
                    continue
                with open(fpath) as f:
                    content = f.read()
                try:
                    tree = ast.parse(content)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    # Check for Call nodes whose func is one of our concrete names
                    if isinstance(node, ast.Call):
                        func = node.func
                        name = None
                        if isinstance(func, ast.Name):
                            name = func.id
                        elif isinstance(func, ast.Attribute):
                            name = func.attr
                        if name in concrete:
                            violations.append(f"{fpath}:{node.lineno} — {name}()")

        self.assertEqual(
            violations,
            [],
            "Concrete tool instantiation found outside main.py:\n"
            + "\n".join(violations),
        )

    # LSP ─────────────────────────────────────────────────────────────────────

    def test_lsp_all_tools_satisfy_base_tool_interface(self) -> None:
        """Every registered tool is a BaseTool subclass with both abstract methods."""
        from tools.built_in.calculator_tool import CalculatorTool
        from tools.built_in.weather_tool import WeatherTool
        from tools.built_in.search_tool import SearchTool
        from tools.built_in.time_tool import TimeTool
        from tools.custom.translate_tool import TranslateTool
        from tools.custom.file_reader_tool import FileReaderTool

        for ToolClass in (
            CalculatorTool,
            WeatherTool,
            SearchTool,
            TimeTool,
            TranslateTool,
            FileReaderTool,
        ):
            tool = ToolClass()
            with self.subTest(tool=tool.name):
                self.assertIsInstance(tool, BaseTool)
                self.assertTrue(callable(tool.execute))
                self.assertTrue(callable(tool.get_declaration))
                self.assertIsInstance(tool.name, str)

    # ISP ─────────────────────────────────────────────────────────────────────

    def test_isp_minimal_observer_needs_only_two_methods(self) -> None:
        """A concrete observer with only on_tool_call + on_response is valid."""

        class _Minimal(BaseObserver):
            def on_tool_call(self, name, args, result) -> None:
                """Minimal on_tool_call."""

            def on_response(self, text) -> None:
                """Minimal on_response."""

        obs = _Minimal()
        # Optional hooks must not raise
        obs.on_agent_start(["calc"])
        obs.on_turn_start("hello")
        obs.on_error("oops", "ctx")
        obs.on_agent_reset()

    # DIP ─────────────────────────────────────────────────────────────────────

    def test_dip_agent_does_not_import_concrete_tools(self) -> None:
        """agent.py must not import any concrete tool module."""
        with open("agent/agent.py") as f:
            src = f.read()
        self.assertNotRegex(
            src,
            r"from tools\.(built_in|custom)\.\w+ import",
            "agent.py imports a concrete tool class",
        )

    def test_dip_agent_does_not_import_concrete_observers(self) -> None:
        """agent.py must not import LoggerObserver."""
        with open("agent/agent.py") as f:
            src = f.read()
        self.assertNotIn("LoggerObserver", src)

    def test_dip_registry_accepts_any_base_tool_subclass(self) -> None:
        """ToolRegistry works with any BaseTool subclass — not just known tools."""
        registry = ToolRegistry()
        registry.register(_make_stub("custom_xyz", "custom result"))
        result = registry.execute("custom_xyz", {})
        self.assertEqual(result, "custom result")


# ─────────────────────────────────────────────────────────────────────────────
# CRITERION 2: Agent functionality
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentFunctionality(unittest.TestCase):
    """ReAct loop: detect → dispatch → observe → final text."""

    def test_react_loop_single_tool_call(self) -> None:
        """Loop detects a function_call, dispatches it, receives text reply."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("calc", {"expression": "1+1"})),
            _response(_text_part("The answer is 2.")),
        ]
        agent = _build_agent([_make_stub("calc", "1+1=2")], session)
        reply = agent.chat("what is 1+1")
        self.assertEqual(reply, "The answer is 2.")
        self.assertEqual(session.send_message.call_count, 2)

    def test_react_loop_tool_result_in_second_call(self) -> None:
        """The tool result is present in the second send_message payload."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("calc", {})),
            _response(_text_part("done")),
        ]
        agent = _build_agent([_make_stub("calc", "42")], session)
        agent.chat("calculate")
        arg = session.send_message.call_args_list[1][0][0]
        self.assertIn("42", str(arg))

    def test_react_loop_two_sequential_tool_calls(self) -> None:
        """Two sequential tool calls both fire before the final text reply."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("time", {"timezone": "Tokyo"})),
            _response(
                _fc_part("translate", {"text": "hello", "target_language": "ja"})
            ),
            _response(_text_part("It is 23:32 in Tokyo; hello is こんにちは.")),
        ]
        agent = _build_agent(
            [
                _make_stub("time", "23:32 JST"),
                _make_stub("translate", "こんにちは"),
            ],
            session,
        )
        reply = agent.chat("What time is it in Tokyo and translate hello to Japanese?")
        self.assertEqual(reply, "It is 23:32 in Tokyo; hello is こんにちは.")
        self.assertEqual(session.send_message.call_count, 3)

    def test_react_loop_no_tool_call_direct_reply(self) -> None:
        """When Gemini answers directly (no tool), chat() returns that text."""
        session = MagicMock()
        session.send_message.return_value = _response(
            _text_part("The capital of France is Paris.")
        )
        agent = _build_agent([], session)
        reply = agent.chat("What is the capital of France?")
        self.assertEqual(reply, "The capital of France is Paris.")
        self.assertEqual(session.send_message.call_count, 1)

    def test_memory_records_user_turn(self) -> None:
        """User input is recorded as the first memory turn."""
        session = MagicMock()
        session.send_message.return_value = _response(_text_part("Hi!"))
        agent = _build_agent([], session)
        agent.chat("Hello agent")
        self.assertEqual(agent.memory.get_history()[0]["role"], "user")
        self.assertIn("Hello agent", agent.memory.get_history()[0]["parts"][0]["text"])

    def test_memory_records_model_turn(self) -> None:
        """Final model reply is recorded as the last memory turn."""
        session = MagicMock()
        session.send_message.return_value = _response(_text_part("Hello human!"))
        agent = _build_agent([], session)
        agent.chat("Hello agent")
        last = agent.memory.last_turn()
        self.assertEqual(last["role"], "model")
        self.assertIn("Hello human!", last["parts"][0]["text"])

    def test_memory_records_function_call_and_response(self) -> None:
        """function_call and function_response turns are both in memory."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("calc", {"expression": "2*3"})),
            _response(_text_part("6")),
        ]
        agent = _build_agent([_make_stub("calc", "2*3=6")], session)
        agent.chat("2 times 3")
        history = agent.memory.get_history()
        roles = [t["role"] for t in history]
        self.assertIn("function", roles)
        fc_turns = [
            t
            for t in history
            if t["role"] == "model" and "function_call" in t.get("parts", [{}])[0]
        ]
        self.assertEqual(len(fc_turns), 1)

    def test_tool_call_cap_prevents_infinite_loop(self) -> None:
        """Loop exits with a graceful message when the tool-call cap is hit."""
        from config.settings import MAX_TOOL_CALLS_PER_TURN

        session = MagicMock()
        session.send_message.return_value = _response(
            _fc_part("calc", {"expression": "1+1"})
        )
        agent = _build_agent([_make_stub("calc", "2")], session)
        reply = agent.chat("keep calling")
        self.assertIsInstance(reply, str)
        self.assertTrue(
            "tool calls" in reply.lower(), f"Expected cap message, got: {reply!r}"
        )

    def test_gemini_client_initialises(self) -> None:
        """Agent.__init__ completes without error with a mocked SDK."""
        session = MagicMock()
        agent = _build_agent([], session)
        self.assertIsNotNone(agent)
        self.assertIsNotNone(agent.memory)
        self.assertIsNotNone(agent.registry)


# ─────────────────────────────────────────────────────────────────────────────
# CRITERION 3: Custom tool implementation
# ─────────────────────────────────────────────────────────────────────────────


class TestCustomToolImplementation(unittest.TestCase):
    """TranslateTool and FileReaderTool — valid schemas, correct behaviour."""

    # TranslateTool ──────────────────────────────────────────────────────────

    def test_translate_tool_is_base_tool_subclass(self) -> None:
        """TranslateTool is a BaseTool subclass."""
        from tools.custom.translate_tool import TranslateTool

        self.assertIsInstance(TranslateTool(), BaseTool)

    def test_translate_tool_name(self) -> None:
        """TranslateTool.name returns 'translate'."""
        from tools.custom.translate_tool import TranslateTool

        self.assertEqual(TranslateTool().name, "translate")

    def test_translate_declaration_name_key(self) -> None:
        """get_declaration()['name'] equals the tool name property."""
        from tools.custom.translate_tool import TranslateTool

        t = TranslateTool()
        self.assertEqual(t.get_declaration()["name"], t.name)

    def test_translate_declaration_has_description(self) -> None:
        """get_declaration() includes a non-empty description."""
        from tools.custom.translate_tool import TranslateTool

        d = TranslateTool().get_declaration()
        self.assertIn("description", d)
        self.assertGreater(len(d["description"]), 10)

    def test_translate_declaration_has_parameters(self) -> None:
        """get_declaration() includes a parameters dict."""
        from tools.custom.translate_tool import TranslateTool

        d = TranslateTool().get_declaration()
        self.assertIn("parameters", d)
        self.assertIn("properties", d["parameters"])

    def test_translate_required_params(self) -> None:
        """'text' and 'target_language' are required parameters."""
        from tools.custom.translate_tool import TranslateTool

        required = TranslateTool().get_declaration()["parameters"]["required"]
        self.assertIn("text", required)
        self.assertIn("target_language", required)

    def test_translate_source_language_optional(self) -> None:
        """'source_language' is in properties but not required."""
        from tools.custom.translate_tool import TranslateTool

        decl = TranslateTool().get_declaration()["parameters"]
        self.assertIn("source_language", decl["properties"])
        self.assertNotIn("source_language", decl["required"])

    def test_translate_executes_and_returns_string(self) -> None:
        """execute() returns a str under happy-path conditions (HTTP mocked)."""
        from tools.custom.translate_tool import TranslateTool

        ok = json.dumps(
            {
                "responseStatus": 200,
                "responseData": {
                    "translatedText": "Bonjour",
                    "detectedLanguage": "en",
                    "match": 1.0,
                },
            }
        )
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.read.return_value = ok.encode()
        with patch("urllib.request.urlopen", return_value=cm):
            result = TranslateTool().execute({"text": "Hello", "target_language": "fr"})
        self.assertIsInstance(result, str)
        self.assertIn("Bonjour", result)

    def test_translate_unknown_language_raises_argument_error(self) -> None:
        """Unrecognised target_language raises ToolArgumentError."""
        from tools.custom.translate_tool import TranslateTool

        with self.assertRaises(ToolArgumentError):
            TranslateTool().execute({"text": "Hello", "target_language": "Klingon"})

    # FileReaderTool ─────────────────────────────────────────────────────────

    def test_file_reader_tool_is_base_tool_subclass(self) -> None:
        """FileReaderTool is a BaseTool subclass."""
        from tools.custom.file_reader_tool import FileReaderTool

        self.assertIsInstance(FileReaderTool(), BaseTool)

    def test_file_reader_tool_name(self) -> None:
        """FileReaderTool.name returns 'file_reader'."""
        from tools.custom.file_reader_tool import FileReaderTool

        self.assertEqual(FileReaderTool().name, "file_reader")

    def test_file_reader_declaration_name_key(self) -> None:
        """get_declaration()['name'] equals the tool name property."""
        from tools.custom.file_reader_tool import FileReaderTool

        t = FileReaderTool()
        self.assertEqual(t.get_declaration()["name"], t.name)

    def test_file_reader_declaration_has_description(self) -> None:
        """get_declaration() includes a non-empty description."""
        from tools.custom.file_reader_tool import FileReaderTool

        d = FileReaderTool().get_declaration()
        self.assertIn("description", d)
        self.assertGreater(len(d["description"]), 10)

    def test_file_reader_required_params(self) -> None:
        """'filepath' is a required parameter."""
        from tools.custom.file_reader_tool import FileReaderTool

        required = FileReaderTool().get_declaration()["parameters"]["required"]
        self.assertIn("filepath", required)

    def test_file_reader_executes_and_returns_string(self) -> None:
        """execute() returns a str for a valid file (uses isolated temp dir)."""
        from tools.custom.file_reader_tool import FileReaderTool

        tmpdir = tempfile.mkdtemp()
        real_tmpdir = os.path.realpath(tmpdir)
        Path(os.path.join(tmpdir, "hello.txt")).write_text("hello world\n")
        try:
            with patch(
                "tools.custom.file_reader_tool.FILE_READER_BASE_DIR", real_tmpdir
            ):
                with patch(
                    "tools.custom.file_reader_tool._RESOLVED_BASE_DIR", real_tmpdir
                ):
                    result = FileReaderTool().execute({"filepath": "hello.txt"})
            self.assertIsInstance(result, str)
            self.assertIn("hello world", result)
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_file_reader_nonexistent_file_raises_execution_error(self) -> None:
        """Non-existent file raises ToolExecutionError."""
        from tools.custom.file_reader_tool import FileReaderTool

        tmpdir = tempfile.mkdtemp()
        real_tmpdir = os.path.realpath(tmpdir)
        try:
            with patch(
                "tools.custom.file_reader_tool.FILE_READER_BASE_DIR", real_tmpdir
            ):
                with patch(
                    "tools.custom.file_reader_tool._RESOLVED_BASE_DIR", real_tmpdir
                ):
                    with self.assertRaises(ToolExecutionError):
                        FileReaderTool().execute({"filepath": "ghost.txt"})
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_file_reader_path_traversal_raises_execution_error(self) -> None:
        """../../etc/passwd traversal raises ToolExecutionError."""
        from tools.custom.file_reader_tool import FileReaderTool

        tmpdir = tempfile.mkdtemp()
        real_tmpdir = os.path.realpath(tmpdir)
        try:
            with patch(
                "tools.custom.file_reader_tool.FILE_READER_BASE_DIR", real_tmpdir
            ):
                with patch(
                    "tools.custom.file_reader_tool._RESOLVED_BASE_DIR", real_tmpdir
                ):
                    with self.assertRaises(ToolExecutionError) as ctx:
                        FileReaderTool().execute({"filepath": "../../etc/passwd"})
            self.assertIn("outside the permitted directory", str(ctx.exception))
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# CRITERION 4a: Code quality — docstrings + return hints
# ─────────────────────────────────────────────────────────────────────────────


class TestCodeQuality(unittest.TestCase):
    """Every non-dunder method has a docstring and a return-type annotation."""

    _SKIP_DUNDER_DOC = frozenset(
        {
            "__repr__",
            "__str__",
            "__len__",
            "__del__",
            "__iter__",
            "__init__",
        }
    )
    _SKIP_RETURN = frozenset({"__init__", "__del__"})

    def _audit(self, path: str) -> list[str]:
        """Return a list of issue strings for the given source file."""
        with open(path) as f:
            src = f.read()
        tree = ast.parse(src, path)
        issues = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            no_doc = not (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            )
            if no_doc and node.name not in self._SKIP_DUNDER_DOC:
                issues.append(f"{path}:{node.lineno}  {node.name}() — no docstring")
            if node.returns is None and node.name not in self._SKIP_RETURN:
                issues.append(f"{path}:{node.lineno}  {node.name}() — no return hint")
        return issues

    def _check_dir(self, directory: str) -> list[str]:
        """Audit all .py files in a directory tree."""
        issues = []
        for root, _, files in os.walk(directory):
            if "__pycache__" in root:
                continue
            for fname in files:
                if fname.endswith(".py") and not fname.startswith("__"):
                    issues.extend(self._audit(os.path.join(root, fname)))
        return issues

    def test_agent_package_fully_annotated(self) -> None:
        """All methods in agent/ have docstrings and return hints."""
        issues = self._check_dir("agent")
        self.assertEqual(issues, [], "\n".join(issues))

    def test_tools_package_fully_annotated(self) -> None:
        """All methods in tools/ have docstrings and return hints."""
        issues = self._check_dir("tools")
        self.assertEqual(issues, [], "\n".join(issues))

    def test_observers_package_fully_annotated(self) -> None:
        """All methods in observers/ have docstrings and return hints."""
        issues = self._check_dir("observers")
        self.assertEqual(issues, [], "\n".join(issues))

    def test_config_package_fully_annotated(self) -> None:
        """All methods in config/ have docstrings and return hints."""
        issues = self._check_dir("config")
        self.assertEqual(issues, [], "\n".join(issues))

    def test_main_fully_annotated(self) -> None:
        """All functions in main.py have docstrings and return hints."""
        issues = self._audit("main.py")
        self.assertEqual(issues, [], "\n".join(issues))


# ─────────────────────────────────────────────────────────────────────────────
# CRITERION 4b: Three-tier error handling
# ─────────────────────────────────────────────────────────────────────────────


class TestThreeTierErrorHandling(unittest.TestCase):
    """All three error tiers are handled gracefully."""

    # Tier 1 — Fatal API errors ───────────────────────────────────────────────

    def test_tier1_api_failure_returns_graceful_message(self) -> None:
        """Permanent API failure returns a user-friendly string (no exception)."""
        from agent.agent import _API_MAX_RETRIES

        session = MagicMock()
        session.send_message.side_effect = ConnectionError("network down")
        agent = _build_agent([], session)
        with patch("agent.agent.time.sleep"):
            reply = agent.chat("hello")
        self.assertIsInstance(reply, str)
        self.assertIn("couldn't reach", reply.lower())

    def test_tier1_retries_fire_before_giving_up(self) -> None:
        """Gemini API is retried _API_MAX_RETRIES times before failing."""
        from agent.agent import _API_MAX_RETRIES

        session = MagicMock()
        session.send_message.side_effect = TimeoutError("timeout")
        agent = _build_agent([], session)
        with patch("agent.agent.time.sleep"):
            agent.chat("hello")
        self.assertEqual(session.send_message.call_count, _API_MAX_RETRIES)

    def test_tier1_transient_error_then_success(self) -> None:
        """A single API failure followed by success returns the correct reply."""
        session = MagicMock()
        session.send_message.side_effect = [
            ConnectionError("blip"),
            _response(_text_part("Recovered.")),
        ]
        agent = _build_agent([], session)
        with patch("agent.agent.time.sleep"):
            reply = agent.chat("hello")
        self.assertEqual(reply, "Recovered.")

    def test_tier1_agent_usable_after_api_failure(self) -> None:
        """After a permanent API failure, the next turn succeeds."""
        from agent.agent import _API_MAX_RETRIES

        session = MagicMock()
        session.send_message.side_effect = [
            ConnectionError("down")
        ] * _API_MAX_RETRIES + [_response(_text_part("Back online."))]
        agent = _build_agent([], session)
        with patch("agent.agent.time.sleep"):
            agent.chat("fail")
            second = agent.chat("hello")
        self.assertEqual(second, "Back online.")

    # Tier 2 — Recoverable tool errors ───────────────────────────────────────

    def test_tier2_tool_execution_error_loop_continues(self) -> None:
        """ToolExecutionError does not crash the loop."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("weather", {"city": "Atlantis"})),
            _response(_text_part("City not found.")),
        ]
        from tools.built_in.weather_tool import WeatherTool

        agent = _build_agent([WeatherTool()], session)
        empty_wttr = json.dumps({"current_condition": [], "nearest_area": []})
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.read.return_value = empty_wttr.encode()
        with patch("tools.built_in.weather_tool.OPENWEATHER_API_KEY", ""):
            with patch("urllib.request.urlopen", return_value=cm):
                reply = agent.chat("weather in Atlantis")
        self.assertIsInstance(reply, str)
        self.assertFalse(reply.startswith("Error"))

    def test_tier2_tool_not_found_error_loop_continues(self) -> None:
        """ToolNotFoundError (hallucinated tool) does not crash the loop."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("fly_to_moon", {})),
            _response(_text_part("I cannot do that.")),
        ]
        agent = _build_agent([_make_stub("calculator", "1")], session)
        reply = agent.chat("fly me to the moon")
        self.assertEqual(reply, "I cannot do that.")

    def test_tier2_tool_argument_error_loop_continues(self) -> None:
        """ToolArgumentError does not crash the loop."""

        class _Bad(BaseTool):
            @property
            def name(self) -> str:
                """Return name."""
                return "bad"

            def execute(self, args: dict) -> str:
                """Always raise."""
                raise ToolArgumentError("Missing args.")

            def get_declaration(self) -> dict:
                """Return schema."""
                return {
                    "name": "bad",
                    "description": "Bad.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }

        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("bad", {})),
            _response(_text_part("Bad args.")),
        ]
        agent = _build_agent([_Bad()], session)
        reply = agent.chat("use bad tool")
        self.assertIsInstance(reply, str)

    def test_tier2_error_observation_injected(self) -> None:
        """The error observation starts with 'Error:' so Gemini re-reasons."""
        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("fly_to_moon", {})),
            _response(_text_part("Cannot.")),
        ]
        agent = _build_agent([_make_stub("calc", "1")], session)
        agent.chat("fly")
        arg = session.send_message.call_args_list[1][0][0]
        result = arg["parts"][0]["function_response"]["response"]["result"]
        self.assertTrue(
            result.startswith("Error:"), f"Expected 'Error:' prefix, got: {result!r}"
        )

    # Tier 3 — Unexpected exceptions ─────────────────────────────────────────

    def test_tier3_unexpected_exception_loop_continues(self) -> None:
        """A bare RuntimeError in a tool does not propagate out of chat()."""

        class _Buggy(BaseTool):
            @property
            def name(self) -> str:
                """Return name."""
                return "buggy"

            def execute(self, args: dict) -> str:
                """Always raise RuntimeError."""
                raise RuntimeError("internal bug")

            def get_declaration(self) -> dict:
                """Return schema."""
                return {
                    "name": "buggy",
                    "description": "Buggy.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }

        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("buggy", {})),
            _response(_text_part("Something went wrong.")),
        ]
        agent = _build_agent([_Buggy()], session)
        reply = agent.chat("use buggy")
        self.assertIsInstance(reply, str)

    def test_tier3_unexpected_error_observation_sent(self) -> None:
        """The unexpected error is wrapped into an observation for Gemini."""

        class _Buggy(BaseTool):
            @property
            def name(self) -> str:
                """Return name."""
                return "buggy"

            def execute(self, args: dict) -> str:
                """Always raise."""
                raise RuntimeError("segfault")

            def get_declaration(self) -> dict:
                """Return schema."""
                return {
                    "name": "buggy",
                    "description": "Buggy.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }

        session = MagicMock()
        session.send_message.side_effect = [
            _response(_fc_part("buggy", {})),
            _response(_text_part("Error noted.")),
        ]
        agent = _build_agent([_Buggy()], session)
        agent.chat("use buggy")
        arg = session.send_message.call_args_list[1][0][0]
        result = arg["parts"][0]["function_response"]["response"]["result"]
        self.assertIn("unexpected error", result.lower())


# ─────────────────────────────────────────────────────────────────────────────
# CRITERION 4c: Supporting infrastructure
# ─────────────────────────────────────────────────────────────────────────────


class TestSupportingInfrastructure(unittest.TestCase):
    """ToolRegistry, MemoryManager, and LoggerObserver work correctly."""

    def test_tool_registry_rejects_non_base_tool(self) -> None:
        """Registering a non-BaseTool raises TypeError."""
        registry = ToolRegistry()
        with self.assertRaises(TypeError):
            registry.execute  # warm up
            registry.register("not a tool")  # type: ignore[arg-type]

    def test_tool_registry_raises_not_found_for_unknown_tool(self) -> None:
        """Executing an unregistered tool raises ToolNotFoundError."""
        registry = ToolRegistry()
        with self.assertRaises(ToolNotFoundError):
            registry.execute("ghost_tool", {})

    def test_tool_registry_raises_on_duplicate_name(self) -> None:
        """Registering two tools with the same name raises ValueError."""
        registry = ToolRegistry()
        registry.register(_make_stub("dup"))
        with self.assertRaises(ValueError):
            registry.register(_make_stub("dup"))

    def test_memory_manager_group_aware_cap(self) -> None:
        """MemoryManager evicts complete groups when the cap is reached."""
        from agent.memory_manager import MemoryManager

        mm = MemoryManager(max_turns=2)
        # Add 3 groups of 2 turns each; cap=2 → first group evicted
        for i in range(3):
            mm.add_turn("user", f"user {i}")
            mm.add_turn("model", f"model {i}")
        # After eviction: should have at most 2 groups = 4 turns
        self.assertLessEqual(mm.group_count(), 2)
        self.assertLessEqual(mm.turn_count(), 4)
        # History must still start with a 'user' turn (Gemini requirement)
        self.assertEqual(mm.get_history()[0]["role"], "user")

    def test_logger_observer_writes_session_log(self) -> None:
        """LoggerObserver creates a log file and writes structured entries."""
        from observers.logger_observer import LoggerObserver

        tmpdir = tempfile.mkdtemp()
        log_path = os.path.join(tmpdir, "test.log")
        try:
            obs = LoggerObserver(log_path=log_path)
            obs.on_agent_start(["calc", "time"])
            obs.on_turn_start("What is 2+2?")
            obs.on_tool_call("calc", {"expression": "2+2"}, "2+2=4")
            obs.on_response("The answer is 4.")
            obs.close()
            with open(log_path) as f:
                content = f.read()
            self.assertIn("SESSION", content)
            self.assertIn("TURN", content)
            self.assertIn("TOOL CALL", content)
            self.assertIn("RESPONSE", content)
            self.assertIn("2+2=4", content)
            self.assertIn("The answer", content)
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_observer_type_check_on_add_observer(self) -> None:
        """agent.add_observer() rejects non-BaseObserver instances."""
        from agent.agent import Agent

        session = MagicMock()
        agent = _build_agent([], session)
        with self.assertRaises(TypeError):
            agent.add_observer("not an observer")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main(verbosity=2)
