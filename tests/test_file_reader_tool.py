"""

Unit tests for FileReaderTool — uses a temporary directory as the sandbox
so no real files on the host are touched.

Coverage:
  Security (path traversal prevention):
    - Relative "../.." escape blocked
    - Absolute path outside base dir blocked
    - Symlink pointing outside base dir blocked
    - Null byte in path blocked (Layer 1)
    - Sibling directory name prefix attack blocked
    - Directory path (not a file) rejected

  Extension allow-list:
    - .txt allowed
    - .py, .exe, .env, no-extension blocked

  Valid reads:
    - Content returned correctly
    - Metadata header present (filename, size, line count)
    - Relative path resolved correctly
    - Nested subdirectory file within base allowed
    - UTF-8 file reads cleanly

  Edge cases:
    - File exceeds MAX_FILE_BYTES -> ToolExecutionError
    - Content exceeds MAX_RETURN_CHARS -> truncated with note
    - Non-UTF-8 bytes -> replacement char warning appended
    - Empty file -> returns header with 0 lines

  Argument validation:
    - Missing filepath, empty, wrong type, None, null byte

  Declaration schema:
    - name, description, filepath parameter, required list
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.base_tool import ToolArgumentError, ToolExecutionError
from tools.custom.file_reader_tool import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_BYTES,
    MAX_RETURN_CHARS,
    FileReaderTool,
)


class FileReaderToolTestBase(unittest.TestCase):
    """Creates an isolated temp directory used as the tool's base dir."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Patch the module-level constants so the tool is confined to tmpdir.
        self._patches = [
            patch(
                "tools.custom.file_reader_tool.FILE_READER_BASE_DIR",
                self.tmpdir,
            ),
            patch(
                "tools.custom.file_reader_tool._RESOLVED_BASE_DIR",
                os.path.realpath(self.tmpdir),
            ),
        ]
        for p in self._patches:
            p.start()
        self.tool = FileReaderTool()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name: str, content: str = "hello\nworld\n") -> str:
        """Write a file inside tmpdir and return its full path."""
        path = os.path.join(self.tmpdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def _execute(self, filepath: str) -> str:
        return self.tool.execute({"filepath": filepath})


# ---------------------------------------------------------------------------
# Valid reads
# ---------------------------------------------------------------------------


class TestFileReaderValidReads(FileReaderToolTestBase):

    def test_returns_string(self):
        self._write("notes.txt", "Hello world\n")
        self.assertIsInstance(self._execute("notes.txt"), str)

    def test_content_present_in_output(self):
        self._write("notes.txt", "Secret recipe\n")
        self.assertIn("Secret recipe", self._execute("notes.txt"))

    def test_metadata_header_contains_filename(self):
        self._write("notes.txt")
        self.assertIn("notes.txt", self._execute("notes.txt"))

    def test_metadata_header_contains_size(self):
        content = "A" * 50
        self._write("size.txt", content)
        result = self._execute("size.txt")
        self.assertIn("50", result)

    def test_metadata_header_contains_line_count(self):
        self._write("lines.txt", "line1\nline2\nline3\n")
        result = self._execute("lines.txt")
        self.assertIn("3 lines", result)

    def test_relative_path_resolved(self):
        self._write("report.txt", "data here\n")
        self.assertIn("data here", self._execute("report.txt"))

    def test_absolute_path_inside_base_allowed(self):
        full_path = self._write("absolute.txt", "abs content\n")
        self.assertIn("abs content", self._execute(full_path))

    def test_nested_subdirectory_allowed(self):
        self._write("sub/deep.txt", "nested content\n")
        self.assertIn("nested content", self._execute("sub/deep.txt"))

    def test_csv_extension_allowed(self):
        self._write("data.csv", "a,b,c\n1,2,3\n")
        result = self._execute("data.csv")
        self.assertIn("a,b,c", result)

    def test_md_extension_allowed(self):
        self._write("readme.md", "# Title\n")
        self.assertIn("# Title", self._execute("readme.md"))

    def test_json_extension_allowed(self):
        self._write("config.json", '{"key": "value"}\n')
        self.assertIn('"key"', self._execute("config.json"))

    def test_empty_file_returns_header(self):
        self._write("empty.txt", "")
        result = self._execute("empty.txt")
        self.assertIn("empty.txt", result)
        self.assertIn("0 lines", result)

    def test_single_line_no_trailing_newline(self):
        self._write("one.txt", "single")
        result = self._execute("one.txt")
        self.assertIn("1 line", result)


# ---------------------------------------------------------------------------
# Security — path traversal prevention
# ---------------------------------------------------------------------------


class TestFileReaderSecurity(FileReaderToolTestBase):

    def test_relative_dotdot_blocked(self):
        """../../etc/passwd style escape must be refused."""
        with self.assertRaises(ToolExecutionError) as ctx:
            self._execute("../../etc/passwd")
        self.assertIn("outside the permitted directory", str(ctx.exception))

    def test_absolute_path_outside_base_blocked(self):
        with self.assertRaises(ToolExecutionError) as ctx:
            self._execute("/etc/passwd")
        self.assertIn("outside the permitted directory", str(ctx.exception))

    def test_unix_root_blocked(self):
        with self.assertRaises(ToolExecutionError) as ctx:
            self._execute("/")
        self.assertIn("outside the permitted directory", str(ctx.exception))

    def test_sibling_directory_prefix_attack_blocked(self):
        """
        A sibling dir whose name starts with the base dir name must be blocked.
        e.g. BASE=/tmp/abc -> /tmp/abc_evil/file.txt must not pass.
        """
        sibling = self.tmpdir + "_evil"
        os.makedirs(sibling, exist_ok=True)
        evil_file = os.path.join(sibling, "secret.txt")
        Path(evil_file).write_text("pwned", encoding="utf-8")
        try:
            with self.assertRaises(ToolExecutionError) as ctx:
                self._execute(evil_file)
            self.assertIn("outside the permitted directory", str(ctx.exception))
        finally:
            import shutil

            shutil.rmtree(sibling, ignore_errors=True)

    def test_symlink_outside_base_blocked(self):
        """A symlink inside base pointing to a file outside must be blocked."""
        # Create a real file outside the sandbox.
        outside = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
        outside.write("outside content")
        outside.close()
        link_path = os.path.join(self.tmpdir, "link.txt")
        os.symlink(outside.name, link_path)
        try:
            with self.assertRaises(ToolExecutionError) as ctx:
                self._execute("link.txt")
            self.assertIn("outside the permitted directory", str(ctx.exception))
        finally:
            os.unlink(outside.name)

    def test_null_byte_in_path_blocked(self):
        """Null bytes are a classic injection vector."""
        with self.assertRaises(ToolArgumentError) as ctx:
            self.tool.execute({"filepath": "notes\x00.txt"})
        self.assertIn("null", str(ctx.exception).lower())

    def test_directory_path_rejected(self):
        """Passing a directory path rather than a file must be refused."""
        subdir = os.path.join(self.tmpdir, "subdir")
        os.makedirs(subdir, exist_ok=True)
        # subdir ends with no extension, so first check is extension.
        # Use a .txt-named directory to exercise the is-a-directory check.
        txtdir = os.path.join(self.tmpdir, "notes.txt")
        os.makedirs(txtdir, exist_ok=True)
        with self.assertRaises(ToolExecutionError) as ctx:
            self._execute("notes.txt")
        self.assertIn("directory", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# Extension allow-list
# ---------------------------------------------------------------------------


class TestFileReaderExtensions(FileReaderToolTestBase):

    def _blocked(self, ext: str):
        """Assert that a file with the given extension is blocked."""
        filename = f"test{ext}"
        self._write(filename, "content")
        with self.assertRaises(ToolExecutionError) as ctx:
            self._execute(filename)
        self.assertIn("not permitted", str(ctx.exception))

    def test_py_blocked(self):
        self._blocked(".py")

    def test_exe_blocked(self):
        self._blocked(".exe")

    def test_env_blocked(self):
        self._blocked(".env")

    def test_sh_blocked(self):
        self._blocked(".sh")

    def test_key_blocked(self):
        self._blocked(".key")

    def test_pem_blocked(self):
        self._blocked(".pem")

    def test_no_extension_blocked(self):
        self._write("noext", "data")
        with self.assertRaises(ToolExecutionError):
            self._execute("noext")

    def test_txt_allowed(self):
        self._write("ok.txt", "fine\n")
        result = self._execute("ok.txt")
        self.assertIn("fine", result)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestFileReaderEdgeCases(FileReaderToolTestBase):

    def test_file_too_large_raises_execution_error(self):
        """Files over MAX_FILE_BYTES must be refused before reading."""
        big = self._write("big.txt", "X" * (MAX_FILE_BYTES + 1))
        with self.assertRaises(ToolExecutionError) as ctx:
            self._execute("big.txt")
        self.assertIn("too large", str(ctx.exception).lower())

    def test_content_over_max_return_chars_truncated(self):
        """Content exceeding MAX_RETURN_CHARS must be truncated."""
        content = "A" * (MAX_RETURN_CHARS + 500)
        self._write("long.txt", content)
        result = self._execute("long.txt")
        self.assertIn("truncated", result.lower())
        # The note uses comma-formatted number e.g. "8,000"
        formatted = f"{MAX_RETURN_CHARS:,}"
        self.assertIn(formatted, result)

    def test_non_utf8_bytes_replaced_with_warning(self):
        """Non-UTF-8 bytes must produce a warning in the output."""
        path = os.path.join(self.tmpdir, "latin1.txt")
        with open(path, "wb") as fh:
            fh.write(b"caf\xe9\n")  # 'café' in Latin-1, invalid UTF-8
        result = self._execute("latin1.txt")
        self.assertIn("Warning", result)
        self.assertIn("\ufffd", result)

    def test_file_not_found_raises_execution_error(self):
        with self.assertRaises(ToolExecutionError) as ctx:
            self._execute("ghost.txt")
        self.assertIn("not found", str(ctx.exception).lower())

    @unittest.skipIf(os.getuid() == 0, "chmod 000 has no effect when running as root")
    def test_permission_denied_raises_execution_error(self):
        path = self._write("locked.txt", "secret\n")
        os.chmod(path, 0o000)
        try:
            with self.assertRaises(ToolExecutionError) as ctx:
                self._execute("locked.txt")
            self.assertIn("permission", str(ctx.exception).lower())
        finally:
            os.chmod(path, 0o644)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestFileReaderArgs(FileReaderToolTestBase):

    def test_missing_filepath_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError) as ctx:
            self.tool.execute({})
        self.assertIn("filepath", str(ctx.exception).lower())

    def test_empty_filepath_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"filepath": "   "})

    def test_none_filepath_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"filepath": None})

    def test_wrong_type_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"filepath": 42})

    def test_null_byte_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"filepath": "file\x00.txt"})


# ---------------------------------------------------------------------------
# Declaration schema
# ---------------------------------------------------------------------------


class TestFileReaderDeclaration(FileReaderToolTestBase):

    def test_name_matches_property(self):
        self.assertEqual(self.tool.get_declaration()["name"], self.tool.name)

    def test_has_description(self):
        d = self.tool.get_declaration()
        self.assertGreater(len(d.get("description", "")), 10)

    def test_filepath_parameter_present(self):
        props = self.tool.get_declaration()["parameters"]["properties"]
        self.assertIn("filepath", props)

    def test_filepath_is_required(self):
        self.assertIn("filepath", self.tool.get_declaration()["parameters"]["required"])

    def test_filepath_type_is_string(self):
        props = self.tool.get_declaration()["parameters"]["properties"]
        self.assertEqual(props["filepath"]["type"], "string")

    def test_allowed_extensions_mentioned_in_description(self):
        desc = self.tool.get_declaration()["description"]
        self.assertIn(".txt", desc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
