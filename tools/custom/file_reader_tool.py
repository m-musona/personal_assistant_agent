"""

FileReaderTool (custom #2) — reads a local plain-text file and returns its
content to the agent.

Security model — directory traversal prevention
-----------------------------------------------
Reading arbitrary files on the host filesystem is the canonical example of
a path-traversal vulnerability. This tool enforces a strict containment
boundary through three independent, layered checks:

  LAYER 1 — Argument sanitisation
      filepath is stripped of leading whitespace and null bytes.
      Any argument that is not a non-empty string is rejected immediately
      with ToolArgumentError before the filesystem is touched.

  LAYER 2 — Extension allow-list
      Only files whose names end with an extension in ALLOWED_EXTENSIONS
      may be read. Binary files (executables, images, archives) and
      sensitive plain-text files (.env, .key, .pem, etc.) are blocked by
      omission — they are simply not on the allow-list.

  LAYER 3 — Canonical path confinement (the primary security gate)
      os.path.realpath() resolves all symlinks and normalises ".." sequences
      BEFORE any filesystem access occurs.  The resolved absolute path is
      then checked with str.startswith() against the resolved BASE_DIR.
      A path that escapes the base directory — whether through "../../etc/passwd",
      symlinks, or encoded sequences — will always resolve to a path that
      does NOT start with BASE_DIR, and is rejected with ToolExecutionError.

      Example:
        BASE_DIR = /home/user/docs          (resolved)
        filepath = ../../../etc/passwd
        realpath = /etc/passwd              <- does NOT start with BASE_DIR
        -> ToolExecutionError: "Access denied: path is outside the allowed directory."

      Note: os.path.realpath() is called before the file exists check,
      so the resolved path is purely a string computation — no filesystem
      access occurs until the containment check passes.

  LAYER 4 — Post-check existence and type validation
      After the path passes the containment check, the file is verified to
      exist and to be a regular file (not a directory, device, or FIFO).
      This prevents information leakage about the filesystem structure.

File size guard
---------------
Files larger than MAX_FILE_BYTES are rejected before being read into memory.
This prevents the agent context window from being accidentally flooded with
multi-megabyte files.

Encoding
--------
Files are read as UTF-8 with `errors="replace"` so that non-UTF-8 bytes
produce a replacement character rather than crashing. A warning is included
in the output when replacement occurred so the user knows the file may have
encoding issues.
"""

from __future__ import annotations

import logging
import os
import unicodedata
from pathlib import Path

from config.settings import FILE_READER_BASE_DIR
from tools.base_tool import BaseTool, ToolArgumentError, ToolExecutionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Maximum file size in bytes that will be read into memory.
# 100 KB is comfortably within a Gemini context window while preventing abuse.
MAX_FILE_BYTES: int = 100_000

# Maximum number of characters included in the returned string.
# Applies after reading; allows slightly more than MAX_FILE_BYTES / 1 byte/char.
MAX_RETURN_CHARS: int = 8_000

# File extensions that are explicitly permitted.
# Everything not on this list is blocked by default (allow-list, not deny-list).
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".log",
        ".xml",
        ".html",
        ".htm",
        ".tex",
    }
)

# Resolved, canonical form of the base directory.
# Computed once at import time so every call uses the same anchor.
_RESOLVED_BASE_DIR: str = os.path.realpath(FILE_READER_BASE_DIR)


class FileReaderTool(BaseTool):
    """
    Reads a local plain-text file within the permitted base directory and
    returns its content.

    The permitted base directory is set via FILE_READER_BASE_DIR in
    config/settings.py (defaults to the process working directory).  Files
    outside this directory are unconditionally refused, regardless of how
    the path is expressed.

    Returned content is truncated to MAX_RETURN_CHARS with a notice so the
    agent can inform the user that the file was partially read.
    """

    @property
    def name(self) -> str:
        return "file_reader"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def execute(self, args: dict) -> str:
        """
        Read the file at args["filepath"] and return its text content.

        Parameters
        ----------
        args : dict
            "filepath" (str, required) — path to the file, relative to the
            base directory or absolute (absolute paths are still checked
            against the base directory).

        Returns
        -------
        str
            File content (truncated to MAX_RETURN_CHARS if necessary), with
            metadata header:
            "File: notes.txt (1 234 bytes, 42 lines)
             ---
             <content>"

        Raises
        ------
        ToolArgumentError
            If "filepath" is missing, empty, contains null bytes, or is not
            a string.
        ToolExecutionError
            If the path escapes the base directory, the extension is not
            allowed, the file does not exist, is not a regular file, exceeds
            MAX_FILE_BYTES, or cannot be read due to OS-level permissions.
        """
        raw_path = self._extract_filepath(args)
        resolved = self._resolve_and_validate(raw_path)
        content = self._read_file(resolved)
        return self._format_output(resolved, content)

    def get_declaration(self) -> dict:
        extensions = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return {
            "name": "file_reader",
            "description": (
                "Reads the content of a local plain-text file and returns it "
                "as a string. Only files within the configured base directory "
                f"are accessible. Allowed extensions: {extensions}. "
                "Use this when the user asks you to read, summarise, or "
                "analyse a local file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": (
                            "Path to the file to read. Can be relative to the "
                            "base directory (e.g. 'notes.txt', 'data/report.csv') "
                            "or an absolute path. The path must remain within "
                            "the permitted base directory."
                        ),
                    },
                },
                "required": ["filepath"],
            },
        }

    # ------------------------------------------------------------------
    # Layer 1 — Argument sanitisation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_filepath(args: dict) -> str:
        """
        Pull, type-check, and sanitise the 'filepath' argument.

        Raises ToolArgumentError for any invalid input before the
        filesystem is touched.
        """
        raw = args.get("filepath")

        if raw is None:
            raise ToolArgumentError(
                "Missing required argument: 'filepath'. "
                "Provide a filename such as 'notes.txt'."
            )
        if not isinstance(raw, str):
            raise ToolArgumentError(
                f"'filepath' must be a string, got {type(raw).__name__!r}."
            )

        # Null bytes in a path are used in some injection attacks.
        if "\x00" in raw:
            raise ToolArgumentError("'filepath' must not contain null bytes.")

        # Strip leading/trailing whitespace; reject if nothing remains.
        cleaned = raw.strip()
        if not cleaned:
            raise ToolArgumentError(
                "'filepath' must not be empty. "
                "Provide a filename such as 'notes.txt'."
            )

        return cleaned

    # ------------------------------------------------------------------
    # Layers 2 + 3 + 4 — Extension check, path confinement, existence
    # ------------------------------------------------------------------

    def _resolve_and_validate(self, raw_path: str) -> str:
        """
        Resolve the path to its canonical absolute form and validate it
        against all security and existence constraints.

        Returns the resolved absolute path string on success.
        Raises ToolExecutionError for any security or filesystem violation.
        """
        # --- Layer 3: canonical path confinement (FIRST) ---------------
        # Resolve the path before any other checks so traversal attempts
        # are caught regardless of the file's extension.
        if not os.path.isabs(raw_path):
            candidate = os.path.join(_RESOLVED_BASE_DIR, raw_path)
        else:
            candidate = raw_path

        resolved = os.path.realpath(candidate)

        safe_base = _RESOLVED_BASE_DIR.rstrip(os.sep) + os.sep
        if resolved != _RESOLVED_BASE_DIR and not resolved.startswith(safe_base):
            logger.warning(
                "Path traversal attempt blocked: raw=%r resolved=%r base=%r",
                raw_path,
                resolved,
                _RESOLVED_BASE_DIR,
            )
            raise ToolExecutionError(
                "Access denied: the path is outside the permitted directory. "
                f"All files must be within: {FILE_READER_BASE_DIR}"
            )

        # --- Layer 2: extension allow-list (SECOND) --------------------
        # Checked after confinement so traversal paths are always blocked
        # with a clear "outside permitted directory" message, not a
        # misleading "file type not permitted" message.
        suffix = Path(resolved).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            raise ToolExecutionError(
                f"File type {suffix!r} is not permitted. "
                f"Allowed extensions: {allowed}. "
                "Only plain-text files may be read."
            )

        # --- Layer 4: existence and type validation --------------------
        if not os.path.exists(resolved):
            # Use a neutral message — don't reveal whether the path WOULD
            # be allowed if it existed (information leakage).
            raise ToolExecutionError(
                f"File not found: {os.path.relpath(resolved, _RESOLVED_BASE_DIR)!r}. "
                "Check the filename and ensure the file is in the base directory."
            )

        if not os.path.isfile(resolved):
            kind = "a directory" if os.path.isdir(resolved) else "not a regular file"
            raise ToolExecutionError(
                f"{os.path.relpath(resolved, _RESOLVED_BASE_DIR)!r} is {kind}. "
                "Only regular files may be read."
            )

        return resolved

    # ------------------------------------------------------------------
    # File reading
    # ------------------------------------------------------------------

    def _read_file(self, resolved_path: str) -> str:
        """
        Read the file at resolved_path and return its text content.

        Enforces MAX_FILE_BYTES before reading and MAX_RETURN_CHARS after.
        Uses UTF-8 with `errors='replace'` so encoding issues never crash
        the agent.

        Raises ToolExecutionError for oversized files and OS permission errors.
        """
        file_size = os.path.getsize(resolved_path)

        if file_size > MAX_FILE_BYTES:
            raise ToolExecutionError(
                f"File is too large to read: "
                f"{file_size:,} bytes (limit: {MAX_FILE_BYTES:,} bytes). "
                "Consider reading a smaller file or splitting it first."
            )

        try:
            with open(resolved_path, encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except PermissionError as exc:
            raise ToolExecutionError(
                f"Permission denied reading "
                f"{os.path.relpath(resolved_path, _RESOLVED_BASE_DIR)!r}. "
                "The file exists but cannot be opened."
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(f"Could not read the file: {exc}") from exc

        # Detect whether replacement characters were introduced.
        has_replacement = "\ufffd" in content
        if has_replacement:
            logger.warning("Non-UTF-8 bytes replaced in %r", resolved_path)

        # Truncate if necessary.
        truncated = False
        if len(content) > MAX_RETURN_CHARS:
            content = content[:MAX_RETURN_CHARS]
            truncated = True

        if has_replacement:
            content += (
                "\n\n[Warning: some characters could not be decoded as UTF-8 "
                "and have been replaced with \ufffd.]"
            )
        if truncated:
            content += (
                f"\n\n[Note: content truncated to {MAX_RETURN_CHARS:,} characters. "
                f"The full file is {file_size:,} bytes.]"
            )

        return content

    # ------------------------------------------------------------------
    # Output formatter
    # ------------------------------------------------------------------

    @staticmethod
    def _format_output(resolved_path: str, content: str) -> str:
        """
        Wrap the file content with a one-line metadata header.

        Returns:
            "File: notes.txt (1 234 bytes, 42 lines)
             ---
             <content>"
        """
        filename = os.path.basename(resolved_path)
        file_size = os.path.getsize(resolved_path)
        line_count = content.count("\n") + (
            1 if content and not content.endswith("\n") else 0
        )

        header = (
            f"File: {filename} "
            f"({file_size:,} bytes, {line_count} line{'s' if line_count != 1 else ''})"
        )
        return f"{header}\n{'-' * len(header)}\n{content}"
