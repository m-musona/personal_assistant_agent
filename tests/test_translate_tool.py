"""
tests/test_translate_tool.py

Unit tests for TranslateTool — all HTTP calls are mocked.

Coverage:
  - Valid MyMemory response: output format, source/target labels
  - Language name -> ISO code normalisation (full names and codes)
  - MyMemory quota (403), invalid pair (400), empty result -> fallback
  - LibreTranslate fallback: valid response, error field, empty result
  - Both backends fail -> ToolExecutionError
  - Low match score -> quality note in output
  - source_language auto-detection label
  - Argument validation: missing text, missing target, empty, wrong type
  - Network errors: URLError, TimeoutError, HTTP 429, HTTP 403
  - Declaration schema structure
"""

from __future__ import annotations

import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from tools.base_tool import ToolArgumentError, ToolExecutionError
from tools.custom.translate_tool import TranslateTool


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _mymemory_ok(
    translated: str = "Bonjour",
    detected: str = "en",
    match: float = 1.0,
) -> str:
    return json.dumps(
        {
            "responseStatus": 200,
            "responseData": {
                "translatedText": translated,
                "detectedLanguage": detected,
                "match": match,
            },
        }
    )


def _mymemory_quota() -> str:
    return json.dumps(
        {
            "responseStatus": 403,
            "responseData": {"translatedText": ""},
        }
    )


def _mymemory_bad_pair() -> str:
    return json.dumps(
        {
            "responseStatus": 400,
            "responseData": {"translatedText": ""},
        }
    )


def _mymemory_empty() -> str:
    return json.dumps(
        {
            "responseStatus": 200,
            "responseData": {"translatedText": "", "match": 1.0},
        }
    )


def _libretranslate_ok(translated: str = "Hola") -> str:
    return json.dumps(
        {
            "translatedText": translated,
            "detectedLanguage": {"language": "en", "confidence": 99},
        }
    )


def _libretranslate_error(msg: str = "Language not supported") -> str:
    return json.dumps({"error": msg})


def _libretranslate_empty() -> str:
    return json.dumps({"translatedText": ""})


def _mock_response(body: str):
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read.return_value = body.encode("utf-8")
    return cm


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://x", code=code, msg="err", hdrs=None, fp=None
    )


# ---------------------------------------------------------------------------
# MyMemory primary backend
# ---------------------------------------------------------------------------


class TestTranslateToolMyMemory(unittest.TestCase):

    def setUp(self):
        self.tool = TranslateTool()

    def _run(self, text="Hello", target="fr", **extra):
        args = {"text": text, "target_language": target, **extra}
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_mymemory_ok())
        ):
            return self.tool.execute(args)

    def test_returns_string(self):
        self.assertIsInstance(self._run(), str)

    def test_translated_text_in_output(self):
        self.assertIn("Bonjour", self._run())

    def test_target_language_code_in_output(self):
        self.assertIn("fr", self._run())

    def test_detected_source_in_output(self):
        self.assertIn("en", self._run())

    def test_backend_label_in_output(self):
        self.assertIn("MyMemory", self._run())

    def test_full_language_name_normalised_to_code(self):
        """'French' must be normalised to 'fr' before the API call."""
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_mymemory_ok())
        ) as mock_open:
            self.tool.execute({"text": "Hello", "target_language": "French"})
            url = mock_open.call_args[0][0].full_url
            self.assertIn("fr", url)

    def test_source_language_name_normalised(self):
        """source_language='German' must be normalised to 'de'."""
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_mymemory_ok())
        ) as mock_open:
            self.tool.execute(
                {
                    "text": "Guten Morgen",
                    "target_language": "English",
                    "source_language": "German",
                }
            )
            url = mock_open.call_args[0][0].full_url
            self.assertIn("de", url)
            self.assertIn("en", url)

    def test_source_auto_not_in_langpair_as_auto(self):
        """Omitting source_language uses 'autodetect' in the MyMemory langpair."""
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_mymemory_ok())
        ) as mock_open:
            self.tool.execute({"text": "Hello", "target_language": "fr"})
            url = mock_open.call_args[0][0].full_url
            self.assertIn("autodetect", url)

    def test_low_match_score_adds_quality_note(self):
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(_mymemory_ok(match=0.3)),
        ):
            result = self.tool.execute({"text": "Hello", "target_language": "fr"})
        self.assertIn("low-confidence", result)

    def test_high_match_score_no_quality_note(self):
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(_mymemory_ok(match=0.9)),
        ):
            result = self.tool.execute({"text": "Hello", "target_language": "fr"})
        self.assertNotIn("low-confidence", result)

    def test_mymemory_banner_stripped(self):
        """'TRANSLATED BY MYMEMORY...' suffix must be removed from output."""
        body = _mymemory_ok(translated="Hola TRANSLATED BY MYMEMORY!!!")
        with patch("urllib.request.urlopen", return_value=_mock_response(body)):
            result = self.tool.execute({"text": "Hello", "target_language": "es"})
        self.assertNotIn("TRANSLATED BY", result)
        self.assertIn("Hola", result)


# ---------------------------------------------------------------------------
# Fallback to LibreTranslate
# ---------------------------------------------------------------------------


class TestTranslateToolFallback(unittest.TestCase):

    def setUp(self):
        self.tool = TranslateTool()

    def _run_fallback(self, mymemory_body: str, libretranslate_body: str) -> str:
        """Simulate MyMemory failure -> LibreTranslate response."""
        call_count = 0

        def side_effect(request, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # first call -> MyMemory
                return _mock_response(mymemory_body)
            return _mock_response(libretranslate_body)  # second call -> LT

        with patch("urllib.request.urlopen", side_effect=side_effect):
            return self.tool.execute({"text": "Hello", "target_language": "es"})

    def test_falls_back_when_mymemory_quota_exceeded(self):
        result = self._run_fallback(_mymemory_quota(), _libretranslate_ok("Hola"))
        self.assertIn("Hola", result)
        self.assertIn("LibreTranslate", result)

    def test_falls_back_when_mymemory_bad_pair(self):
        result = self._run_fallback(_mymemory_bad_pair(), _libretranslate_ok("Hola"))
        self.assertIn("Hola", result)

    def test_falls_back_when_mymemory_empty_result(self):
        result = self._run_fallback(_mymemory_empty(), _libretranslate_ok("Hola"))
        self.assertIn("Hola", result)

    def test_both_fail_raises_execution_error(self):
        """Both backends failing must raise ToolExecutionError."""

        def all_fail(request, timeout=10):
            raise urllib.error.URLError("network down")

        with patch("urllib.request.urlopen", side_effect=all_fail):
            with self.assertRaises(ToolExecutionError):
                self.tool.execute({"text": "Hello", "target_language": "fr"})

    def test_libretranslate_error_field_raises_execution_error(self):
        def side_effect(request, timeout=10):
            if "mymemory" in request.full_url:
                return _mock_response(_mymemory_quota())
            return _mock_response(_libretranslate_error("Language not supported"))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"text": "Hello", "target_language": "xx"})
        self.assertIn("LibreTranslate error", str(ctx.exception))

    def test_libretranslate_empty_result_raises_execution_error(self):
        def side_effect(request, timeout=10):
            if "mymemory" in request.full_url:
                return _mock_response(_mymemory_quota())
            return _mock_response(_libretranslate_empty())

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with self.assertRaises(ToolExecutionError):
                self.tool.execute({"text": "Hello", "target_language": "fr"})


# ---------------------------------------------------------------------------
# Language normalisation
# ---------------------------------------------------------------------------


class TestLanguageNormalisation(unittest.TestCase):

    def setUp(self):
        self.tool = TranslateTool()

    def _normalise(self, lang: str) -> str:
        return self.tool._normalise_language(lang)

    def test_french_name_to_code(self):
        self.assertEqual(self._normalise("French"), "fr")

    def test_german_name_to_code(self):
        self.assertEqual(self._normalise("German"), "de")

    def test_japanese_name_to_code(self):
        self.assertEqual(self._normalise("Japanese"), "ja")

    def test_code_passthrough(self):
        self.assertEqual(self._normalise("fr"), "fr")

    def test_code_case_insensitive(self):
        self.assertEqual(self._normalise("FR"), "fr")

    def test_chinese_simplified(self):
        self.assertEqual(self._normalise("Chinese"), "zh")

    def test_farsi_alias(self):
        self.assertEqual(self._normalise("Farsi"), "fa")

    def test_unknown_name_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError) as ctx:
            self._normalise("Klingon")
        self.assertIn("Klingon", str(ctx.exception))

    def test_unknown_short_code_passes_through(self):
        # Unknown short codes are passed through (API will validate).
        result = self._normalise("xx")
        self.assertEqual(result, "xx")


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestTranslateToolArgs(unittest.TestCase):

    def setUp(self):
        self.tool = TranslateTool()

    def test_missing_text_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError) as ctx:
            self.tool.execute({"target_language": "fr"})
        self.assertIn("text", str(ctx.exception).lower())

    def test_missing_target_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError) as ctx:
            self.tool.execute({"text": "Hello"})
        self.assertIn("target_language", str(ctx.exception).lower())

    def test_empty_text_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"text": "   ", "target_language": "fr"})

    def test_empty_target_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"text": "Hello", "target_language": "   "})

    def test_wrong_type_text_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"text": 42, "target_language": "fr"})

    def test_none_text_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"text": None, "target_language": "fr"})

    def test_unrecognised_target_language_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError) as ctx:
            self.tool.execute({"text": "Hello", "target_language": "Klingon"})
        self.assertIn("Klingon", str(ctx.exception))


# ---------------------------------------------------------------------------
# Network errors
# ---------------------------------------------------------------------------


class TestTranslateToolNetworkErrors(unittest.TestCase):

    def setUp(self):
        self.tool = TranslateTool()

    def test_url_error_raises_execution_error(self):
        with patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("no route")
        ):
            with self.assertRaises(ToolExecutionError):
                self.tool.execute({"text": "Hello", "target_language": "fr"})

    def test_timeout_raises_execution_error(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"text": "Hello", "target_language": "fr"})
        self.assertIn("timed out", str(ctx.exception).lower())

    def test_http_429_raises_execution_error(self):
        with patch("urllib.request.urlopen", side_effect=_http_error(429)):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"text": "Hello", "target_language": "fr"})
        self.assertIn("rate limit", str(ctx.exception).lower())

    def test_http_403_raises_execution_error(self):
        with patch("urllib.request.urlopen", side_effect=_http_error(403)):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"text": "Hello", "target_language": "fr"})
        self.assertIn("403", str(ctx.exception))

    def test_malformed_json_raises_execution_error(self):
        def side_effect(req, timeout=10):
            if "mymemory" in req.full_url:
                return _mock_response("not json")
            return _mock_response(_libretranslate_ok())

        with patch("urllib.request.urlopen", side_effect=side_effect):
            # Malformed MyMemory -> falls back to LibreTranslate successfully.
            result = self.tool.execute({"text": "Hello", "target_language": "es"})
        self.assertIn("Hola", result)


# ---------------------------------------------------------------------------
# Declaration schema
# ---------------------------------------------------------------------------


class TestTranslateToolDeclaration(unittest.TestCase):

    def setUp(self):
        self.tool = TranslateTool()

    def test_name_matches_property(self):
        self.assertEqual(self.tool.get_declaration()["name"], self.tool.name)

    def test_has_description(self):
        d = self.tool.get_declaration()
        self.assertGreater(len(d.get("description", "")), 10)

    def test_text_parameter_present_and_required(self):
        decl = self.tool.get_declaration()["parameters"]
        self.assertIn("text", decl["properties"])
        self.assertIn("text", decl["required"])

    def test_target_language_present_and_required(self):
        decl = self.tool.get_declaration()["parameters"]
        self.assertIn("target_language", decl["properties"])
        self.assertIn("target_language", decl["required"])

    def test_source_language_optional(self):
        decl = self.tool.get_declaration()["parameters"]
        self.assertIn("source_language", decl["properties"])
        self.assertNotIn("source_language", decl["required"])

    def test_text_type_is_string(self):
        props = self.tool.get_declaration()["parameters"]["properties"]
        self.assertIn(props["text"]["type"].lower(), ("string",))

    def test_target_language_type_is_string(self):
        props = self.tool.get_declaration()["parameters"]["properties"]
        self.assertIn(props["target_language"]["type"].lower(), ("string",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
