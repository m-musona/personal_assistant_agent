"""

Unit tests for SearchTool — all HTTP calls are mocked.

Coverage:
  - Wikipedia standard article: title, description, extract, URL in output
  - Wikipedia extract truncation at MAX_EXTRACT_CHARS
  - Wikipedia disambiguation falls through to search titles
  - Wikipedia search-title fallback resolves a real article
  - DuckDuckGo fallback when Wikipedia returns nothing
  - No results from either backend -> ToolExecutionError
  - Network errors on Wikipedia -> DuckDuckGo attempted
  - Network errors on both -> ToolExecutionError
  - Timeout -> ToolExecutionError
  - language arg changes Wikipedia subdomain
  - Argument validation: missing, empty, wrong type
  - Declaration schema structure
"""

from __future__ import annotations

import json
import unittest
import urllib.error
from unittest.mock import MagicMock, call, patch

from tools.base_tool import ToolArgumentError, ToolExecutionError
from tools.built_in.search_tool import MAX_EXTRACT_CHARS, SearchTool


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _wiki_standard(
    title: str = "Python",
    description: str = "programming language",
    extract: str = "Python is a high-level language.",
    url: str = "https://en.wikipedia.org/wiki/Python",
) -> str:
    return json.dumps(
        {
            "type": "standard",
            "title": title,
            "description": description,
            "extract": extract,
            "content_urls": {"desktop": {"page": url}},
        }
    )


def _wiki_disambiguation() -> str:
    return json.dumps({"type": "disambiguation", "title": "Python"})


def _wiki_not_found() -> str:
    """Wikipedia returns a 404-shaped JSON for missing pages."""
    return json.dumps(
        {"type": "https://mediawiki.org/wiki/HyperSwitch/errors/not_found"}
    )


def _wiki_search_results(*titles: str) -> str:
    return json.dumps({"query": {"search": [{"title": t} for t in titles]}})


def _ddg_result(
    heading: str = "Python",
    abstract: str = "Python is a programming language.",
    source: str = "Wikipedia",
    url: str = "https://en.wikipedia.org/wiki/Python",
) -> str:
    return json.dumps(
        {
            "Heading": heading,
            "AbstractText": abstract,
            "AbstractSource": source,
            "AbstractURL": url,
        }
    )


def _ddg_empty() -> str:
    return json.dumps({"AbstractText": "", "Heading": "", "AbstractSource": ""})


def _mock_response(body: str):
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read.return_value = body.encode("utf-8")
    return cm


def _http_404():
    return urllib.error.HTTPError(
        url="http://x", code=404, msg="Not Found", hdrs=None, fp=None
    )


def _url_error():
    return urllib.error.URLError("connection refused")


# ---------------------------------------------------------------------------
# Wikipedia primary backend
# ---------------------------------------------------------------------------


class TestSearchToolWikipedia(unittest.TestCase):

    def setUp(self):
        self.tool = SearchTool()

    def test_returns_string(self):
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_wiki_standard())
        ):
            result = self.tool.execute({"query": "Python"})
        self.assertIsInstance(result, str)

    def test_contains_title(self):
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(_wiki_standard(title="Python")),
        ):
            result = self.tool.execute({"query": "Python"})
        self.assertIn("Python", result)

    def test_contains_description(self):
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(
                _wiki_standard(description="programming language")
            ),
        ):
            result = self.tool.execute({"query": "Python"})
        self.assertIn("programming language", result)

    def test_contains_extract(self):
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(
                _wiki_standard(extract="Python is a language.")
            ),
        ):
            result = self.tool.execute({"query": "Python"})
        self.assertIn("Python is a language.", result)

    def test_contains_source_url(self):
        url = "https://en.wikipedia.org/wiki/Python"
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(_wiki_standard(url=url)),
        ):
            result = self.tool.execute({"query": "Python"})
        self.assertIn(url, result)

    def test_long_extract_truncated(self):
        long_extract = "A" * (MAX_EXTRACT_CHARS + 200)
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(_wiki_standard(extract=long_extract)),
        ):
            result = self.tool.execute({"query": "Python"})
        # Output must be shorter than raw extract + header overhead
        self.assertLessEqual(len(result), MAX_EXTRACT_CHARS + 300)
        self.assertIn("…", result)

    def test_disambiguation_triggers_search_fallback(self):
        """
        Disambiguation page on direct lookup -> search API called ->
        second title resolves to a real article.
        """
        responses = [
            _mock_response(_wiki_disambiguation()),  # direct title lookup
            _mock_response(_wiki_search_results("Python (programming language)")),
            _mock_response(_wiki_standard(title="Python (programming language)")),
        ]
        with patch("urllib.request.urlopen", side_effect=responses):
            result = self.tool.execute({"query": "Python"})
        self.assertIn("Python", result)

    def test_language_arg_changes_subdomain(self):
        """language='fr' must hit fr.wikipedia.org."""
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_wiki_standard())
        ) as mock_open:
            self.tool.execute({"query": "Python", "language": "fr"})
            url_used = mock_open.call_args[0][0].full_url
            self.assertIn("fr.wikipedia.org", url_used)


# ---------------------------------------------------------------------------
# DuckDuckGo fallback
# ---------------------------------------------------------------------------


class TestSearchToolDuckDuckGo(unittest.TestCase):

    def setUp(self):
        self.tool = SearchTool()

    def _run_with_ddg_fallback(self, ddg_body: str) -> str:
        """
        Simulate: Wikipedia 404 on direct lookup + empty search results
        -> DuckDuckGo fallback.
        """

        def side_effect(request, timeout=10):
            url = request.full_url
            if "wikipedia.org" in url:
                raise _http_404()
            return _mock_response(ddg_body)

        with patch("urllib.request.urlopen", side_effect=side_effect):
            return self.tool.execute({"query": "some topic"})

    def test_ddg_result_returned_when_wiki_empty(self):
        result = self._run_with_ddg_fallback(_ddg_result())
        self.assertIn("Python is a programming language", result)

    def test_ddg_heading_in_output(self):
        result = self._run_with_ddg_fallback(_ddg_result(heading="Python"))
        self.assertIn("Python", result)

    def test_ddg_source_url_in_output(self):
        result = self._run_with_ddg_fallback(
            _ddg_result(url="https://en.wikipedia.org/wiki/Python")
        )
        self.assertIn("https://en.wikipedia.org/wiki/Python", result)

    def test_ddg_long_abstract_truncated(self):
        long_abstract = "B" * (MAX_EXTRACT_CHARS + 300)
        result = self._run_with_ddg_fallback(_ddg_result(abstract=long_abstract))
        self.assertIn("…", result)

    def test_both_backends_empty_raises_execution_error(self):
        """No results from Wikipedia or DuckDuckGo -> ToolExecutionError."""

        def side_effect(request, timeout=10):
            url = request.full_url
            if "wikipedia.org" in url:
                raise _http_404()
            return _mock_response(_ddg_empty())

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"query": "xyzzy_nonexistent_topic"})
        self.assertIn("No results", str(ctx.exception))


# ---------------------------------------------------------------------------
# Network error handling
# ---------------------------------------------------------------------------


class TestSearchToolNetworkErrors(unittest.TestCase):

    def setUp(self):
        self.tool = SearchTool()

    def test_wiki_network_error_raises_execution_error(self):
        """A network-level URLError must surface immediately as ToolExecutionError."""
        with patch("urllib.request.urlopen", side_effect=_url_error()):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"query": "Python"})
        self.assertIn("connection refused", str(ctx.exception).lower())

    def test_both_backends_network_error_raises_execution_error(self):
        with patch("urllib.request.urlopen", side_effect=_url_error()):
            with self.assertRaises(ToolExecutionError):
                self.tool.execute({"query": "Python"})

    def test_timeout_raises_execution_error(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"query": "Python"})
        self.assertIn("timed out", str(ctx.exception).lower())

    def test_http_500_raises_execution_error(self):
        err = urllib.error.HTTPError(
            url="http://x", code=500, msg="Server Error", hdrs=None, fp=None
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(ToolExecutionError) as ctx:
                self.tool.execute({"query": "Python"})
        self.assertIn("500", str(ctx.exception))

    def test_malformed_json_falls_through(self):
        """Bad JSON from Wikipedia -> DuckDuckGo fallback attempted."""

        def side_effect(request, timeout=10):
            url = request.full_url
            if "wikipedia.org" in url:
                return _mock_response("not json at all")
            return _mock_response(_ddg_result())

        with patch("urllib.request.urlopen", side_effect=side_effect):
            result = self.tool.execute({"query": "Python"})
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestSearchToolArgs(unittest.TestCase):

    def setUp(self):
        self.tool = SearchTool()

    def test_missing_query_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError) as ctx:
            self.tool.execute({})
        self.assertIn("query", str(ctx.exception).lower())

    def test_empty_query_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"query": "   "})

    def test_none_query_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"query": None})

    def test_wrong_type_raises_argument_error(self):
        with self.assertRaises(ToolArgumentError):
            self.tool.execute({"query": 123})

    def test_language_defaults_to_en(self):
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_wiki_standard())
        ) as mock_open:
            self.tool.execute({"query": "Python"})
            url_used = mock_open.call_args[0][0].full_url
            self.assertIn("en.wikipedia.org", url_used)

    def test_invalid_language_still_attempts_request(self):
        """An unrecognised language code is passed through; Wikipedia returns 404."""

        def side_effect(request, timeout=10):
            if "wikipedia.org" in request.full_url:
                raise _http_404()
            return _mock_response(_ddg_result())

        with patch("urllib.request.urlopen", side_effect=side_effect):
            # Should not raise ToolArgumentError — language validation is lax.
            result = self.tool.execute({"query": "Python", "language": "xx"})
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# Declaration schema
# ---------------------------------------------------------------------------


class TestSearchToolDeclaration(unittest.TestCase):

    def setUp(self):
        self.tool = SearchTool()

    def test_name_matches_property(self):
        self.assertEqual(self.tool.get_declaration()["name"], self.tool.name)

    def test_has_description(self):
        d = self.tool.get_declaration()
        self.assertIn("description", d)
        self.assertGreater(len(d["description"]), 10)

    def test_query_parameter_present(self):
        props = self.tool.get_declaration()["parameters"]["properties"]
        self.assertIn("query", props)

    def test_query_is_required(self):
        self.assertIn("query", self.tool.get_declaration()["parameters"]["required"])

    def test_language_is_optional(self):
        decl = self.tool.get_declaration()["parameters"]
        self.assertIn("language", decl["properties"])
        self.assertNotIn("language", decl["required"])

    def test_query_type_is_string(self):
        props = self.tool.get_declaration()["parameters"]["properties"]
        self.assertEqual(props["query"]["type"], "string")


if __name__ == "__main__":
    unittest.main(verbosity=2)
