"""

SearchTool - retrieves a concise summary for a query using two backends
in a priority cascade:

  1. Wikipedia REST API  (primary)
     Hits the /page/summary/{title} endpoint.  Returns the first paragraph
     of the article: title, description, and extract (≤ MAX_EXTRACT_CHARS).
     No API key required.

  2. DuckDuckGo Instant Answer API  (fallback)
     Used when Wikipedia returns no article or the topic is not encyclopedic
     (current events, products, places without a Wikipedia page, etc.).
     Returns the "AbstractText" from DuckDuckGo's zero-click JSON endpoint.
     No API key required.

Backend cascade logic
---------------------
  a. Try Wikipedia for the query as-is.
  b. If Wikipedia returns a disambiguation page, try each listed alternative
     until one resolves to a real article (up to MAX_DISAMBIG_TRIES).
  c. If Wikipedia still yields nothing, fall back to DuckDuckGo.
  d. If DuckDuckGo also has no result, raise ToolExecutionError with a
     suggestion to rephrase the query.

Error handling
--------------
  - Network / timeout  -> ToolExecutionError (retry suggestion)
  - No results found   -> ToolExecutionError (rephrase suggestion)
  - Missing argument   -> ToolArgumentError
"""

from __future__ import annotations

import json
import logging
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from tools.base_tool import BaseTool, ToolArgumentError, ToolExecutionError

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT: int = 10
MAX_EXTRACT_CHARS: int = 600  # trim long Wikipedia extracts to this length
MAX_DISAMBIG_TRIES: int = 3  # max alternatives to try on disambiguation

_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
_WIKI_SEARCH_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=query&list=search&srsearch={}&srlimit=3&format=json"
)
_DDG_URL = "https://api.duckduckgo.com/?q={}&format=json&no_redirect=1&no_html=1"


class SearchTool(BaseTool):
    """
    Searches Wikipedia (with DuckDuckGo fallback) and returns a short summary.

    Designed to give the agent quick factual context without consuming many
    tokens.  Long Wikipedia extracts are truncated to MAX_EXTRACT_CHARS and
    a source URL is appended so users can read more.
    """

    @property
    def name(self) -> str:
        """Return the unique tool identifier used by ToolRegistry."""
        return "search"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def execute(self, args: dict) -> str:
        """
        Search for a query and return a concise plain-text summary.

        Parameters
        ----------
        args : dict
            "query" (str, required) - the topic or question to search for.
            "language" (str, optional) - Wikipedia language code, default "en".

        Returns
        -------
        str
            A short summary with source attribution, e.g.:
            "Python (programming language) - Wikipedia
             Python is a high-level, general-purpose programming language...
             Source: https://en.wikipedia.org/wiki/Python_(programming_language)"

        Raises
        ------
        ToolArgumentError
            If "query" is missing or empty.
        ToolExecutionError
            If no results are found on either backend, or the network fails.
        """
        query, language = self._extract_args(args)

        # --- 1. Try Wikipedia -------------------------------------------
        result = self._try_wikipedia(query, language)
        if result:
            return result

        # --- 2. Fall back to DuckDuckGo ---------------------------------
        logger.debug("Wikipedia found nothing for %r - trying DuckDuckGo.", query)
        result = self._try_duckduckgo(query)
        if result:
            return result

        raise ToolExecutionError(
            f"No results found for '{query}'. "
            "Try rephrasing the query, using a more specific term, or asking "
            "about a different aspect of the topic."
        )

    def get_declaration(self) -> dict:
        """Return the Gemini function-calling schema for this tool."""
        return {
            "name": "search",
            "description": (
                "Searches Wikipedia (with DuckDuckGo as a fallback) and returns "
                "a concise factual summary about a topic. Use this to look up "
                "definitions, historical facts, biographical information, "
                "scientific concepts, current events, or anything the user asks "
                "about that you are uncertain of or that may have changed since "
                "your training. Do not use this for calculations or weather."
            ),
            "parameters": {
                # "type": "object",
                "type": "OBJECT",
                "properties": {
                    "query": {
                        # "type": "string",
                        "type": "STRING",
                        "description": (
                            "The topic or question to search for. "
                            "Use clear, specific terms. "
                            "Examples: 'Python programming language', "
                            "'Marie Curie', 'photosynthesis', "
                            "'FIFA World Cup 2022 winner'."
                        ),
                    },
                    "language": {
                        # "type": "string",
                        "type": "STRING",
                        "description": (
                            "Wikipedia language edition to search. "
                            "Use a BCP-47 code such as 'en', 'fr', 'de', 'es'. "
                            "Defaults to 'en' if omitted."
                        ),
                    },
                },
                "required": ["query"],
            },
        }

    # ------------------------------------------------------------------
    # Argument extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_args(args: dict) -> tuple[str, str]:
        """Validate and return (query, language) from the args dict."""
        query = args.get("query")
        if query is None:
            raise ToolArgumentError(
                "Missing required argument: 'query'. "
                "Provide a search term such as 'Albert Einstein'."
            )
        if not isinstance(query, str):
            raise ToolArgumentError(
                f"'query' must be a string, got {type(query).__name__!r}."
            )
        query = query.strip()
        if not query:
            raise ToolArgumentError("'query' must not be empty.")

        language = str(args.get("language", "en")).strip().lower() or "en"
        return query, language

    # ------------------------------------------------------------------
    # Wikipedia backend
    # ------------------------------------------------------------------

    def _try_wikipedia(self, query: str, language: str = "en") -> str | None:
        """
        Attempt to fetch a Wikipedia summary for the query.

        Returns a formatted string on success, or None when no article is
        found (so the caller can fall through to the DuckDuckGo backend).
        """
        # Step 1: direct title lookup (handles most well-known topics).
        result = self._wiki_summary_by_title(query, language)
        if result:
            return result

        # Step 2: full-text search to find the best matching article title.
        titles = self._wiki_search_titles(query, language)
        for title in titles[:MAX_DISAMBIG_TRIES]:
            result = self._wiki_summary_by_title(title, language)
            if result:
                return result

        return None

    def _wiki_summary_by_title(self, title: str, language: str = "en") -> str | None:
        """
        Fetch the /page/summary/{title} endpoint for a Wikipedia article.

        Returns a formatted string, or None if the article does not exist or
        is a disambiguation page.
        """
        encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
        base = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{encoded}"

        try:
            raw = self._http_get(base)
        except ToolExecutionError as exc:
            # Only swallow "not found" (404); re-raise network/timeout/5xx
            # so they surface to the caller rather than silently falling through.
            if "404" in str(exc):
                return None
            raise

        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            return None

        page_type = data.get("type", "")
        if page_type == "disambiguation":
            logger.debug("Wikipedia disambiguation page for %r.", title)
            return None
        if page_type not in ("standard", ""):
            return None

        extract: str = data.get("extract", "").strip()
        if not extract:
            return None

        page_title: str = data.get("title", title)
        description: str = data.get("description", "")
        page_url: str = (
            data.get("content_urls", {})
            .get("desktop", {})
            .get("page", f"https://{language}.wikipedia.org/wiki/{encoded}")
        )

        # Trim long extracts and add an ellipsis.
        if len(extract) > MAX_EXTRACT_CHARS:
            extract = extract[:MAX_EXTRACT_CHARS].rsplit(" ", 1)[0] + "…"

        header = f"{page_title}"
        if description:
            header += f" - {description}"

        return f"{header}\n\n{extract}\n\nSource: {page_url}"

    def _wiki_search_titles(self, query: str, language: str = "en") -> list[str]:
        """
        Use Wikipedia's search API to find article titles matching `query`.

        Returns a list of title strings (may be empty on error or no results).
        """
        encoded = urllib.parse.quote_plus(query)
        url = (
            f"https://{language}.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={encoded}"
            f"&srlimit={MAX_DISAMBIG_TRIES}&format=json"
        )
        try:
            raw = self._http_get(url)
            data = json.loads(raw)
            results = data.get("query", {}).get("search", [])
            return [r["title"] for r in results]
        except ToolExecutionError as exc:
            if "404" in str(exc):
                return []
            raise
        except (json.JSONDecodeError, KeyError):
            return []

    # ------------------------------------------------------------------
    # DuckDuckGo fallback backend
    # ------------------------------------------------------------------

    def _try_duckduckgo(self, query: str) -> str | None:
        """
        Query DuckDuckGo's Instant Answer (zero-click) JSON API.

        Returns a formatted string on success, or None when DuckDuckGo has
        no abstract text for the query.
        """
        encoded = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_redirect=1&no_html=1"

        try:
            raw = self._http_get(url)
        except ToolExecutionError as exc:
            if "404" in str(exc):
                return None
            raise

        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            return None

        abstract: str = data.get("AbstractText", "").strip()
        if not abstract:
            return None

        source: str = data.get("AbstractSource", "DuckDuckGo")
        url_result = data.get("AbstractURL", "")
        heading: str = data.get("Heading", query)

        if len(abstract) > MAX_EXTRACT_CHARS:
            abstract = abstract[:MAX_EXTRACT_CHARS].rsplit(" ", 1)[0] + "…"

        lines = [f"{heading} - {source}", "", abstract]
        if url_result:
            lines.append(f"\nSource: {url_result}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _http_get(self, url: str) -> str:
        """
        Perform a GET request and return the response body as a UTF-8 string.

        Raises ToolExecutionError on HTTP errors, network failures, and
        timeouts so callers never have to handle urllib exceptions directly.
        """
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "PersonalAssistantAgent/1.0 (educational project)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                return resp.read().decode("utf-8")

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # 404 = article not found; callers treat this as "no result"
                # and fall through to the next backend.
                raise ToolExecutionError(f"Page not found (HTTP 404): {url}") from exc
            # 5xx and other HTTP errors are fatal - surface them immediately.
            raise ToolExecutionError(
                f"HTTP {exc.code} from search service. Please try again later."
            ) from exc

        except urllib.error.URLError as exc:
            # Network-level failure - fatal, surface immediately.
            raise ToolExecutionError(
                f"Could not reach the search service: {exc.reason}. "
                "Check your internet connection."
            ) from exc

        except TimeoutError as exc:
            # Timeout - fatal, surface immediately.
            raise ToolExecutionError(
                "The search service timed out. Please try again in a moment."
            ) from exc
