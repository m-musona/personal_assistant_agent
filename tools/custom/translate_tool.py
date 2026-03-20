"""

TranslateTool (custom #1) — translates text using a two-backend cascade:

  Backend 1 — MyMemory  (primary, no key required)
      Free REST API from Translated.com. No registration needed.
      Endpoint: https://api.mymemory.translated.net/get
      Limit: 500 words/day anonymous, 1 000 words/day with a registered email.
      Returns the translated string and a quality match score.

  Backend 2 — LibreTranslate  (fallback, configurable)
      Open-source, self-hostable. Public instances exist but may be throttled.
      Endpoint: LIBRETRANSLATE_URL from config/settings.py
      Activated automatically when MyMemory returns an error response or when
      LIBRETRANSLATE_URL points to a private instance.

Architecture notes
------------------
  Strategy Pattern (OCP / DIP)
      The two backends are implemented as private methods (_translate_mymemory,
      _translate_libretranslate) behind a single execute() entry point.
      Adding a third backend (DeepL, Argos) means adding one method and one
      entry in the cascade — zero changes to the Agent, ToolRegistry, or
      PromptBuilder.

  Language normalisation
      Both ISO 639-1 ("fr") and full names ("French") are accepted.
      _normalise_language() maps common names to codes before the API call,
      so "Translate to French" works as well as "Translate to fr".

Error handling
--------------
  - Missing / empty arguments        -> ToolArgumentError
  - Unsupported language             -> ToolExecutionError (lists supported)
  - Rate limit / quota exceeded      -> ToolExecutionError (retry suggestion)
  - Both backends fail               -> ToolExecutionError
  - Network / timeout                -> ToolExecutionError
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from config.settings import LIBRETRANSLATE_URL
from tools.base_tool import BaseTool, ToolArgumentError, ToolExecutionError

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT: int = 10

# ---------------------------------------------------------------------------
# Language name → ISO 639-1 code mapping
# Covers the most common names a user or LLM might supply.
# ---------------------------------------------------------------------------
_LANGUAGE_ALIASES: dict[str, str] = {
    "afrikaans": "af",
    "albanian": "sq",
    "arabic": "ar",
    "azerbaijani": "az",
    "basque": "eu",
    "belarusian": "be",
    "bengali": "bn",
    "bosnian": "bs",
    "bulgarian": "bg",
    "catalan": "ca",
    "chinese": "zh",
    "chinese simplified": "zh",
    "chinese traditional": "zh-TW",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "esperanto": "eo",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "haitian creole": "ht",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "irish": "ga",
    "italian": "it",
    "japanese": "ja",
    "kannada": "kn",
    "kazakh": "kk",
    "korean": "ko",
    "kurdish": "ku",
    "latvian": "lv",
    "lithuanian": "lt",
    "macedonian": "mk",
    "malay": "ms",
    "maltese": "mt",
    "marathi": "mr",
    "nepali": "ne",
    "norwegian": "no",
    "persian": "fa",
    "farsi": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "sinhala": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "spanish": "es",
    "sundanese": "su",
    "swahili": "sw",
    "swedish": "sv",
    "tagalog": "tl",
    "filipino": "tl",
    "tajik": "tg",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
    "xhosa": "xh",
    "yiddish": "yi",
    "yoruba": "yo",
    "zulu": "zu",
}

# Languages supported by MyMemory's auto-detect + translate route.
# Used to validate the target_language argument before making an API call.
_MYMEMORY_SUPPORTED: frozenset[str] = frozenset(_LANGUAGE_ALIASES.values()) | {
    "zh-TW",
    "zh-CN",
}


class TranslateTool(BaseTool):
    """
    Translates text into a target language using MyMemory (primary) with
    LibreTranslate as a configurable fallback.

    Accepts both ISO 639-1 codes ("fr", "de", "ja") and full language names
    ("French", "German", "Japanese") in the target_language argument.

    Returns the translated text with the detected source language and which
    backend was used, e.g.:

        Translation (English -> French) via MyMemory:
        "Bonjour, comment allez-vous?"
    """

    @property
    def name(self) -> str:
        """Return the unique tool identifier used by ToolRegistry."""
        return "translate"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def execute(self, args: dict) -> str:
        """
        Translate text["text"] into args["target_language"].

        Parameters
        ----------
        args : dict
            "text"            (str, required) — the text to translate.
            "target_language" (str, required) — language name or ISO code.
            "source_language" (str, optional) — source language (default: auto).

        Returns
        -------
        str
            Formatted result string including detected language, backend used,
            and the translated text.

        Raises
        ------
        ToolArgumentError
            If "text" or "target_language" is missing or empty.
        ToolExecutionError
            If the target language is unsupported, the quota is exceeded,
            the network is unavailable, or both backends fail.
        """
        text, target_code, source_code = self._extract_args(args)

        # --- Backend 1: MyMemory ----------------------------------------
        try:
            result = self._translate_mymemory(text, target_code, source_code)
            logger.info("MyMemory translated %d chars -> %r", len(text), target_code)
            return result
        except ToolExecutionError as exc:
            logger.warning("MyMemory failed (%s) — trying LibreTranslate.", exc)

        # --- Backend 2: LibreTranslate ----------------------------------
        result = self._translate_libretranslate(text, target_code, source_code)
        logger.info("LibreTranslate translated %d chars -> %r", len(text), target_code)
        return result

    def get_declaration(self) -> dict:
        """Return the Gemini function-calling schema for this tool."""
        return {
            "name": "translate",
            "description": (
                "Translates text from one language to another. "
                "Accepts both ISO 639-1 language codes (e.g. 'fr', 'de', 'ja') "
                "and full language names (e.g. 'French', 'German', 'Japanese'). "
                "The source language is detected automatically if not specified. "
                "Use this whenever the user asks to translate text or asks what "
                "a phrase means in another language."
            ),
            "parameters": {
                # "type": "object",
                "type": "OBJECT",
                "properties": {
                    "text": {
                        # "type": "string",
                        "type": "STRING",
                        "description": (
                            "The text to translate. Can be a word, phrase, "
                            "sentence, or short paragraph."
                        ),
                    },
                    "target_language": {
                        # "type": "string",
                        "type": "STRING",
                        "description": (
                            "The language to translate into. "
                            "Use a full name ('French') or ISO 639-1 code ('fr'). "
                            "Examples: 'Spanish', 'Japanese', 'ar', 'zh'."
                        ),
                    },
                    "source_language": {
                        # "type": "string",
                        "type": "STRING",
                        "description": (
                            "The language of the input text. "
                            "Omit to let the API detect it automatically. "
                            "Use a full name or ISO 639-1 code."
                        ),
                    },
                },
                "required": ["text", "target_language"],
            },
        }

    # ------------------------------------------------------------------
    # Argument extraction & language normalisation
    # ------------------------------------------------------------------

    def _extract_args(self, args: dict) -> tuple[str, str, str]:
        """
        Validate and extract (text, target_code, source_code).

        target_code and source_code are normalised to ISO 639-1 before return.
        Raises ToolArgumentError on missing / invalid inputs.
        """
        # --- text -------------------------------------------------------
        text = args.get("text")
        if text is None:
            raise ToolArgumentError(
                "Missing required argument: 'text'. "
                "Provide the text you want to translate."
            )
        if not isinstance(text, str):
            raise ToolArgumentError(
                f"'text' must be a string, got {type(text).__name__!r}."
            )
        text = text.strip()
        if not text:
            raise ToolArgumentError("'text' must not be empty.")

        # --- target_language --------------------------------------------
        raw_target = args.get("target_language")
        if raw_target is None:
            raise ToolArgumentError(
                "Missing required argument: 'target_language'. "
                "Provide a language name such as 'French' or a code like 'fr'."
            )
        if not isinstance(raw_target, str) or not raw_target.strip():
            raise ToolArgumentError("'target_language' must be a non-empty string.")
        target_code = self._normalise_language(raw_target.strip())

        # --- source_language (optional, default "auto") -----------------
        raw_source = str(args.get("source_language", "auto")).strip()
        source_code = (
            "auto"
            if raw_source.lower() in ("", "auto", "detect", "automatic")
            else self._normalise_language(raw_source)
        )

        return text, target_code, source_code

    @staticmethod
    def _normalise_language(lang: str) -> str:
        """
        Convert a language name or code to an ISO 639-1 code.

        Accepts:
          - ISO 639-1 codes as-is ("fr", "zh-TW")
          - Full English names via _LANGUAGE_ALIASES ("French" -> "fr")

        Raises ToolArgumentError for unrecognised values.
        """
        # Already a short ISO code (2–3 chars, e.g. "fr", "de", "zh-TW").
        # Codes are at most 3 alpha chars or 5 with a hyphen (e.g. "zh-TW").
        is_short_code = (
            len(lang) <= 5
            and lang.replace("-", "").isalpha()
            and (len(lang) <= 3 or "-" in lang)
        )
        if is_short_code:
            code = lang.lower()
            # Check against the union of alias values for validation.
            if code in _MYMEMORY_SUPPORTED or code == "auto":
                return code
            # Allow unknown short codes through — the API will validate.
            logger.debug("Unrecognised language code %r — passing through.", lang)
            return code

        # Full name lookup.
        code = _LANGUAGE_ALIASES.get(lang.lower())
        if code:
            return code

        raise ToolArgumentError(
            f"Unrecognised language: {lang!r}. "
            "Use a full name such as 'French' or an ISO 639-1 code such as 'fr'. "
            f"Supported names include: "
            + ", ".join(sorted(_LANGUAGE_ALIASES)[:20])
            + ", and more."
        )

    # ------------------------------------------------------------------
    # Backend 1: MyMemory
    # ------------------------------------------------------------------

    def _translate_mymemory(self, text: str, target: str, source: str) -> str:
        """
        Translate via MyMemory's free REST API.

        Endpoint: GET https://api.mymemory.translated.net/get
        Parameters:
            q      — text to translate
            langpair — "source|target" (use "auto|target" for auto-detect)

        MyMemory error signals
        ----------------------
        The API always returns HTTP 200 but signals errors in the JSON:
            responseStatus == 403  -> quota exceeded
            responseStatus == 400  -> invalid language pair
            match < 0              -> translation engine refused the request
        """
        lang_pair = f"{source}|{target}" if source != "auto" else f"autodetect|{target}"
        params = urllib.parse.urlencode(
            {
                "q": text,
                "langpair": lang_pair,
            }
        )
        url = f"https://api.mymemory.translated.net/get?{params}"

        raw = self._http_post_or_get(url, method="GET")

        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                "MyMemory returned an unreadable response. " "Please try again later."
            ) from exc

        status = data.get("responseStatus", 200)

        if status == 403:
            raise ToolExecutionError(
                "MyMemory daily quota exceeded. "
                "Please try again tomorrow or use a shorter text."
            )
        if status == 400:
            raise ToolExecutionError(
                f"MyMemory does not support the language pair "
                f"'{source}' -> '{target}'. "
                "Try a different target language."
            )
        if str(status) != "200":
            raise ToolExecutionError(
                f"MyMemory returned an unexpected status: {status}."
            )

        translated: str = data.get("responseData", {}).get("translatedText", "")
        if not translated:
            raise ToolExecutionError(
                "MyMemory returned an empty translation. "
                "The text may be too short or the language unsupported."
            )

        # MyMemory includes "TRANSLATED BY MYMEMORY..." in some responses.
        if "TRANSLATED BY" in translated.upper():
            translated = translated.split("TRANSLATED BY")[0].strip()

        # Detect the source language that MyMemory identified.
        detected = (
            data.get("responseData", {}).get("detectedLanguage", source) or source
        )
        match_score = data.get("responseData", {}).get("match", 1.0)
        quality_note = (
            " (low-confidence translation)" if float(match_score) < 0.5 else ""
        )

        src_label = detected if detected != "auto" else "auto-detected"
        return (
            f"Translation ({src_label} -> {target}) via MyMemory"
            f'{quality_note}:\n"{translated}"'
        )

    # ------------------------------------------------------------------
    # Backend 2: LibreTranslate
    # ------------------------------------------------------------------

    def _translate_libretranslate(self, text: str, target: str, source: str) -> str:
        """
        Translate via LibreTranslate's POST /translate endpoint.

        LibreTranslate uses "auto" for source language detection natively,
        which maps cleanly to our internal convention.

        Raises ToolExecutionError if the endpoint returns a non-200 response,
        the JSON is malformed, or the network is unreachable.
        """
        payload = json.dumps(
            {
                "q": text,
                "source": source if source != "auto" else "auto",
                "target": target,
                "format": "text",
            }
        ).encode("utf-8")

        try:
            raw = self._http_post_or_get(
                LIBRETRANSLATE_URL, method="POST", body=payload
            )
        except ToolExecutionError as exc:
            raise ToolExecutionError(
                f"LibreTranslate is unavailable: {exc}. "
                "Both translation backends have failed. "
                "Please check your network connection and try again."
            ) from exc

        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                "LibreTranslate returned an unreadable response."
            ) from exc

        if "error" in data:
            raise ToolExecutionError(
                f"LibreTranslate error: {data['error']}. "
                "Both translation backends have failed."
            )

        translated: str = data.get("translatedText", "")
        if not translated:
            raise ToolExecutionError(
                "LibreTranslate returned an empty translation. "
                "Both backends have failed."
            )

        detected = data.get("detectedLanguage", {}).get("language", source)
        src_label = detected if detected not in ("auto", None, "") else "auto-detected"
        return (
            f"Translation ({src_label} -> {target}) via LibreTranslate:\n"
            f'"{translated}"'
        )

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _http_post_or_get(
        self,
        url: str,
        method: str = "GET",
        body: bytes | None = None,
    ) -> str:
        """
        Perform a GET or POST request and return the response body as a string.

        Raises ToolExecutionError for HTTP errors, network failures, timeouts.
        """
        headers: dict[str, str] = {
            "User-Agent": "PersonalAssistantAgent/1.0",
            "Accept": "application/json",
        }
        if method == "POST" and body:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                return resp.read().decode("utf-8")

        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise ToolExecutionError(
                    "Rate limit reached. Please wait a moment and try again."
                ) from exc
            if exc.code == 403:
                raise ToolExecutionError(
                    "Access denied (HTTP 403). "
                    "The API key may be missing or the quota exceeded."
                ) from exc
            raise ToolExecutionError(
                f"HTTP {exc.code} from translation service. " "Please try again later."
            ) from exc

        except urllib.error.URLError as exc:
            raise ToolExecutionError(
                f"Could not reach the translation service: {exc.reason}. "
                "Check your internet connection."
            ) from exc

        except TimeoutError as exc:
            raise ToolExecutionError(
                "The translation service timed out. " "Please try again in a moment."
            ) from exc
