"""Translator agent: detects non-English ticket language and translates the response.

Translation runs ONLY on the `response` field; `justification` always stays in English.
Uses the existing LLM gateway for translation quality.  Fails open — if detection or
translation fails the original English response is returned unchanged.

Caching strategy
----------------
- ``detect_language`` — lru_cache (process-lifetime) + diskcache (cross-run).
  Key is the raw text; language codes are tiny and safe to store.
- ``translate_response`` — diskcache only (expensive LLM call).
  Key is (response_text, target_lang) post-PII-scrubbing.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache

from .logger import get_logger
from . import llm
from .cache import disk
from .models import SupportTicket, TriageResult

logger = get_logger(__name__)

# Minimum text length in characters to bother running language detection.
_MIN_DETECT_LEN = 15

_TRANSLATE_SYSTEM = (
    "You are a professional translator. "
    "Translate the user's message into the target language. "
    "Output ONLY the translated text — no explanations, no punctuation changes, no added content.\n"
    "- Preserve all formatting: bullet points, numbered lists, bold markers, line breaks.\n"
    "- Preserve all proper nouns exactly as-is: product names (HackerRank, Claude, Visa), "
    "UI element names, button labels, setting names, and email addresses.\n"
    "- Preserve all URLs, file paths, and source citations exactly as-is.\n"
    "- Do not translate technical terms that are commonly used untranslated in the target language.\n"
    "- If the text is already in the target language, return it unchanged."
)


@lru_cache(maxsize=512)
@disk("lang_detect")
def detect_language(text: str) -> str | None:
    """Returns a BCP-47 language code (e.g. 'fr', 'es', 'hi') or None if detection fails.

    Cached in-process (lru_cache) and on disk (diskcache) so repeated tickets
    never re-run detection.
    """
    t = text.strip()
    if len(t) < _MIN_DETECT_LEN:
        return None
    # ASCII-only text is almost certainly English — skip detection
    ascii_ratio = sum(1 for c in t if ord(c) < 128) / len(t)
    if ascii_ratio > 0.95:
        return "en"
    try:
        from langdetect import DetectorFactory
        DetectorFactory.seed = 42  # deterministic
        from langdetect import detect_langs
        results = detect_langs(t)
        top = results[0]
        if top.prob < 0.90:  # ignore low-confidence detections
            return None
        return top.lang
    except Exception:
        return None


@disk("translate")
def translate_response(text: str, target_lang: str, ticket_text: str) -> str:
    """Calls the LLM to translate `text` into `target_lang`.  Returns original on failure.

    Disk-cached so the same (response, lang) pair never costs another LLM call.
    Key is (text, target_lang) — callers must pass post-PII-scrubbing content.
    """
    try:
        prompt = f"Translate the following response to language code '{target_lang}':\n{text}\n User's query in original language: \n{ticket_text}"
        translated = llm.call(system=_TRANSLATE_SYSTEM, user=prompt)
        return translated.strip()
    except Exception as exc:
        logger.warning("Translation to %r failed, using original: %s", target_lang, exc)
        return text


def maybe_translate(ticket: SupportTicket, result: TriageResult) -> TriageResult:
    """Detects the ticket's language and translates the response if it is not English.

    Rules:
    - Justification is always kept in English (the evaluator reads it).
    - Only the `response` field is translated.
    - If detection returns 'en' or None, the result is returned as-is with zero extra calls.
    - If the response is already in the target language (LLM already responded correctly),
      skip the extra translation call.
    """
    ticket_text = ticket.full_text()
    lang = detect_language(ticket_text)

    if not lang or lang == "en":
        return result

    result_lang = detect_language(result.response)
    if not result_lang or result_lang == lang:
        return result

    logger.info(
        "[%s] Detected non-English language=%r; translating response.",
        ticket.id,
        lang,
    )

    translated_response = translate_response(result.response, lang, ticket.full_text())
    return dataclasses.replace(result, response=translated_response)
