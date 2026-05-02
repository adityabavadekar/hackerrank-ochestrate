"""Lightweight local PII scrubber for outbound LLM prompts."""

from dataclasses import dataclass
from functools import lru_cache
import re

from . import config
from .logger import get_logger

try:
    if not config.PRESIDIO_ENABLED:
        raise Exception("Presidio is disabled")
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
except Exception:  # pragma: no cover - optional runtime dependency
    AnalyzerEngine = None
    Pattern = None
    PatternRecognizer = None
    RecognizerRegistry = None
    AnonymizerEngine = None
    OperatorConfig = None

logger = get_logger(__name__)


def log_pii_mode() -> None:
    """Logs the active PII scrubbing mode at startup (P2.3)."""
    if not config.PRESIDIO_ENABLED:
        logger.info("PII mode: regex-only (Presidio disabled via config)")
        return
    analyzer, _, _ = _get_presidio_engines()
    if analyzer is not None:
        logger.info("PII mode: Presidio + regex (Presidio initialized successfully)")
    else:
        logger.info("PII mode: regex-only (Presidio unavailable at runtime)")




@dataclass(frozen=True)
class ScrubResult:
    text: str
    replacements: int


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._=-]{8,}\b", re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(
    r'(["\']?)'
    r"(password|passwd|pwd|username|user(name)?|api[_ -]?key|token|secret|authorization)"
    r'\1'
    r"(\s*(?:is|=|:)\s*)"
    r'(["\']?)([^\s,;}{\]\[)(]+?)\5(?=(?:[\s,;}{\]\[]|$))',
    re.IGNORECASE,
)
_LONG_SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{12,}|ghp_[A-Za-z0-9]{20,}|[A-Za-z0-9_\-]{24,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})\b"
)


def _replace_and_count(pattern: re.Pattern[str], text: str, repl) -> ScrubResult:
    replaced = 0

    def _inner(match: re.Match[str]) -> str:
        nonlocal replaced
        replaced += 1
        if callable(repl):
            return repl(match)
        return repl

    return ScrubResult(text=pattern.sub(_inner, text), replacements=replaced)


@lru_cache(maxsize=1)
def _get_presidio_engines():
    """Builds a lightweight Presidio setup using regex recognizers only.

    This avoids a hard runtime dependency on large spaCy pipelines while still
    using Presidio's analyzer/anonymizer stack as the primary PII detector.
    """
    if (
        AnalyzerEngine is None
        or Pattern is None
        or PatternRecognizer is None
        or RecognizerRegistry is None
        or AnonymizerEngine is None
        or OperatorConfig is None
    ):
        return None, None, None

    try:
        registry = RecognizerRegistry()
        recognizers = [
            (
                "EMAIL_ADDRESS",
                [_EMAIL_RE.pattern],
                config.PII_EMAIL_TOKEN,
            ),
            (
                "PHONE_NUMBER",
                [_PHONE_RE.pattern],
                config.PII_PHONE_TOKEN,
            ),
            (
                "IP_ADDRESS",
                [_IP_RE.pattern],
                config.PII_IP_TOKEN,
            ),
            (
                "CREDIT_CARD",
                [_CARD_RE.pattern],
                config.PII_CARD_TOKEN,
            ),
            (
                "API_SECRET",
                [_BEARER_RE.pattern, _LONG_SECRET_RE.pattern],
                config.PII_SECRET_TOKEN,
            ),
        ]

        operators = {}
        for entity_name, regexes, token in recognizers:
            patterns = [
                Pattern(name=f"{entity_name.lower()}_{idx}", regex=regex, score=0.9)
                for idx, regex in enumerate(regexes, start=1)
            ]
            registry.add_recognizer(
                PatternRecognizer(
                    supported_entity=entity_name,
                    patterns=patterns,
                    supported_language="en",
                )
            )
            operators[entity_name] = OperatorConfig("replace", {"new_value": token})

        analyzer = AnalyzerEngine(
            registry=registry,
            supported_languages=["en"],
            nlp_engine=None,
        )
        anonymizer = AnonymizerEngine()
        return analyzer, anonymizer, operators
    except Exception as exc:
        logger.warning("Presidio initialization failed, falling back to regex-only scrubber: %s", exc)
        return None, None, None


def _presidio_scrub(text: str) -> ScrubResult:
    analyzer, anonymizer, operators = _get_presidio_engines()
    if analyzer is None or anonymizer is None or operators is None:
        return ScrubResult(text=text, replacements=0)

    try:
        results = analyzer.analyze(text=text, language="en")
        if not results:
            return ScrubResult(text=text, replacements=0)
        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )
        return ScrubResult(text=anonymized.text, replacements=len(results))
    except Exception as exc:
        logger.warning("Presidio scrub failed, falling back to regex-only scrubber: %s", exc)
        return ScrubResult(text=text, replacements=0)


def scrub_text(text: str) -> ScrubResult:
    """Masks common PII and secret-bearing values before external model calls.

    Presidio runs first for standard entity detection/anonymization.
    A structured regex pass runs after that to preserve keys like
    `password`, `username`, and JSON field names while masking only values.
    """
    total = 0
    current = text
    if config.PRESIDIO_ENABLED:
        presidio_result = _presidio_scrub(text)
        total = presidio_result.replacements
        current = presidio_result.text

    # Keep a regex fallback for environments where Presidio is unavailable and
    # as a second-pass cleanup for anything Presidio misses.
    for pattern, token in (
        (_EMAIL_RE, config.PII_EMAIL_TOKEN),
        (_PHONE_RE, config.PII_PHONE_TOKEN),
        (_IP_RE, config.PII_IP_TOKEN),
        (_CARD_RE, config.PII_CARD_TOKEN),
        (_BEARER_RE, f"Bearer {config.PII_SECRET_TOKEN}"),
        (_LONG_SECRET_RE, config.PII_SECRET_TOKEN),
    ):
        result = _replace_and_count(pattern, current, token)
        current = result.text
        total += result.replacements

    def _secret_repl(match: re.Match[str]) -> str:
        quote = match.group(1)
        key = match.group(2)
        sep = match.group(4)
        value_quote = match.group(5)
        key_lower = key.lower()
        if "user" in key_lower:
            token = config.PII_USERNAME_TOKEN
        elif "pass" in key_lower or key_lower == "pwd":
            token = config.PII_PASSWORD_TOKEN
        else:
            token = config.PII_SECRET_TOKEN
        return f"{quote}{key}{quote}{sep}{value_quote}{token}{value_quote}"

    result = _replace_and_count(_SECRET_VALUE_RE, current, _secret_repl)
    current = result.text
    total += result.replacements

    return ScrubResult(text=current, replacements=total)
