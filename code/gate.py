"""Stage 1: Pre-LLM gate - injection detection, hard escalation, multi-intent splitting."""

import json
from dataclasses import asdict
from .phrases import OFF_TOPIC_PHRASES, INJECTION_PHRASES
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, List, Optional

from litellm import completion
import numpy as np
try:
    from detoxify import Detoxify
except Exception:  # pragma: no cover - optional dependency at runtime
    Detoxify = None
try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency at runtime
    fuzz = None
try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency at runtime
    pipeline = None

from .logger import get_logger
from .models import (Company, RequestType, SafetyResult, Status, SupportTicket, TriageResult)
from . import validator
from . import config
from .cache import lru
from .pii import scrub_text

logger = get_logger(__name__)

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
    r"disregard\s+(the\s+)?(above|previous|prior)",
    r"forget\s+(all\s+)?(previous|prior)\s+instructions?",
    r"you\s+are\s+now\s+",
    r"\bact\s+as\s+(a\s+)?(system|assistant|admin|ai|bot)\b",
    r"new\s+prompt\s*:",
    r"system\s*:\s*you",
    r"override\s+(instructions?|prompt)",
]


_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    re.IGNORECASE,
)


_HARD_ESCALATION_PATTERNS = [
    (r"\baccount\s+(was\s+|has\s+been\s+|is\s+)?hacked\b", "account hacked"),
    (r"\baccount\s+(was\s+|has\s+been\s+|is\s+)?compromised\b", "account compromised"),
    (r"\baccount\s+(was\s+|has\s+been\s+)?breached\b", "account breached"),
    (r"\baccount\s+(was\s+|has\s+been\s+)?taken\s+over\b", "account taken over"),
    (r"\bunauthorized\s+(charge|transaction|payment)s?\b", "unauthorized charge"),
    (r"\bunknown\s+(charge|transaction)s?\b", "unknown charge"),
    (r"\bfraud(ulent)?\b", "fraud"),
    (r"\bfraudulent\s+(activity|transaction|charge)s?\b", "fraudulent activity"),
    (r"\bidentity\s+theft\b", "identity theft"),
    (r"\bdata\s+breach\b", "data breach"),
    (r"\bsecurity\s+breach\b", "security breach"),
    (r"\bchargeback\b", "chargeback"),
    (r"\blegal\s+action\b", "legal action"),
    (r"\blawsuit\b", "lawsuit"),
    (r"\bsue(d|ing)?\b", "legal threat"),
]

# Soft-warn patterns: log a warning but let the pipeline proceed so the
# retriever can answer from corpus (e.g. Visa lost/stolen card docs).
_SOFT_ESCALATION_PATTERNS = [
    (r"\bstolen\s+(card|cheque)s?\b", "stolen card"),
    (r"\bcard\s+(was\s+|has\s+been\s+)?stolen\b", "card stolen"),
    (r"\b(lost|missing)\s+(card|cheque)s?\b", "lost card"),
]

_CONVERSATIONAL_PATTERNS = [
    r"^(hi+|hey+|hello+|hiya|howdy|yo+|sup|what'?s up)\W*$",
    r"^how are (you|u|ya)\??$",
    r"^(good|fine|ok+|okay|great|nice|cool|lol|lmao|hehe|haha|xd)\W*$",
    r"^(bye|goodbye|cya|see ya|later)\W*$",
    r"^(yes|no|maybe|sure|nope|yep|nah)\W*$",
    r"^(me is|i am|i'm)\s+\w+$",
]

_CONVERSATIONAL_RE = re.compile(
    "|".join(_CONVERSATIONAL_PATTERNS),
    re.IGNORECASE,
)

_GRATITUDE_PATTERNS = [
    r"^(thanks?|thank you|thx|ty)( so much| very much| for helping me| for the help)?\W*$",
    r"^(i )?appreciate (it|the help)\W*$",
]

_GRATITUDE_RE = re.compile(
    "|".join(_GRATITUDE_PATTERNS),
    re.IGNORECASE,
)

_OFF_TOPIC_PATTERNS = [
    r"\b(write|generate|create|code|program|script)\s+(me\s+)?(a\s+)?(python|java|javascript|sql|code|function|class|script)\b",
    r"\b(what is|explain|tell me about|describe)\s+(the\s+)?(meaning|history|capital|population)\b",
    r"\b(recipe|weather|joke|poem|song|story)\b",
    r"\b(delete|wipe|remove|erase|destroy)\b.{0,40}\b(all\s+)?files?\b",
    r"\b(rm\s+-rf|format\s+c:|drop\s+database|delete\s+all\s+files)\b",
]

_OFF_TOPIC_RE = re.compile(
    "|".join(_OFF_TOPIC_PATTERNS),
    re.IGNORECASE,
)

_INTENT_SPLITTERS = re.compile(
    r"(?:also|additionally|furthermore|secondly|another question|and also)[,:]?\s+",
    re.IGNORECASE,
)

_NOISE_RE = re.compile(r"^[^a-zA-Z0-9]+$")  # only symbols like ----~~~~
_LOW_SIGNAL_RE = re.compile(r"^[a-zA-Z]{20,}$")  # long continuous gibberish letters

_SUPPORT_KEYWORDS = (
    "account", "payment", "charged", "charge", "refund", "test", "assessment",
    "workspace", "login", "merchant", "transaction", "submission", "dashboard",
    "candidate", "recruiter", "card", "visa", "claude", "hackerrank",
)


@dataclass
class GateResult:
    """Outcome of the Stage 1 gate check."""

    passed: bool
    safety: SafetyResult
    sub_issues: List[str] = field(default_factory=list)
    early_result: Optional[TriageResult] = None

def run(ticket: SupportTicket, retriever_or_model: Any | None = None) -> GateResult:
    """Runs all Stage 1 checks on a ticket and returns an early result or passes it through.

    Checks are ordered cheapest-first: injection -> hard keywords -> multi-intent.

    Args:
        ticket: The raw support ticket to evaluate.
        retriever_or_model: Retriever instance or raw sentence-transformer model for semantic checks.

    Returns:
        GateResult with either an early_result (short-circuit) or sub_issues for Stage 2.
    """
    # Accept either a Retriever or the raw model directly
    model = getattr(retriever_or_model, "_model", retriever_or_model)
    text = ticket.full_text()

    if not text.strip():
        logger.warning("Empty ticket text for ticket: %r", ticket.subject)
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=False, reason="empty_text"),
            early_result=validator._escalation_fallback(
                "No text content found in ticket. Blocked before LLM.",
                status=Status.REPLIED,
                request_type=RequestType.INVALID,
                response="Your message appears to be empty. Please provide details about your issue.",
            ),
        )

    if _INJECTION_RE.search(text):
        logger.warning("Prompt injection detected in ticket: %r", ticket.subject)
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=False, reason="prompt_injection"),
            early_result=validator._escalation_fallback(
                "Prompt injection pattern detected in ticket text. Blocked before LLM.",
                status=Status.REPLIED,
                request_type=RequestType.INVALID,
                response="Your message could not be processed because it contains content that conflicts with our support guidelines. Please rephrase and try again.",
            ),
        )

    fuzzy_reason = _check_fuzzy_invalid_pattern(text)
    if fuzzy_reason == "prompt_injection":
        logger.warning("RapidFuzz prompt injection detected in ticket: %r", ticket.subject)
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=False, reason="prompt_injection_fuzzy"),
            early_result=validator._escalation_fallback(
                "Prompt injection pattern detected via fuzzy matching. Blocked before LLM.",
                status=Status.REPLIED,
                request_type=RequestType.INVALID,
                response="Your message could not be processed because it contains content that conflicts with our support guidelines. Please rephrase and try again.",
            ),
        )

    if _CONVERSATIONAL_RE.search(text):
        logger.warning("Conversational filler detected in ticket: %r", ticket.subject)
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=False, reason="conversational_filler"),
            early_result=validator._escalation_fallback(
                "Conversational/filler message detected. Blocked before LLM.",
                status=Status.REPLIED,
                request_type=RequestType.INVALID,
                response="I am a support assistant. Please provide a relevant question or issue for me to help you with.",
            ),
        )

    if _GRATITUDE_RE.search(text):
        logger.warning("Gratitude message detected in ticket: %r", ticket.subject)
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=False, reason="gratitude"),
            early_result=validator._escalation_fallback(
                "User expressed gratitude. Replied and blocked before LLM.",
                status=Status.REPLIED,
                request_type=RequestType.INVALID,
                response="Happy to help",
            ),
        )

    if _OFF_TOPIC_RE.search(text):
        logger.warning("Off-topic request detected in ticket: %r", ticket.subject)
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=False, reason="off_topic"),
            early_result=validator._escalation_fallback(
                "Off-topic request detected. Blocked before LLM.",
                status=Status.REPLIED,
                request_type=RequestType.INVALID,
                response="I am a support assistant. Please provide a relevant question or issue for me to help you with.",
            ),
        )

    if fuzzy_reason == "off_topic":
        logger.warning("RapidFuzz off-topic request detected in ticket: %r", ticket.subject)
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=False, reason="off_topic_fuzzy"),
            early_result=validator._escalation_fallback(
                "Off-topic or harmful non-support request detected via fuzzy matching. Blocked before LLM.",
                status=Status.REPLIED,
                request_type=RequestType.INVALID,
                response="I am a support assistant. Please provide a relevant question or issue for me to help you with.",
            ),
        )

    matched_keyword = _check_hard_escalation(text)
    if matched_keyword:
        logger.warning(
            "Hard escalation keyword '%s' in ticket: %r",
            matched_keyword,
            ticket.subject,
        )
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=True, reason="hard_keyword"),
            early_result=validator._escalation_fallback(
                f"Hard escalation keyword matched: '{matched_keyword}'. Human review required.",
                product_area="fraud",
                response=config.ESCALATION_RESPONSE,
            ),
        )

    # Soft-warn: log but let the pipeline continue so the retriever can
    # answer from corpus (e.g. lost/stolen card procedures).
    soft_keyword = _check_soft_escalation(text)
    if soft_keyword:
        logger.warning(
            "Soft escalation keyword '%s' in ticket (pipeline continues): %r",
            soft_keyword,
            ticket.subject,
        )
    
    if _is_low_quality(text):
        logger.warning(
            "Low-quality or noise ticket text for ticket: %r", ticket.subject
        )
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=False, reason="low_quality_text"),
            early_result=validator._escalation_fallback(
                "Low-quality or non-informative text detected. Blocked before LLM.",
                status=Status.REPLIED,
                request_type=RequestType.INVALID,
                response="Your message appears to be incomplete or unclear. Please provide more details about your issue.",
            ),
        )

    if _is_detoxify_invalid(text):
        logger.warning("Detoxify flagged the ticket as abusive/off-topic: %r", ticket.subject)
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=False, reason="detoxify_invalid"),
            early_result=validator._escalation_fallback(
                "Toxic or abusive non-support message detected by Detoxify. Blocked before LLM.",
                status=Status.REPLIED,
                request_type=RequestType.INVALID,
                response="I am a support assistant. Please provide a relevant question or issue for me to help you with.",
            ),
        )

    if _is_mnli_invalid(text):
        logger.warning("MNLI classifier marked ticket as invalid/off-topic: %r", ticket.subject)
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=False, reason="mnli_invalid"),
            early_result=validator._escalation_fallback(
                "Zero-shot gate classified the message as invalid or off-topic. Blocked before LLM.",
                status=Status.REPLIED,
                request_type=RequestType.INVALID,
                response="I am a support assistant. Please provide a relevant question or issue for me to help you with.",
            ),
        )

    # ML-based injection check (expensive so keep at last)
    if _check_llama_guard(text):
        logger.warning("Llama-Guard injection detected in ticket: %r", ticket.subject)
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=False, reason="prompt_injection_llm"),
            early_result=validator._escalation_fallback(
                "Unsafe content detected by ML safety model. Blocked before LLM.",
                status=Status.REPLIED,
                request_type=RequestType.INVALID,
                response="Your message could not be processed because it contains content that conflicts with our support guidelines. Please rephrase and try again.",
            ),
        )

    if model is not None and _is_semantically_invalid(text, model):
        logger.warning("Semantically invalid ticket: %r", ticket.subject)
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=False, reason="semantic_invalid"),
            early_result=validator._escalation_fallback(
                "Semantically invalid ticket detected. Blocked before LLM.",
                status=Status.REPLIED,
                request_type=RequestType.INVALID,
                response="I am a support assistant. Please provide a relevant question or issue for me to help you with.",
            ),
        )

    sub_issues = _split_intents(text)
    if len(sub_issues) > 1:
        logger.info("Multi-intent ticket split into %d sub-issues", len(sub_issues))

    logger.debug(
        "Gate passed for ticket: %r (%d sub-issues)", ticket.subject, len(sub_issues)
    )
    return GateResult(
        passed=True,
        safety=SafetyResult(should_escalate=False, reason="clean"),
        sub_issues=sub_issues,
    )


def _check_llama_guard(text: str) -> bool:
    """Uses Llama-Guard to detect unsafe prompts or injections."""
    try:
        api_key = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_API_KEY", ""))
        if not api_key:
            return False
        scrubbed = scrub_text(text)
            
        response = completion(
            model="huggingface/meta-llama/Llama-Guard-4-12B",
            messages=[{"role": "user", "content": scrubbed.text}],
            api_key=api_key,
            max_tokens=20,
            temperature=0.0
        )
        content = response.choices[0].message.content.strip().lower()
        return "unsafe" in content
    except Exception as e:
        logger.warning("Llama-Guard check failed (likely rate-limited), ignoring: %s", e)
        return False


def _contains_support_keywords(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in _SUPPORT_KEYWORDS)


def _check_fuzzy_invalid_pattern(text: str) -> Optional[str]:
    """Uses RapidFuzz to catch paraphrases of known invalid patterns."""
    if fuzz is None:
        return None

    lower = text.lower()

    for phrase in INJECTION_PHRASES:
        if fuzz.partial_ratio(lower, phrase) >= config.GATE_FUZZY_THRESHOLD:
            return "prompt_injection"

    for phrase in OFF_TOPIC_PHRASES:
        if fuzz.partial_ratio(lower, phrase) >= config.GATE_FUZZY_THRESHOLD:
            return "off_topic"

    return None


@lru_cache(maxsize=1)
def _get_mnli_classifier():
    if pipeline is None:
        return None
    try:
        return pipeline(
            "zero-shot-classification",
            model=config.GATE_MNLI_MODEL,
        )
    except Exception as exc:
        logger.warning("MNLI gate model unavailable, ignoring: %s", exc)
        return None


def _is_mnli_invalid(text: str) -> bool:
    """Uses a zero-shot classifier to catch invalid/off-topic support requests."""
    classifier = _get_mnli_classifier()
    if classifier is None:
        return False

    labels = [
        "technical support request",
        "off-topic request",
        "harmful code request",
        "greeting or chit-chat",
        "abusive message",
    ]
    try:
        result = classifier(text, candidate_labels=labels, multi_label=False)
        label = result["labels"][0]
        score = float(result["scores"][0])
        logger.debug("MNLI gate label=%s score=%.3f", label, score)
        return (
            label in {"off-topic request", "harmful code request", "greeting or chit-chat", "abusive message"}
            and score >= config.GATE_MNLI_THRESHOLD
            and not _contains_support_keywords(text)
        )
    except Exception as exc:
        logger.warning("MNLI gate classification failed, ignoring: %s", exc)
        return False


@lru_cache(maxsize=1)
def _get_detoxify_model():
    if Detoxify is None:
        return None
    try:
        return Detoxify("original")
    except Exception as exc:
        logger.warning("Detoxify model unavailable, ignoring: %s", exc)
        return None


def _is_detoxify_invalid(text: str) -> bool:
    """Blocks highly toxic non-support prompts while failing open for normal tickets."""
    model = _get_detoxify_model()
    if model is None:
        return False
    try:
        scores = model.predict(text)
        toxicity = float(scores.get("toxicity", 0.0))
        insult = float(scores.get("insult", 0.0))
        threat = float(scores.get("threat", 0.0))
        severe = max(toxicity, insult, threat)
        logger.debug(
            "Detoxify gate toxicity=%.3f insult=%.3f threat=%.3f",
            toxicity,
            insult,
            threat,
        )
        return (
            severe >= config.GATE_DETOXIFY_THRESHOLD
            and not _contains_support_keywords(text)
        )
    except Exception as exc:
        logger.warning("Detoxify gate failed, ignoring: %s", exc)
        return False


def _is_low_quality(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if _NOISE_RE.fullmatch(t):
        return True

    tokens = t.split()

    # Single long token: vowel ratio + consonant-cluster
    if len(tokens) == 1 and len(t) >= 7:
        vowels = sum(1 for c in t.lower() if c in "aeiouy")
        if vowels == 0:
            return True
            
        vowel_ratio = vowels / len(t)
        consonant_runs = re.findall(r"[bcdfghjklmnpqrstvwxz]{5,}", t.lower())
        
        if vowel_ratio < 0.15 or consonant_runs:
            return True
    
    if len(tokens) == 1 and len(set(t)) <= 1:
        return True

    if len(tokens) >= 5:
        unique_words = len(set(tokens))
        if unique_words / len(tokens) <= 0.2:
            return True

    return False


def _check_hard_escalation(text: str) -> Optional[str]:
    lower = text.lower()
    for pattern, label in _HARD_ESCALATION_PATTERNS:
        if re.search(pattern, lower):
            return label
    return None


def _check_soft_escalation(text: str) -> Optional[str]:
    """Returns the matched label if a soft-warn pattern matches, else None.

    Soft patterns log a warning but do NOT block the pipeline — the retriever
    may have corpus content that can answer the ticket (e.g. Visa card procedures).
    """
    lower = text.lower()
    for pattern, label in _SOFT_ESCALATION_PATTERNS:
        if re.search(pattern, lower):
            return label
    return None




def _split_intents(text: str) -> List[str]:
    """Splits a ticket text into sub-issues by detecting multi-intent conjunctions.

    Returns the original text as a single-element list when no split point is found,
    so callers can always iterate without branching.
    """
    parts = _INTENT_SPLITTERS.split(text)
    cleaned = [p.strip() for p in parts if len(p.strip()) >= 5]
    return cleaned if cleaned else [text]


_INVALID_ANCHORS = [
    "hello how are you",
    "hehe lol bye",
    "write me a python script",
    "what is the capital of france",
    "tell me a joke",
    "me is me",
    "hi there",
    "write code for me",
    "idiot stupid",
    "good morning",
]

_VALID_ANCHORS = [
    "my account is locked and I cannot login",
    "I was charged twice for my subscription",
    "the assessment is not loading",
    "I need to dispute a transaction on my visa card",
    "I cannot access my dashboard",
    "my payment failed but I was still charged",
    "the proctoring feature is not working",
]

_invalid_anchor_embs: np.ndarray | None = None
_valid_anchor_embs: np.ndarray | None = None

def init_semantic_gate(model: Any) -> None:
    """Call once at startup with your already-loaded retriever model."""
    global _invalid_anchor_embs, _valid_anchor_embs
    if _invalid_anchor_embs is not None and _valid_anchor_embs is not None:
        return
    _invalid_anchor_embs = model.encode(_INVALID_ANCHORS, normalize_embeddings=True)
    _valid_anchor_embs = model.encode(_VALID_ANCHORS, normalize_embeddings=True)

def _is_semantically_invalid(text: str, model: Any, threshold: float = 0.55) -> bool:
    init_semantic_gate(model)
    if _invalid_anchor_embs is None or _valid_anchor_embs is None:
        return False  # fail open if not initialized

    emb = _encode_text_gate(model, text)
    invalid_score = float(np.max(_invalid_anchor_embs @ emb))
    valid_score = float(np.max(_valid_anchor_embs @ emb))

    return invalid_score > threshold and invalid_score > valid_score


@lru(maxsize=512)
def _encode_text_gate(model: Any, text: str) -> "np.ndarray":
    """Encodes text for semantic gate checks; lru_cache prevents duplicate encodes."""
    return model.encode([text], normalize_embeddings=True)[0]


if __name__ == "__main__":
    print("Enter a support ticket text (type 'exit' to quit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            break

        result = run(
            SupportTicket(issue=user_input, subject="", company=Company.CLAUDE)
        )
        print(json.dumps(asdict(result), indent=2))
