"""Stage 1: Pre-LLM gate - injection detection, hard escalation, multi-intent splitting."""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .logger import get_logger
from .models import SafetyResult, Status, SupportTicket, TriageResult, RequestType

logger = get_logger(__name__)

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
    r"disregard\s+(the\s+)?(above|previous|prior)",
    r"forget\s+(all\s+)?(previous|prior)\s+instructions?",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(if\s+you\s+are\s+)?",
    r"new\s+prompt\s*:",
    r"system\s*:\s*you",
    r"override\s+(instructions?|prompt)",
]

_HARD_ESCALATION_KEYWORDS = [
    "fraud",
    "unauthorized charge",
    "account hacked",
    "account compromised",
    "chargeback",
    "legal action",
    "data breach",
    "suspended wrongly",
    "wrongfully suspended",
    "identity theft",
    "stolen card",
]

_INTENT_SPLITTERS = re.compile(
    r"(?:also|additionally|furthermore|secondly|another question|and also)[,:]?\s+",
    re.IGNORECASE,
)

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    re.IGNORECASE,
)


@dataclass
class GateResult:
    """Outcome of the Stage 1 gate check."""

    passed: bool
    safety: SafetyResult
    sub_issues: List[str] = field(default_factory=list)
    early_result: Optional[TriageResult] = None


def _canned_injection_reply() -> TriageResult:
    return TriageResult(
        status=Status.REPLIED,
        product_area="general",
        request_type=RequestType.INVALID,
        response=(
            "Your message could not be processed because it contains content "
            "that conflicts with our support guidelines. Please rephrase and try again."
        ),
        justification="Prompt injection pattern detected in ticket text. Blocked before LLM.",
    )


def _canned_escalation_reply(keyword: str) -> TriageResult:
    return TriageResult(
        status=Status.ESCALATED,
        product_area="fraud",
        request_type=RequestType.PRODUCT_ISSUE,
        response=(
            "Your request has been escalated to our specialized team who will "
            "contact you directly. Do not share sensitive information over email."
        ),
        justification=f"Hard escalation keyword matched: '{keyword}'. Human review required.",
    )


def run(ticket: SupportTicket) -> GateResult:
    """Runs all Stage 1 checks on a ticket and returns an early result or passes it through.

    Checks are ordered cheapest-first: injection -> hard keywords -> multi-intent.

    Args:
        ticket: The raw support ticket to evaluate.

    Returns:
        GateResult with either an early_result (short-circuit) or sub_issues for Stage 2.
    """
    text = ticket.full_text()

    if _INJECTION_RE.search(text):
        logger.warning("Prompt injection detected in ticket: %r", ticket.subject)
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=False, reason="prompt_injection"),
            early_result=_canned_injection_reply(),
        )

    matched_keyword = _check_hard_escalation(text)
    if matched_keyword:
        logger.warning("Hard escalation keyword '%s' in ticket: %r", matched_keyword, ticket.subject)
        return GateResult(
            passed=False,
            safety=SafetyResult(should_escalate=True, reason=f"hard_keyword:{matched_keyword}"),
            early_result=_canned_escalation_reply(matched_keyword),
        )

    sub_issues = _split_intents(text)
    if len(sub_issues) > 1:
        logger.info("Multi-intent ticket split into %d sub-issues", len(sub_issues))

    logger.debug("Gate passed for ticket: %r (%d sub-issues)", ticket.subject, len(sub_issues))
    return GateResult(
        passed=True,
        safety=SafetyResult(should_escalate=False, reason="clean"),
        sub_issues=sub_issues,
    )


def _check_hard_escalation(text: str) -> Optional[str]:
    lower = text.lower()
    for kw in _HARD_ESCALATION_KEYWORDS:
        if kw in lower:
            return kw
    return None


def _split_intents(text: str) -> List[str]:
    """Splits a ticket text into sub-issues by detecting multi-intent conjunctions.

    Returns the original text as a single-element list when no split point is found,
    so callers can always iterate without branching.
    """
    parts = _INTENT_SPLITTERS.split(text)
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned if cleaned else [text]
