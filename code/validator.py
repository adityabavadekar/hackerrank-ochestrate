"""Post-generation validator: JSON parsing, enum correction, confidence override, multi-intent merge."""

import json
from typing import Any, Dict, List, Optional
import ast
import re
import unicodedata
from typing import Union

from .logger import get_logger
from .models import (
    RequestType,
    Status,
    SupportTicket,
    TriageResult,
)

logger = get_logger(__name__)

_LOW_CONFIDENCE_THRESHOLD = 2
_VALID_STATUSES = {s.value for s in Status}
_VALID_REQUEST_TYPES = {r.value for r in RequestType}


def parse_and_validate(raw_json: str, ticket: SupportTicket) -> Optional[TriageResult]:
    """Parses LLM JSON output, corrects enum values, and applies confidence override.

    Returns None only when parsing fails after a retry signal — callers should
    treat None as an escalation.

    Args:
        raw_json: Raw string output from the LLM.
        ticket: Original ticket, used for fallback justification text.

    Returns:
        Validated TriageResult, potentially with status overridden to escalated.
    """
    data = parse_string_as_json(raw_json)
    if data is None:
        logger.error("JSON parse failed for ticket %r; escalating", ticket.subject)
        return _escalation_fallback("Could not parse LLM response as JSON.")

    data = _fill_missing_fields(data)
    data = _correct_enums(data)
    result = _build_result(data)
    result = _apply_confidence_override(result, data)
    return result


def merge_sub_results(results: List[TriageResult]) -> TriageResult:
    """Merges multiple sub-issue results into one ticket result.

    If any sub-issue is escalated, the whole ticket escalates — the worst case wins.

    Args:
        results: Per-sub-issue TriageResults.

    Returns:
        Single merged TriageResult.
    """
    if not results:
        return _escalation_fallback("No sub-issue results to merge.")

    any_escalated = any(r.status == Status.ESCALATED for r in results)
    merged_status = Status.ESCALATED if any_escalated else results[0].status
    combined_response = " | ".join(r.response for r in results)
    combined_justification = " | ".join(r.justification for r in results)

    if any_escalated:
        logger.info("Multi-intent merge: escalating because at least one sub-issue escalated")

    return TriageResult(
        status=merged_status,
        product_area=results[0].product_area,
        request_type=results[0].request_type,
        response=combined_response,
        justification=combined_justification,
    )



def parse_string_as_json(
    response: str,
    *,
    prefer_largest: bool = True,
    strict_ast: bool = False,
) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Extract and parse JSON from messy LLM output.
    """
    # from source.assistant_logging import Logger
    # logger = Logger(name="extract_json_from_llm")

    if not response.strip():
        raise ValueError("Input string is empty or whitespace-only.")

    text = response.strip()

    try:

        def _try_json(s: str) -> Optional[Any]:
            try:
                return json.loads(s)
            except (json.JSONDecodeError, ValueError):
                return None

        def _normalize_unicode(s: str) -> str:
            replacements = {
                "\u2018": "'",
                "\u2019": "'",  # '' curly single
                "\u201c": '"',
                "\u201d": '"',  # "" curly double
                "\u201e": '"',
                "\u201f": '"',  # „‟ low/high double
                "\u2010": "-",
                "\u2011": "-",  # non-breaking / figure dash
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2015": "-",
                "\u00a0": " ",  # non-breaking space
            }
            for src, dst in replacements.items():
                s = s.replace(src, dst)
            return unicodedata.normalize("NFC", s)

        def _remove_comments(s: str) -> str:
            s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
            s = re.sub(r"(?m)//[^\n]*$", "", s)
            return s

        def _remove_trailing_commas(s: str) -> str:
            return re.sub(r",\s*([\]}])", r"\1", s)

        def _fix_literal_newlines_in_strings(s: str) -> str:
            def _escape_nl(m: re.Match) -> str:
                inner = m.group(1).replace("\n", "\\n").replace("\r", "\\r")
                return f'"{inner}"'

            return re.sub(r'"((?:[^"\\]|\\.)*)"', _escape_nl, s)

        def _clean(s: str) -> str:
            s = _normalize_unicode(s)
            s = _remove_comments(s)
            s = _remove_trailing_commas(s)
            return s.strip()

        def _try_parse(s: str, *, allow_ast: bool = True) -> Optional[Any]:
            result = _try_json(s)
            if result is not None:
                return result

            cleaned = _clean(s)
            result = _try_json(cleaned)
            if result is not None:
                return result

            fixed = _fix_literal_newlines_in_strings(cleaned)
            result = _try_json(fixed)
            if result is not None:
                return result

            if "'" in cleaned and '"' not in cleaned:
                swapped = cleaned.replace("'", '"')
                result = _try_json(swapped)
                if result is not None:
                    return result

            completed = _try_complete_brackets(cleaned)
            if completed:
                result = _try_json(completed)
                if result is not None:
                    return result

            if allow_ast and not strict_ast:
                for src in (s, cleaned):
                    try:
                        obj = ast.literal_eval(src)
                        if isinstance(obj, (dict, list)):
                            return obj
                    except Exception:
                        pass

            return None

        def _try_complete_brackets(s: str) -> Optional[str]:
            stack: List[str] = []
            in_string = False
            escape_next = False

            for ch in s:
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                elif ch in "}]":
                    if stack and stack[-1] == ch:
                        stack.pop()

            if not stack:
                return None
            return s + "".join(reversed(stack))

        def _extract_code_fence_blocks(s: str) -> List[str]:
            pattern = r"```(?:[a-zA-Z0-9_+-]*)?\s*(.*?)```"
            return re.findall(pattern, s, re.DOTALL | re.IGNORECASE)

        def _extract_inline_backtick(s: str) -> List[str]:
            pattern = r"`([{\[].*?[}\]])`"
            return re.findall(pattern, s, re.DOTALL)

        def _extract_balanced_candidates(s: str) -> List[str]:
            candidates: List[str] = []
            stack: List[str] = []
            start: Optional[int] = None
            in_string = False
            escape_next = False

            for i, ch in enumerate(s):
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue

                if ch in "{[":
                    if not stack:
                        start = i
                    stack.append("}" if ch == "{" else "]")
                elif ch in "}]":
                    if stack and ch == stack[-1]:
                        stack.pop()
                        if not stack and start is not None:
                            candidates.append(s[start : i + 1])
                            start = None
                    else:
                        stack.clear()
                        start = None

            return candidates

        def _score(obj: Any) -> int:
            if isinstance(obj, dict):
                return len(obj)
            if isinstance(obj, list):
                return len(obj)
            return 0

        valid_results: List[Any] = []

        def _collect(candidate: str) -> None:
            obj = _try_parse(candidate)
            if obj is not None:
                valid_results.append(obj)

        obj = _try_parse(text)
        if obj is not None:
            return obj

        for block in _extract_code_fence_blocks(text):
            _collect(block.strip())

        for span in _extract_inline_backtick(text):
            _collect(span.strip())

        for candidate in _extract_balanced_candidates(text):
            _collect(candidate)

        normalised = _normalize_unicode(text)
        if normalised != text:
            for candidate in _extract_balanced_candidates(normalised):
                _collect(candidate)

        _collect(_clean(text))

        if not valid_results:
            preview = text[:500] + ("…" if len(text) > 500 else "")
            raise ValueError(
                "No valid JSON structure found in the LLM response.\n"
                f"Response preview:\n{preview}"
            )

        if not prefer_largest or len(valid_results) == 1:
            return valid_results[0]

        return max(valid_results, key=_score)
    except:  # noqa: E722
        return None



def _fill_missing_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "status": Status.ESCALATED.value,
        "product_area": "general",
        "request_type": RequestType.PRODUCT_ISSUE.value,
        "response": "Unable to generate a response. This ticket has been escalated.",
        "justification": "Missing field filled with safe default.",
        "confidence": 1,
    }
    for key, default in defaults.items():
        if key not in data or data[key] is None:
            logger.warning("Missing field '%s' in LLM output; using default: %r", key, default)
            data[key] = default
    return data


def _correct_enums(data: Dict[str, Any]) -> Dict[str, Any]:
    status_val = str(data.get("status", "")).lower()
    if status_val not in _VALID_STATUSES:
        corrected = Status.ESCALATED.value
        logger.warning("Invalid status %r corrected to %r", status_val, corrected)
        data["status"] = corrected

    rtype_val = str(data.get("request_type", "")).lower()
    if rtype_val not in _VALID_REQUEST_TYPES:
        corrected = RequestType.PRODUCT_ISSUE.value
        logger.warning("Invalid request_type %r corrected to %r", rtype_val, corrected)
        data["request_type"] = corrected

    return data


def _build_result(data: Dict[str, Any]) -> TriageResult:
    product_area = str(data.get("product_area", "general"))
    return TriageResult(
        status=Status(data["status"]),
        product_area=product_area,
        request_type=RequestType(data["request_type"]),
        response=str(data["response"]),
        justification=str(data["justification"]),
    )


def _apply_confidence_override(result: TriageResult, data: Dict[str, Any]) -> TriageResult:
    """Overrides status to ESCALATED when LLM confidence is too low."""
    try:
        confidence = int(data.get("confidence", 5))
    except (ValueError, TypeError):
        confidence = 5

    if confidence <= _LOW_CONFIDENCE_THRESHOLD and result.status != Status.ESCALATED:
        logger.info(
            "Confidence %d <= %d; overriding status from %r to escalated",
            confidence,
            _LOW_CONFIDENCE_THRESHOLD,
            result.status.value,
        )
        return TriageResult(
            status=Status.ESCALATED,
            product_area=result.product_area,
            request_type=result.request_type,
            response=result.response,
            justification=result.justification + f" [escalated: low confidence={confidence}]",
        )

    return result


def _escalation_fallback(reason: str) -> TriageResult:
    return TriageResult(
        status=Status.ESCALATED,
        product_area="general",
        request_type=RequestType.PRODUCT_ISSUE,
        response="Your request has been escalated for human review.",
        justification=reason,
    )
