"""Sample-set evaluation script (P1.2).

Compares the agent's output.csv against the expected signals in sample_support_tickets.csv
and prints aggregate accuracy metrics for status, request_type, and product_area.

Usage:
    uv run python -m code.eval_sample
    # or with explicit paths:
    uv run python -m code.eval_sample \
        --expected support_tickets/sample_support_tickets.csv \
        --actual support_tickets/output.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _normalize(val: Optional[str]) -> str:
    return (val or "").strip().lower()


def _col(row: Dict[str, str], *candidates: str) -> str:
    """Tries multiple column name spellings, returns the first match."""
    for c in candidates:
        if c in row:
            return _normalize(row[c])
    return ""


# Scoring

_STATUS_WEIGHT = 0.35
_REQTYPE_WEIGHT = 0.30
_AREA_WEIGHT = 0.20
_RESPONSE_PRESENT_WEIGHT = 0.15  # whether agent produced any non-empty response


def _score_row(expected: Dict[str, str], actual: Dict[str, str]) -> Dict[str, float]:
    exp_status = _col(expected, "Status", "status")
    exp_rtype = _col(expected, "Request Type", "request_type")
    exp_area = _col(expected, "Product Area", "product_area")

    act_status = _col(actual, "Status", "status")
    act_rtype = _col(actual, "Request Type", "request_type")
    act_area = _col(actual, "Product Area", "product_area")
    act_response = _col(actual, "Response", "response")

    status_ok = float(exp_status == act_status) if exp_status else 0.0
    rtype_ok = float(exp_rtype == act_rtype) if exp_rtype else 0.0
    area_ok = float(exp_area == act_area) if exp_area else 0.0
    response_ok = float(bool(act_response))

    weighted = (
        status_ok * _STATUS_WEIGHT
        + rtype_ok * _REQTYPE_WEIGHT
        + area_ok * _AREA_WEIGHT
        + response_ok * _RESPONSE_PRESENT_WEIGHT
    )

    return {
        "status_ok": status_ok,
        "rtype_ok": rtype_ok,
        "area_ok": area_ok,
        "response_ok": response_ok,
        "weighted": weighted,
        "exp_status": exp_status,
        "act_status": act_status,
        "exp_rtype": exp_rtype,
        "act_rtype": act_rtype,
        "exp_area": exp_area,
        "act_area": act_area,
    }


# Matching expected <-> actual rows

def _match_rows(
    expected: List[Dict[str, str]],
    actual: List[Dict[str, str]],
) -> List[tuple[Dict[str, str], Dict[str, str]]]:
    """Matches expected rows to actual rows.

    Matching priority:
      1. Subject (case-insensitive, stripped)
      2. First 80 chars of Issue text (for cases where the actual CSV lacks subjects)
    """
    actual_by_subject: Dict[str, Dict[str, str]] = {}
    actual_by_issue: Dict[str, Dict[str, str]] = {}
    for row in actual:
        subj = _col(row, "Subject", "subject")
        if subj:
            actual_by_subject[subj] = row
        issue_key = _col(row, "Issue", "issue")[:80]
        if issue_key:
            actual_by_issue[issue_key] = row

    matched = []
    unmatched_expected = []
    for exp_row in expected:
        subj = _col(exp_row, "Subject", "subject")
        issue_key = _col(exp_row, "Issue", "issue")[:80]

        act_row = actual_by_subject.get(subj) or actual_by_issue.get(issue_key)
        if act_row is not None:
            matched.append((exp_row, act_row))
        else:
            unmatched_expected.append(subj or issue_key or "(empty)")

    return matched, unmatched_expected


def _print_report(
    matched: List[tuple[Dict[str, str], Dict[str, str]]],
    unmatched: List[str],
    verbose: bool = False,
) -> float:
    if not matched:
        print("No matched rows found. Check that Subject columns align between files.")
        return 0.0

    scores = [_score_row(exp, act) for exp, act in matched]
    n = len(scores)

    status_acc = sum(s["status_ok"] for s in scores) / n
    rtype_acc = sum(s["rtype_ok"] for s in scores) / n
    area_acc = sum(s["area_ok"] for s in scores) / n
    response_pct = sum(s["response_ok"] for s in scores) / n
    overall = sum(s["weighted"] for s in scores) / n

    print("=" * 60)
    print(f"  SAMPLE-SET EVALUATION — {n} matched rows")
    print("=" * 60)
    print(f"  Status accuracy      : {status_acc * 100:.1f}%  (weight {_STATUS_WEIGHT:.0%})")
    print(f"  Request type accuracy: {rtype_acc * 100:.1f}%  (weight {_REQTYPE_WEIGHT:.0%})")
    print(f"  Product area accuracy: {area_acc * 100:.1f}%  (weight {_AREA_WEIGHT:.0%})")
    print(f"  Response non-empty   : {response_pct * 100:.1f}%  (weight {_RESPONSE_PRESENT_WEIGHT:.0%})")
    print("  ─────────────────────────────────────────")
    print(f"  WEIGHTED SCORE       : {overall * 100:.1f}%")
    print("=" * 60)

    if unmatched:
        print(f"\n  ⚠ {len(unmatched)} expected rows had no matching subject in output:")
        for s in unmatched:
            print(f"    - {s!r}")

    if verbose:
        print("\n  Per-row breakdown:")
        print(f"  {'Subject':<40} {'Status':^7} {'ReqType':^9} {'Area':^9}")
        print(f"  {'-' * 40} {'-' * 7} {'-' * 9} {'-' * 9}")
        for (exp, _), s in zip(matched, scores):
            subj = _col(exp, "Subject", "subject")[:38]
            status_sym = "✓" if s["status_ok"] else f"✗({s['act_status'][:6]})"
            rtype_sym = "✓" if s["rtype_ok"] else f"✗({s['act_rtype'][:6]})"
            area_sym = "✓" if s["area_ok"] else f"✗({s['act_area'][:6]})"
            print(f"  {subj:<40} {status_sym:^7} {rtype_sym:^9} {area_sym:^9}")
        print()

    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent output vs sample expected outputs")
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path("support_tickets/sample_support_tickets.csv"),
        help="Path to the sample CSV with expected signals",
    )
    parser.add_argument(
        "--actual",
        type=Path,
        default=Path("support_tickets/output.csv"),
        help="Path to the agent's output CSV",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-row breakdown",
    )
    args = parser.parse_args()

    if not args.expected.exists():
        print(f"ERROR: Expected CSV not found: {args.expected}", file=sys.stderr)
        sys.exit(1)
    if not args.actual.exists():
        print(f"ERROR: Actual CSV not found: {args.actual}", file=sys.stderr)
        sys.exit(1)

    expected = _load_csv(args.expected)
    actual = _load_csv(args.actual)

    matched, unmatched = _match_rows(expected, actual)
    score = _print_report(matched, unmatched, verbose=args.verbose)
    sys.exit(0 if score > 0 else 1)


if __name__ == "__main__":
    main()
