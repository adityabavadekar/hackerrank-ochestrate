"""Writes the final output CSV with one row per processed ticket."""

from code.models import Company
import csv
from pathlib import Path
from typing import List, Tuple

from .logger import get_logger
from .models import SupportTicket, TriageResult

logger = get_logger(__name__)

_FIELDNAMES = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]

def map_company(company: Company) -> str:
    match company:
        case Company.HACKERRANK:
            return "HackerRank"
        case Company.CLAUDE:
            return "Claude"
        case Company.VISA:
            return "Visa"
        case Company.NONE:
            return "None"


def write(results: List[Tuple[SupportTicket, TriageResult]], output_path: Path) -> None:
    """Writes all triage results to a CSV file, creating parent dirs if needed.

    Args:
        results: Ordered list of (Ticket, TriageResult) pairs.
        output_path: Destination path for the CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for ticket, result in results:
            writer.writerow({
                "issue": ticket.issue,
                "subject": ticket.subject or "",
                "company": map_company(ticket.company),
                "response": result.response,
                "product_area": result.product_area,
                "status": result.status.to_expected_form(),
                "request_type": result.request_type.value,
                "justification": result.justification,
            })

    logger.info("Output written to %s (%d rows)", output_path, len(results))
