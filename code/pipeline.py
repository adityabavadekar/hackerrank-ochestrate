"""Core pipeline orchestration logic."""

import csv
import sys
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

from . import config

# Load .env before any module reads
load_dotenv(config.ENV_FILE)

from .agent import TriageAgent
from .corpus_loader import CorpusLoader
from . import gate, output_writer
from .logger import get_logger
from .models import Company, Document, SupportTicket, TriageResult
from .retriever import Retriever
from . import validator

logger = get_logger(__name__)


def _parse_company(value: str) -> Company:
    try:
        return Company(value)
    except ValueError:
        return Company.NONE


def _load_tickets(path: Path) -> List[SupportTicket]:
    if not path.exists():
        logger.error("Input CSV not found: %s", path)
        sys.exit(1)

    tickets = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append(
                SupportTicket(
                    issue=row.get("Issue", ""),
                    subject=row.get("Subject"),
                    company=_parse_company(row.get("Company", "None")),
                )
            )

    logger.info("Loaded %d tickets from %s", len(tickets), path)
    return tickets


def _process_ticket(
    ticket: SupportTicket,
    retriever: Retriever,
    agent: TriageAgent,
) -> TriageResult:
    """Runs one ticket through all three pipeline stages.

    Returns an early result at each stage if a condition short-circuits the pipeline.
    """
    gate_result = gate.run(ticket)
    if not gate_result.passed:
        return gate_result.early_result

    sub_issues = gate_result.sub_issues
    sub_results: List[TriageResult] = []

    for sub_text in sub_issues:
        sub_ticket = SupportTicket(
            issue=sub_text,
            subject=ticket.subject,
            company=ticket.company,
        )

        docs, threshold_met = retriever.retrieve(sub_text, ticket.company)
        if not threshold_met:
            sub_results.append(
                validator._escalation_fallback("No corpus support found above similarity threshold.")
            )
            continue

        result = agent.process(sub_ticket, docs)
        sub_results.append(result)

    if len(sub_results) == 1:
        return sub_results[0]

    return validator.merge_sub_results(sub_results)


def run_pipeline(args) -> None:
    if args.model:
        logger.info("Model overridden via CLI: %s", args.model)

    logger.info("Starting pipeline | input=%s | output=%s", args.input, args.output)

    documents: List[Document] = CorpusLoader().load()
    retriever = Retriever(documents)
    agent = TriageAgent()

    tickets = _load_tickets(args.input)

    if args.tickets is not None:
        tickets = tickets[: args.tickets]
        logger.info("Limiting run to first %d ticket(s).", args.tickets)
    results: List[Tuple[SupportTicket, TriageResult]] = []

    for i, ticket in enumerate(tickets, start=1):
        logger.info("[%d/%d] Processing [%s]: %r (company=%s)", i, len(tickets), ticket.id, ticket.subject, ticket.company.value)
        result = _process_ticket(ticket, retriever, agent)
        results.append((ticket, result))
        logger.debug("Result: status=%s request_type=%s", result.status.value, result.request_type.value)

    output_writer.write(results, args.output)
    logger.info("Done. %d tickets processed.", len(results))
