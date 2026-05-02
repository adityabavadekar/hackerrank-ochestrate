"""Core pipeline orchestration logic."""

import csv
import sys
import time
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
from .pii import log_pii_mode
from .retriever import Retriever
from .translator import maybe_translate
from . import validator

logger = get_logger(__name__)


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
                    company=Company.try_from_str(row.get("Company", "GENERAL")),
                )
            )

    logger.info("Loaded %d tickets from %s", len(tickets), path)
    return tickets


def _process_ticket(
    ticket: SupportTicket,
    retriever: Retriever,
    agent: TriageAgent,
) -> TriageResult:
    result = _resolve_ticket(ticket, retriever, agent)
    return maybe_translate(ticket, result)


def _resolve_ticket(
    ticket: SupportTicket,
    retriever: Retriever,
    agent: TriageAgent,
) -> TriageResult:
    """Runs one ticket through all three pipeline stages.

    Returns an early result at each stage if a condition short-circuits the pipeline.
    """
    gate_result = gate.run(ticket, retriever)
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
            best_score = docs[0].score if docs else 0.0
            sub_results.append(agent.process_low_retrieval(sub_ticket, docs, best_score))
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
    log_pii_mode()

    documents: List[Document] = CorpusLoader().load()
    retriever = Retriever(documents)
    agent = TriageAgent(retriever)
    retriever.init_semantic_gate()

    tickets = _load_tickets(args.input)

    if args.tickets is not None:
        tickets = tickets[: args.tickets]
        logger.info("Limiting run to first %d ticket(s).", args.tickets)
    results: List[Tuple[SupportTicket, TriageResult]] = []
    cooldown_every = getattr(args, "cooldown_every", 0)
    cooldown_seconds = getattr(args, "cooldown_seconds", 0.0)

    for i, ticket in enumerate(tickets, start=1):
        logger.info("[%d/%d] Processing [%s]: %r (company=%s)", i, len(tickets), ticket.id, ticket.subject, ticket.company.value)
        result = _process_ticket(ticket, retriever, agent)
        results.append((ticket, result))
        logger.debug("Result: status=%s request_type=%s", result.status.value, result.request_type.value)

        if (
            cooldown_every > 0
            and cooldown_seconds > 0
            and i < len(tickets)
            and i % cooldown_every == 0
        ):
            logger.info(
                "Cooldown triggered after %d ticket(s); sleeping for %.1f second(s) to reduce API rate limiting.",
                i,
                cooldown_seconds,
            )
            time.sleep(cooldown_seconds)

    output_writer.write(results, args.output)
    logger.info("Done. %d tickets processed.", len(results))
