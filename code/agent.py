"""Stage 3: LLM triage — builds a grounded prompt and calls the LLM wrapper in two stages."""

import dataclasses
import json
from collections import Counter
from typing import List
import re

from .logger import get_logger
from . import config, llm, validator
from .models import Company, RetrievedDoc, SupportTicket, TriageResult, Status, RequestType
from .pii import scrub_text
from .retriever import Retriever

logger = get_logger(__name__)

_MAX_CONTEXT_CHUNKS = 5

_ROUTER_SYSTEM_PROMPT = """You are a classification and routing agent for a support platform.
Your job is to read the user ticket and the retrieved context, and output a JSON classification.

# RULES:
1. You MUST select the `product_area` strictly from the provided CANDIDATE LIST.
2. If none of the candidates fit, or if confidence is low, fallback to "general".
3. If the ticket describes a real user complaint but retrieved context is irrelevant or unhelpful, set request_type="product_issue" and status="escalated" - never "invalid"
4. `status`:
- MUST be "escalated" ONLY if:
  - The issue involves: fraud, unauthorized account access, billing disputes, legal threats, data breaches, or critical system outages
  - OR the context is completely irrelevant or missing and you cannot even infer a safe or useful next step
- MUST be "replied" if:
  - The issue is a standard `product_issue`
  - OR it is an `invalid` (pleasantries, greetings, thanks, insults, pure conversational filler, or completely off-topic questions)
  - OR the context provides enough information to guide the user.
5. `inferred_company`:
   - If the ticket specifies "Company: None", infer the company from the ticket content.
   - Use vocabulary clues: "assessment", "proctoring", "recruiter" -> "hackerrank"; "claude", "prompt", "anthropic", "api key" -> "claude"; "card", "transaction", "merchant" -> "visa".
   - If genuinely ambiguous, output "none".
   - If the company is already known (not "none"), still output it - do not override it.
6. `invalid_reason`:
   - If `request_type` is "invalid", provide a reason categorization.
7. Output valid JSON only, exactly matching the requested format.
"""

_ROUTER_OUTPUT_FORMAT = """\
OUTPUT FORMAT (JSON only, no other text):
{
  "product_area": "<string from candidates list>",
  "request_type": "product_issue" | "feature_request" | "bug" | "invalid",
  "status": "replied" | "escalated",
  "inferred_company": "hackerrank" | "claude" | "visa" | "none",
  "confidence": <integer 1-5>,
  "invalid_reason": "greeting" | "off_topic" | "conversational" | "gibberish" | "insult" | "out_of_scope_service" | null
}"""

_RESPONDER_SYSTEM_PROMPT = """You are a support responder agent for three products: HackerRank, Claude, and Visa.

Your goal is to generate a clear, helpful, and professional response to the user based ONLY on the provided CONTEXT.

# Language rules
- If the user's language is not English, respond in the same language. Maintain the language throughout the response. Justification language MUST be English.

# RULES:
1. You ONLY use information from the CONTEXT provided. Never use outside knowledge.
2. If the context does not fully support an answer, your justification should explain what is missing.
3. Your response must not promise actions you cannot take (e.g. "we will contact your admin"). Only tell the user what they can do themselves or who they should contact.
4. Response Quality:
    - Be clear, structured, and easy to follow.
    - Prefer step-by-step instructions when applicable.
    - Use short paragraphs or bullet points for readability.
    - Avoid unnecessary repetition or filler.
5. Tone & Style:
    - Maintain a professional, polite, and neutral tone.
    - Be helpful and solution-oriented.
    - Do NOT sound robotic or overly formal.
    - Do NOT use slang or casual phrases.
    - Do NOT include apologies unless the situation clearly warrants it.
6. Your justification MUST:
    - Be concise (1–2 sentences maximum). Minimal 1 sentence required other than the sources.
    - Explain why the response is correct based on the CONTEXT.
    - Include source references using this format:
        [sources: <id>;<id>;<id>]
    - Always use [sources: ] format for sources. Never repeat this block.
    - Do not repeat same source. Only 1 source block is allowed which should contain all source you used for the response.
7. Your response should maintain a professional, polite, and neutral tone at all times.
"""

_RESPONDER_OUTPUT_FORMAT = """\
OUTPUT FORMAT (JSON only, no other text):
{
  "response": "<user-facing response>",
  "justification": "<internal reasoning, cite sources by filename>"
}"""

_INVALID_RESPONSES = {
    "greeting":      "Happy to help! If you need anything else, feel free to ask.",
    "conversational":"Please describe the issue you're experiencing and I'll do my best to assist.",
    "gibberish":     "Your message was unclear. Please describe your issue in detail so I can help.",
    "insult":        "I'm here to help with product support. Please describe your issue and I'll be happy to assist.",
    "off_topic":     "I'm sorry, this is outside the scope of what I can help with. I'm a support assistant for HackerRank, Claude, and Visa.",
    "out_of_scope_service":    "I can only assist with HackerRank, Claude, and Visa products. For other services, please contact their support directly.",
    None:            "I'm a support assistant. Please describe your issue and I'll be happy to help.",
}
_DEFAULT_INVALID_RESPONSE = "I am a support assistant. Please provide a relevant question or issue for me to help you with."


class TriageAgent:
    """Orchestrates Stage 3: Two-agent pipeline (Router -> Responder)."""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        logger.info("TriageAgent ready (model from LLM_MODEL env)")

    def process(
        self,
        ticket: SupportTicket,
        retrieved_docs: List[RetrievedDoc],
    ) -> TriageResult:
        """Runs the two-agent pipeline."""
        candidates_pool = [doc.document.meta.get("product_area", "general") for doc in retrieved_docs]
        counts = Counter(candidates_pool)
        top_candidates = [area for area, _ in counts.most_common(30)]
        if "general" not in top_candidates:
            top_candidates.append("general")

        # TODO: temporary logging
        logger.debug("[%s] Top candidates passed to Router: %s", ticket.id, top_candidates)

        router_data = self._run_router(ticket, retrieved_docs, top_candidates)

        if router_data["status"] == "escalated":
            justification_text = self._generate_escalation_justification(
                router_data.get("product_area", "general"),
                router_data.get("request_type", "product_issue"),
                router_data.get("confidence", 0)
            )
            return TriageResult(
                status=Status.ESCALATED,
                product_area=router_data["product_area"],
                request_type=RequestType(router_data["request_type"]),
                response=config.ESCALATION_RESPONSE,
                justification=justification_text,
            )

        if router_data.get("request_type") == "invalid":
            logger.debug("[%s] Router classified ticket as invalid, skipping Responder", ticket.id)
            invalid_reason = router_data.get("invalid_reason")
            response_text = _INVALID_RESPONSES.get(invalid_reason, _DEFAULT_INVALID_RESPONSE)
            return TriageResult(
                status=Status.REPLIED,
                product_area="general",
                request_type=RequestType.INVALID,
                response=response_text,
                justification=f"Blocked invalid/conversational ticket before generation. (Reason: {invalid_reason}, Confidence: {router_data.get('confidence', 5)}/5)",
            )
        
        if ticket.company == Company.NONE:
            inferred = router_data.get("inferred_company", "none")
            if inferred not in {"hackerrank", "claude", "visa", "none"}:
                inferred = "none"
            inferred_company = Company.try_from_str(inferred)
            if inferred_company != Company.NONE:
                ticket = dataclasses.replace(ticket, company=inferred_company)
                # Filter already-retrieved docs to prefer inferred company (no extra retrieval call)
                company_docs = [d for d in retrieved_docs if d.document.source == inferred_company]
                if company_docs:
                    logger.debug("[%s] Filtered to %d docs for inferred company=%s (no re-retrieve)",
                                 ticket.id, len(company_docs), inferred_company.value)
                    retrieved_docs = company_docs

        product_area = router_data.get("product_area", "general")
        if product_area and product_area != "general":
            retrieved_docs = self._retriever.boost_by_product_area(retrieved_docs, product_area)

        return self._run_responder(ticket, retrieved_docs, router_data)

    def process_low_retrieval(
        self,
        ticket: SupportTicket,
        retrieved_docs: List[RetrievedDoc],
        best_score: float,
    ) -> TriageResult:
        """Lets the router classify low-retrieval tickets before defaulting to escalation.

        This catches clearly invalid or off-topic prompts that should be replied to
        as invalid rather than escalated just because retrieval had no good match.
        """
        logger.info(
            "[%s] Low-retrieval fallback: routing ticket before escalation (best_score=%.3f)",
            ticket.id,
            best_score,
        )

        # Extract candidate product areas even from low-score docs so the router
        # can still produce a specific classification instead of defaulting to general.
        candidates_pool = [d.document.meta.get("product_area", "general") for d in retrieved_docs]
        counts = Counter(candidates_pool)
        top_candidates = [area for area, _ in counts.most_common(10)]
        if "general" not in top_candidates:
            top_candidates.append("general")

        router_data = self._run_router(ticket, retrieved_docs, top_candidates)

        if ticket.company == Company.NONE:
            inferred = router_data.get("inferred_company", "none")
            inferred_company = Company.try_from_str(inferred)
            if inferred_company != Company.NONE:
                ticket = dataclasses.replace(ticket, company=inferred_company)
                company_docs = [d for d in retrieved_docs if d.document.source == inferred_company]
                if company_docs:
                    logger.debug("[%s] Filtered to %d docs for inferred company=%s (low retrieval)",
                                 ticket.id, len(company_docs), inferred_company.value)
                    retrieved_docs = company_docs

        if router_data.get("request_type") == "invalid":
            logger.debug("[%s] Low-retrieval ticket classified as invalid by router", ticket.id)
            invalid_reason = router_data.get("invalid_reason")
            response_text = _INVALID_RESPONSES.get(invalid_reason, _DEFAULT_INVALID_RESPONSE)
            return TriageResult(
                status=Status.REPLIED,
                product_area="general",
                request_type=RequestType.INVALID,
                response=response_text,
                justification=(
                    "Router classified the ticket as invalid/off-topic despite low retrieval support. "
                    f"(Reason: {invalid_reason}, Confidence: {router_data.get('confidence', 5)}/5)"
                ),
            )

        product_area = router_data.get("product_area", "general")
        request_type_str = router_data.get("request_type", "product_issue")
        confidence = router_data.get("confidence", 0)

        # High-confidence bypass: if the router is certain this is a real support
        # ticket, attempt to answer it even though retrieval was weak.
        if confidence >= 4:
            logger.info(
                "[%s] Low-retrieval bypass: router confidence=%d >= 4, running responder (score=%.3f)",
                ticket.id, confidence, best_score,
            )
            return self._run_responder(ticket, retrieved_docs, router_data)

        justification = (
            f"Retrieval confidence too low to answer safely (best match score: {best_score:.2f}). "
            f"Router classified this as {request_type_str} in {product_area}. "
            f"Escalated for human review. (Confidence: {confidence}/5)"
        )
        return TriageResult(
            status=Status.ESCALATED,
            product_area=product_area,
            request_type=RequestType(request_type_str),
            response=config.ESCALATION_RESPONSE,
            justification=justification,
        )


    def _run_router(self, ticket: SupportTicket, retrieved_docs: List[RetrievedDoc], top_candidates: List[str]) -> dict:
        router_prompt = self._build_router_prompt(ticket, retrieved_docs, top_candidates)
        logger.debug("[%s] Calling Router Agent for ticket: %r", ticket.id, ticket.subject)
        
        # TODO: temporary logging
        # logger.debug("Router User prompt: \n===START PROMPT===\n%s\n===END PROMPT===", router_prompt)
        
        router_data = None
        for attempt in range(1, 3):
            try:
                raw = llm.call(system=_ROUTER_SYSTEM_PROMPT, user=router_prompt)
                
                # TODO: temporary logging
                # logger.debug("Router LLM raw response: \n===START LLM RAW===\n%s\n===END LLM RAW===", raw)
                
                router_data = validator.parse_string_as_json(raw)
                if router_data is not None:
                    # TODO: temporary logging
                    logger.debug(f"Router LLM parsed response: {json.dumps(router_data, indent=2)}")
                    break
                logger.warning("[%s] Attempt %d: Router output did not parse, retrying", ticket.id, attempt)
            except RuntimeError as exc:
                logger.error("[%s] Router call failed: %s", ticket.id, exc)
                return {"status": "escalated", "product_area": "general", "request_type": "product_issue", "confidence": 0}
        
        if not router_data:
            logger.error("[%s] Router failed to produce valid JSON after 2 attempts.", ticket.id)
            return {"status": "escalated", "product_area": "general", "request_type": "product_issue", "confidence": 0}

        product_area = router_data.get("product_area", "general")
        if product_area not in top_candidates:
            product_area = "general"
            
        request_type = str(router_data.get("request_type", "product_issue")).lower()
        if request_type not in {"product_issue", "feature_request", "bug", "invalid"}:
            request_type = "product_issue"
            
        status_val = str(router_data.get("status", "escalated")).lower()
        if status_val not in {"replied", "escalated"}:
            status_val = "escalated"

        inferred_company = str(router_data.get("inferred_company", "none")).lower()
        if inferred_company not in {"hackerrank", "claude", "visa", "none"}:
            inferred_company = "none"

        confidence = router_data.get("confidence", 5)
        try:
            confidence = int(confidence)
        except (ValueError, TypeError):
            confidence = 5
            
        if confidence < 3:
            product_area = "general"

        return {
            "status": status_val,
            "product_area": product_area,
            "request_type": request_type,
            "invalid_reason": router_data.get("invalid_reason"),
            "inferred_company": inferred_company,
            "confidence": confidence
        }

    def _run_responder(self, ticket: SupportTicket, retrieved_docs: List[RetrievedDoc], router_data: dict) -> TriageResult:
        responder_prompt = self._build_responder_prompt(ticket, retrieved_docs, router_data)
        logger.debug("[%s] Calling Responder Agent for ticket: %r", ticket.id, ticket.subject)
        
        # TODO: temporary logging
        # logger.debug("Responder User prompt: \n===START PROMPT===\n%s\n===END PROMPT===", responder_prompt)
        
        responder_data = None
        for attempt in range(1, 3):
            try:
                raw = llm.call(system=_RESPONDER_SYSTEM_PROMPT, user=responder_prompt)
                
                # TODO: temporary logging
                # logger.debug("Responder LLM raw response: \n===START LLM RAW===\n%s\n===END LLM RAW===", raw)
                
                responder_data = validator.parse_string_as_json(raw)
                if responder_data is not None:
                    # TODO: temporary logging
                    logger.debug(f"Responder LLM parsed response: {json.dumps(responder_data, indent=2)}")
                    break
                logger.warning("[%s] Attempt %d: Responder output did not parse, retrying", ticket.id, attempt)
            except RuntimeError as exc:
                logger.error("[%s] Responder call failed: %s", ticket.id, exc)
                return validator._escalation_fallback(str(exc))
                
        if not responder_data:
            logger.error("[%s] Responder failed to produce valid JSON after 2 attempts.", ticket.id)
            return validator._escalation_fallback("Responder failed to produce valid JSON after 2 attempts.")

        response = str(responder_data.get("response", config.ESCALATION_RESPONSE))
        justification = str(responder_data.get("justification", "Missing justification field."))
        
        # Post-process sources: extract, validate against retrieved docs, deduplicate, enforce format
        valid_doc_ids: set[str] = {rd.document.id for rd in retrieved_docs}
        raw_sources = set(re.findall(r'\[sources?:?\s*([^\]]+)\]', justification, flags=re.IGNORECASE))
        cleaned_sources: list[str] = []
        if raw_sources:
            for src_group in raw_sources:
                for src in re.split(r'[,;]\s*', src_group):
                    src = src.strip()
                    if src and src in valid_doc_ids:
                        cleaned_sources.append(src)
                    elif src:
                        logger.debug("[%s] Dropping invalid/hallucinated source: %r", ticket.id, src)

        # Fallback: if no valid sources were cited, use the top retrieved docs
        if not cleaned_sources and retrieved_docs and retrieved_docs[0].score > 0.4:
            cleaned_sources = [rd.document.id for rd in retrieved_docs[:2]]
            logger.debug("[%s] No valid sources cited; using top-%d retrieved docs", ticket.id, len(cleaned_sources))

        # Remove all raw source blocks and append a single unified block
        justification = re.sub(r'\s*\[sources?:?[^\]]+\]', '', justification, flags=re.IGNORECASE).strip()
        if len(cleaned_sources) > 0:
            justification += f" [sources: {';'.join(sorted(set(cleaned_sources)))}]"

        return TriageResult(
            status=Status(router_data["status"]),
            product_area=router_data["product_area"],
            request_type=RequestType(router_data["request_type"]),
            response=response,
            justification=justification,
        )

    def _build_context_string(self, docs: List[RetrievedDoc]) -> str:
        context_blocks = []
        total_len = 0
        max_chars = 12000  # Hard context size limit to prevent overflow
        
        for rd in docs[:_MAX_CONTEXT_CHUNKS]:
            block = f"[source: {rd.document.id}]\n{rd.document.content}"
            if total_len + len(block) > max_chars:
                block = block[:max_chars - total_len] + "...\n[TRUNCATED]"
                context_blocks.append(block)
                break
            context_blocks.append(block)
            total_len += len(block)
            
        return "\n\n".join(context_blocks)

    def _generate_escalation_justification(self, product_area: str, request_type: str, confidence: int) -> str:
        """Builds a descriptive hardcoded escalation justification based on semantic keys."""
        area_clean = product_area.replace("_", " ").title()
        type_clean = request_type.replace("_", " ")
        
        base = f"Escalated {type_clean} regarding {area_clean} for human review."
        
        if request_type == "bug":
            base = f"Escalated bug report in the {area_clean} area to the engineering team for further investigation."
        elif request_type == "feature_request":
            base = f"Escalated feature request for {area_clean} to the product management team for review."
            
        return f"{base} (Confidence: {confidence}/5)"

    def _masked_ticket_text(self, ticket: SupportTicket) -> str:
        scrubbed = scrub_text(ticket.full_text())
        if scrubbed.replacements:
            logger.info(
                "[%s] PII scrubber masked %d sensitive value(s) in ticket text before prompt assembly.",
                ticket.id,
                scrubbed.replacements,
            )
        return scrubbed.text

    def _build_router_prompt(self, ticket: SupportTicket, docs: List[RetrievedDoc], candidates: List[str]) -> str:
        context = self._build_context_string(docs)
        masked_issue = self._masked_ticket_text(ticket)
        
        candidates_str = ", ".join(f'"{c}"' for c in candidates)
        
        return (
            f"CONTEXT:\n{context}\n\n"
            f"TICKET:\n"
            f"Company: {ticket.company.value}\n"
            f"Issue: {masked_issue}\n\n"
            f"CANDIDATE PRODUCT AREAS:\n[{candidates_str}]\n\n"
            f"{_ROUTER_OUTPUT_FORMAT}"
        )

    def _build_responder_prompt(self, ticket: SupportTicket, docs: List[RetrievedDoc], router_data: dict) -> str:
        context = self._build_context_string(docs)
        masked_issue = self._masked_ticket_text(ticket)
        
        return (
            f"CONTEXT:\n{context}\n\n"
            f"TICKET:\n"
            f"Company: {ticket.company.value}\n"
            f"Issue: {masked_issue}\n\n"
            f"ROUTER CLASSIFICATION:\n"
            f"Product Area: {router_data.get('product_area')}\n"
            f"Request Type: {router_data.get('request_type')}\n\n"
            f"{_RESPONDER_OUTPUT_FORMAT}"
        )
