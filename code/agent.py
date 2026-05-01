"""Stage 3: LLM triage — builds a grounded prompt and calls the LLM wrapper in two stages."""

import json
from collections import Counter
from typing import List

from .logger import get_logger
from . import llm, validator
from .models import RetrievedDoc, SupportTicket, TriageResult, Status, RequestType

logger = get_logger(__name__)

_MAX_CONTEXT_CHUNKS = 5

_ROUTER_SYSTEM_PROMPT = """You are a classification and routing agent for a support platform.
Your job is to read the user ticket and the retrieved context, and output a JSON classification.

# RULES:
1. You MUST select the `product_area` strictly from the provided CANDIDATE LIST.
2. If none of the candidates fit, or if confidence is low, fallback to "general".
3. `status`:
- MUST be "escalated" ONLY if:
  - The issue involves: fraud, unauthorized account access, billing disputes, legal threats, data breaches, or critical system outages
  - OR the context is completely irrelevant or missing and you cannot even infer a safe or useful next step
- MUST be "replied" if:
  - The issue is a standard `product_issue`
  - OR it is an `invalid` (off-topic)
  - OR the context provides enough information to guide the user.
4. Output valid JSON only, exactly matching the requested format.
"""

_ROUTER_OUTPUT_FORMAT = """\
OUTPUT FORMAT (JSON only, no other text):
{
  "product_area": "<string from candidates list>",
  "request_type": "product_issue" | "feature_request" | "bug" | "invalid",
  "status": "replied" | "escalated",
  "confidence": <integer 1-5>
}"""

_RESPONDER_SYSTEM_PROMPT = """You are a support responder agent for three products: HackerRank, Claude, and Visa.

# RULES:
1. You ONLY use information from the CONTEXT provided. Never use outside knowledge.
2. If the context does not fully support an answer, your justification should explain what is missing.
3. Your response must not promise actions you cannot take (e.g. "we will contact your admin"). Only tell the user what they can do themselves or who they should contact.
4. Your justification MUST:
   - Be concise (1–2 sentences maximum).
   - Explain why the response is correct based on the CONTEXT.
   - Include source references using this format:
     [sources: <id>;<id>;<id>]
   - Do not repeat same source. Only 1 source block is allowed which should contain all source you used for the response.
5. Your response should maintain a professional, polite, and neutral tone at all times.
"""

_RESPONDER_OUTPUT_FORMAT = """\
OUTPUT FORMAT (JSON only, no other text):
{
  "response": "<user-facing response>",
  "justification": "<internal reasoning, cite sources by filename>"
}"""


class TriageAgent:
    """Orchestrates Stage 3: Two-agent pipeline (Router -> Responder)."""

    def __init__(self) -> None:
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
        logger.debug("Candidates pool: %s", candidates_pool)
        logger.debug("Top candidates passed to Router: %s", top_candidates)

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
                response="Your request has been escalated for human review.",
                justification=justification_text,
            )

        return self._run_responder(ticket, retrieved_docs, router_data)

    def _run_router(self, ticket: SupportTicket, retrieved_docs: List[RetrievedDoc], top_candidates: List[str]) -> dict:
        router_prompt = self._build_router_prompt(ticket, retrieved_docs, top_candidates)
        logger.debug("Calling Router Agent for ticket: %r", ticket.subject)
        
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
                logger.warning("Attempt %d: Router output did not parse, retrying", attempt)
            except RuntimeError as exc:
                logger.error("Router call failed: %s", exc)
                return {"status": "escalated", "product_area": "general", "request_type": "product_issue", "confidence": 0}
        
        if not router_data:
            logger.error("Router failed to produce valid JSON after 2 attempts.")
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
            "confidence": confidence
        }

    def _run_responder(self, ticket: SupportTicket, retrieved_docs: List[RetrievedDoc], router_data: dict) -> TriageResult:
        responder_prompt = self._build_responder_prompt(ticket, retrieved_docs, router_data)
        logger.debug("Calling Responder Agent for ticket: %r", ticket.subject)
        
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
                logger.warning("Attempt %d: Responder output did not parse, retrying", attempt)
            except RuntimeError as exc:
                logger.error("Responder call failed: %s", exc)
                return validator._escalation_fallback(str(exc))
                
        if not responder_data:
            return validator._escalation_fallback("Responder failed to produce valid JSON after 2 attempts.")

        response = str(responder_data.get("response", "Unable to generate a response. This ticket has been escalated."))
        justification = str(responder_data.get("justification", "Missing justification field."))

        return TriageResult(
            status=Status(router_data["status"]),
            product_area=router_data["product_area"],
            request_type=RequestType(router_data["request_type"]),
            response=response,
            justification=justification,
        )

    def _build_context_string(self, docs: List[RetrievedDoc]) -> str:
        context_blocks = [
            f"[source: {rd.document.id}]\n{rd.document.content}"
            for rd in docs[:_MAX_CONTEXT_CHUNKS]
        ]
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

    def _build_router_prompt(self, ticket: SupportTicket, docs: List[RetrievedDoc], candidates: List[str]) -> str:
        context = self._build_context_string(docs)
        
        candidates_str = ", ".join(f'"{c}"' for c in candidates)
        
        return (
            f"CONTEXT:\n{context}\n\n"
            f"TICKET:\n"
            f"Company: {ticket.company.value}\n"
            f"Issue: {ticket.full_text()}\n\n"
            f"CANDIDATE PRODUCT AREAS:\n[{candidates_str}]\n\n"
            f"{_ROUTER_OUTPUT_FORMAT}"
        )

    def _build_responder_prompt(self, ticket: SupportTicket, docs: List[RetrievedDoc], router_data: dict) -> str:
        context = self._build_context_string(docs)
        
        return (
            f"CONTEXT:\n{context}\n\n"
            f"TICKET:\n"
            f"Company: {ticket.company.value}\n"
            f"Issue: {ticket.full_text()}\n\n"
            f"ROUTER CLASSIFICATION:\n"
            f"Product Area: {router_data.get('product_area')}\n"
            f"Request Type: {router_data.get('request_type')}\n\n"
            f"{_RESPONDER_OUTPUT_FORMAT}"
        )