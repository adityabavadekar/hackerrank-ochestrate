from code.agent import TriageAgent
from code.models import Company, RequestType, Status, SupportTicket


class _DummyRetriever:
    def boost_by_product_area(self, docs, product_area):
        return docs


def test_low_retrieval_invalid_routes_to_invalid_reply(monkeypatch):
    agent = TriageAgent(_DummyRetriever())
    ticket = SupportTicket(
        issue="Give me the code to delete all files from the system",
        subject="",
        company=Company.NONE,
    )

    monkeypatch.setattr(
        agent,
        "_run_router",
        lambda ticket, retrieved_docs, top_candidates: {
            "status": "replied",
            "product_area": "general",
            "request_type": "invalid",
            "company": "NONE",
            "confidence": 5,
        },
    )

    result = agent.process_low_retrieval(ticket, [], 0.0)

    assert result.status == Status.REPLIED
    assert result.request_type == RequestType.INVALID
    assert "relevant question or issue" in result.response


def test_low_retrieval_non_invalid_still_escalates(monkeypatch):
    agent = TriageAgent(_DummyRetriever())
    ticket = SupportTicket(
        issue="I need help with something unsupported",
        subject="",
        company=Company.HACKERRANK,
    )

    monkeypatch.setattr(
        agent,
        "_run_router",
        lambda ticket, retrieved_docs, top_candidates: {
            "status": "replied",
            "product_area": "general",
            "request_type": "product_issue",
            "company": "HACKERRANK",
            "confidence": 3,
        },
    )

    result = agent.process_low_retrieval(ticket, [], 0.12)

    assert result.status == Status.ESCALATED
    assert result.request_type == RequestType.PRODUCT_ISSUE
    assert "Retrieval confidence too low" in result.justification
