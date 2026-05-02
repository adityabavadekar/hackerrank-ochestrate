from code.agent import TriageAgent
from code.models import Company, Document, RetrievedDoc, SupportTicket


class _DummyRetriever:
    def boost_by_product_area(self, docs, product_area):
        return docs


def test_router_prompt_scrubs_ticket_but_not_context():
    agent = TriageAgent(_DummyRetriever())
    ticket = SupportTicket(
        issue="username: adi password=secret123",
        subject="Login issue",
        company=Company.CLAUDE,
    )
    docs = [
        RetrievedDoc(
            document=Document(
                id="claude/doc.md",
                content="Support doc content with literal password reset wording intact.",
                source=Company.CLAUDE,
                meta={"product_area": "account_management"},
            ),
            score=0.9,
        )
    ]

    prompt = agent._build_router_prompt(ticket, docs, ["account_management", "general"])

    assert "Issue: Login issue username: [USERNAME] password=[PASSWORD]" in prompt
    assert "Support doc content with literal password reset wording intact." in prompt
    assert "secret123" not in prompt
