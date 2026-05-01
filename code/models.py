import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class Status(str, Enum):
    REPLIED = "replied"
    ESCALATED = "escalated"


class RequestType(str, Enum):
    PRODUCT_ISSUE = "product_issue"
    FEATURE_REQUEST = "feature_request"
    BUG = "bug"
    INVALID = "invalid"



class Company(str, Enum):
    HACKERRANK = "HackerRank"
    CLAUDE = "Claude"
    VISA = "Visa"
    NONE = "None"


@dataclass(frozen=True)
class SupportTicket:
    issue: str
    subject: Optional[str]
    company: Company
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def full_text(self) -> str:
        return f"{self.subject or ''} {self.issue}".strip()


@dataclass(frozen=True)
class TriageResult:
    status: Status
    product_area: str
    response: str
    justification: str
    request_type: RequestType

@dataclass(frozen=True)
class Document:
    id: str
    content: str
    source: Company
    meta: Dict[str, str]


@dataclass(frozen=True)
class RetrievedDoc:
    document: Document
    score: float


@dataclass(frozen=True)
class ClassificationResult:
    request_type: RequestType
    product_area: str


@dataclass(frozen=True)
class SafetyResult:
    should_escalate: bool
    reason: str


class SourceType(str, Enum):
    CLAUDE = "claude"
    HACKERRANK = "hackerrank"
    VISA = "visa"