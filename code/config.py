"""Central registry for all project paths and file locations."""

from pathlib import Path

# Path to the code/ directory.
CODE_DIR: Path = Path(__file__).resolve().parent

# Repository root.
PROJECT_ROOT: Path = CODE_DIR.parent

CORPUS_DIR: Path = PROJECT_ROOT / "data"

TICKETS_DIR: Path = PROJECT_ROOT / "support_tickets"
DEFAULT_INPUT_CSV: Path = TICKETS_DIR / "support_tickets.csv"
DEFAULT_OUTPUT_CSV: Path = TICKETS_DIR / "output.csv"
SAMPLE_SUPPORT_TICKETS_CSV: Path = TICKETS_DIR / "sample_support_tickets.csv"

ENV_FILE: Path = PROJECT_ROOT / ".env"

# Log paths
LOGS_DIR: Path = PROJECT_ROOT / "logs"
APP_LOG_FILE: Path = LOGS_DIR / "orchestrate.log"

# Cache paths
CACHE_DIR: Path = PROJECT_ROOT / ".cache"
EMBEDDINGS_CACHE_DIR: Path = CACHE_DIR / "embeddings"

# Environment Variable Keys
ENV_LLM_MODEL: str = "LLM_MODEL"
ENV_LLM_MAX_TOKENS: str = "LLM_MAX_TOKENS"
ENV_LLM_TEMPERATURE: str = "LLM_TEMPERATURE"
ENV_LLM_API_BASE: str = "LLM_API_BASE"

# Static response strings
ESCALATION_RESPONSE: str = "Escalate to a human"

# Gate model configuration
GATE_MNLI_MODEL: str = "facebook/bart-large-mnli"
GATE_MNLI_THRESHOLD: float = 0.87
GATE_DETOXIFY_THRESHOLD: float = 0.92
GATE_FUZZY_THRESHOLD: float = 88.0

# PII scrubbing placeholders
PII_EMAIL_TOKEN: str = "[EMAIL]"
PII_PHONE_TOKEN: str = "[PHONE]"
PII_PASSWORD_TOKEN: str = "[PASSWORD]"
PII_USERNAME_TOKEN: str = "[USERNAME]"
PII_CARD_TOKEN: str = "[CARD_NUMBER]"
PII_IP_TOKEN: str = "[IP_ADDRESS]"
PII_SECRET_TOKEN: str = "[SECRET]"
PRESIDIO_ENABLED: bool = False  # disabled due to time it takes 