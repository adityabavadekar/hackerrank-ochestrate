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

ENV_FILE: Path = PROJECT_ROOT / ".env"

# Cache paths
CACHE_DIR: Path = PROJECT_ROOT / ".cache"
EMBEDDINGS_CACHE_DIR: Path = CACHE_DIR / "embeddings"

# Environment Variable Keys
ENV_LLM_MODEL: str = "LLM_MODEL"
ENV_LLM_MAX_TOKENS: str = "LLM_MAX_TOKENS"
ENV_LLM_TEMPERATURE: str = "LLM_TEMPERATURE"
ENV_LLM_API_BASE: str = "LLM_API_BASE"
