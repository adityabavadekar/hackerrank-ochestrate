# HackerRank Orchestrate: AI Triage Agent

This is the source code for the AI Support Agent built for the HackerRank Orchestrate 24-hour hackathon (May 2026).

## Overview
This agent processes customer support tickets and autonomously categorizes, resolves, or escalates them using a multi-stage LLM pipeline.
It leverages a **Hybrid Retrieval-Augmented Generation (RAG)** approach:
1. **Gate (Stage 1):** Pre-LLM safety checks for prompt injection and hard keyword-based escalation (fraud, legal, data breach).
2. **Retriever (Stage 2):** Fast, cached BM25 + dense vector hybrid retrieval over the provided `data/` knowledge corpus, including dynamic metadata extraction and deduplication.
3. **LLM Triage (Stage 3):** A two-agent sequential workflow. A **Router Agent** strictly categorizes the ticket and assesses confidence. If safe to proceed, a **Responder Agent** drafts the final user-facing reply grounded *only* in the retrieved context.

## Requirements & Setup

This project uses `uv` for lightning-fast dependency management.

1. Install dependencies from the lockfile:
   ```bash
   uv sync
   ```
2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
3. Set up your environment variables. Copy the example file and add your API keys:
   ```bash
   cp ../.env.example ../.env
   ```
   *Note: Ensure you set `LLM_MODEL` (e.g., `anthropic/claude-haiku-3-5` or `openai/gpt-4o`) and the corresponding provider API key (e.g., `ANTHROPIC_API_KEY`).*

## Running the Pipeline

The primary entry point is `code/main.py`, which is optimized for fast CLI invocation.

To run the pipeline on the default `support_tickets.csv` and generate `output.csv`:

```bash
# From the repository root:
python -m code.main
```

### Options:

- **Specify a custom input/output file:**
  ```bash
  python -m code.main --input custom_tickets.csv --output results.csv
  ```
- **Override the LLM model on the fly:**
  ```bash
  python -m code.main --model "openai/gpt-4o"
  ```
- **Smoke test on a small subset of tickets (e.g., first 5):**
  ```bash
  python -m code.main --tickets 5
  ```

Run `python -m code.main --help` to see all available options.

## Determinism
To ensure maximum reproducibility, this agent runs with LLM temperature strictly pinned to `0.0` and utilizes explicit seed parameters (`seed=42`) via the LiteLLM wrapper. Document embeddings are deterministically cached to local disk via SHA256 hashes to guarantee stable context retrieval.
