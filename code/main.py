"""Fast CLI entry point. Parses arguments before loading heavy dependencies."""

import argparse
import os
from pathlib import Path

from . import config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HackerRank Orchestrate: AI-powered support ticket triage pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=config.DEFAULT_INPUT_CSV,
        help="Path to input CSV file containing support tickets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DEFAULT_OUTPUT_CSV,
        help="Path where the output CSV will be written.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "LiteLLM model string to use, e.g. 'openrouter/elephant-alpha', "
            "'openai/gpt-4o', 'anthropic/claude-haiku-3-5'. "
            "Overrides LLM_MODEL from .env."
        ),
    )
    parser.add_argument(
        "--tickets",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N tickets. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--cooldown-every",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Pause after every N processed tickets to reduce API burstiness. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=15.0,
        metavar="S",
        help=(
            "How long to sleep when the cooldown triggers. "
            "Used only when --cooldown-every is greater than 0."
        ),
    )
    args = parser.parse_args()

    if args.cooldown_every < 0:
        parser.error("--cooldown-every must be 0 or a positive integer.")
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be 0 or a positive number.")

    return args


def main() -> None:
    # 1. Parse args (super fast, no heavy imports yet)
    args = _parse_args()

    # 2. Set environment overrides early
    if args.model:
        os.environ[config.ENV_LLM_MODEL] = args.model

    # 3. Load heavy modules and run pipeline
    from .pipeline import run_pipeline
    run_pipeline(args)


if __name__ == "__main__":
    main()
