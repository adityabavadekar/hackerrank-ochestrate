"""Thin LiteLLM wrapper — routes any model/provider from env, no hardcoding."""

import os
import time
from typing import Optional

import litellm
from litellm import completion, RateLimitError, AuthenticationError, BadRequestError

from .logger import get_logger
from . import config
from .cache import disk, lru

logger = get_logger(__name__)

# Suppress litellm's own verbose logging; our wrapper owns the log output.
litellm.suppress_debug_info = True

# Only needed for local Ollama; all hosted providers (OpenRouter, Anthropic,
# OpenAI, Groq, Gemini, etc.) are resolved automatically from their own key vars.

_DEFAULT_MODEL = "anthropic/claude-haiku-3-5"
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.0

_MAX_RETRIES = 2
_RETRY_DELAY_SECONDS = 3


def _get_model() -> str:
    return os.environ.get(config.ENV_LLM_MODEL, _DEFAULT_MODEL)


def _get_max_tokens() -> int:
    try:
        return int(os.environ.get(config.ENV_LLM_MAX_TOKENS, _DEFAULT_MAX_TOKENS))
    except ValueError:
        logger.warning("Invalid %s value; using default %d", config.ENV_LLM_MAX_TOKENS, _DEFAULT_MAX_TOKENS)
        return _DEFAULT_MAX_TOKENS


def _get_temperature() -> float:
    try:
        return float(os.environ.get(config.ENV_LLM_TEMPERATURE, _DEFAULT_TEMPERATURE))
    except ValueError:
        logger.warning("Invalid %s value; using default %.1f", config.ENV_LLM_TEMPERATURE, _DEFAULT_TEMPERATURE)
        return _DEFAULT_TEMPERATURE


def call(
    system: str,
    user: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Calls any LLM via LiteLLM using model and API keys from environment variables.

    Model name follows LiteLLM conventions: ``provider/model-name``.
    Examples: ``anthropic/claude-haiku-3-5``, ``openai/gpt-4o``,
    ``groq/llama-3.3-70b-versatile``, ``gemini/gemini-1.5-flash``.

    The relevant API key env var (e.g. ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``)
    is picked up automatically by LiteLLM from the environment. No configuration
    beyond setting the right key is needed to switch providers.

    Responses are disk-cached (key = system+user+model+tokens+temp) so identical
    prompts never hit the API twice. Cache is bypassed when temperature > 0.

    Args:
        system: System prompt text.
        user: User message text (the grounded prompt).
        model: Override the model from env. If None, reads ``LLM_MODEL``.
        max_tokens: Override token limit from env. If None, reads ``LLM_MAX_TOKENS``.
        temperature: Override temperature from env. If None, reads ``LLM_TEMPERATURE``.

    Returns:
        Raw text content from the LLM response.

    Raises:
        RuntimeError: When all retry attempts are exhausted or an unrecoverable
            error occurs (auth failure, bad request).
    """
    resolved_model = model or _get_model()
    resolved_max_tokens = max_tokens or _get_max_tokens()
    resolved_temperature = temperature if temperature is not None else _get_temperature()
    return _call_cached(system, user, resolved_model, resolved_max_tokens, resolved_temperature)


def _call_cached(
    system: str,
    user: str,
    resolved_model: str,
    resolved_max_tokens: int,
    resolved_temperature: float,
) -> str:
    """Routes through disk cache for deterministic (temp=0) calls; bypasses for temp>0."""
    if resolved_temperature > 0:
        # Non-deterministic output — skip cache entirely.
        return _call_uncached(system, user, resolved_model, resolved_max_tokens, resolved_temperature)
    return _call_cached_deterministic(system, user, resolved_model, resolved_max_tokens, resolved_temperature)


@disk("llm")
def _call_cached_deterministic(
    system: str,
    user: str,
    resolved_model: str,
    resolved_max_tokens: int,
    resolved_temperature: float,
) -> str:
    """Disk-cached inner call for deterministic (temperature=0) prompts."""
    return _call_uncached(system, user, resolved_model, resolved_max_tokens, resolved_temperature)


def _call_uncached(
    system: str,
    user: str,
    resolved_model: str,
    resolved_max_tokens: int,
    resolved_temperature: float,
) -> str:

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    logger.debug(
        "LLM call | model=%s max_tokens=%d temperature=%.2f",
        resolved_model,
        resolved_max_tokens,
        resolved_temperature,
    )

    # api_base is only required for local Ollama. For all hosted providers
    # (OpenRouter, Anthropic, OpenAI, Groq, etc.) LiteLLM resolves the URL
    # and key automatically from the environment.
    api_base: Optional[str] = os.environ.get(config.ENV_LLM_API_BASE) or None

    if api_base:
        logger.debug("Using custom api_base=%s", api_base)

    last_error: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = completion(
                model=resolved_model,
                messages=messages,
                max_tokens=resolved_max_tokens,
                temperature=resolved_temperature,
                seed=42,
                api_base=api_base,
            )
            text = response.choices[0].message.content
            logger.debug(
                "LLM response received | attempt=%d tokens_used=%s",
                attempt,
                getattr(response.usage, "total_tokens", "unknown"),
            )
            return text

        except AuthenticationError as exc:
            # Auth errors won't fix themselves on retry.
            logger.error("Authentication failed for model=%s: %s", resolved_model, exc)
            raise RuntimeError(
                f"API key missing or invalid for model '{resolved_model}'. "
                "Check your .env file and set the correct key (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY)."
            ) from exc

        except BadRequestError as exc:
            # Bad prompt or unsupported param — retrying won't help.
            logger.error("Bad request to model=%s: %s", resolved_model, exc)
            raise RuntimeError(f"Bad request to '{resolved_model}': {exc}") from exc

        except RateLimitError as exc:
            logger.warning(
                "Rate limit hit on attempt %d/%d for model=%s; retrying in %ds",
                attempt,
                _MAX_RETRIES,
                resolved_model,
                _RETRY_DELAY_SECONDS,
            )
            last_error = exc
            time.sleep(_RETRY_DELAY_SECONDS)

        except Exception as exc:
            logger.warning(
                "LLM call failed on attempt %d/%d: %s",
                attempt,
                _MAX_RETRIES,
                exc,
            )
            last_error = exc
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"LLM call to '{resolved_model}' failed after {_MAX_RETRIES} attempts."
    ) from last_error


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from colorama import Fore, Style, init as _colorama_init
    from . import config

    _colorama_init(autoreset=True)
    load_dotenv(config.ENV_FILE, override=True)

    _model = os.environ.get(config.ENV_LLM_MODEL, _DEFAULT_MODEL)
    _base = os.environ.get(config.ENV_LLM_API_BASE) or "(not set)"
    _user_msg = "Hi"
    _system_msg = "You are a helpful assistant. Reply in 5 words or fewer."

    print(f"{Fore.CYAN}LLM smoke test{Style.RESET_ALL}")
    print(f"  model   : {Fore.YELLOW}{_model}{Style.RESET_ALL}")
    print(f"  api_base: {Fore.YELLOW}{_base}{Style.RESET_ALL}")
    print(f"  input   : {Fore.WHITE}{_user_msg!r}{Style.RESET_ALL}")

    try:
        _response = call(system=_system_msg, user=_user_msg, max_tokens=10)
        print(f"  response: {Fore.GREEN}{_response!r}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Smoke test passed.{Style.RESET_ALL}")
    except RuntimeError as _exc:
        print(f"{Fore.RED}Smoke test failed: {_exc}{Style.RESET_ALL}")
        sys.exit(1)
