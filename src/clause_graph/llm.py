"""The only module that talks to an LLM (step 6+), like embeddings.py for embeddings.

Data rule (CLAUDE.md, ADR 0003/0004): LLM API calls are allowed only because CUAD is
public. Callers send only minimised candidate text, never whole contracts, and log
model + prompt version on everything the LLM produces.

Provider via .env: LLM_PROVIDER=anthropic|ollama (ollama: step 7c), LLM_MODEL,
COMPARE_MODEL, ANTHROPIC_API_KEY. The interface is provider-neutral: complete_json()
takes a system prompt, a user message and a JSON schema, and returns the parsed JSON
object plus usage. Both providers constrain decoding to the schema (Anthropic
structured outputs; Ollama's `format` JSON-schema mode), so callers validate the
same way regardless of provider.

Why structured outputs, not a forced tool: Claude Sonnet 5.5 and Opus 5.5 reject
tool_choice {"type": "tool"} with a 400 error (platform.claude.com, tool-use docs,
checked 2026-11-14); structured outputs give the same guarantee (schema-valid
JSON) and map 1:1 onto Ollama's JSON-schema mode. See ADR 0004.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_COMPARE_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 8000
TIMEOUT_S = 300

# USD per million tokens (input, output), base rates without batch or cache discounts.
# Source: https://platform.claude.com/docs/en/about-claude/pricing
PRICES_AS_OF = "2026-11-14"
PRICES = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-5-5": (0.10, 0.50),  # "from" price
}

# Lowest thinking setting per model. Extended (manual) thinking is never enabled.
# Sonnet 5.5 rejects "disabled"; "between_tools" turns off up-front thinking (no tools
# here, so no thinking at all). Opus 5.5 cannot turn thinking off: it stays adaptive
# and its thinking tokens are billed as output (visible in usage).
THINKING = {
    "claude-sonnet-5-5": {"type": "between_tools"},
    "claude-haiku-5-5": {"type": "disabled"},
}


class LLMError(Exception):
    """The LLM is misconfigured, unreachable, or returned no usable answer."""


@dataclass(frozen=True)
class Config:
    provider: str
    model: str
    compare_model: str
    effort: str


@dataclass(frozen=True)
class Completion:
    data: dict | None  # parsed JSON object, None if the call produced none
    raw_text: str  # exactly what the model returned (cached for audit)
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    model: str
    error: str | None = None  # e.g. "refusal", "max_tokens", "invalid_json"


def config() -> Config:
    load_dotenv(ROOT / ".env")
    return Config(
        provider=os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER),
        model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        compare_model=os.environ.get("COMPARE_MODEL", DEFAULT_COMPARE_MODEL),
        effort=os.environ.get("LLM_EFFORT", DEFAULT_EFFORT),
    )


def has_api_key() -> bool:
    load_dotenv(ROOT / ".env")
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def complete_json(
    system: str, user: str, schema: dict, model: str | None = None, cfg: Config | None = None
) -> Completion:
    """One call constrained to `schema`. Never raises for a bad answer (refusal,
    truncation, invalid JSON): that comes back in Completion.error for escalation.
    Raises LLMError only for configuration or transport problems."""
    cfg = cfg or config()
    model = model or cfg.model
    if cfg.provider == "anthropic":
        return _anthropic(system, user, schema, model, cfg)
    if cfg.provider == "ollama":
        raise LLMError(
            "LLM_PROVIDER=ollama is planned for step 7c (Ollama `format` JSON-schema mode); "
            "use LLM_PROVIDER=anthropic for now."
        )
    raise LLMError(f"Unknown LLM_PROVIDER '{cfg.provider}' (expected anthropic or ollama).")


def _anthropic(system: str, user: str, schema: dict, model: str, cfg: Config) -> Completion:
    import anthropic

    if not has_api_key():
        raise LLMError("ANTHROPIC_API_KEY is not set (expected in .env).")
    client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=3)
    kwargs: dict = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {
            "format": {"type": "json_schema", "schema": schema},
            "effort": cfg.effort,
        },
    }
    if model in THINKING:
        kwargs["thinking"] = THINKING[model]
    try:
        response = client.messages.create(**kwargs)
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError(f"Anthropic API not reachable: {e}") from e

    text = "".join(b.text for b in response.content if b.type == "text")
    usage = response.usage
    error, data = None, None
    if response.stop_reason == "refusal":
        error = "refusal"
    elif response.stop_reason == "max_tokens":
        error = "max_tokens"
    else:
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                data, error = None, "invalid_json"
        except json.JSONDecodeError:
            error = "invalid_json"
    return Completion(
        data=data,
        raw_text=text,
        stop_reason=response.stop_reason,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        model=response.model,
        error=error,
    )
