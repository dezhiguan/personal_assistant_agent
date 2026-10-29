"""Versioned, tier-aware Qwen list-price estimates, separate from actual billing."""

import json
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

from personal_assistant_agent.core.config import Settings

PRICING = json.loads(Path(__file__).with_name("qwen_pricing.json").read_text())
MILLION = Decimal(1000000)


def estimate_cost(usage: dict, settings: Settings) -> dict | None:
    if (
        settings.model_name != PRICING["model"]
        or urlparse(settings.base_url).hostname != PRICING["base_url_host"]
    ):
        return None
    inputs, outputs = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    cached = int(usage.get("input_token_details", {}).get("cache_read", 0))
    if min(inputs, outputs, cached) < 0 or cached > inputs:
        raise ValueError("Invalid token usage")
    tier = next((t for t in PRICING["tiers"] if inputs <= t["max_input_tokens"]), None)
    if tier is None:
        return None
    input_cny = (
        (inputs - cached) * Decimal(tier["input_per_million"])
        + cached * Decimal(tier["cache_read_per_million"])
    ) / MILLION
    output_cny = outputs * Decimal(tier["output_per_million"]) / MILLION
    fx = Decimal(PRICING["fx"]["CNY_per_USD"])
    return {
        "input_cost_cny": float(input_cny),
        "output_cost_cny": float(output_cny),
        "total_cost_cny": float(input_cny + output_cny),
        "input_cost": float(input_cny / fx),
        "output_cost": float(output_cny / fx),
        "total_cost": float((input_cny + output_cny) / fx),
        "cost_is_estimate": True,
        "pricing_verified_on": PRICING["verified_on"],
        "pricing_source": PRICING["source"],
        "fx_date": PRICING["fx"]["date"],
        "cny_per_usd": str(fx),
    }


def estimate_calls(usages: Iterable[dict], settings: Settings) -> dict | None:
    """Price each call before summing; tiers apply per call, not per conversation."""
    estimates = [estimate_cost(usage, settings) for usage in usages]
    if not estimates or any(value is None for value in estimates):
        return None
    result = dict(estimates[0])
    for key in ("input_cost", "output_cost", "total_cost"):
        for field in (key, key + "_cny"):
            result[field] = float(sum(Decimal(str(value[field])) for value in estimates))
    result["model_calls"] = len(estimates)
    return result
