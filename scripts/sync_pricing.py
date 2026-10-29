"""Register versioned Qwen list-price estimates in the workspace model price map."""

from decimal import Decimal

import httpx
import truststore

from personal_assistant_agent.agents.model import PRICING_PROVIDER
from personal_assistant_agent.core.config import get_settings
from personal_assistant_agent.observability.costs import MILLION, PRICING


def main():
    truststore.inject_into_ssl()
    settings = get_settings()
    if not settings.langsmith_api_key:
        raise SystemExit("LANGSMITH_API_KEY is required")
    fx = Decimal(PRICING["fx"]["CNY_per_USD"])

    # API prices are USD per TOKEN, whereas the vendor snapshot is CNY per million.
    def usd(rate):
        return format(Decimal(rate) / MILLION / fx, "f")

    with httpx.Client(
        base_url=settings.langsmith_endpoint,
        headers={"x-api-key": settings.langsmith_api_key},
        timeout=30,
    ) as client:
        response = client.get("/model-price-map")
        response.raise_for_status()
        existing = {entry["name"]: entry for entry in response.json()}
        minimum = 0
        for index, tier in enumerate(PRICING["tiers"]):
            name = f"assistant-qwen-plus-beijing-estimate-{PRICING['verified_on']}-tier{index}"
            payload = {
                "name": name,
                "start_time": PRICING["verified_on"] + "T00:00:00Z",
                "match_pattern": "^qwen-plus$",
                "provider": PRICING_PROVIDER,
                "prompt_cost": usd(tier["input_per_million"]),
                "completion_cost": usd(tier["output_per_million"]),
                "prompt_cost_details": {"cache_read": usd(tier["cache_read_per_million"])},
                "prompt_tokens_above": minimum,
            }
            previous = existing.get(name)
            if previous:
                response = client.put("/model-price-map/" + previous["id"], json=payload)
            else:
                response = client.post("/model-price-map", json=payload)
            response.raise_for_status()
            print("Pricing estimate verified:", name)
            minimum = tier["max_input_tokens"]
        print("List-price estimates, not Alibaba invoices; USD uses the stored FX snapshot.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Pricing synchronization failed: {type(error).__name__}") from None
