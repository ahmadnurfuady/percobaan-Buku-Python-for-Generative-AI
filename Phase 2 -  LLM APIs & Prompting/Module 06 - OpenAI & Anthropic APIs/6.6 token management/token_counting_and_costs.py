# 6.6 Token Management and Cost Estimation
PRICING = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},  # per 1M tokens
    "claude-opus-4-5": {"input": 15.00, "output": 75.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

CONTEXT_LIMITS = {
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gemini-1.5-pro": 1_000_000,
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost in USD based on per-1M-token pricing."""
    if model not in PRICING:
        raise ValueError(f"Unknown model: {model}")
    p = PRICING[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

def fits_in_context(model: str, token_count: int, reserve_for_output: int = 2048) -> bool:
    limit = CONTEXT_LIMITS.get(model, 128_000)
    return token_count + reserve_for_output <= limit

if __name__ == "__main__":
    cost = estimate_cost("claude-sonnet-4-5", input_tokens=500, output_tokens=300)
    print(f"Estimated cost: ${cost:.6f}")
    print("Fits in gpt-4o (50k tokens):", fits_in_context("gpt-4o", 50_000))
