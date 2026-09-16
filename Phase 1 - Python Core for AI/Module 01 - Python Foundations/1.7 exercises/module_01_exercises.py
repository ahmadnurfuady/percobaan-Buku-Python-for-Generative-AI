# 1.7 Module 01 Exercises
import functools
import time

# 1. Token cost calculation
def token_cost(tokens: int, model: str) -> float:
    """Calculate estimated cost using a dict of costs per 1K tokens."""
    costs = {
        "gpt-4o": 0.0025,
        "gpt-4o-mini": 0.00015,
        "claude-3-5-sonnet": 0.003,
        "gemini-1.5-pro": 0.00125,
        "gemini-1.5-flash": 0.000075,
    }
    if model not in costs:
        raise ValueError(f"Unknown model: '{model}'. Available models: {list(costs.keys())}")
    return (tokens / 1000) * costs[model]


# 2. Retry decorator
def retry(max_attempts: int = 3, delay: float = 0.5):
    """Decorator to retry a function up to `max_attempts` times on any exception."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"[Attempt {attempt}/{max_attempts}] {func.__name__} failed with error: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            if last_exception is not None:
                raise last_exception
            raise RuntimeError(f"Function {func.__name__} was not executed.")
        return wrapper
    return decorator


# 3. Temperature label classifier
def temperature_label(t: float) -> str:
    """Map temperature range to label: 0.0-0.3 precise, 0.3-0.7 balanced, 0.7-1.0 creative."""
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Temperature {t} is outside the valid range (0.0 - 1.0)")
    if t <= 0.3:
        return "precise"
    elif t <= 0.7:
        return "balanced"
    else:
        return "creative"


# 4. Parse pricing string using string methods only
def parse_pricing_str(pricing_str: str) -> tuple[int, float]:
    """Extract token count (int) and cost (float) using only string methods."""
    # Example format: "128000 tokens, 0.005 USD per 1K"
    token_part, cost_part = pricing_str.split(",")
    tokens = int(token_part.strip().split()[0])
    cost = float(cost_part.strip().split()[0])
    return tokens, cost


if __name__ == "__main__":
    print("=== Exercise 1: Token Cost ===")
    print("Cost for 2500 tokens (gpt-4o):", token_cost(2500, "gpt-4o"))

    print("\n=== Exercise 2: Retry Decorator ===")
    # Test function that fails the first 2 times and succeeds on the 3rd attempt
    attempts = 0

    @retry(max_attempts=3, delay=0.1)
    def flaky_api_call():
        global attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError(f"Temporary network error on attempt #{attempts}")
        return f"Success on attempt #{attempts}!"

    result = flaky_api_call()
    print("Result:", result)

    print("\n=== Exercise 3: Temperature Labels ===")
    for temp in [0.1, 0.5, 0.9]:
        print(f"Temperature {temp} -> {temperature_label(temp)}")

    print("\n=== Exercise 4: Parse String ===")
    raw_str = "128000 tokens, 0.005 USD per 1K"
    tokens, cost = parse_pricing_str(raw_str)
    print(f"Input: '{raw_str}'")
    print(f"Extracted -> Tokens: {tokens} (type: {type(tokens).__name__}), Cost: {cost} (type: {type(cost).__name__})")

