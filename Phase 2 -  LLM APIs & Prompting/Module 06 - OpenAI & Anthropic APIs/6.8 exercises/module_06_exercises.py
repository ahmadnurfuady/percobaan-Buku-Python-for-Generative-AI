# 6.8 Module 06 Exercises
import time
import asyncio

# Exercise 1: Retry on rate limit helper
def retry_on_rate_limit(call_fn, max_retries: int = 5, backoff_base: float = 2.0):
    for attempt in range(1, max_retries + 1):
        try:
            return call_fn()
        except Exception as e:
            err_msg = str(e).lower()
            if ("rate" in err_msg or "429" in err_msg) and attempt < max_retries:
                wait = backoff_base ** attempt
                print(f"Rate limited (attempt {attempt}). Retrying in {wait}s...")
                time.sleep(wait * 0.01)
            else:
                raise

# Exercise 2: TokenBudgetManager class
class BudgetExceeded(Exception):
    pass

class TokenBudgetManager:
    def __init__(self, max_budget_tokens: int):
        self.max_budget = max_budget_tokens
        self.used_tokens = 0

    def add_usage(self, tokens: int) -> None:
        if self.used_tokens + tokens > self.max_budget:
            raise BudgetExceeded(f"Budget of {self.max_budget} exceeded! Tried adding {tokens} to {self.used_tokens}.")
        self.used_tokens += tokens

    @property
    def remaining(self) -> int:
        return max(0, self.max_budget - self.used_tokens)

if __name__ == "__main__":
    mgr = TokenBudgetManager(1000)
    mgr.add_usage(450)
    print("Remaining tokens:", mgr.remaining)
