# 7.2 System Prompts - Weak vs Strong
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

WEAK_SYSTEM = "You are an AI assistant."

STRONG_SYSTEM = """You are a senior Python engineer reviewing code for a production AI pipeline.
Your job:
- Identify bugs, security issues, and performance problems
- Suggest concrete improvements with code examples
- Explain WHY each issue matters

Rules:
- Be direct. Do not pad with compliments.
- If code is correct, say so briefly and move on.
- Always include the corrected code when suggesting a fix.

Format:
Return your review as a numbered list. Each item: Issue -> Impact -> Fix."""

snippet = """
def get_user(user_id):
    key = os.getenv('DB_KEY')
    result = requests.get(f'http://db/{user_id}?key={key}')
    return result.json()
"""

if __name__ == "__main__":
    print("=== Strong System Prompt Definition ===")
    print(STRONG_SYSTEM)
