# 6.2 The Anthropic SDK - Basic Message Call
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "mock-key"))

def call_anthropic_basic(prompt: str) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    # Response text resides in the first content block
    return message.content[0].text

if __name__ == "__main__":
    try:
        ans = call_anthropic_basic("What is retrieval-augmented generation?")
        print(ans)
    except Exception as e:
        print(f"Demo run (set ANTHROPIC_API_KEY to execute live call): {e}")
