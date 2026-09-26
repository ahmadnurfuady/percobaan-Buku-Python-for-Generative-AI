# 6.2 The Anthropic SDK - Multi-turn Conversation Loop
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "mock-key"))

def chat_step(history: list[dict], user_input: str, system: str = "You are a helpful Python tutor.") -> str:
    """The Anthropic API is stateless; append turns to maintain history."""
    history.append({"role": "user", "content": user_input})
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=system,
        messages=history,
    )
    assistant_text = response.content[0].text
    history.append({"role": "assistant", "content": assistant_text})
    return assistant_text

if __name__ == "__main__":
    conversation = []
    print("Multi-turn chat session initialized.")
