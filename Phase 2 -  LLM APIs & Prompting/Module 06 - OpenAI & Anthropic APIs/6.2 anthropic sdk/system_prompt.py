# 6.2 The Anthropic SDK - System Prompt
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "mock-key"))

def call_with_system_prompt(system_instruction: str, user_question: str) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system=system_instruction,
        messages=[{"role": "user", "content": user_question}],
    )
    return message.content[0].text

if __name__ == "__main__":
    sys_prompt = "You are a concise technical writer. Answer in plain English, no jargon."
    user_msg = "Explain what a vector database does."
    try:
        print(call_with_system_prompt(sys_prompt, user_msg))
    except Exception as e:
        print(f"Demo run: {e}")
