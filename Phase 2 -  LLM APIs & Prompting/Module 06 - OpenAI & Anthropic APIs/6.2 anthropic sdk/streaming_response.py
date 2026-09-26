# 6.2 The Anthropic SDK - Streaming Responses
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "mock-key"))

def stream_claude_response(prompt: str):
    """Stream tokens directly using stream context manager."""
    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()
        final = stream.get_final_message()
        print(f"\nTotal tokens: {final.usage.input_tokens + final.usage.output_tokens}")

if __name__ == "__main__":
    try:
        stream_claude_response("List 5 use cases for vector databases.")
    except Exception as e:
        print(f"Demo run: {e}")
