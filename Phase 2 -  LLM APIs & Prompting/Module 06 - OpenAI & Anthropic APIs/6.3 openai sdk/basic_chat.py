# 6.3 The OpenAI SDK - Chat Completions
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "mock-key"))

def call_openai_basic(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": "You are a concise technical assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    print(f"Tokens used: {response.usage.total_tokens}")
    return content

if __name__ == "__main__":
    try:
        ans = call_openai_basic("What is the difference between RAG and fine-tuning?")
        print(ans)
    except Exception as e:
        print(f"Demo run: {e}")
