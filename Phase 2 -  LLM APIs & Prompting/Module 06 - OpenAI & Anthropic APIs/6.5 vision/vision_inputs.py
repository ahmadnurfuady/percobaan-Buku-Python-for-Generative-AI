# 6.5 Vision - Images as Input (URL & Base64)
import anthropic
import os
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "mock-key"))

# Option A: Image via URL
def describe_image_url(url: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": url}},
                {"type": "text", "text": "Describe what you see in this image."}
            ]
        }]
    )
    return response.content[0].text

# Option B: Image via Base64 for local files
def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.standard_b64encode(data).decode()
    ext = Path(path).suffix.lstrip(".").lower()
    media_type = f"image/{ext}"

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": "What is in this image?"}
            ]
        }]
    )
    return response.content[0].text

if __name__ == "__main__":
    print("Vision helpers ready for URL and base64 files.")
