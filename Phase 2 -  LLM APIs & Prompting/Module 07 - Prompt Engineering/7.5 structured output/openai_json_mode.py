# 7.5 Approach 2: OpenAI JSON Mode
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "mock-key"))

def extract_entities_json(user_text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": 'Extract entities. Return JSON with this schema: {"people": [string], "organizations": [string], "locations": [string]}',
            },
            {"role": "user", "content": user_text},
        ],
    )
    return json.loads(response.choices[0].message.content)

if __name__ == "__main__":
    print("JSON Mode query helper ready.")
