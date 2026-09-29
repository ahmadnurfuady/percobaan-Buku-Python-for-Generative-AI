# 7.5 Structured Output (Approach 2: OpenAI JSON mode)
from openai import OpenAI
import os, json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.xkiro.com/v1")

response = client.chat.completions.create(
    model="qwen/qwen3.8-max:free",
    response_format={"type": "json_object"},  # enforces valid JSON
    messages=[
        {
            "role": "system",
            "content": """Extract entities. Return JSON with this schema:
{"people": [string], "organizations": [string], "locations": [string]}"""
        },
        {
            "role": "user",
            "content": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla."
        }
    ]
)

result = json.loads(response.choices[0].message.content)
print(result)
# {'people': ['Elon Musk'], 'organizations': ['SpaceX', 'Tesla'], 'locations': ['Hawthorne, California']}
