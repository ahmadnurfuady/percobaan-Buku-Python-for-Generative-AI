# 7.5 Approach 1: Prompt-Enforced JSON Output
import json

SYSTEM = """You are a data extractor. Extract information and return ONLY a JSON object.
No markdown, no explanation, no code fences. Raw JSON only.
Schema:
{
  "company": string,
  "founded": integer or null,
  "products": [string],
  "headquarters": string or null,
  "is_public": boolean
}"""

def clean_json_response(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)

if __name__ == "__main__":
    raw = '```json\n{"company": "Anthropic", "founded": 2021, "products": ["Claude"], "headquarters": "San Francisco", "is_public": false}\n```'
    parsed = clean_json_response(raw)
    print("Parsed JSON dictionary:", parsed)
