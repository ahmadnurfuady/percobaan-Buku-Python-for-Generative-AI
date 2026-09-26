# 7.8 Module 07 Exercises
import json

# Exercise 3: safe_json_parse
def safe_json_parse(text: str) -> dict:
    """Attempt json.loads, then strip markdown code fences."""
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)

if __name__ == "__main__":
    fenced = "```json\n{\"status\": \"success\", \"code\": 200}\n```"
    print("Parsed fenced JSON:", safe_json_parse(fenced))
