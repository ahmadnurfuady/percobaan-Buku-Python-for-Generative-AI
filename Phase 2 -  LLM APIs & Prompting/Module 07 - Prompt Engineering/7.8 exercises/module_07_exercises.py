# 7.8 Module 07 Exercises
import json
from dataclasses import dataclass, asdict

# 1. System prompt variations for code review
BASIC_PROMPT = "Review this code for bugs."
INTERMEDIATE_PROMPT = "You are a code reviewer. Find bugs and suggest fixes with code."
EXPERT_PROMPT = """You are a Principal Software Engineer conducting a strict code review.
Evaluate code for correctness, security, performance, and maintainability.
Format as: Issue -> Impact -> Fix."""

# 2. Prompt Library class
class PromptLibrary:
    def __init__(self):
        self._templates: dict[str, dict] = {}

    def register(self, template) -> None:
        self._templates[template.name] = asdict(template)

    def save_to_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._templates, f, indent=2)

    def load_from_json(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            self._templates = json.load(f)

# 3. Automatic JSON repair helper
def safe_json_parse(text: str) -> dict:
    """First try json.loads, then strip markdown fences."""
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)

if __name__ == "__main__":
    fenced_example = "```json\n{\"status\": \"success\", \"items\": [1, 2, 3]}\n```"
    print("Parsed JSON:", safe_json_parse(fenced_example))
