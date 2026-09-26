# 7.7 Systematic Prompt Evaluation
from dataclasses import dataclass
import json

@dataclass
class EvalCase:
    input_text: str
    expected_keywords: list[str]
    must_be_json: bool = False

def evaluate_prompt_local(mock_responses: dict[str, str], cases: list[EvalCase]) -> dict:
    results = []
    for case in cases:
        text = mock_responses.get(case.input_text, "").strip()
        keyword_hit = any(kw.lower() in text.lower() for kw in case.expected_keywords)

        json_valid = True
        if case.must_be_json:
            try:
                json.loads(text)
            except Exception:
                json_valid = False

        passed = keyword_hit and json_valid
        results.append({
            "input": case.input_text[:60],
            "passed": passed,
            "response": text[:60],
        })

    pass_rate = sum(r["passed"] for r in results) / len(results) if results else 0.0
    return {"pass_rate": pass_rate, "results": results}

if __name__ == "__main__":
    cases = [
        EvalCase("Predict whether an email is spam.", ["CLASSIFICATION"]),
        EvalCase("Write a description.", ["GENERATION"]),
    ]
    mocks = {
        "Predict whether an email is spam.": "CLASSIFICATION",
        "Write a description.": "GENERATION",
    }
    report = evaluate_prompt_local(mocks, cases)
    print(f"Pass rate: {report['pass_rate']:.0%}")
