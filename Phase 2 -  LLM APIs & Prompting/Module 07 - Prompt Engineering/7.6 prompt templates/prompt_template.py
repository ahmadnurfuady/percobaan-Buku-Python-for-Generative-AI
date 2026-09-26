# 7.6 Reusable PromptTemplate Dataclass with Variable Auto-detection
from dataclasses import dataclass, field
from string import Formatter
from typing import Any

@dataclass
class PromptTemplate:
    """A reusable, versioned prompt template with placeholder validation."""
    name: str
    system: str
    user: str
    version: str = "1.0"
    required_vars: list[str] = field(default_factory=list)

    def __post_init__(self):
        formatter = Formatter()
        combined = self.system + self.user
        self.required_vars = [fname for _, fname, _, _ in formatter.parse(combined) if fname is not None]

    def render(self, **kwargs: Any) -> tuple[str, str]:
        missing = set(self.required_vars) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return self.system.format(**kwargs), self.user.format(**kwargs)

# QA Template Constant
QA_TEMPLATE = PromptTemplate(
    name="question_answering",
    version="1.2",
    system="You are a {domain} expert. Answer questions accurately and concisely.",
    user="Question: {question}\n\nContext:\n{context}",
)

if __name__ == "__main__":
    sys, usr = QA_TEMPLATE.render(
        domain="machine learning",
        question="What is backpropagation?",
        context="Backpropagation computes gradients using the chain rule."
    )
    print("Rendered System:", sys)
    print("Rendered User:", usr)
