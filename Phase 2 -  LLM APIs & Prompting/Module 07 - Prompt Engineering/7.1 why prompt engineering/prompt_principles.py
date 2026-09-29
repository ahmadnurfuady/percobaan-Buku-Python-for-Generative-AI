# 7.1 Why Prompt Engineering Matters
"""
The core principles are:
- Be explicit - state the format, tone, length, and constraints you want
- Give examples - a few well-chosen examples beat long instructions
- Think step by step - ask the model to reason before concluding
- Separate concerns - put instructions in the system prompt, data in the user message
"""

def print_principles():
    principles = [
        "Be explicit - state the format, tone, length, and constraints you want",
        "Give examples - a few well-chosen examples beat long instructions",
        "Think step by step - ask the model to reason before concluding",
        "Separate concerns - put instructions in the system prompt, data in the user message",
    ]
    print("Core Principles of Prompt Engineering:")
    for p in principles:
        print(f" - {p}")

if __name__ == "__main__":
    print_principles()
