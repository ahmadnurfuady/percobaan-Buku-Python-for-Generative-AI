# 6.4 Tool Calling (Function Calling) with Anthropic
import anthropic
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "mock-key"))

# 1. Define the tool schema
tools = [
    {
        "name": "get_model_info",
        "description": "Returns context window size and cost per 1K tokens for a given LLM.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "The model identifier, e.g. 'gpt-4o' or 'claude-sonnet-4-5'."
                }
            },
            "required": ["model_name"],
        },
    }
]

# 2. Local tool execution function
def get_model_info(model_name: str) -> dict:
    db = {
        "claude-sonnet-4-5": {"context_k": 200, "cost_input": 3.00, "cost_output": 15.00},
        "gpt-4o": {"context_k": 128, "cost_input": 2.50, "cost_output": 10.00},
        "gemini-1.5-pro": {"context_k": 1000, "cost_input": 1.25, "cost_output": 5.00},
    }
    return db.get(model_name, {"error": f"Unknown model: {model_name}"})

def run_tool_calling_flow():
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": "How large is the context window of claude-sonnet-4-5?"}],
    )
    if response.stop_reason == "tool_use":
        tool_block = next(b for b in response.content if b.type == "tool_use")
        tool_name = tool_block.name
        tool_input = tool_block.input
        tool_use_id = tool_block.id

        result = get_model_info(**tool_input)
        print(f"Tool called: {tool_name} with {tool_input} -> {result}")

        final = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            tools=tools,
            messages=[
                {"role": "user", "content": "How large is the context window of claude-sonnet-4-5?"},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json.dumps(result),
                        }
                    ],
                },
            ],
        )
        print("Final answer:", final.content[0].text)

if __name__ == "__main__":
    try:
        run_tool_calling_flow()
    except Exception as e:
        print(f"Demo run: {e}")
