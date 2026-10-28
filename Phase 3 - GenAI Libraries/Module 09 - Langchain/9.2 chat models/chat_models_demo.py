# 9.2 Chat Models (Halaman 101-103)
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI model
llm_oai = ChatOpenAI(
    model="qwen/qwen3.8-max:free",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.xkiro.com/v1",
)

# Anthropic model - same interface
llm_ant = ChatAnthropic(
    model="qwen/qwen3.8-max:free",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.xkiro.com",
)

# Both accept the same message types
messages = [
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is LangChain Expression Language?"),
]

if __name__ == "__main__":
    # Invoke (blocking)
    print("=== Invoke Anthropic ===")
    response = llm_ant.invoke(messages)
    print(response.content)

    # Stream
    print("\n=== Stream Anthropic ===")
    for chunk in llm_ant.stream(messages):
        print(chunk.content, end="", flush=True)
    print()

    # Batch - run multiple inputs in parallel
    print("\n=== Batch Anthropic ===")
    responses = llm_ant.batch([
        [HumanMessage(content="What is RAG?")],
        [HumanMessage(content="What is a vector database?")],
    ])
    for r in responses:
        print(r.content[:80])
