# 9.1 What LangChain Is
"""
LangChain is a framework for composing LLM-powered applications from reusable building blocks.
Its core idea is the chain - a sequence of steps where the output of one step feeds into the next.
The modern way to build chains is LCEL (LangChain Expression Language), a pipe-based syntax
that makes composition explicit and readable.

When to use LangChain: RAG pipelines, conversational applications, document processing workflows,
and any situation where you need to chain multiple LLM calls with structured prompts.
"""

def print_langchain_overview():
    print("=== LangChain Core Concepts ===")
    print("1. LCEL: Expression Language using the '|' pipe operator")
    print("2. Unified Model Interface: ChatOpenAI and ChatAnthropic")
    print("3. Modular components: Prompts, Output Parsers, Document Loaders, Memory, Retrievers")

if __name__ == "__main__":
    print_langchain_overview()
