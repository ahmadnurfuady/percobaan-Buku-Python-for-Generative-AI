# 10.1 What LlamaIndex Is (Halaman 115)
"""
LlamaIndex specialises in data ingestion and indexing for LLM applications.
Core concepts:
- Node: a chunk of text with metadata, the atomic unit
- Index: a data structure built from nodes (VectorStoreIndex, SummaryIndex, KeywordIndex)
- Query Engine: takes a question, retrieves nodes, synthesises an answer
- Retriever: the retrieval component, usable independently from the query engine
"""

def print_llamaindex_concepts():
    print("=== LlamaIndex Abstractions ===")
    print("Node -> Index -> Retriever -> ResponseSynthesizer -> QueryEngine")

if __name__ == "__main__":
    print_llamaindex_concepts()
