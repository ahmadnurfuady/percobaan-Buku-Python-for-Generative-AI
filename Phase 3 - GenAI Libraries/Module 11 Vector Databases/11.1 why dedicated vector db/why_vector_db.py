# 11.1 Why a Dedicated Vector Database (Halaman 125)
"""
Production systems need:
- Persistence: data survives restarts
- Scale: millions of vectors without loading everything into RAM
- Filtering: combine vector search with metadata conditions
- Updates: add, delete, and update documents efficiently
- Multi-tenancy: isolate collections per user or tenant
"""
def print_db_comparison():
    print("ChromaDB: simple, embedded/local, open-source")
    print("Pinecone: fully managed, serverless, production-grade")
    print("Weaviate: open-source, built-in hybrid BM25 + vector search")

if __name__ == "__main__":
    print_db_comparison()
