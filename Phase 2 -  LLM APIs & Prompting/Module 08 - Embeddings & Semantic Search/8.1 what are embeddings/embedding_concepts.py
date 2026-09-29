# 8.1 What Are Embeddings?
"""
An embedding is a fixed-length vector of floating-point numbers that represents a piece of text.
Semantically similar texts produce vectors that are geometrically close.
This property powers semantic search, clustering, deduplication, and the retrieval step in RAG.

"What is RAG?"                      -> [0.021, -0.143, 0.892, ...]  (1536 numbers)
"Retrieval Augmented Generation"    -> [0.019, -0.139, 0.881, ...]  (geometrically close)
"Who won the cricket match?"        -> [-0.312, 0.401, 0.203, ...]  (geometrically far)

Key properties:
- Dimensionality: 768 to 3072 dimensions depending on the model
- Magnitude: vectors are usually L2-normalised (length = 1)
- Comparison: cosine similarity is the standard metric (dot product of normalised vectors)
"""

def print_embedding_concepts():
    print("=== Key Properties of Text Embeddings ===")
    print("1. Dimensionality: 768 to 3072 dimensions depending on model")
    print("2. Magnitude: Typically L2-normalised (length = 1.0)")
    print("3. Comparison: Cosine similarity (dot product of unit vectors)")

if __name__ == "__main__":
    print_embedding_concepts()
