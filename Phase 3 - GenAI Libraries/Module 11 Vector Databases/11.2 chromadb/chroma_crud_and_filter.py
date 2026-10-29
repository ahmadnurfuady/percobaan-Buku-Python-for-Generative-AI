# 11.2 ChromaDB (Halaman 125-128)
import chromadb
from chromadb.config import Settings
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Gunakan OpenRouter jika tersedia di .env (sebagaimana modul sebelumnya) atau OpenAI default
api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
base_url = "https://openrouter.ai/api/v1" if os.environ.get("OPENROUTER_API_KEY") else None
embed_model = "nvidia/llama-nemotron-embed-vl-1b-v2:free" if os.environ.get("OPENROUTER_API_KEY") else "text-embedding-3-small"

openai_client = OpenAI(api_key=api_key, base_url=base_url)

def embed(texts: list[str]) -> list[list[float]]:
    resp = openai_client.embeddings.create(model=embed_model, input=texts)
    return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]

# Persistent client (saves to disk)
client = chromadb.PersistentClient(path="./chroma_db")

# Get or create a collection
collection = client.get_or_create_collection(
    name="ai_knowledge_base",
    metadata={"hnsw:space": "cosine"}, # cosine similarity
)

# Add documents with pre-computed embeddings
documents = [
    "RAG connects LLMs to external knowledge bases at inference time.",
    "Fine-tuning adapts model weights using task-specific training data.",
    "Prompt engineering guides LLM outputs without any model training.",
    "Vector databases index embeddings for fast similarity search.",
    "Agents use LLMs to plan and execute multi-step tasks with tools.",
]
ids = [f"doc_{i}" for i in range(len(documents))]
metadatas = [{"category": "technique", "index": i} for i in range(len(documents))]

if __name__ == "__main__":
    embeddings = embed(documents)
    # Gunakan upsert agar aman jika dijalankan berulang kali
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Collection size: {collection.count()}")

    # Query
    query = "how do I add knowledge to an LLM without retraining?"
    q_embed = embed([query])[0]
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=3,
        include=["documents", "distances", "metadatas"],
    )
    print("\nTop results:")
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        print(f" [{dist:.4f}] {doc}")
        print(f"  category: {meta['category']}")

    # Metadata filtering
    results_filtered = collection.query(
        query_embeddings=[q_embed],
        n_results=3,
        where={"category": {"$eq": "technique"}}, # pre-filter
        include=["documents", "distances"],
    )

    # Update a document
    collection.update(
        ids=["doc_0"],
        documents=["RAG retrieves external documents and injects them into the LLM context."],
        embeddings=embed(["RAG retrieves external documents and injects them into the LLM context."]),
    )

    # Delete a document
    try:
        collection.delete(ids=["doc_4"])
    except Exception:
        pass
    print(f"After delete, count: {collection.count()}")
