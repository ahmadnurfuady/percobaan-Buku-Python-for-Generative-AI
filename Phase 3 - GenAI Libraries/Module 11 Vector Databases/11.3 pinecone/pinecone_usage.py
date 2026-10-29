# 11.3 Pinecone (Halaman 128-130)
# pip install pinecone
from pinecone import Pinecone, ServerlessSpec
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Gunakan OpenRouter jika tersedia di .env atau OpenAI default
api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
base_url = "https://openrouter.ai/api/v1" if os.environ.get("OPENROUTER_API_KEY") else None
embed_model = "nvidia/llama-nemotron-embed-vl-1b-v2:free" if os.environ.get("OPENROUTER_API_KEY") else "text-embedding-3-small"

openai_client = OpenAI(api_key=api_key, base_url=base_url)

def embed(texts: list[str]) -> list[list[float]]:
    resp = openai_client.embeddings.create(model=embed_model, input=texts)
    return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]

if __name__ == "__main__":
    if "PINECONE_API_KEY" in os.environ:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        INDEX_NAME = "ai-knowledge-base"

        # Deteksi dimensi embedding secara dinamis sesuai model yang dipakai
        sample_dim = len(embed(["test"])[0])

        # Cek apakah index sudah ada dengan dimensi yang berbeda
        existing_indexes = pc.list_indexes().names()
        if INDEX_NAME in existing_indexes:
            index_desc = pc.describe_index(INDEX_NAME)
            if index_desc.dimension != sample_dim:
                print(f"Dimensi index lama ({index_desc.dimension}) != dimensi model ({sample_dim}). Menghapus index lama...")
                pc.delete_index(INDEX_NAME)
                while INDEX_NAME in pc.list_indexes().names():
                    time.sleep(2)

        # Buat index jika belum ada
        if INDEX_NAME not in pc.list_indexes().names():
            print(f"Membuat index '{INDEX_NAME}' dengan dimensi {sample_dim}...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=sample_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(INDEX_NAME).status['ready']:
                time.sleep(1)

        index = pc.Index(INDEX_NAME)

        documents = [
            {"id": "d1", "text": "RAG connects LLMs to external knowledge bases."},
            {"id": "d2", "text": "Fine-tuning updates model weights on task-specific data."},
            {"id": "d3", "text": "ChromaDB is an open-source local vector database."},
        ]

        vectors = [
            {
                "id": doc["id"],
                "values": embed([doc["text"]])[0],
                "metadata": {"text": doc["text"], "source": "guide"},
            }
            for doc in documents
        ]
        index.upsert(vectors=vectors)
        print("Index stats:", index.describe_index_stats())

        # Query
        q_vector = embed(["how to add external knowledge to an LLM?"])[0]
        results = index.query(
            vector=q_vector,
            top_k=3,
            include_metadata=True,
        )
        for match in results["matches"]:
            print(f"[{match['score']:.4f}] {match['metadata']['text']}")
    else:
        print("PINECONE_API_KEY not configured. Set it in .env to run live Pinecone calls.")
