# 11.4 Weaviate (Halaman 130-133)
# pip install weaviate-client
"""
Weaviate is an open-source vector database with built-in hybrid search (vector +
BM25 keyword), a GraphQL API, and a Python client.

Start locally via Docker:
docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
"""
import os
import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter, MetadataQuery
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup Embedding Client (OpenRouter atau OpenAI default)
api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
base_url = "https://openrouter.ai/api/v1" if os.environ.get("OPENROUTER_API_KEY") else None
embed_model = "nvidia/llama-nemotron-embed-vl-1b-v2:free" if os.environ.get("OPENROUTER_API_KEY") else "text-embedding-3-small"

openai_client = OpenAI(api_key=api_key, base_url=base_url)

def embed(texts: list[str]) -> list[list[float]]:
    resp = openai_client.embeddings.create(model=embed_model, input=texts)
    return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]

def get_weaviate_client():
    # Cek apakah konfigurasi Cloud tersedia di .env
    weaviate_url = os.environ.get("WEAVIATE_URL")
    weaviate_key = os.environ.get("WEAVIATE_API_KEY")

    if weaviate_url and weaviate_key:
        print(f"Menghubungkan ke Weaviate Cloud: {weaviate_url}...")
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_key),
        )
    else:
        print("Menghubungkan ke Weaviate Lokal (Docker localhost:8080)...")
        return weaviate.connect_to_local()

def main():
    client = get_weaviate_client()

    try:
        if not client.is_ready():
            print("Gagal: Weaviate belum siap menerima request.")
            return

        print("Status Weaviate: Ready!\n")

        collection_name = "AIDocument"

        # 1. Reset collection jika sudah ada agar bisa dijalankan berulang
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        # 2. Create collection (schema) dengan konfigurasi custom embedding
        collection = client.collections.create(
            name=collection_name,
            properties=[
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT),
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # custom vectors
        )
        print(f"Collection '{collection_name}' berhasil dibuat.")

        # 3. Data Dokumen
        documents = [
            {"text": "RAG connects LLMs to external knowledge bases at inference time.", "source": "guide", "category": "technique"},
            {"text": "Prompt engineering guides LLM outputs without any model training.", "source": "guide", "category": "technique"},
            {"text": "Fine-tuning adapts model weights using task-specific training data.", "source": "guide", "category": "technique"},
            {"text": "ChromaDB and Weaviate are open-source and scalable vector databases.", "source": "docs", "category": "database"},
            {"text": "Pinecone is a fully managed cloud vector database service.", "source": "docs", "category": "database"},
        ]

        print("Melakukan embedding dan batch insert...")
        with collection.batch.dynamic() as batch:
            for doc in documents:
                vec = embed([doc["text"]])[0]
                batch.add_object(
                    properties={
                        "text": doc["text"],
                        "source": doc["source"],
                        "category": doc["category"],
                    },
                    vector=vec,
                )

        if len(collection.batch.failed_objects) > 0:
            print(f"Warning: {len(collection.batch.failed_objects)} objek gagal di-insert.")
        else:
            print(f"Berhasil meng-insert {len(documents)} objek ke Weaviate.\n")

        query_text = "how to add external knowledge to an LLM?"
        q_vector = embed([query_text])[0]

        # 4. Pure Vector Search (near_vector)
        print(f"--- 1. Vector Search: '{query_text}' ---")
        vector_results = collection.query.near_vector(
            near_vector=q_vector,
            limit=2,
            return_properties=["text", "category", "source"],
            return_metadata=MetadataQuery(distance=True),
        )
        for obj in vector_results.objects:
            print(f" [Distance: {obj.metadata.distance:.4f}] {obj.properties['text']}")

        # 5. Hybrid Search (BM25 keyword + vector)
        # alpha = 0.5 (seimbang antara kecocokan kata kunci BM25 dan semantic vector)
        print(f"\n--- 2. Hybrid Search (alpha=0.5): 'external knowledge' ---")
        hybrid_results = collection.query.hybrid(
            query="external knowledge",
            vector=q_vector,
            alpha=0.5,
            limit=2,
            return_properties=["text", "category"],
            return_metadata=MetadataQuery(score=True),
        )
        for obj in hybrid_results.objects:
            print(f" [Score: {obj.metadata.score:.4f}] {obj.properties['text']}")

        # 6. Metadata Filter Search
        print(f"\n--- 3. Filtered Vector Search (category == 'database') ---")
        filtered_results = collection.query.near_vector(
            near_vector=q_vector,
            limit=2,
            filters=Filter.by_property("category").equal("database"),
            return_properties=["text", "category"],
            return_metadata=MetadataQuery(distance=True),
        )
        for obj in filtered_results.objects:
            print(f" [Distance: {obj.metadata.distance:.4f}] {obj.properties['text']} ({obj.properties['category']})")

    finally:
        client.close()
        print("\nKoneksi Weaviate ditutup dengan aman.")

if __name__ == "__main__":
    main()
