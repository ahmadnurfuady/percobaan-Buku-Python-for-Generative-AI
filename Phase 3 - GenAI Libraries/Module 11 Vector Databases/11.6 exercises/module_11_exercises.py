# 11.6 Module 11 Exercises (Halaman 133-134)
import time
import json
import random
import chromadb

# ----------------------------------------------------------------------
# Exercise 1: VectorDBBenchmark
# ----------------------------------------------------------------------
class VectorDBBenchmark:
    """Benchmark performa ChromaDB untuk insert dan query dalam jumlah tertentu."""
    def __init__(self, client=None, dimension: int = 16):
        self.client = client or chromadb.Client()
        self.dimension = dimension

    def _random_vector(self) -> list[float]:
        return [random.random() for _ in range(self.dimension)]

    def run_benchmark(self, num_docs: int = 1000, num_queries: int = 100) -> dict:
        col = self.client.get_or_create_collection(f"bench_{int(time.time()*1000)}")
        
        # 1. Insert documents
        ids = [f"id_{i}" for i in range(num_docs)]
        docs = [f"Benchmark document content {i}" for i in range(num_docs)]
        embeddings = [self._random_vector() for _ in range(num_docs)]
        
        t0 = time.perf_counter()
        col.add(ids=ids, documents=docs, embeddings=embeddings)
        insert_time = time.perf_counter() - t0

        # 2. Query latency tests
        query_vectors = [self._random_vector() for _ in range(num_queries)]
        latencies = []
        for q_vec in query_vectors:
            q_t0 = time.perf_counter()
            col.query(query_embeddings=[q_vec], n_results=5)
            latencies.append(time.perf_counter() - q_t0)

        latencies.sort()
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = latencies[int(len(latencies) * 0.95)]
        queries_per_sec = len(latencies) / sum(latencies)

        return {
            "num_docs": num_docs,
            "insert_time_sec": round(insert_time, 4),
            "num_queries": num_queries,
            "avg_latency_ms": round(avg_latency * 1000, 2),
            "p95_latency_ms": round(p95_latency * 1000, 2),
            "throughput_qps": round(queries_per_sec, 2),
        }

# ----------------------------------------------------------------------
# Exercise 2: MultiCollectionSearch
# ----------------------------------------------------------------------
class MultiCollectionSearch:
    """Mencari ke beberapa collection ChromaDB sekaligus dan menggabungkan hasil berdasarkan skor jarak."""
    def __init__(self, client):
        self.client = client

    def search(self, collection_names: list[str], query_embedding: list[float], top_k: int = 5) -> list[dict]:
        all_results = []
        for name in collection_names:
            try:
                col = self.client.get_collection(name)
                res = col.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "distances", "metadatas"],
                )
                if res["documents"] and res["documents"][0]:
                    for doc, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0] if res["metadatas"] else [{}] * len(res["documents"][0])):
                        all_results.append({
                            "collection": name,
                            "document": doc,
                            "distance": dist,
                            "metadata": meta,
                        })
            except Exception:
                continue

        # Urutkan berdasarkan distance terendah (paling mirip)
        all_results.sort(key=lambda x: x["distance"])
        return all_results[:top_k]

# ----------------------------------------------------------------------
# Exercise 3: TenantVectorStore
# ----------------------------------------------------------------------
class TenantVectorStore:
    """Store yang mengisolasi data per tenant menggunakan collection terpisah."""
    def __init__(self, client):
        self.client = client

    def _get_collection(self, tenant_id: str):
        safe_tenant = tenant_id.replace("-", "_")
        return self.client.get_or_create_collection(
            name=f"tenant_{safe_tenant}",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, tenant_id: str, ids: list[str], documents: list[str], embeddings: list[list[float]], metadatas: list[dict] = None):
        col = self._get_collection(tenant_id)
        col.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def search(self, tenant_id: str, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        col = self._get_collection(tenant_id)
        res = col.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "distances"])
        output = []
        if res["documents"] and res["documents"][0]:
            for doc, dist in zip(res["documents"][0], res["distances"][0]):
                output.append({"document": doc, "distance": dist})
        return output

# ----------------------------------------------------------------------
# Exercise 4: JSON Export and Import Migration Helper
# ----------------------------------------------------------------------
def export_collection_to_json(collection) -> str:
    """Mengekspor seluruh isi collection ke format JSON string."""
    all_info = collection.get()
    ids = all_info.get("ids", [])
    if not ids:
        return "[]"
    data = collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])
    records = []
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    embeddings = data.get("embeddings", [])

    for i in range(len(ids)):
        emb = None
        if embeddings is not None and i < len(embeddings):
            item = embeddings[i]
            emb = item.tolist() if hasattr(item, "tolist") else item

        rec = {
            "id": ids[i],
            "document": docs[i] if docs is not None and i < len(docs) else None,
            "metadata": metas[i] if metas is not None and i < len(metas) else None,
            "embedding": emb,
        }
        records.append(rec)
    return json.dumps(records, indent=2)

def import_collection_from_json(collection, json_str: str):
    """Mengimpor kembali data dari format JSON ke collection."""
    records = json.loads(json_str)
    if not records:
        return
    ids = [r["id"] for r in records]
    documents = [r["document"] for r in records]
    metadatas = [r["metadata"] for r in records]
    embeddings = [r["embedding"] for r in records if r.get("embedding") is not None]

    kwargs = {"ids": ids, "documents": documents, "metadatas": metadatas}
    if len(embeddings) == len(ids):
        kwargs["embeddings"] = embeddings
    collection.upsert(**kwargs)

# ----------------------------------------------------------------------
# Demonstrasi Eksekusi
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo Module 11 Exercises ===\n")
    client = chromadb.Client()

    # 1. Test Exercise 1: Benchmark
    print("[1] Testing VectorDBBenchmark:")
    benchmark = VectorDBBenchmark(client, dimension=8)
    stats = benchmark.run_benchmark(num_docs=100, num_queries=20)
    print(f" - Insert {stats['num_docs']} docs time : {stats['insert_time_sec']}s")
    print(f" - Avg query latency       : {stats['avg_latency_ms']} ms")
    print(f" - P95 query latency       : {stats['p95_latency_ms']} ms")
    print(f" - Throughput              : {stats['throughput_qps']} QPS\n")

    # 2. Test Exercise 2: Multi-Collection Search
    print("[2] Testing MultiCollectionSearch:")
    col_a = client.get_or_create_collection("docs_ai")
    col_b = client.get_or_create_collection("docs_db")
    col_a.add(ids=["a1"], documents=["RAG Architecture overview"], embeddings=[[0.1]*8])
    col_b.add(ids=["b1"], documents=["PostgreSQL pgvector tuning"], embeddings=[[0.12]*8])

    multi_search = MultiCollectionSearch(client)
    merged = multi_search.search(["docs_ai", "docs_db"], query_embedding=[0.1]*8, top_k=2)
    for r in merged:
        print(f" - [{r['collection']}] (dist={r['distance']:.4f}): {r['document']}")
    print()

    # 3. Test Exercise 3: TenantVectorStore
    print("[3] Testing TenantVectorStore:")
    tenant_store = TenantVectorStore(client)
    tenant_store.add_documents("tenant-alpha", ["t1"], ["Data rahasia Alpha"], [[0.2]*8])
    tenant_store.add_documents("tenant-beta", ["t1"], ["Data rahasia Beta"], [[0.2]*8])
    alpha_results = tenant_store.search("tenant-alpha", query_embedding=[0.2]*8)
    print(f" - Tenant Alpha query result: {alpha_results[0]['document']}")
    beta_results = tenant_store.search("tenant-beta", query_embedding=[0.2]*8)
    print(f" - Tenant Beta query result : {beta_results[0]['document']}\n")

    # 4. Test Exercise 4: JSON Export and Import
    print("[4] Testing JSON Export & Import:")
    export_json = export_collection_to_json(col_a)
    col_restore = client.get_or_create_collection("docs_ai_restored")
    import_collection_from_json(col_restore, export_json)
    print(f" - Restored collection count: {col_restore.count()}")
    print(" - First document restored :", col_restore.get()["documents"][0])

    print("\nSemua latihan Module 11 berhasil dijalankan.")
