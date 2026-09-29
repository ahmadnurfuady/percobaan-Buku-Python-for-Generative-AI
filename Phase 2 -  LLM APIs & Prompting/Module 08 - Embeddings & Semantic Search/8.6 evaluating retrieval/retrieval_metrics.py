import numpy as np
from dataclasses import dataclass

# --- TAMBAHAN: Mock Class untuk simulasi Vector Store dari Module 8.4 ---
class MockDocument:
    def __init__(self, doc_id):
        self.id = doc_id

class MockSearchResult:
    def __init__(self, doc_id):
        self.document = MockDocument(doc_id)

class MockVectorStore:
    def search(self, query: str, k: int):
        import random
        all_ids = ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10"]
        selected = random.sample(all_ids, k)
        return [MockSearchResult(doc_id) for doc_id in selected]

# PERBAIKAN: Taruh @dataclass TEPAT sebelum deklarasi class-nya
@dataclass
class RetrievalEvalCase:
    query: str
    relevant_doc_ids: list[str]   # ground-truth relevant documents

def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = retrieved_ids[:k]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of all relevant docs found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)

def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """MRR: reciprocal of the rank of the first relevant result."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval(
    store,               
    eval_cases: list[RetrievalEvalCase],
    k: int = 5,) -> dict:
    """Run all eval cases and return aggregate metrics."""

    p_scores, r_scores, mrr_scores = [], [], []
    for case in eval_cases:
        results = store.search(case.query, k=k)
        retrieved_ids = [r.document.id for r in results]
        p_scores.append(precision_at_k(retrieved_ids, case.relevant_doc_ids, k))
        r_scores.append(recall_at_k(retrieved_ids, case.relevant_doc_ids, k))
        mrr_scores.append(mean_reciprocal_rank(retrieved_ids, case.relevant_doc_ids))
    return {
        f"precision@{k}":  round(float(np.mean(p_scores)),4),
        f"recall@{k}":  round(float(np.mean(r_scores)),4),
        f"MRR":  round(float(np.mean(mrr_scores)),4),
    }

# Menggunakan data kasus evaluasi
eval_cases = [
    RetrievalEvalCase("How does RAG work?", ["d01", "d08"]),
    RetrievalEvalCase("What are vector databases?", ["d02", "d06"]),
    RetrievalEvalCase("How do agents use language models?", ["d10"]),
    RetrievalEvalCase("What is fine-tuning?", ["d03"]),
    RetrievalEvalCase("How do transformers model token relationships?", ["d09"]),
]

if __name__ == "__main__":
    store = MockVectorStore()
    metrics = evaluate_retrieval(store, eval_cases, k=3)
    print("--- Hasil Evaluasi Menggunakan Mock Store ---")
    print(metrics)
