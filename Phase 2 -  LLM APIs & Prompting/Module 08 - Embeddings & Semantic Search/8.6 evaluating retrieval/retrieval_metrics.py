# 8.6 Evaluating Retrieval Quality - Precision@k, Recall@k, MRR
import numpy as np
from dataclasses import dataclass

@dataclass
class RetrievalEvalCase:
    query: str
    relevant_doc_ids: list[str]

def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)

def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

if __name__ == "__main__":
    retrieved = ["d1", "d3", "d5", "d7"]
    ground_truth = ["d3", "d7"]
    print("Precision@2:", precision_at_k(retrieved, ground_truth, k=2))
    print("Recall@4:", recall_at_k(retrieved, ground_truth, k=4))
    print("MRR:", mean_reciprocal_rank(retrieved, ground_truth))
