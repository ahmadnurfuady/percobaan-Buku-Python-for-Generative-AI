# 8.7 Metadata Filtered Search
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np

@dataclass
class FilteredDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

class FilteredVectorStore:
    def __init__(self):
        self._docs: list[FilteredDocument] = []

    def add(self, docs: list[FilteredDocument], embeddings: np.ndarray) -> None:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.where(norms == 0, 1, norms)
        for doc, emb in zip(docs, normed):
            doc.embedding = emb
            self._docs.append(doc)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter_fn: Optional[Callable[[FilteredDocument], bool]] = None,
    ) -> list[tuple[FilteredDocument, float]]:
        candidates = self._docs if filter_fn is None else [d for d in self._docs if filter_fn(d)]
        if not candidates:
            return []

        q_norm = np.linalg.norm(query_vector)
        q_normed = query_vector / (q_norm if q_norm > 0 else 1)

        matrix = np.array([d.embedding for d in candidates], dtype=np.float32)
        scores = matrix @ q_normed
        k = min(k, len(candidates))
        top_idx = np.argsort(scores)[::-1][:k]
        return [(candidates[i], float(scores[i])) for i in top_idx]

if __name__ == "__main__":
    store = FilteredVectorStore()
    rng = np.random.default_rng(42)
    docs = [
        FilteredDocument("a1", "GPT-4o vision", {"category": "openai"}),
        FilteredDocument("a2", "Claude 3.5 Sonnet coding", {"category": "anthropic"}),
    ]
    store.add(docs, rng.standard_normal((2, 8)))
    results = store.search(rng.standard_normal(8), k=1, filter_fn=lambda d: d.metadata["category"] == "anthropic")
    for doc, score in results:
        print(f"Found: {doc.id} - {doc.text} (score: {score:.4f})")
