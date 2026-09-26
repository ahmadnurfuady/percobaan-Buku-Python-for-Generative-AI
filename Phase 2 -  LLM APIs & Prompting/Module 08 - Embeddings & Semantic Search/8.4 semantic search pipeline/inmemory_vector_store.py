# 8.4 In-Memory VectorStore with Semantic Search
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int

class VectorStore:
    def __init__(self, dim: int = 8):
        self.dim = dim
        self._documents: list[Document] = []
        self._matrix: Optional[np.ndarray] = None

    def add_documents(self, documents: list[Document], embeddings: np.ndarray) -> None:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = (embeddings / norms).astype(np.float32)

        for doc, vec in zip(documents, normed):
            doc.embedding = vec
            self._documents.append(doc)

        self._matrix = np.array([d.embedding for d in self._documents], dtype=np.float32)

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[SearchResult]:
        if self._matrix is None or len(self._documents) == 0:
            return []
        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            return []
        q_normed = (query_vector / q_norm).astype(np.float32)

        scores = self._matrix @ q_normed
        k = min(k, len(self._documents))
        top_idx = np.argsort(scores)[::-1][:k]

        return [
            SearchResult(document=self._documents[int(i)], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(top_idx)
        ]

if __name__ == "__main__":
    store = VectorStore(dim=8)
    rng = np.random.default_rng(42)
    docs = [Document("d1", "What is RAG?"), Document("d2", "Vector databases store embeddings.")]
    mock_embs = rng.standard_normal((2, 8))
    store.add_documents(docs, mock_embs)
    results = store.search(rng.standard_normal(8), k=2)
    for r in results:
        print(f"Rank {r.rank} (score: {r.score:.4f}): {r.document.text}")
