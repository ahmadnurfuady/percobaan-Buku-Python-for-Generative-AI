# 8.8 Module 08 Exercises
import os, sqlite3, hashlib, json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 1. DuplicateDetector class
class DuplicateDetector:
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def find_near_duplicates(self, embeddings: np.ndarray) -> list[tuple[int, int, float]]:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.where(norms == 0, 1, norms)
        sim_matrix = normed @ normed.T

        duplicates = []
        n = embeddings.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self.threshold:
                    duplicates.append((i, j, float(sim_matrix[i, j])))
        return duplicates

# 2. HybridSearch blending semantic and keyword scores
class HybridSearch:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.docs = []

    def set_corpus(self, docs: list[str]):
        self.docs = docs

    def score(self, query: str, semantic_scores: list[float]) -> list[tuple[str, float]]:
        query_words = set(query.lower().split())
        results = []
        for doc, sem_score in zip(self.docs, semantic_scores):
            doc_words = doc.lower().split()
            # Simple term-overlap keyword score
            kw_score = sum(1 for w in doc_words if w in query_words) / max(len(doc_words), 1)
            final_score = self.alpha * sem_score + (1 - self.alpha) * kw_score
            results.append((doc, final_score))
        return sorted(results, key=lambda x: x[1], reverse=True)

# 3. Update & Delete extensions for VectorStore
class MutableVectorStore:
    def __init__(self):
        self._docs = {}  # doc_id -> (text, embedding)

    def add(self, doc_id: str, text: str, embedding: np.ndarray):
        norm = np.linalg.norm(embedding)
        normed = embedding / (norm if norm > 0 else 1)
        self._docs[doc_id] = (text, normed)

    def delete(self, doc_id: str) -> bool:
        if doc_id in self._docs:
            del self._docs[doc_id]
            return True
        return False

    def update(self, doc_id: str, new_text: str, new_embedding: np.ndarray):
        self.add(doc_id, new_text, new_embedding)

# 4. embed_with_cache using SQLite
def embed_with_cache(text: str, model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free", db_path: str = "embed_cache.db") -> list[float]:
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, embedding TEXT)")
    
    key = hashlib.sha256(f"{model}:{text}".encode()).hexdigest()
    cur = conn.execute("SELECT embedding FROM cache WHERE hash = ?", (key,))
    row = cur.fetchone()
    if row:
        conn.close()
        return json.loads(row[0])

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://openrouter.ai/api/v1")  
    resp = client.embeddings.create(input=[text], model=model)
    emb = resp.data[0].embedding
    
    conn.execute("INSERT INTO cache VALUES (?, ?)", (key, json.dumps(emb)))
    conn.commit()
    conn.close()
    return emb

if __name__ == "__main__":
    print("Module 08 Exercises implementations initialized.")
