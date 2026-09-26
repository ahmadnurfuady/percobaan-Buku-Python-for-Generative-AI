# 8.3 Cosine Similarity and Pairwise Matrices
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def pairwise_similarity(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix of shape (n, n)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = matrix / norms
    return (normed @ normed.T).astype(np.float32)

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    sample_vecs = rng.standard_normal((4, 8)).astype(np.float32)
    sim_matrix = pairwise_similarity(sample_vecs)
    print("Pairwise similarity shape:", sim_matrix.shape)
    print("Diagonal values (self-similarity):", np.diag(sim_matrix))
