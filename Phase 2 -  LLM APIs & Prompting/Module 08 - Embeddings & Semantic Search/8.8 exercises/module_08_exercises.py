# 8.8 Module 08 Exercises
import numpy as np

# Exercise 1: Duplicate Detector
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

if __name__ == "__main__":
    detector = DuplicateDetector(threshold=0.90)
    print("Duplicate detector ready.")
