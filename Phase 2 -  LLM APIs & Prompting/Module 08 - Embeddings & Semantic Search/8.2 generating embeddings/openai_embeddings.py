# 8.2 Generating Embeddings with OpenAI
from openai import OpenAI
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "mock-key"))

def embed_texts_openai(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed a list of texts and return float32 numpy array of shape (n, dim)."""
    response = client.embeddings.create(input=texts, model=model)
    vectors = sorted(response.data, key=lambda e: e.index)
    return np.array([v.embedding for v in vectors], dtype=np.float32)

if __name__ == "__main__":
    print("OpenAI embed_texts ready.")
