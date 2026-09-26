# 8.5 Chunking Text for Embedding (Sentence-Aware)
import re
from dataclasses import dataclass

@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int

def chunk_by_sentences(
    text: str,
    doc_id: str,
    max_chars: int = 1000,
    overlap_chars: int = 100,
) -> list[Chunk]:
    """Split text into chunks respecting sentence boundaries with overlap."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[Chunk] = []
    current = ""
    char_offset = 0
    chunk_idx = 0

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) > max_chars and current:
            end = char_offset + len(current)
            chunks.append(Chunk(doc_id, chunk_idx, current.strip(), char_offset, end))
            chunk_idx += 1

            overlap_start = max(0, len(current) - overlap_chars)
            overlap_text = current[overlap_start:]
            current = (overlap_text + " " + sentence).strip()
            char_offset = end - len(overlap_text)
        else:
            current = candidate

    if current.strip():
        end = char_offset + len(current)
        chunks.append(Chunk(doc_id, chunk_idx, current.strip(), char_offset, end))

    return chunks

if __name__ == "__main__":
    doc = "Large language models are neural networks. They learn to predict the next token. RAG extends LLMs by connecting them to external knowledge."
    chunks = chunk_by_sentences(doc, "doc_1", max_chars=80, overlap_chars=20)
    for c in chunks:
        print(f"Chunk {c.chunk_index} ({c.char_start}-{c.char_end}): {c.text}")
