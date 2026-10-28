# 9.9 Module 09 Exercises (Halaman 114-115)
import hashlib
from langchain_core.documents import Document

# 1. MultiQueryRetriever manual
def multi_query_retrieve(query: str, retriever, llm_paraphrase_fn) -> list[Document]:
    paraphrases = llm_paraphrase_fn(query) # list of 3 paraphrased queries
    seen_hashes = set()
    unique_docs = []
    for q in [query] + paraphrases:
        docs = retriever.invoke(q)
        for d in docs:
            h = hashlib.sha256(d.page_content.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_docs.append(d)
    return unique_docs

# 2. Map-Reduce Summarizer
def map_reduce_summarize(chunks: list[str], summarize_fn, final_summarize_fn) -> str:
    chunk_summaries = [summarize_fn(c) for c in chunks]
    combined = "\n".join(chunk_summaries)
    return final_summarize_fn(combined)

# 3. Groundedness scorer
def score_groundedness(answer: str, context: str, judge_llm_fn) -> float:
    prompt = f"Context: {context}\nAnswer: {answer}\nScore from 0 to 10 how grounded the answer is."
    return judge_llm_fn(prompt)

if __name__ == "__main__":
    print("Module 09 Exercises implementations compiled.")
