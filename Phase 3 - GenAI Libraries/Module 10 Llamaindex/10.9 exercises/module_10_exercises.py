# 10.9 Module 10 Exercises (Halaman 124-125)
from llama_index.core import VectorStoreIndex, SummaryIndex, Document, Settings
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

# Konfigurasi LlamaIndex Global Settings
Settings.llm = OpenAILike(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    is_chat_model=True,
    temperature=0.1,
    max_tokens=1024,
)
Settings.embed_model = OpenAIEmbedding(
    model_name="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
)

# ----------------------------------------------------------------------
# Exercise 1: Document Q&A dengan Sumber File (ask_with_sources)
# ----------------------------------------------------------------------
def ask_with_sources(query_engine, question: str) -> dict:
    """Mengajukan pertanyaan ke query engine dan mengembalikan jawaban beserta nama file sumber."""
    resp = query_engine.query(question)
    sources = [node.metadata.get("source", "unknown") for node in resp.source_nodes]
    return {
        "answer": str(resp),
        "sources": list(set(sources)),
    }

# ----------------------------------------------------------------------
# Exercise 2: Perbandingan VectorStoreIndex vs SummaryIndex
# ----------------------------------------------------------------------
def compare_indices(documents: list[Document], questions: list[str]) -> dict:
    """Membandingkan VectorStoreIndex (pencarian fakta spesifik) vs SummaryIndex (sintesis menyeluruh)."""
    vector_index = VectorStoreIndex.from_documents(documents)
    summary_index = SummaryIndex.from_documents(documents)

    vector_engine = vector_index.as_query_engine(similarity_top_k=2)
    summary_engine = summary_index.as_query_engine()

    comparison = []
    for q in questions:
        v_ans = vector_engine.query(q)
        s_ans = summary_engine.query(q)
        comparison.append({
            "question": q,
            "vector_answer": str(v_ans),
            "summary_answer": str(s_ans),
        })
    return comparison

# ----------------------------------------------------------------------
# Exercise 3: Metadata-Filtered Retriever
# ----------------------------------------------------------------------
def get_metadata_filtered_retriever(index: VectorStoreIndex, year: int, topic: str, similarity_top_k: int = 3):
    """Membuat retriever yang memfilter dokumen berdasarkan kesesuaian metadata 'year' dan 'topic'."""
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="year", value=year),
            MetadataFilter(key="topic", value=topic),
        ]
    )
    return index.as_retriever(similarity_top_k=similarity_top_k, filters=filters)

# ----------------------------------------------------------------------
# Exercise 4: Two-Stage Pipeline (Retrieval + LLM Re-Ranking)
# ----------------------------------------------------------------------
def rerank_nodes_with_llm(nodes, question: str, llm=None, top_n: int = 3):
    """Menilai ulang relevansi setiap node dengan skor LLM (0-10) lalu mengambil top-n terbaik."""
    llm = llm or Settings.llm
    scored_nodes = []

    for node in nodes:
        prompt = (
            f"Question: {question}\n"
            f"Passage: {node.text}\n"
            "On a scale of 0 to 10, how relevant is this passage to the question? "
            "Reply with ONLY a single integer number between 0 and 10."
        )
        try:
            score_text = str(llm.complete(prompt)).strip()
            # Ambil angka pertama dari respon
            score = float("".join(c for c in score_text if c.isdigit() or c == ".") or "0")
        except Exception:
            score = 0.0

        scored_nodes.append((node, score))

    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    return [node for node, score in scored_nodes[:top_n]]

# ----------------------------------------------------------------------
# Demonstrasi Eksekusi
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo Module 10 Exercises ===\n")

    # Data dokumen sampel untuk pengujian
    sample_docs = [
        Document(
            text="In 2024, modern vector embeddings became essential for RAG systems.",
            metadata={"source": "rag_report_2024.txt", "year": 2024, "topic": "embeddings"}
        ),
        Document(
            text="In 2023, prompt engineering was the primary focus before agent frameworks matured.",
            metadata={"source": "prompting_2023.txt", "year": 2023, "topic": "prompting"}
        ),
        Document(
            text="In 2024, embedding compression via product quantisation reduced RAM costs significantly.",
            metadata={"source": "compression_2024.txt", "year": 2024, "topic": "embeddings"}
        ),
    ]

    index = VectorStoreIndex.from_documents(sample_docs)

    # 1. Test Exercise 1: ask_with_sources
    print("[1] Testing ask_with_sources:")
    query_engine = index.as_query_engine(similarity_top_k=2)
    result = ask_with_sources(query_engine, "What happened to embeddings in 2024?")
    print(f"Jawaban: {result['answer']}")
    print(f"Sumber  : {result['sources']}\n")

    # 2. Test Exercise 3: Metadata Filtering (year=2024, topic='embeddings')
    print("[2] Testing Metadata-Filtered Retriever (year=2024, topic='embeddings'):")
    filtered_retriever = get_metadata_filtered_retriever(index, year=2024, topic="embeddings")
    retrieved_nodes = filtered_retriever.retrieve("tell me about AI advances")
    for n in retrieved_nodes:
        print(f" - [{n.metadata.get('source')}] (year={n.metadata.get('year')}, topic={n.metadata.get('topic')}): {n.text}")
    print()

    # 3. Test Exercise 4: Two-stage Re-ranking
    print("[3] Testing Two-Stage Re-ranking:")
    all_retriever = index.as_retriever(similarity_top_k=3)
    initial_nodes = all_retriever.retrieve("How did compression help embeddings?")
    reranked = rerank_nodes_with_llm(initial_nodes, "How did compression help embeddings?", top_n=2)
    print("Top Re-ranked Nodes:")
    for idx, node in enumerate(reranked, start=1):
        print(f" {idx}. {node.text}")

    print("\nSemua latihan Module 10 berhasil dijalankan.")
