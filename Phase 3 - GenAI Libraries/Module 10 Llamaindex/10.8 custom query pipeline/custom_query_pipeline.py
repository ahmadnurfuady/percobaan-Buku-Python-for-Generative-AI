# 10.8 Custom Query Pipeline (Halaman 123-124)
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

Settings.llm = OpenAILike(
    model="qwen/qwen3.8-max:free",
    api_key=os.environ["OPENAI_API_KEY"],
    api_base="https://api.xkiro.com/v1",
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

documents = [
    Document(text="HNSW (Hierarchical Navigable Small World) is the graph algorithm powering most modern vector search."),
    Document(text="IVF (Inverted File Index) partitions vectors into clusters for fast approximate search."),
    Document(text="PQ (Product Quantisation) compresses vectors to reduce memory usage at the cost of recall."),
    Document(text="Exact nearest-neighbour search guarantees correctness but is O(n) - unusable at scale."),
]

index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=2)

if __name__ == "__main__":
    # Different response modes
    for mode in [ResponseMode.COMPACT, ResponseMode.TREE_SUMMARIZE]:
        synthesiser = get_response_synthesizer(response_mode=mode)
        engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesiser)
        resp = engine.query("What algorithms are used in vector search?")
        print(f"[{mode}] {str(resp)[:120]}\n")
