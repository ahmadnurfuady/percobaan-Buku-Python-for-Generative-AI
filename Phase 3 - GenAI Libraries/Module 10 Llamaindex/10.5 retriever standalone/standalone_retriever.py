# 10.5 Retriever - Used Independently (Halaman 119-120)
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
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
    Document(text="ChromaDB is an open-source embedding database with a simple Python API.", metadata={"tag": "vector-db"}),
    Document(text="Pinecone is a managed vector database designed for production workloads.", metadata={"tag": "vector-db"}),
    Document(text="FAISS is a library by Meta for efficient similarity search on dense vectors.", metadata={"tag": "vector-search"}),
    Document(text="pgvector adds vector similarity search to PostgreSQL.", metadata={"tag": "vector-db"}),
    Document(text="Weaviate is an open-source vector database with a GraphQL API.", metadata={"tag": "vector-db"}),
]

index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

if __name__ == "__main__":
    # Retrieve without synthesising an answer
    nodes = retriever.retrieve("open source vector databases")
    print("Retrieved nodes:")
    for node in nodes:
        print(f" [{node.score:.4f}] {node.text}")
        print(f"  tag: {node.metadata.get('tag')}")
