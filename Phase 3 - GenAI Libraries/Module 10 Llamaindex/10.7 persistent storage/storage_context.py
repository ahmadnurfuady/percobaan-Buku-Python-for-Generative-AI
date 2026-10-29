# 10.7 Persistent Storage (Halaman 122-123)
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
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

PERSIST_DIR = "./storage"

documents = [
    Document(text="LlamaIndex persists indexes to disk as JSON files."),
    Document(text="Loading a persisted index avoids re-embedding on every run."),
]

if __name__ == "__main__":
    # Build and save index
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Index saved to {PERSIST_DIR}")

    # Load index back - no re-embedding needed
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    loaded_index = load_index_from_storage(storage_context)
    query_engine = loaded_index.as_query_engine()
    response = query_engine.query("How does LlamaIndex handle persistence?")
    print("After reload:", response)
