# 10.2 Setup and Settings (Halaman 116)
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

# Configure global settings - applied to all indexes and query engines
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
Settings.chunk_size = 512       # tokens per chunk
Settings.chunk_overlap = 64     # token overlap between chunks

if __name__ == "__main__":
    print("LlamaIndex Global Settings configured successfully.")
