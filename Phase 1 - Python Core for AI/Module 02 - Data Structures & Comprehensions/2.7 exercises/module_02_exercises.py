# 2.7 Module 02 Exercises
from typing import Iterable, Generator, Any

# 1. Filter dan Sort API Responses menggunakan List Comprehension
def filter_and_sort_responses(responses: list[dict]) -> list[dict]:
    """
    Mengembalikan daftar respons yang memiliki latency < 500ms,
    diurutkan berdasarkan jumlah token secara ascending.
    """
    filtered = [r for r in responses if r.get("latency_ms", 0) < 500]
    return sorted(filtered, key=lambda r: r.get("tokens", 0))


# 2. Statistik Percakapan
def conversation_stats(messages: list[dict]) -> dict:
    """
    Menghitung statistik pesan percakapan:
    - total_messages: Jumlah total pesan
    - user_turns: Jumlah pesan dari user
    - assistant_turns: Jumlah pesan dari assistant
    - avg_words_per_message: Rata-rata kata per pesan
    """
    total = len(messages)
    if total == 0:
        return {
            "total_messages": 0,
            "user_turns": 0,
            "assistant_turns": 0,
            "avg_words_per_message": 0.0,
        }

    user_turns = sum(1 for m in messages if m.get("role") == "user")
    assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
    total_words = sum(len(m.get("content", "").split()) for m in messages)

    return {
        "total_messages": total,
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "avg_words_per_message": round(total_words / total, 2),
    }


# 3. Generator Batching Items
def batch_items(items: Iterable[Any], batch_size: int) -> Generator[list[Any], None, None]:
    """
    Generator yang menghasilkan batch berukuran batch_size dari iterable apapun.
    Menangani sisa item (batch terakhir yang tidak penuh) dengan benar.
    """
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# 4. Irisan Set Model Cepat & Murah
def find_fast_and_cheap_models(fast_models: list[str], cheap_models: list[str]) -> set[str]:
    """
    Menggunakan operasi irisan set (&) untuk menemukan model yang cepat sekaligus murah.
    """
    return set(fast_models) & set(cheap_models)


if __name__ == "__main__":
    print("=== Exercise 1: Filter & Sort Responses ===")
    sample_responses = [
        {"model": "gpt-4o", "tokens": 1500, "latency_ms": 650},
        {"model": "gpt-4o-mini", "tokens": 300, "latency_ms": 120},
        {"model": "claude-haiku", "tokens": 800, "latency_ms": 250},
        {"model": "gemini-flash", "tokens": 150, "latency_ms": 90},
        {"model": "custom-llm", "tokens": 2000, "latency_ms": 510},
    ]
    result1 = filter_and_sort_responses(sample_responses)
    for res in result1:
        print(f"Model: {res['model']}, Tokens: {res['tokens']}, Latency: {res['latency_ms']}ms")

    print("\n=== Exercise 2: Conversation Stats ===")
    conversation = [
        {"role": "user", "content": "Halo, jelaskan apa itu Generative AI dalam 2 kalimat."},
        {"role": "assistant", "content": "Generative AI adalah cabang kecerdasan buatan yang mampu membuat konten baru seperti teks, gambar, atau kode. Teknologi ini memanfaatkan model deep learning yang dilatih pada dataset besar."},
        {"role": "user", "content": "Berikan contoh aplikasinya."},
        {"role": "assistant", "content": "Contoh aplikasinya meliputi ChatGPT untuk teks, Midjourney untuk gambar, dan GitHub Copilot untuk membantu penulisan kode program."},
    ]
    stats = conversation_stats(conversation)
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n=== Exercise 3: Batch Generator ===")
    dataset = list(range(1, 11))  # 10 item
    print(f"Data asli: {dataset}")
    print("Batch (ukuran 3):")
    for i, batch in enumerate(batch_items(dataset, batch_size=3), 1):
        print(f"  Batch {i}: {batch}")

    print("\n=== Exercise 4: Fast & Cheap Models (Set Operations) ===")
    fast_models = ["gpt-4o-mini", "claude-3-5-haiku", "gemini-1.5-flash", "gpt-4o"]
    cheap_models = ["gpt-4o-mini", "claude-3-5-haiku", "gemini-1.5-flash", "deepseek-v3"]
    common_models = find_fast_and_cheap_models(fast_models, cheap_models)
    print("Model Cepat:", fast_models)
    print("Model Murah:", cheap_models)
    print("Model Cepat & Murah (Intersection):", common_models)

