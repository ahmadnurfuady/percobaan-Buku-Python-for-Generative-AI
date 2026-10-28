# 6.2 The Anthropic SDK - Multi-turn Conversation Loop
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], base_url="https://api.xkiro.com")

def chat_step(history: list[dict], user_input: str, system: str = "You are a helpful Python tutor.") -> str:
    """The Anthropic API is stateless; append turns to maintain history."""
    history.append({"role": "user", "content": user_input})
    response = client.messages.create(
        model="qwen/qwen3.8-max:free",
        max_tokens=1024,
        system=system,
        messages=history,
    )
    assistant_text = response.content[0].text
    history.append({"role": "assistant", "content": assistant_text})
    return assistant_text

if __name__ == "__main__":
    conversation = []
    print("Multi-turn chat session initialized.")
    print("Ketik 'history' untuk melihat riwayat percakapan, 'quit' untuk keluar.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Sesi chat berakhir.")
            break

        if user_input.lower() == "history":
            if not conversation:
                print("  (Belum ada riwayat percakapan.)\n")
            else:
                print("\n=== Riwayat Percakapan ===")
                for i, turn in enumerate(conversation, start=1):
                    role = "You" if turn["role"] == "user" else "AI"
                    print(f"[{i}] {role}: {turn['content']}")
                print("==========================\n")
            continue

        # Kirim ke AI dan tampilkan balasan
        reply = chat_step(conversation, user_input)
        print(f"AI: {reply}\n")
