# 3.6 Module 03 Exercises
import functools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from string import Formatter

# -------------------------------------------------------------------
# 1. RateLimiter Class
# -------------------------------------------------------------------
class RateLimiter:
    """RateLimiter membatasi jumlah panggilan maksimal N kali per menit (sleeping as needed)."""
    def __init__(self, max_calls_per_minute: int = 60):
        self.max_calls = max_calls_per_minute
        self.interval = 60.0 / max_calls_per_minute  # Interval dalam detik
        self.last_call = 0.0

    def check_and_wait(self) -> None:
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            sleep_time = self.interval - elapsed
            print(f"[RateLimiter] Menunggu {sleep_time:.3f} detik...")
            time.sleep(sleep_time)
        self.last_call = time.time()


# -------------------------------------------------------------------
# 2. PromptTemplate Dataclass
# -------------------------------------------------------------------
@dataclass
class PromptTemplate:
    """
    PromptTemplate dataclass dengan method render(**kwargs) yang mengisi placeholder
    menggunakan str.format_map dan melempar ValueError jika ada placeholder yang kurang.
    """
    template: str

    def get_placeholders(self) -> set[str]:
        formatter = Formatter()
        return {fname for _, fname, _, _ in formatter.parse(self.template) if fname is not None}

    def render(self, **kwargs) -> str:
        placeholders = self.get_placeholders()
        missing = placeholders - set(kwargs.keys())
        if missing:
            raise ValueError(f"Placeholder berikut belum disediakan: {sorted(list(missing))}")
        return self.template.format_map(kwargs)


# -------------------------------------------------------------------
# 3. Parametrised Retry Decorator
# -------------------------------------------------------------------
def retry(max_attempts: int = 3, delay: float = 0.1):
    """Decorator terparameterisasi untuk melakukan retry hingga max_attempts kali."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_err: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    print(f"[Retry {attempt}/{max_attempts}] {func.__name__} gagal: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            if last_err is not None:
                raise last_err
            raise RuntimeError(f"Fungsi {func.__name__} gagal dieksekusi.")
        return wrapper
    return decorator


if __name__ == "__main__":
    print("=== Exercise 1: RateLimiter ===")
    limiter = RateLimiter(max_calls_per_minute=300)  # max 300 panggilan/menit (0.2 detik/panggilan)
    print("Memanggil check_and_wait() 5 kali secara cepat:")
    start_time = time.time()
    for i in range(1, 6):
        limiter.check_and_wait()
        print(f"  Panggilan ke-{i} berhasil pada t={time.time() - start_time:.3f}s")

    print("\n=== Exercise 2: PromptTemplate ===")
    template = PromptTemplate(
        "Halo {name}, Anda terdaftar sebagai {role} di sistem {system}."
    )
    # Render berhasil
    rendered = template.render(name="Ahmad", role="AI Engineer", system="Generative AI Platform")
    print("Hasil render:", rendered)

    # Uji validasi placeholder missing
    try:
        template.render(name="Ahmad")  # Kurang 'role' dan 'system'
    except ValueError as e:
        print("Validasi Berhasil (Error Tertangkap):", e)

    print("\n=== Exercise 3: Parametrised Retry Decorator ===")
    call_count = 0

    @retry(max_attempts=3, delay=0.1)
    def flaky_function():
        global call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError(f"Kegagalan koneksi pada percobaan #{call_count}")
        return f"Berhasil mendapatkan respons LLM pada percobaan #{call_count}!"

    res = flaky_function()
    print("Hasil:", res)

    print("\n=== Exercise 4: Package Import (llm_package) ===")
    # Menambahkan path folder '3.5 modules and packages' ke sys.path
    package_dir = Path(__file__).resolve().parent.parent / "3.5 modules and packages"
    if str(package_dir) not in sys.path:
        sys.path.append(str(package_dir))

    import importlib
    llm_pkg = importlib.import_module("llm_package")
    ConversationHistory = llm_pkg.ConversationHistory
    LLMConfig = llm_pkg.LLMConfig

    config = LLMConfig(model="gpt-4o", temperature=0.5, system_prompt="You are a helpful assistant.")
    history = ConversationHistory(max_turns=3, system_prompt=config.system_prompt or "")

    history.add("user", "Jelaskan apa itu PyTorch!")
    history.add("assistant", "PyTorch adalah framework open-source machine learning yang dikembangkan oleh Meta.")

    print("Config Dict:", config.as_dict)
    print("Conversation History repr:", repr(history))
    print("Payload API:", history.to_api_payload())
