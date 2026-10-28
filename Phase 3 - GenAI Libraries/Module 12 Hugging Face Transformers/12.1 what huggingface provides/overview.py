# 12.1 What Hugging Face Provides (Halaman 134)
"""
When to use Hugging Face over hosted APIs:
- Data privacy: model runs locally, no data leaves your infrastructure
- Cost at scale: after the inference hardware cost, no per-token fees
- Customisation: fine-tune on your domain data
- Offline: works without internet after model download
"""
def print_hf_tools():
    print("1. Pipeline API: high-level inference for 30+ tasks in 2-3 lines of code")
    print("2. AutoModel / AutoTokenizer: load any model from the Hub")
    print("3. PEFT / LoRA: parameter-efficient fine-tuning")

if __name__ == "__main__":
    print_hf_tools()
