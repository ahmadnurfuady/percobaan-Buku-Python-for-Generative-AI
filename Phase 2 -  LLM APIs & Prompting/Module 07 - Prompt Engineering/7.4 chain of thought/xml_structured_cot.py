# 7.4 Structured CoT with XML Tags
import re

SYSTEM_XML = """Solve problems using this exact format:
<thinking>
Step-by-step reasoning here.
</thinking>
<answer>
The final answer only, no reasoning.
</answer>"""

def parse_xml_sections(response_text: str) -> tuple[str, str]:
    thinking = re.search(r"<thinking>(.*?)</thinking>", response_text, re.DOTALL)
    answer = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
    reasoning = thinking.group(1).strip() if thinking else "not found"
    ans = answer.group(1).strip() if answer else "not found"
    return reasoning, ans

if __name__ == "__main__":
    sample_llm_output = """
    <thinking>
    Context tokens: 5 * 400 = 2000.
    Total input = 2000 + 50 = 2050.
    Remaining = 4096 - 2050 = 2046 tokens.
    </thinking>
    <answer>
    2046 tokens
    </answer>
    """
    reasoning, final_ans = parse_xml_sections(sample_llm_output)
    print("Reasoning:", reasoning)
    print("Answer:", final_ans)
