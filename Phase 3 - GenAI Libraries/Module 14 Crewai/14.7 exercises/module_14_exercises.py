# 14.7 Module 14 Exercises (Halaman 163)
import os
import json
import time
from typing import Type
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from dotenv import load_dotenv

load_dotenv()

# Setup default LLM
api_key = os.environ.get("OPENAI_API_KEY")
if api_key and api_key.startswith("sk-xt-"):
    default_llm = LLM(
        model="openai/qwen/qwen3.8-max:free",
        api_key=api_key,
        base_url="https://api.xkiro.com/v1",
    )
else:
    default_llm = "gpt-4o"

# ----------------------------------------------------------------------
# Exercise 3: Custom WebSearchTool wired to an Agent
# ----------------------------------------------------------------------
class SearchQueryInput(BaseModel):
    query: str = Field(description="Search query string")

class CustomWebSearchTool(BaseTool):
    """Tool pencarian web mock yang mengembalikan hasil fakta terstruktur."""
    name: str = "Web Search Tool"
    description: str = "Search the web for technical documentation, research, and benchmarks."
    args_schema: Type[BaseModel] = SearchQueryInput

    def _run(self, query: str) -> str:
        q_lower = query.lower()
        if "fine-tuning" in q_lower or "rag" in q_lower:
            return (
                "RAG provides dynamic retrieval from external vector DBs with zero model weight changes. "
                "Fine-tuning adapts internal parameters via LoRA for specific tasks or styles."
            )
        return f"Results for '{query}': Found relevant architectural patterns and implementation guides."

# ----------------------------------------------------------------------
# Exercise 1: Three-Agent Content Pipeline (Researcher -> Outline -> Full Writer)
# ----------------------------------------------------------------------
def build_content_pipeline_crew(topic: str, llm=None):
    llm = llm or default_llm
    
    researcher = Agent(
        role="Topic Researcher",
        goal=f"Extract core facts and technical comparisons for '{topic}'",
        backstory="Deep-dive researcher specialising in AI architectures.",
        llm=llm,
        verbose=False,
    )
    outliner = Agent(
        role="Outline Architect",
        goal="Structure the researched facts into a logical 3-section outline",
        backstory="Senior editor organizing complex technical ideas.",
        llm=llm,
        verbose=False,
    )
    writer = Agent(
        role="Technical Post Writer",
        goal="Write a concise publication-ready technical post",
        backstory="Developer-advocate writing engaging and accurate engineering blogs.",
        llm=llm,
        verbose=False,
    )

    t1 = Task(
        description=f"Research the key trade-offs and concepts for: '{topic}'",
        expected_output="Bulleted list of key facts and trade-offs.",
        agent=researcher,
    )
    t2 = Task(
        description="Using the research output, create a 3-part structured article outline.",
        expected_output="An outline with Part 1, Part 2, and Part 3 headers.",
        agent=outliner,
        context=[t1],
    )
    t3 = Task(
        description="Expand the outline into a finished post (around 200 words).",
        expected_output="Complete markdown blog post.",
        agent=writer,
        context=[t2],
    )

    return Crew(
        agents=[researcher, outliner, writer],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=False,
    )

# ----------------------------------------------------------------------
# Exercise 2: Code Review Crew (Developer, Security, Performance, Lead)
# ----------------------------------------------------------------------
def build_code_review_crew(code_snippet: str, llm=None):
    llm = llm or default_llm

    security_reviewer = Agent(
        role="Security Reviewer",
        goal="Identify injection, secret leaks, and insecure function calls.",
        backstory="AppSec engineer auditing Python code.",
        llm=llm,
        verbose=False,
    )
    perf_reviewer = Agent(
        role="Performance Reviewer",
        goal="Identify algorithmic bottlenecks, unnecessary loops, and memory hogs.",
        backstory="High-throughput systems optimization engineer.",
        llm=llm,
        verbose=False,
    )
    lead_reviewer = Agent(
        role="Lead Code Reviewer",
        goal="Consolidate feedback into an actionable, categorized review summary.",
        backstory="Staff engineer issuing final pull request approvals.",
        llm=llm,
        verbose=False,
    )

    t_sec = Task(
        description=f"Audit this code for security vulnerabilities:\n\n{code_snippet}",
        expected_output="Security findings list.",
        agent=security_reviewer,
    )
    t_perf = Task(
        description=f"Analyze this code for performance efficiency:\n\n{code_snippet}",
        expected_output="Performance findings list.",
        agent=perf_reviewer,
    )
    t_lead = Task(
        description="Synthesize both security and performance findings into a final report.",
        expected_output="Unified markdown report with verdict (Approve/Changes Requested).",
        agent=lead_reviewer,
        context=[t_sec, t_perf],
    )

    return Crew(
        agents=[security_reviewer, perf_reviewer, lead_reviewer],
        tasks=[t_sec, t_perf, t_lead],
        process=Process.sequential,
        verbose=False,
    )

# ----------------------------------------------------------------------
# Exercise 4: Sequential vs Hierarchical Comparison Matrix
# ----------------------------------------------------------------------
def get_process_comparison() -> dict:
    """Mendokumentasikan kapan Process.sequential vs Process.hierarchical sebaiknya dipilih."""
    return {
        "Process.sequential": {
            "execution": "Tugas berjalan linier satu per satu sesuai urutan context.",
            "overhead": "Rendah, tidak ada LLM manager orchestrator tambahan.",
            "best_for": "Alur kerja deterministik: Research -> Outline -> Write -> Review.",
        },
        "Process.hierarchical": {
            "execution": "Manager agent memutuskan agen mana yang mengerjakan tugas dan mendelegasikan balik.",
            "overhead": "Tinggi, setiap delegasi memerlukan call LLM manager.",
            "best_for": "Tugas kompleks dengan kebutuhan revisi dinamis, alokasi tugas ad-hoc.",
        }
    }

# ----------------------------------------------------------------------
# Demonstrasi Eksekusi
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo Module 14 Exercises ===\n")

    # 1. Test Exercise 3: WebSearchTool execution
    print("[1] Testing Custom WebSearchTool:")
    search_tool = CustomWebSearchTool()
    tool_output = search_tool._run("fine-tuning vs RAG")
    print(f" - Tool Result: {tool_output}\n")

    # 2. Test Exercise 1: Content Pipeline Assembly
    print("[2] Testing Three-Agent Content Pipeline Assembly:")
    topic = "The practical guide to fine-tuning vs RAG"
    content_crew = build_content_pipeline_crew(topic)
    print(f" - Pipeline topic: {topic}")
    print(f" - Crew agents ({len(content_crew.agents)}): {[a.role for a in content_crew.agents]}")
    print(f" - Crew tasks  ({len(content_crew.tasks)}): {[t.expected_output[:30] + '...' for t in content_crew.tasks]}\n")

    # 3. Test Exercise 2: Code Review Crew Assembly
    print("[3] Testing Code Review Crew Assembly:")
    sample_code = "def get_user(query): return eval(query)"
    review_crew = build_code_review_crew(sample_code)
    print(f" - Target Code: {sample_code}")
    print(f" - Reviewers  : {[a.role for a in review_crew.agents]}")
    print(f" - Tasks count: {len(review_crew.tasks)}\n")

    # 4. Test Exercise 4: Sequential vs Hierarchical Comparison
    print("[4] Testing Process.sequential vs Process.hierarchical Comparison:")
    comparison = get_process_comparison()
    for proc_name, details in comparison.items():
        print(f" * {proc_name}:")
        print(f"   - Mekanisme: {details['execution']}")
        print(f"   - Overhead : {details['overhead']}")
        print(f"   - Cocok utk: {details['best_for']}")

    print("\nSemua latihan Module 14 berhasil dijalankan.")
