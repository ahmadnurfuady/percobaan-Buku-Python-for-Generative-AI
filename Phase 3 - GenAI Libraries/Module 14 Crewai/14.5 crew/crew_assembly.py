# 14.5 Crew - Putting It Together (Halaman 159-161)
import sys
from pathlib import Path
from crewai import Crew, Process, Agent, LLM
import os
from dotenv import load_dotenv

load_dotenv()

# Tambahkan path ke modul tools dan tasks yang berada di folder tetangga
current_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(current_dir / "14.3 tools"))
sys.path.append(str(current_dir / "14.4 tasks"))

from custom_agent_tools import cost_tool, context_tool
from dependent_tasks import make_tasks

api_key = os.environ.get("OPENAI_API_KEY")
if api_key and api_key.startswith("sk-xt-"):
    llm = LLM(
        model="openai/qwen/qwen3.8-max:free",
        api_key=api_key,
        base_url="https://api.xkiro.com/v1",
    )
else:
    llm = "gpt-4o"

researcher = Agent(
    role="AI Research Analyst",
    goal="Find accurate information on AI topics.",
    backstory="Senior AI research analyst with expertise in LLM systems.",
    llm=llm,
    tools=[context_tool], # can look up context windows
    verbose=False,
    max_iter=5,
)
writer = Agent(
    role="Technical Content Writer",
    goal="Write clear technical content for developers.",
    backstory="Technical writer with a software engineering background.",
    llm=llm,
    tools=[cost_tool], # can look up costs for examples
    verbose=False,
    max_iter=5,
)
reviewer = Agent(
    role="Content Quality Reviewer",
    goal="Ensure content is accurate and publication-ready.",
    backstory="Meticulous editor who reviews technical AI content.",
    llm=llm,
    verbose=False,
    max_iter=5,
)

research_task, writing_task, review_task = make_tasks(researcher, writer, reviewer)

crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential, # tasks run in order; use Process.hierarchical for manager-led
    verbose=False,
    memory=False, # set True to enable cross-task memory
)

if __name__ == "__main__":
    print("Sequential Crew assembled successfully.")
    print("Agents:", [a.role for a in crew.agents])
    print("Tasks :", [t.description[:40] + "..." for t in crew.tasks])
    # Untuk menjalankan pipeline lengkap (memerlukan kuota API):
    # result = crew.kickoff()
    # print("=== CREW OUTPUT ===")
    # print(result)
