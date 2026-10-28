# 14.1 What CrewAI Is (Halaman 152-153)
"""
CrewAI is a framework for orchestrating multiple AI agents that collaborate on a shared goal.
Key mental model:
- Agent: a specialist (researcher, writer, reviewer) with defined expertise and tools
- Task: a concrete piece of work with an expected output
- Crew: a team of agents that executes tasks sequentially or hierarchically
"""

def print_crewai_overview():
    print("CrewAI Architecture: Agents -> Tools -> Tasks -> Crew")

if __name__ == "__main__":
    print_crewai_overview()
