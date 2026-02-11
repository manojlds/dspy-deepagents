"""Deep agent for codebase exploration and analysis.

Demonstrates ``build_deep_agent(root=...)`` which gives the RLM filesystem
tools (list_dir, grep, glob_search, read_file_lines, stat, replace_lines)
rooted at a codebase directory.  The agent recursively explores a repository,
delegates module-level analysis to sub-agents, and synthesizes a structured
architectural overview.
"""

import os
import sys

import dspy

from dspy_deepagents import build_deep_agent


def main() -> None:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this example")

    codebase_path = sys.argv[1] if len(sys.argv) > 1 else "."

    dspy.configure(lm=dspy.LM(f"openai/{model}", api_key=api_key))

    agent = build_deep_agent(
        root=codebase_path,
        max_iterations=40,
        max_llm_calls=60,
        max_depth=2,
    )

    task = (
        "Analyze the architecture and design patterns of this codebase. "
        "Identify the main components and how they interact. Produce a "
        "structured analysis with the following sections:\n"
        "1. Architecture Overview — high-level structure and module layout\n"
        "2. Key Components — purpose and responsibility of each major module\n"
        "3. Design Patterns — patterns and idioms used (e.g., factory, "
        "strategy, dependency injection)\n"
        "4. Potential Improvements — areas where the design could be "
        "simplified, extended, or made more robust"
    )
    context = (
        "You are a senior software architect performing a codebase review. "
        "Use the exploration tools to map the repository, then delegate "
        "module-level deep dives to sub-agents. Synthesize findings into a "
        "clear, actionable report."
    )

    result = agent(task=task, context=context)

    print("Result:\n", result.result)
    print("Trajectory steps:", len(result.trajectory))


if __name__ == "__main__":
    main()
