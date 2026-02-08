"""Basic deep agent example.

Demonstrates the simplest usage of ``build_deep_agent``: give it a task
and let the RLM loop handle planning, execution, and synthesis via REPL
iterations.
"""

import os

import dspy

from dspy_deepagents import build_deep_agent


def main() -> None:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this example")

    dspy.configure(lm=dspy.LM(f"openai/{model}", api_key=api_key))

    agent = build_deep_agent(max_iterations=20, max_llm_calls=30)
    result = agent(
        task="Summarize how recursion is used in Deep Agents.",
        context="Focus on how sub-agent delegation provides context isolation.",
    )

    print("Result:\n", result.result)
    print("Trajectory steps:", len(result.trajectory))


if __name__ == "__main__":
    main()
