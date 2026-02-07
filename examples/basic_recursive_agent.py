import os

import dspy

from dspy_deepagents import RecursionConfig, RecursiveAgent, RecursiveMemory


def main() -> None:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this example")

    dspy.settings.configure(lm=dspy.OpenAI(model=model, api_key=api_key))

    agent = RecursiveAgent(config=RecursionConfig(max_depth=2, budget=4))
    memory = RecursiveMemory()
    result = agent(
        task="Summarize how recursion is used in Deep Agents.",
        context="Focus on planner/executor/reviewer roles.",
        state=memory,
    )

    print("Result:\n", result.result)
    print("Confidence:", result.confidence)
    print("Memory:\n", result.memory)
    print("Trace events:", len(result.trace))


if __name__ == "__main__":
    main()
