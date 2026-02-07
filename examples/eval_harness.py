import os

import dspy

from dspy_deepagents import RecursionConfig, RecursiveAgent, Tool, ToolRegistry


def echo_tool(text: str) -> str:
    return text


def main() -> None:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this example")

    dspy.settings.configure(lm=dspy.OpenAI(model=model, api_key=api_key))

    tools = ToolRegistry(
        tools=[Tool(name="echo", description="Echo input text", func=echo_tool)]
    )

    agent = RecursiveAgent(
        config=RecursionConfig(max_depth=2, budget=4, review_threshold=0.8),
        tools=tools,
    )

    tasks = [
        "Draft a two-step plan for onboarding a new engineer.",
        "Summarize why recursive decomposition helps tool-using agents.",
        "Use the echo tool to repeat 'hello deep agents'.",
    ]

    for task in tasks:
        result = agent(task=task, context="")
        print("=" * 80)
        print("Task:", task)
        print("Result:\n", result.result)
        print("Confidence:", result.confidence)
        print("Trace events:", len(result.trace))
        print("Memory:\n", result.memory)


if __name__ == "__main__":
    main()
