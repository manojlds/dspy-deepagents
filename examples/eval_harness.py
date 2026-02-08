"""Evaluation harness for comparing deep agent against baselines.

Runs a set of tasks through the deep agent and prints results with
trajectory statistics. Can be extended with ground-truth scoring
and DSPy optimizer integration.
"""

import os

import dspy

from dspy_deepagents import build_deep_agent


def echo(text: str) -> str:
    """Echo the input text back unchanged.

    Args:
        text: Text to echo.
    """
    return text


def main() -> None:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this example")

    dspy.configure(lm=dspy.LM(f"openai/{model}", api_key=api_key))

    agent = build_deep_agent(
        max_iterations=20,
        max_llm_calls=30,
        extra_tools=[echo],
    )

    tasks = [
        "Draft a two-step plan for onboarding a new engineer.",
        "Summarize why recursive decomposition helps tool-using agents.",
        "Use the echo tool to repeat 'hello deep agents'.",
    ]

    for task in tasks:
        result = agent(task=task)
        print("=" * 80)
        print("Task:", task)
        print("Result:\n", result.result)
        print("Trajectory steps:", len(result.trajectory))


if __name__ == "__main__":
    main()
