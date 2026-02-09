"""Deep agent example with custom domain tools.

Demonstrates registering extra tool functions via ``extra_tools``.
The agent's RLM REPL can call these tools directly alongside the
built-in planning and workspace tools.
"""

import os

import dspy

from dspy_deepagents import build_deep_agent


def word_count(text: str) -> str:
    """Count the number of words in the given text.

    Args:
        text: The text to count words in.
    """
    return str(len(text.split()))


def main() -> None:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this example")

    dspy.configure(lm=dspy.LM(f"openai/{model}", api_key=api_key))

    agent = build_deep_agent(
        max_iterations=15,
        extra_tools=[word_count],
    )

    result = agent(
        task="Use the word_count tool to count words in: 'Deep agents use recursion'.",
    )

    print("Result:\n", result.result)
    print("Trajectory steps:", len(result.trajectory))


if __name__ == "__main__":
    main()
