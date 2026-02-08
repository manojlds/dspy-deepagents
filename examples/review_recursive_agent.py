"""Deep agent example with cross-agent review.

Demonstrates the ``include_review=True`` option which adds a
``review_draft`` tool.  The agent can call this tool to have an
independent reviewer (separate RLM with isolated context) evaluate
its draft before submitting.
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

    agent = build_deep_agent(
        max_iterations=25,
        include_review=True,
    )

    result = agent(
        task=(
            "Explain the trade-offs of recursive agent decomposition in one paragraph."
        ),
        context="Prioritize concise and actionable insights.",
    )

    print("Result:\n", result.result)
    print("Trajectory steps:", len(result.trajectory))


if __name__ == "__main__":
    main()
