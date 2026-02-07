import os

import dspy

from dspy_deepagents import RecursionConfig, RecursiveAgent


def main() -> None:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this example")

    dspy.settings.configure(lm=dspy.OpenAI(model=model, api_key=api_key))

    agent = RecursiveAgent(
        config=RecursionConfig(
            max_depth=1, budget=2, review_threshold=0.9, reflection_rounds=2
        )
    )

    result = agent(
        task=(
            "Explain the trade-offs of recursive agent decomposition in one paragraph."
        ),
        context="Prioritize concise and actionable insights.",
    )

    print("Result:\n", result.result)
    print("Confidence:", result.confidence)
    print("Trace events:", len(result.trace))


if __name__ == "__main__":
    main()
