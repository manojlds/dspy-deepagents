import os

import dspy

from dspy_deepagents import RecursionConfig, RecursiveAgent, Tool, ToolRegistry


def length_tool(text: str) -> str:
    return str(len(text.split()))


def main() -> None:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this example")

    dspy.settings.configure(lm=dspy.OpenAI(model=model, api_key=api_key))

    tools = ToolRegistry(
        tools=[
            Tool(
                name="word_count",
                description="Count words in text",
                func=length_tool,
            )
        ]
    )

    agent = RecursiveAgent(
        config=RecursionConfig(max_depth=1, budget=2),
        tools=tools,
    )

    result = agent(
        task="Use the word_count tool to count words in: 'Deep agents use recursion'.",
        context="",
    )

    print("Result:\n", result.result)
    print("Trace:\n", result.trace)


if __name__ == "__main__":
    main()
