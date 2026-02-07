import json
import os
from urllib.parse import quote
from urllib.request import Request, urlopen

import dspy

from dspy_deepagents import RecursionConfig, RecursiveAgent, Tool, ToolRegistry


def wikipedia_summary(topic: str) -> str:
    safe_topic = quote(topic.strip().replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_topic}"
    request = Request(url, headers={"User-Agent": "dspy-deepagents-example/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - example script
        return f"Error fetching summary for '{topic}': {exc}"

    extract = payload.get("extract") or "No summary available."
    page_url = payload.get("content_urls", {}).get("desktop", {}).get("page")
    if page_url:
        return f"Summary: {extract}\nSource: {page_url}"
    return f"Summary: {extract}"


def main() -> None:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this example")

    dspy.settings.configure(lm=dspy.OpenAI(model=model, api_key=api_key))

    tools = ToolRegistry(
        tools=[
            Tool(
                name="wikipedia_summary",
                description=(
                    "Fetch a short Wikipedia summary and source URL for a topic. "
                    "Input should be a concise topic name."
                ),
                func=wikipedia_summary,
            )
        ]
    )

    agent = RecursiveAgent(
        config=RecursionConfig(max_depth=2, max_children=3, budget=6),
        tools=tools,
    )

    task = (
        "Create a deep research brief on LLM alignment techniques. Cover RLHF, "
        "constitutional AI, and supervised fine-tuning. Use the wikipedia_summary "
        "tool for each technique, then synthesize into a 3-section brief with "
        "citations for each section."
    )
    context = (
        "You are a research analyst. Use tools to gather source-backed notes, "
        "then summarize clearly with bullet points and citations."
    )

    result = agent(task=task, context=context)

    print("Result:\n", result.result)
    print("Confidence:", result.confidence)
    print("Trace:\n", result.trace)
    print("Memory:\n", result.memory)


if __name__ == "__main__":
    main()
