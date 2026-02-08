"""Deep research agent with Wikipedia tool and sub-agent delegation.

Demonstrates a realistic research workflow where the agent:
1. Plans research steps using write_todos()
2. Delegates each research subtopic to a sub-agent via delegate()
3. Sub-agents use the wikipedia_summary tool to gather facts
4. Parent synthesizes findings from the shared workspace
"""

import json
import os
from urllib.parse import quote
from urllib.request import Request, urlopen

import dspy

from dspy_deepagents import build_deep_agent


def wikipedia_summary(topic: str) -> str:
    """Fetch a short Wikipedia summary and source URL for a topic.

    Args:
        topic: A concise topic name to look up on Wikipedia.
    """
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

    dspy.configure(lm=dspy.LM(f"openai/{model}", api_key=api_key))

    agent = build_deep_agent(
        max_iterations=40,
        max_llm_calls=60,
        max_depth=2,
        extra_tools=[wikipedia_summary],
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
    print("Trajectory steps:", len(result.trajectory))


if __name__ == "__main__":
    main()
