from dspy_deepagents.tools import Tool, ToolRegistry


def test_tool_registry_runs_tool() -> None:
    registry = ToolRegistry(
        tools=[Tool(name="echo", description="Echo input", func=lambda x: x)]
    )

    result = registry.run("echo", "hello")

    assert result == "hello"
