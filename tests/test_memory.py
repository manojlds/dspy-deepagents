from dspy_deepagents.memory import RecursiveMemory


def test_memory_serializes_recent_entries() -> None:
    memory = RecursiveMemory()
    memory.add("planner", "Task A", depth=0)
    memory.add("executor", "Result A", depth=1)

    serialized = memory.serialize()

    assert "[planner depth=0] Task A" in serialized
    assert "[executor depth=1] Result A" in serialized
