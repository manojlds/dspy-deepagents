from dspy_deepagents.trace import TraceEvent


def test_trace_event_to_dict() -> None:
    event = TraceEvent(
        event="execute",
        depth=1,
        node_id="1-1",
        task="do thing",
        data={"result": "done"},
    )

    payload = event.to_dict()

    assert payload["event"] == "execute"
    assert payload["depth"] == 1
    assert payload["node_id"] == "1-1"
    assert payload["task"] == "do thing"
    assert payload["result"] == "done"
