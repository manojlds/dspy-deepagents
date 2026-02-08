"""Unit tests for tool functions (no LLM required).

These test the four pillar tool classes independently:
- TodoStore (planning)
- Workspace (filesystem)
- SubAgentDelegator (delegation depth limits)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from dspy_deepagents.tools import SubAgentDelegator, TodoStore, Workspace


class TestTodoStore:
    def test_empty_read(self) -> None:
        store = TodoStore()
        assert store.read_todos() == "No todos yet."

    def test_write_and_read_roundtrip(self) -> None:
        store = TodoStore()
        todos = [
            {"content": "step 1", "status": "pending"},
            {"content": "step 2", "status": "done"},
        ]
        result = store.write_todos(json.dumps(todos))
        assert "2 todos" in result

        read_back = json.loads(store.read_todos())
        assert len(read_back) == 2
        assert read_back[0]["content"] == "step 1"
        assert read_back[1]["status"] == "done"

    def test_write_replaces_previous(self) -> None:
        store = TodoStore()
        store.write_todos('[{"content": "old", "status": "pending"}]')
        store.write_todos('[{"content": "new", "status": "pending"}]')
        read_back = json.loads(store.read_todos())
        assert len(read_back) == 1
        assert read_back[0]["content"] == "new"


class TestWorkspace:
    def test_write_and_read_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            ws.write_file("notes.md", "hello world")
            assert ws.read_file("notes.md") == "hello world"

    def test_read_nonexistent_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            result = ws.read_file("missing.txt")
            assert "Error" in result
            assert "not found" in result

    def test_list_files_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            assert ws.list_files() == "Workspace is empty."

    def test_list_files_after_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            ws.write_file("a.txt", "aaa")
            ws.write_file("sub/b.txt", "bbb")
            listing = ws.list_files()
            assert "a.txt" in listing
            assert "sub/b.txt" in listing

    def test_nested_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            ws.write_file("deep/nested/file.txt", "content")
            assert ws.read_file("deep/nested/file.txt") == "content"

    def test_path_traversal_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            result = ws.write_file("../../escape.txt", "bad")
            assert "Error" in result
            assert "escapes" in result

    def test_read_path_traversal_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            result = ws.read_file("../../etc/passwd")
            assert "Error" in result
            assert "escapes" in result


class TestSubAgentDelegator:
    def test_depth_limit_reached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            delegator = SubAgentDelegator(
                workspace=ws,
                max_depth=2,
                current_depth=2,
            )
            result = delegator.delegate("do something")
            assert "Error" in result
            assert "max sub-agent depth" in result

    def test_no_factory_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            delegator = SubAgentDelegator(
                workspace=ws,
                max_depth=3,
                current_depth=0,
                agent_factory=None,
            )
            result = delegator.delegate("do something")
            assert "Error" in result
            assert "not configured" in result

    def test_depth_below_limit_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            delegator = SubAgentDelegator(
                workspace=ws,
                max_depth=3,
                current_depth=1,
            )
            # At depth 1 with max 3, delegation should be allowed
            # (not depth-limited). Without a factory it returns
            # "not configured" instead of the depth error.
            result = delegator.delegate("do something")
            assert "max sub-agent depth" not in result

    def test_inherited_params_forwarded_to_child(self) -> None:
        """Verify extra_tools, signature, sub_lm, include_review reach children."""
        captured: dict[str, object] = {}

        def fake_factory(**kwargs: object) -> None:
            captured.update(kwargs)
            # Raise to stop execution before RLM forward() is called
            raise RuntimeError("stop")

        def my_tool(x: str) -> str:
            return x

        sentinel_sig = type("FakeSig", (), {})
        sentinel_lm = object()

        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            delegator = SubAgentDelegator(
                workspace=ws,
                max_depth=3,
                current_depth=0,
                agent_factory=fake_factory,
                extra_tools=[my_tool],
                signature=sentinel_sig,
                sub_lm=sentinel_lm,
                include_review=True,
            )

            try:
                delegator.delegate("child task", context="ctx")
            except RuntimeError:
                pass

            # Factory should have been called with inherited params
            assert captured["signature"] is sentinel_sig
            assert captured["sub_lm"] is sentinel_lm
            assert captured["include_review"] is True
            assert captured["extra_tools"] == [my_tool]
            # Depth should be incremented
            assert captured["current_depth"] == 1
            assert captured["workspace"] is ws

    def test_inherited_params_default_to_none(self) -> None:
        """Without inherited params, children get None/False defaults."""
        captured: dict[str, object] = {}

        def fake_factory(**kwargs: object) -> None:
            captured.update(kwargs)
            raise RuntimeError("stop")

        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            delegator = SubAgentDelegator(
                workspace=ws,
                max_depth=3,
                current_depth=0,
                agent_factory=fake_factory,
            )

            try:
                delegator.delegate("child task")
            except RuntimeError:
                pass

            assert captured["extra_tools"] is None
            assert captured["signature"] is None
            assert captured["sub_lm"] is None
            assert captured["include_review"] is False
