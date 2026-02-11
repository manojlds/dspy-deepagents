from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from dspy_deepagents.codebase_tools import CodebaseTools, FilesystemTools


class TestListDir:
    def test_list_dir_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.list_dir())
            assert result["items"] == []
            assert result["total"] == 0

    def test_list_dir_with_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("aaa")
            (Path(tmp) / "b.py").write_text("bbb")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.list_dir())
            paths = [item["path"] for item in result["items"]]
            assert "a.txt" in paths
            assert "b.py" in paths

    def test_list_dir_dirs_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "z_file.txt").write_text("")
            (Path(tmp) / "a_dir").mkdir()
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.list_dir())
            assert result["items"][0]["type"] == "dir"
            assert result["items"][1]["type"] == "file"

    def test_list_dir_pagination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                (Path(tmp) / f"file{i}.txt").write_text("")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.list_dir(limit=2, offset=0))
            assert len(result["items"]) == 2
            assert result["total"] == 5
            assert result["truncated"] is True

            result2 = json.loads(tools.list_dir(limit=2, offset=4))
            assert len(result2["items"]) == 1
            assert result2["truncated"] is False

    def test_list_dir_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "a" / "b"
            nested.mkdir(parents=True)
            (nested / "deep.txt").write_text("")
            tools = CodebaseTools(Path(tmp))

            shallow = json.loads(tools.list_dir(depth=1))
            deep_paths = [item["path"] for item in shallow["items"]]
            assert any("deep.txt" in p for p in deep_paths) is False

            deep = json.loads(tools.list_dir(depth=3))
            deep_paths = [item["path"] for item in deep["items"]]
            assert any("deep.txt" in p for p in deep_paths)

    def test_list_dir_skips_hidden_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            (Path(tmp) / ".git" / "config").write_text("")
            (Path(tmp) / "src").mkdir()
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.list_dir(depth=2))
            paths = [item["path"] for item in result["items"]]
            assert not any(".git" in p for p in paths)
            assert "src" in paths


class TestGlobSearch:
    def test_glob_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "main.py").write_text("")
            (Path(tmp) / "readme.md").write_text("")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.glob_search("*.py"))
            assert result["matches"] == ["main.py"]

    def test_glob_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp) / "sub"
            sub.mkdir()
            (sub / "notes.txt").write_text("")
            (Path(tmp) / "top.txt").write_text("")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.glob_search("**/*.txt"))
            assert len(result["matches"]) == 2

    def test_glob_pagination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                (Path(tmp) / f"f{i}.py").write_text("")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.glob_search("*.py", limit=2, offset=0))
            assert len(result["matches"]) == 2
            assert result["total"] == 5
            assert result["truncated"] is True

    def test_glob_skips_skip_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nm = Path(tmp) / "node_modules"
            nm.mkdir()
            (nm / "pkg.js").write_text("")
            (Path(tmp) / "app.js").write_text("")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.glob_search("**/*.js"))
            assert result["matches"] == ["app.js"]


class TestGrep:
    def test_grep_basic_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "code.py").write_text("def hello():\n    pass\n")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.grep("hello"))
            assert result["total_matches"] == 1
            assert result["matches"][0]["text"] == "def hello():"

    def test_grep_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("Hello\nhello\nHELLO\n")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.grep("hello"))
            assert result["total_matches"] == 3

    def test_grep_case_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("Hello\nhello\nHELLO\n")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.grep("hello", case_sensitive=True))
            assert result["total_matches"] == 1

    def test_grep_with_context_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            content = "line1\nline2\ntarget\nline4\nline5\n"
            (Path(tmp) / "f.txt").write_text(content)
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.grep("target", context_lines=1))
            match = result["matches"][0]
            assert match["context_before"] == ["line2"]
            assert match["context_after"] == ["line4"]

    def test_grep_max_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lines = "\n".join(f"match {i}" for i in range(20))
            (Path(tmp) / "f.txt").write_text(lines)
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.grep("match", max_matches=5))
            assert len(result["matches"]) == 5
            assert result["truncated"] is True

    def test_grep_max_matches_per_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lines = "\n".join(f"hit {i}" for i in range(20))
            (Path(tmp) / "f.txt").write_text(lines)
            tools = CodebaseTools(Path(tmp))
            result = json.loads(
                tools.grep("hit", max_matches=50, max_matches_per_file=3)
            )
            assert len(result["matches"]) == 3

    def test_grep_with_glob_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "code.py").write_text("target\n")
            (Path(tmp) / "notes.txt").write_text("target\n")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.grep("target", glob_pattern="*.py"))
            assert result["total_matches"] == 1
            assert result["matches"][0]["path"] == "code.py"

    def test_grep_skips_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "bin.dat").write_bytes(b"target\x00binary")
            (Path(tmp) / "text.txt").write_text("target\n")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.grep("target"))
            paths = [m["path"] for m in result["matches"]]
            assert "bin.dat" not in paths
            assert "text.txt" in paths

    def test_grep_invalid_regex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.grep("[invalid"))
            assert "error" in result
            assert "invalid regex" in result["error"]


class TestReadFileLines:
    def test_read_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            content = "\n".join(f"line {i}" for i in range(1, 11))
            (Path(tmp) / "f.txt").write_text(content)
            tools = CodebaseTools(Path(tmp))
            result = json.loads(
                tools.read_file_lines("f.txt", start_line=1, end_line=5)
            )
            assert result["start_line"] == 1
            assert result["end_line"] == 5
            assert "line 1" in result["content"]
            assert "line 5" in result["content"]

    def test_read_with_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("alpha\nbeta\ngamma\n")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(
                tools.read_file_lines("f.txt", start_line=1, end_line=3)
            )
            assert "1: alpha" in result["content"]
            assert "2: beta" in result["content"]
            assert "3: gamma" in result["content"]

    def test_read_has_more(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            content = "\n".join(f"line {i}" for i in range(1, 21))
            (Path(tmp) / "f.txt").write_text(content)
            tools = CodebaseTools(Path(tmp))
            result = json.loads(
                tools.read_file_lines("f.txt", start_line=1, end_line=5)
            )
            assert result["has_more"] is True

            result2 = json.loads(
                tools.read_file_lines("f.txt", start_line=1, end_line=20)
            )
            assert result2["has_more"] is False

    def test_read_beyond_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("one\ntwo\nthree\n")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(
                tools.read_file_lines("f.txt", start_line=1, end_line=999)
            )
            assert result["end_line"] == result["total_lines"]

    def test_read_nonexistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.read_file_lines("missing.txt"))
            assert "error" in result


class TestStat:
    def test_stat_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("hello\nworld\n")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.stat("f.txt"))
            assert result["type"] == "file"
            assert result["size"] > 0
            assert result["lines"] == 2

    def test_stat_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "subdir").mkdir()
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.stat("subdir"))
            assert result["type"] == "dir"

    def test_stat_nonexistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.stat("nope"))
            assert "error" in result


class TestReplaceLines:
    def test_replace_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("aaa\nbbb\nccc\nddd\n")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.replace_lines("f.txt", 2, 3, "XXX\nYYY"))
            assert result["old_lines"] == 2
            assert result["new_lines"] == 2
            content = (Path(tmp) / "f.txt").read_text()
            assert "XXX" in content
            assert "YYY" in content
            assert "bbb" not in content
            assert "ccc" not in content

    def test_replace_with_more_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("aaa\nbbb\nccc\n")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.replace_lines("f.txt", 2, 2, "X\nY\nZ"))
            assert result["old_lines"] == 1
            assert result["new_lines"] == 3
            assert result["total_lines_after"] == 5

    def test_replace_with_fewer_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("aaa\nbbb\nccc\nddd\n")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.replace_lines("f.txt", 2, 3, "SINGLE"))
            assert result["old_lines"] == 2
            assert result["new_lines"] == 1
            assert result["total_lines_after"] == 3

    def test_replace_invalid_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("aaa\nbbb\n")
            tools = CodebaseTools(Path(tmp))
            result = json.loads(tools.replace_lines("f.txt", 5, 10, "nope"))
            assert "error" in result


class TestPathTraversal:
    def test_resolve_blocks_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = CodebaseTools(Path(tmp))
            with pytest.raises(ValueError, match="escapes"):
                tools._resolve("../../etc/passwd")

    def test_list_dir_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = CodebaseTools(Path(tmp))
            with pytest.raises(ValueError, match="escapes"):
                tools.list_dir("../../etc")

    def test_read_file_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = CodebaseTools(Path(tmp))
            with pytest.raises(ValueError, match="escapes"):
                tools.read_file_lines("../../etc/passwd")


def test_filesystem_tools_alias() -> None:
    assert FilesystemTools is CodebaseTools
