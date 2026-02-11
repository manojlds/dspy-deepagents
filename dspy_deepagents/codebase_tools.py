"""Filesystem middleware — paginated, bounded file tools for Deep Agents.

Implements the filesystem pillar of the Deep Agents architecture.  Provides
navigation, search, read, and edit tools that all Deep Agents get by default,
mirroring LangChain Deep Agents' filesystem middleware.

All methods return JSON-formatted strings so the REPL can parse them
programmatically.
"""

from __future__ import annotations

import fnmatch
import json
import re
from datetime import UTC, datetime
from pathlib import Path

_SKIP_DIRS = frozenset({
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    "dist",
    "build",
})

_MAX_BINARY_CHECK = 8192
_MAX_SIZE_FOR_LINE_COUNT = 10 * 1024 * 1024


class FilesystemTools:
    """Paginated filesystem tools rooted at a directory.

    Part of the default Deep Agent tool set (filesystem pillar).
    Every path argument is resolved relative to *root* and validated so
    that it cannot escape the root via ``..`` or symlinks.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _resolve(self, path: str) -> Path:
        """Resolve *path* relative to root and ensure it stays inside.

        Args:
            path: Relative path within the codebase root.

        Returns:
            Resolved absolute ``Path``.

        Raises:
            ValueError: If the resolved path escapes root.
        """
        resolved = (self.root / path).resolve()
        if not str(resolved).startswith(str(self.root)):
            raise ValueError(f"path '{path}' escapes codebase root")
        return resolved

    def list_dir(
        self,
        path: str = ".",
        depth: int = 1,
        limit: int = 200,
        offset: int = 0,
    ) -> str:
        """List directory entries with pagination.

        Args:
            path: Relative directory path to list.
            depth: How many levels deep to recurse (1 = immediate children).
            limit: Maximum number of entries to return.
            offset: Number of entries to skip for pagination.
        """
        target = self._resolve(path)
        if not target.is_dir():
            return json.dumps({"error": f"'{path}' is not a directory"})

        items: list[dict[str, object]] = []
        self._collect_entries(target, target, depth, items)

        items.sort(key=lambda e: (e["type"] != "dir", str(e["path"])))

        total = len(items)
        page = items[offset : offset + limit]
        return json.dumps({
            "path": path,
            "items": page,
            "total": total,
            "offset": offset,
            "limit": limit,
            "truncated": offset + limit < total,
        })

    def _collect_entries(
        self,
        directory: Path,
        base: Path,
        remaining_depth: int,
        out: list[dict[str, object]],
    ) -> None:
        if remaining_depth <= 0:
            return
        try:
            children = sorted(directory.iterdir())
        except PermissionError:
            return
        for child in children:
            if child.name.startswith(".") and child.is_dir():
                continue
            if child.is_dir() and child.name in _SKIP_DIRS:
                continue
            rel = str(child.relative_to(base))
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
            entry_type = "dir" if child.is_dir() else "file"
            out.append({"path": rel, "type": entry_type, "size": size})
            if child.is_dir():
                self._collect_entries(child, base, remaining_depth - 1, out)

    def glob_search(
        self,
        pattern: str,
        limit: int = 200,
        offset: int = 0,
    ) -> str:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern relative to the codebase root (e.g. ``**/*.py``).
            limit: Maximum number of matches to return.
            offset: Number of matches to skip for pagination.
        """
        matches: list[str] = []
        for p in sorted(self.root.glob(pattern)):
            if not str(p.resolve()).startswith(str(self.root)):
                continue
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            if p.is_file():
                matches.append(str(p.relative_to(self.root)))

        total = len(matches)
        page = matches[offset : offset + limit]
        return json.dumps({
            "pattern": pattern,
            "matches": page,
            "total": total,
            "offset": offset,
            "limit": limit,
            "truncated": offset + limit < total,
        })

    def grep(
        self,
        pattern: str,
        path: str = ".",
        glob_pattern: str | None = None,
        case_sensitive: bool = False,
        max_matches: int = 50,
        max_matches_per_file: int = 10,
        context_lines: int = 0,
    ) -> str:
        """Search for a text pattern in files.

        Args:
            pattern: Regex pattern to search for.
            path: Relative directory (or file) to search within.
            glob_pattern: If provided, only search files matching this glob.
            case_sensitive: Whether the search is case-sensitive.
            max_matches: Maximum total matches to return.
            max_matches_per_file: Maximum matches per individual file.
            context_lines: Number of context lines before and after each match.
        """
        target = self._resolve(path)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            return json.dumps({"error": f"invalid regex: {exc}"})

        if target.is_file():
            files = [target]
        else:
            files = sorted(target.rglob("*"))

        results: list[dict[str, object]] = []
        total_matches = 0

        for fp in files:
            if not fp.is_file():
                continue
            if any(part in _SKIP_DIRS for part in fp.relative_to(self.root).parts):
                continue
            rel_parts = fp.relative_to(self.root).parts
            if any(
                p.startswith(".") and idx < len(rel_parts) - 1
                for idx, p in enumerate(rel_parts)
            ):
                continue
            if glob_pattern and not fnmatch.fnmatch(
                str(fp.relative_to(self.root)), glob_pattern
            ):
                continue
            if self._is_binary(fp):
                continue

            file_hits = 0
            try:
                lines = fp.read_text(errors="replace").splitlines()
            except (OSError, UnicodeDecodeError):
                continue

            for i, line in enumerate(lines):
                if compiled.search(line):
                    if total_matches >= max_matches:
                        return json.dumps({
                            "pattern": pattern,
                            "matches": results,
                            "total_matches": total_matches,
                            "truncated": True,
                        })
                    if file_hits >= max_matches_per_file:
                        break
                    if context_lines:
                        before = lines[max(0, i - context_lines) : i]
                        after = lines[i + 1 : i + 1 + context_lines]
                    else:
                        before = []
                        after = []
                    results.append({
                        "path": str(fp.relative_to(self.root)),
                        "line": i + 1,
                        "text": line,
                        "context_before": before,
                        "context_after": after,
                    })
                    file_hits += 1
                    total_matches += 1

        return json.dumps({
            "pattern": pattern,
            "matches": results,
            "total_matches": total_matches,
            "truncated": False,
        })

    def read_file_lines(
        self,
        path: str,
        start_line: int = 1,
        end_line: int = 200,
    ) -> str:
        """Read a specific range of lines from a file.

        Args:
            path: Relative path to the file.
            start_line: First line to read (1-indexed).
            end_line: Last line to read (1-indexed, inclusive).
        """
        target = self._resolve(path)
        if not target.is_file():
            return json.dumps({"error": f"'{path}' is not a file"})
        try:
            all_lines = target.read_text(errors="replace").splitlines()
        except OSError as exc:
            return json.dumps({"error": str(exc)})

        total = len(all_lines)
        start_line = max(1, start_line)
        end_line = min(end_line, total)

        selected = all_lines[start_line - 1 : end_line]
        numbered = "\n".join(
            f"{start_line + i}: {line}" for i, line in enumerate(selected)
        )
        return json.dumps({
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": total,
            "content": numbered,
            "has_more": end_line < total,
        })

    def stat(self, path: str) -> str:
        """Get file or directory metadata.

        Args:
            path: Relative path to inspect.
        """
        target = self._resolve(path)
        if not target.exists():
            return json.dumps({"error": f"'{path}' does not exist"})

        st = target.stat()
        entry_type = "dir" if target.is_dir() else "file"
        modified = datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat()

        result: dict[str, object] = {
            "path": path,
            "type": entry_type,
            "size": st.st_size,
            "modified": modified,
        }
        if entry_type == "file" and st.st_size <= _MAX_SIZE_FOR_LINE_COUNT:
            try:
                result["lines"] = len(target.read_text(errors="replace").splitlines())
            except OSError:
                pass

        return json.dumps(result)

    def replace_lines(
        self,
        path: str,
        start_line: int,
        end_line: int,
        new_text: str,
    ) -> str:
        """Replace a range of lines in a file.

        Args:
            path: Relative path to the file.
            start_line: First line to replace (1-indexed, inclusive).
            end_line: Last line to replace (1-indexed, inclusive).
            new_text: Replacement text (may contain newlines).
        """
        target = self._resolve(path)
        if not target.is_file():
            return json.dumps({"error": f"'{path}' is not a file"})

        try:
            lines = target.read_text(errors="replace").splitlines(keepends=True)
        except OSError as exc:
            return json.dumps({"error": str(exc)})

        total = len(lines)
        if start_line < 1 or end_line < start_line or start_line > total:
            return json.dumps({
                "error": (
                    f"invalid range {start_line}-{end_line} "
                    f"for file with {total} lines"
                ),
            })
        end_line = min(end_line, total)

        old_count = end_line - start_line + 1
        new_lines = new_text.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        lines[start_line - 1 : end_line] = new_lines
        target.write_text("".join(lines))

        return json.dumps({
            "path": path,
            "old_lines": old_count,
            "new_lines": len(new_lines),
            "total_lines_after": len(lines),
        })

    @staticmethod
    def _is_binary(path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                chunk = f.read(_MAX_BINARY_CHECK)
        except OSError:
            return True
        return b"\x00" in chunk


CodebaseTools = FilesystemTools
