from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecursiveMemory:
    entries: list[MemoryEntry] = field(default_factory=list)

    def add(self, role: str, content: str, **metadata: Any) -> None:
        self.entries.append(MemoryEntry(role=role, content=content, metadata=metadata))

    def serialize(self, max_entries: int = 12) -> str:
        if not self.entries:
            return ""
        selected = self.entries[-max_entries:]
        lines: list[str] = []
        for entry in selected:
            meta = " ".join(f"{key}={value}" for key, value in entry.metadata.items())
            header = f"[{entry.role}{' ' + meta if meta else ''}]"
            lines.append(f"{header} {entry.content}")
        return "\n".join(lines)
