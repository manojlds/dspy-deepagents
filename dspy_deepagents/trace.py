from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    event: str
    depth: int
    node_id: str
    task: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "depth": self.depth,
            "node_id": self.node_id,
            "task": self.task,
            **self.data,
        }
