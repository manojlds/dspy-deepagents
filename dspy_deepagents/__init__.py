"""DSPy-native Deep Agents with RLM-backed orchestration.

The core API is ``build_deep_agent()``, which returns an RLM module
wired with tools implementing the four Deep Agents pillars:

1. **Detailed system prompts** -- Signature docstrings
2. **Planning tool** -- ``write_todos`` / ``read_todos``
3. **Sub-agent delegation** -- ``delegate()`` with context isolation
4. **Filesystem** -- ``list_dir`` / ``grep`` / ``glob_search`` /
   ``read_file_lines`` / ``stat`` / ``replace_lines`` + workspace
   ``write_file`` / ``read_file`` / ``list_files``
"""

from dspy_deepagents.agent import build_coding_agent, build_deep_agent
from dspy_deepagents.codebase_tools import CodebaseTools, FilesystemTools
from dspy_deepagents.signature import (
    CodingAgentSignature,
    DeepAgentSignature,
    ResearchAgentSignature,
    ReviewSignature,
)
from dspy_deepagents.tools import (
    SubAgentDelegator,
    TodoStore,
    Workspace,
    make_review_tool,
)

__all__ = [
    "build_coding_agent",
    "build_deep_agent",
    "CodebaseTools",
    "CodingAgentSignature",
    "DeepAgentSignature",
    "FilesystemTools",
    "ResearchAgentSignature",
    "ReviewSignature",
    "SubAgentDelegator",
    "TodoStore",
    "Workspace",
    "make_review_tool",
]
