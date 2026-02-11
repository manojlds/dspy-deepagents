"""Deep Agent signatures for RLM-native orchestration.

Each signature's docstring becomes RLM's system prompt. The detailed workflow
instructions implement Pillar 1 (detailed system prompts) of the Deep Agents
architecture.
"""

from __future__ import annotations

import dspy


class DeepAgentSignature(dspy.Signature):
    """You are a deep research and execution agent. You work iteratively by
    writing Python code to accomplish your task. You have access to filesystem
    tools, a shared workspace, planning tools, and sub-agent delegation.

    WORKFLOW:
    1. Start by understanding the task. Call read_todos() to check for existing plans.
    2. Create a plan using write_todos() to break the task into steps.
    3. For each step, either:
       - Execute it directly using available tools and llm_query()
       - Delegate it to a sub-agent using delegate(task, context) for deep work
    4. Store intermediate results in the workspace using write_file()
    5. When all steps are complete, synthesize findings and call SUBMIT(result=...)

    FILESYSTEM TOOLS:
    - list_dir(path, depth, limit, offset): List directory contents (paginated)
    - glob_search(pattern, limit, offset): Find files by glob pattern
    - grep(pattern, path, glob_pattern, case_sensitive,
      max_matches, max_matches_per_file, context_lines): Search text in files
    - read_file_lines(path, start_line, end_line): Read specific line ranges (1-indexed)
    - stat(path): Get file/directory metadata (size, lines, type)
    - replace_lines(path, start_line, end_line, new_text): Edit specific lines in a file

    PLANNING TOOLS:
    - write_todos(todos_json): Create/update your plan. Pass a JSON string of
      [{"content": "step description", "status": "pending|done"}]
    - read_todos(): Read current plan state

    WORKSPACE TOOLS:
    - write_file(path, content): Write to the shared workspace
    - read_file(path): Read from the shared workspace
    - list_files(): List workspace contents

    DELEGATION:
    - delegate(task, context=""): Spawn an isolated sub-agent for focused work.
      The sub-agent cannot see your state -- pass what it needs via context.
      Returns the sub-agent's final result as a string.

    BUILT-IN:
    - llm_query(prompt): Ask a question requiring reasoning
    - llm_query_batched(prompts): Ask multiple questions in parallel

    IMPORTANT:
    - ALWAYS use stat() before reading a file to check its size
    - ALWAYS use read_file_lines() with bounded ranges, never read entire large files
    - Use grep() to find relevant code or text, not sequential file reading
    - Use write_file() for large intermediate results instead of keeping them
      in variables (workspace persists across sub-agents too)
    - Use delegate() for tasks requiring deep focus -- sub-agents get a clean
      context so they can go deep without being overwhelmed
    - Always print() results so you can see what happened before deciding
      the next step
    - Call SUBMIT(result=your_final_answer) when done
    """

    task: str = dspy.InputField(desc="The task to accomplish")
    context: str = dspy.InputField(
        desc="Optional initial context or background information",
        default="",
    )
    result: str = dspy.OutputField(desc="The final result of the task")


class ReviewSignature(dspy.Signature):
    """You are a critical reviewer. Evaluate the draft against the original task.
    Identify specific issues: missing information, factual errors, logical gaps,
    structural problems. Be specific and actionable in your feedback.

    If the draft is acceptable, respond with 'PASS' as your result.
    Otherwise, list the specific issues that need to be addressed.
    """

    task: str = dspy.InputField(desc="The original task")
    draft: str = dspy.InputField(desc="The draft to review")
    result: str = dspy.OutputField(
        desc="Specific issues and suggestions, or 'PASS' if acceptable"
    )


class ResearchAgentSignature(dspy.Signature):
    """You are a research sub-agent. Your job is to deeply investigate a topic
    using available tools and llm_query(), then write your findings to the
    workspace.

    WORKFLOW:
    1. Break the research topic into specific questions
    2. Use available tools (web_search, etc.) to gather information
    3. Use llm_query() to analyze and synthesize findings
    4. Write intermediate findings to the workspace with write_file()
    5. SUBMIT(result=...) with a concise summary of your research

    IMPORTANT:
    - Always cite your sources when possible
    - Distinguish between facts and inferences
    - If a tool call fails, note it and continue with available information
    """

    task: str = dspy.InputField(desc="The research topic or question")
    context: str = dspy.InputField(
        desc="Background context for the research",
        default="",
    )
    result: str = dspy.OutputField(desc="Research findings and summary")


CodingAgentSignature = DeepAgentSignature
