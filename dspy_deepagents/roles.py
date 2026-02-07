from __future__ import annotations

import dspy
from dspy.predict import RLM


class PlanSignature(dspy.Signature):
    """Plan child tasks for a recursive agent node."""

    task: str = dspy.InputField(desc="Primary task to solve.")
    context: str = dspy.InputField(desc="Additional context for planning.")
    memory: str = dspy.InputField(
        desc="Serialized memory/state passed through recursion."
    )
    sub_tasks: list[str] = dspy.OutputField(desc="List of child tasks to delegate.")


class ExecuteSignature(dspy.Signature):
    """Execute a leaf task (no further recursion)."""

    task: str = dspy.InputField(desc="Leaf task to solve.")
    context: str = dspy.InputField(desc="Additional context for execution.")
    memory: str = dspy.InputField(
        desc="Serialized memory/state passed through recursion."
    )
    tool_result: str = dspy.InputField(desc="Result from a tool call, if used.")
    result: str = dspy.OutputField(desc="Result of executing the task.")
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1.")


class SynthesizeSignature(dspy.Signature):
    """Synthesize child results into a final answer."""

    task: str = dspy.InputField(desc="Parent task to solve.")
    child_results: list[str] = dspy.InputField(desc="Child outputs to merge.")
    memory: str = dspy.InputField(
        desc="Serialized memory/state passed through recursion."
    )
    result: str = dspy.OutputField(desc="Final synthesized result.")
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1.")


class ReviewSignature(dspy.Signature):
    """Review a draft answer and decide whether to retry."""

    task: str = dspy.InputField(desc="Task being solved.")
    draft: str = dspy.InputField(desc="Current draft answer.")
    context: str = dspy.InputField(desc="Additional context for review.")
    memory: str = dspy.InputField(
        desc="Serialized memory/state passed through recursion."
    )
    critique: str = dspy.OutputField(desc="Critique or improvement guidance.")
    should_retry: str = dspy.OutputField(desc="'yes' if a retry is needed, else 'no'.")
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1.")


class ToolSelectSignature(dspy.Signature):
    """Select a tool for leaf execution when available."""

    task: str = dspy.InputField(desc="Task to solve.")
    context: str = dspy.InputField(desc="Additional context.")
    memory: str = dspy.InputField(
        desc="Serialized memory/state passed through recursion."
    )
    tools: str = dspy.InputField(desc="Available tools and descriptions.")
    use_tool: str = dspy.OutputField(desc="'yes' if a tool should be used, else 'no'.")
    tool_name: str = dspy.OutputField(desc="Name of the selected tool.")
    tool_input: str = dspy.OutputField(desc="Input to pass to the tool.")


class PlannerAgent(dspy.Module):
    def __init__(
        self,
        predictor: dspy.Module | None = None,
        rlm: RLM | None = None,
        use_rlm: bool = True,
    ) -> None:
        super().__init__()
        if predictor is not None:
            self.predictor = predictor
        elif use_rlm:
            self.predictor = rlm or RLM(PlanSignature, max_iterations=10)
        else:
            self.predictor = dspy.Predict(PlanSignature)

    def forward(self, task: str, context: str, memory: str) -> dspy.Prediction:
        return self.predictor(task=task, context=context, memory=memory)


class ExecutorAgent(dspy.Module):
    def __init__(
        self,
        predictor: dspy.Module | None = None,
        rlm: RLM | None = None,
        use_rlm: bool = True,
    ) -> None:
        super().__init__()
        if predictor is not None:
            self.predictor = predictor
        elif use_rlm:
            self.predictor = rlm or RLM(ExecuteSignature, max_iterations=10)
        else:
            self.predictor = dspy.Predict(ExecuteSignature)

    def forward(
        self,
        task: str,
        context: str,
        memory: str,
        tool_result: str = "",
    ) -> dspy.Prediction:
        return self.predictor(
            task=task,
            context=context,
            memory=memory,
            tool_result=tool_result,
        )


class ReviewerAgent(dspy.Module):
    def __init__(
        self,
        predictor: dspy.Module | None = None,
        rlm: RLM | None = None,
        use_rlm: bool = True,
    ) -> None:
        super().__init__()
        if predictor is not None:
            self.predictor = predictor
        elif use_rlm:
            self.predictor = rlm or RLM(ReviewSignature, max_iterations=10)
        else:
            self.predictor = dspy.Predict(ReviewSignature)

    def forward(
        self, task: str, draft: str, context: str, memory: str
    ) -> dspy.Prediction:
        return self.predictor(task=task, draft=draft, context=context, memory=memory)


class SynthesizerAgent(dspy.Module):
    def __init__(
        self,
        predictor: dspy.Module | None = None,
        rlm: RLM | None = None,
        use_rlm: bool = True,
    ) -> None:
        super().__init__()
        if predictor is not None:
            self.predictor = predictor
        elif use_rlm:
            self.predictor = rlm or RLM(SynthesizeSignature, max_iterations=10)
        else:
            self.predictor = dspy.Predict(SynthesizeSignature)

    def forward(
        self, task: str, child_results: list[str], memory: str
    ) -> dspy.Prediction:
        return self.predictor(task=task, child_results=child_results, memory=memory)


class ToolSelectorAgent(dspy.Module):
    def __init__(
        self,
        predictor: dspy.Module | None = None,
        rlm: RLM | None = None,
        use_rlm: bool = True,
    ) -> None:
        super().__init__()
        if predictor is not None:
            self.predictor = predictor
        elif use_rlm:
            self.predictor = rlm or RLM(ToolSelectSignature, max_iterations=6)
        else:
            self.predictor = dspy.Predict(ToolSelectSignature)

    def forward(
        self, task: str, context: str, memory: str, tools: str
    ) -> dspy.Prediction:
        return self.predictor(
            task=task,
            context=context,
            memory=memory,
            tools=tools,
        )
