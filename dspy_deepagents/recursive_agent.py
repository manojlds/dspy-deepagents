from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import dspy
from dspy.predict import RLM


class PlanSignature(dspy.Signature):
    """Plan child tasks for a recursive agent node."""

    task: str = dspy.InputField(desc="Primary task to solve.")
    context: str = dspy.InputField(desc="Additional context for planning.")
    memory: str = dspy.InputField(desc="Serialized memory/state passed through recursion.")
    sub_tasks: list[str] = dspy.OutputField(desc="List of child tasks to delegate.")


class ExecuteSignature(dspy.Signature):
    """Execute a leaf task (no further recursion)."""

    task: str = dspy.InputField(desc="Leaf task to solve.")
    context: str = dspy.InputField(desc="Additional context for execution.")
    memory: str = dspy.InputField(desc="Serialized memory/state passed through recursion.")
    result: str = dspy.OutputField(desc="Result of executing the task.")
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1.")


class SynthesizeSignature(dspy.Signature):
    """Synthesize child results into a final answer."""

    task: str = dspy.InputField(desc="Parent task to solve.")
    child_results: list[str] = dspy.InputField(desc="Child outputs to merge.")
    result: str = dspy.OutputField(desc="Final synthesized result.")
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1.")


@dataclass(frozen=True)
class RecursionConfig:
    max_depth: int = 2
    max_children: int = 4
    budget: int = 10


class RecursiveAgent(dspy.Module):
    """Minimal recursive agent skeleton backed by DSPy RLM.

    This module wires together planner, executor, and synthesizer roles and uses
    recursion limits defined by RecursionConfig. RLM is used as a leaf executor
    by default, but can be overridden with a custom executor module.
    """

    def __init__(
        self,
        config: RecursionConfig | None = None,
        planner: dspy.Module | None = None,
        executor: dspy.Module | None = None,
        synthesizer: dspy.Module | None = None,
        rlm: RLM | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RecursionConfig()
        self.planner = planner or dspy.Predict(PlanSignature)
        self.executor = executor or dspy.Predict(ExecuteSignature)
        self.synthesizer = synthesizer or dspy.Predict(SynthesizeSignature)
        self.rlm = rlm or RLM("context, query -> output", max_iterations=10)

    def forward(
        self,
        task: str,
        context: str = "",
        memory: str = "",
    ) -> dspy.Prediction:
        trace: list[dict[str, str | int | float | list[str]]] = []
        result, confidence = self._run_node(
            task=task,
            context=context,
            memory=memory,
            depth=0,
            budget=self.config.budget,
            trace=trace,
        )
        return dspy.Prediction(result=result, confidence=confidence, trace=trace)

    def _run_node(
        self,
        task: str,
        context: str,
        memory: str,
        depth: int,
        budget: int,
        trace: list[dict[str, str | int | float | list[str]]],
    ) -> tuple[str, float]:
        if depth >= self.config.max_depth or budget <= 0:
            return self._execute_leaf(task, context, memory, depth, trace)

        plan = self.planner(task=task, context=context, memory=memory)
        sub_tasks = self._normalize_sub_tasks(plan.sub_tasks)
        sub_tasks = sub_tasks[: self.config.max_children]
        trace.append(
            {
                "event": "plan",
                "depth": depth,
                "task": task,
                "sub_tasks": sub_tasks,
            }
        )

        if not sub_tasks:
            return self._execute_leaf(task, context, memory, depth, trace)

        child_results: list[str] = []
        child_confidences: list[float] = []
        for sub_task in sub_tasks:
            child_result, child_confidence = self._run_node(
                task=sub_task,
                context=context,
                memory=memory,
                depth=depth + 1,
                budget=budget - 1,
                trace=trace,
            )
            child_results.append(child_result)
            child_confidences.append(child_confidence)

        synthesis = self.synthesizer(task=task, child_results=child_results)
        trace.append(
            {
                "event": "synthesize",
                "depth": depth,
                "task": task,
                "child_results": child_results,
                "confidence": synthesis.confidence,
            }
        )
        return synthesis.result, float(synthesis.confidence)

    def _execute_leaf(
        self,
        task: str,
        context: str,
        memory: str,
        depth: int,
        trace: list[dict[str, str | int | float | list[str]]],
    ) -> tuple[str, float]:
        rlm_result = self.rlm(context=context, query=task)
        result = rlm_result.output
        trace.append(
            {
                "event": "execute",
                "depth": depth,
                "task": task,
                "result": result,
            }
        )
        return result, 0.5

    @staticmethod
    def _normalize_sub_tasks(sub_tasks: Iterable[str] | str | None) -> list[str]:
        if sub_tasks is None:
            return []
        if isinstance(sub_tasks, str):
            return [line.strip() for line in sub_tasks.split("\n") if line.strip()]
        return [task for task in sub_tasks if task]
