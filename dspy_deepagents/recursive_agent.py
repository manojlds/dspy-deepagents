from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import count

import dspy

from dspy_deepagents.memory import RecursiveMemory
from dspy_deepagents.roles import (
    ExecutorAgent,
    PlannerAgent,
    ReviewerAgent,
    SynthesizerAgent,
    ToolSelectorAgent,
)
from dspy_deepagents.tools import ToolRegistry
from dspy_deepagents.trace import TraceEvent


@dataclass(frozen=True)
class RecursionConfig:
    max_depth: int = 2
    max_children: int = 4
    budget: int = 10
    review_threshold: float = 0.7
    reflection_rounds: int = 1


class RecursiveAgent(dspy.Module):
    """Minimal recursive agent skeleton backed by DSPy RLM.

    This module wires together planner, executor, and synthesizer roles and uses
    recursion limits defined by RecursionConfig. RLM is used as a leaf executor
    by default, but can be overridden with a custom executor module.
    """

    def __init__(
        self,
        config: RecursionConfig | None = None,
        planner: PlannerAgent | None = None,
        executor: ExecutorAgent | None = None,
        reviewer: ReviewerAgent | None = None,
        synthesizer: SynthesizerAgent | None = None,
        tool_selector: dspy.Module | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RecursionConfig()
        self.planner = planner or PlannerAgent()
        self.executor = executor or ExecutorAgent()
        self.reviewer = reviewer or ReviewerAgent()
        self.synthesizer = synthesizer or SynthesizerAgent()
        self.tool_selector = tool_selector or ToolSelectorAgent()
        self.tools = tools or ToolRegistry()
        self._node_counter = count(1)

    def forward(
        self,
        task: str,
        context: str = "",
        memory: str = "",
        state: RecursiveMemory | None = None,
    ) -> dspy.Prediction:
        trace: list[TraceEvent] = []
        state = state or RecursiveMemory()
        result, confidence = self._run_node(
            task=task,
            context=context,
            memory=memory,
            state=state,
            depth=0,
            budget=self.config.budget,
            trace=trace,
        )
        return dspy.Prediction(
            result=result,
            confidence=confidence,
            trace=[event.to_dict() for event in trace],
            memory=state.serialize(),
        )

    def _run_node(
        self,
        task: str,
        context: str,
        memory: str,
        state: RecursiveMemory,
        depth: int,
        budget: int,
        trace: list[TraceEvent],
    ) -> tuple[str, float]:
        if depth >= self.config.max_depth or budget <= 0:
            return self._execute_leaf(task, context, memory, state, depth, trace)

        serialized_memory = self._serialize_memory(memory, state)
        plan = self.planner(task=task, context=context, memory=serialized_memory)
        sub_tasks = self._normalize_sub_tasks(plan.sub_tasks)
        sub_tasks = sub_tasks[: self.config.max_children]
        node_id = self._next_node_id(depth)
        state.add("planner", f"Subtasks: {sub_tasks}", task=task, depth=depth)
        trace.append(
            TraceEvent(
                event="plan",
                depth=depth,
                node_id=node_id,
                task=task,
                data={"sub_tasks": sub_tasks},
            )
        )

        if not sub_tasks:
            return self._execute_leaf(task, context, memory, state, depth, trace)

        child_results: list[str] = []
        child_confidences: list[float] = []
        for sub_task in sub_tasks:
            child_result, child_confidence = self._run_node(
                task=sub_task,
                context=context,
                memory=memory,
                state=state,
                depth=depth + 1,
                budget=budget - 1,
                trace=trace,
            )
            child_results.append(child_result)
            child_confidences.append(child_confidence)

        synthesis = self.synthesizer(
            task=task, child_results=child_results, memory=serialized_memory
        )
        state.add("synthesizer", synthesis.result, task=task, depth=depth)
        trace.append(
            TraceEvent(
                event="synthesize",
                depth=depth,
                node_id=node_id,
                task=task,
                data={
                    "child_results": child_results,
                    "confidence": synthesis.confidence,
                },
            )
        )
        return self._review_and_refine(
            task=task,
            draft=synthesis.result,
            confidence=float(synthesis.confidence),
            context=context,
            memory=memory,
            state=state,
            depth=depth,
            trace=trace,
        )

    def _execute_leaf(
        self,
        task: str,
        context: str,
        memory: str,
        state: RecursiveMemory,
        depth: int,
        trace: list[TraceEvent],
    ) -> tuple[str, float]:
        serialized_memory = self._serialize_memory(memory, state)
        tool_result = ""
        if self.tools.names():
            selection = self.tool_selector(
                task=task,
                context=context,
                memory=serialized_memory,
                tools=self.tools.describe(),
            )
            use_tool = str(selection.use_tool).strip().lower() == "yes"
            tool_name = str(selection.tool_name).strip()
            if use_tool and tool_name:
                tool_input = str(selection.tool_input).strip()
                tool_result = self.tools.run(tool_name, tool_input)
                state.add(
                    "tool",
                    f"{tool_name}({tool_input}) -> {tool_result}",
                    task=task,
                    depth=depth,
                )
                trace.append(
                    TraceEvent(
                        event="tool",
                        depth=depth,
                        node_id=self._next_node_id(depth),
                        task=task,
                        data={
                            "tool_name": tool_name,
                            "tool_input": tool_input,
                            "tool_result": tool_result,
                        },
                    )
                )

        execution = self.executor(
            task=task,
            context=context,
            memory=serialized_memory,
            tool_result=tool_result,
        )
        state.add("executor", execution.result, task=task, depth=depth)
        trace.append(
            TraceEvent(
                event="execute",
                depth=depth,
                node_id=self._next_node_id(depth),
                task=task,
                data={
                    "result": execution.result,
                    "confidence": execution.confidence,
                },
            )
        )
        return self._review_and_refine(
            task=task,
            draft=execution.result,
            confidence=float(execution.confidence),
            context=context,
            memory=memory,
            state=state,
            depth=depth,
            trace=trace,
        )

    @staticmethod
    def _normalize_sub_tasks(sub_tasks: Iterable[str] | str | None) -> list[str]:
        if sub_tasks is None:
            return []
        if isinstance(sub_tasks, str):
            return [line.strip() for line in sub_tasks.split("\n") if line.strip()]
        return [task for task in sub_tasks if task]

    def _review_and_refine(
        self,
        task: str,
        draft: str,
        confidence: float,
        context: str,
        memory: str,
        state: RecursiveMemory,
        depth: int,
        trace: list[TraceEvent],
    ) -> tuple[str, float]:
        if confidence >= self.config.review_threshold:
            return draft, confidence

        serialized_memory = self._serialize_memory(memory, state)
        current_result = draft
        current_confidence = confidence
        for round_index in range(self.config.reflection_rounds):
            review = self.reviewer(
                task=task,
                draft=current_result,
                context=context,
                memory=serialized_memory,
            )
            trace.append(
                TraceEvent(
                    event="review",
                    depth=depth,
                    node_id=self._next_node_id(depth),
                    task=task,
                    data={
                        "critique": review.critique,
                        "should_retry": review.should_retry,
                        "confidence": review.confidence,
                        "round": round_index + 1,
                    },
                )
            )
            state.add(
                "reviewer",
                review.critique,
                task=task,
                depth=depth,
                round=round_index + 1,
            )
            should_retry = str(review.should_retry).strip().lower() == "yes"
            current_confidence = float(review.confidence)
            if not should_retry:
                break
            refinement_context = (
                f"{context}\n\nReviewer feedback:\n{review.critique}"
                if review.critique
                else context
            )
            execution = self.executor(
                task=task,
                context=refinement_context,
                memory=serialized_memory,
                tool_result="",
            )
            current_result = execution.result
            current_confidence = float(execution.confidence)
            trace.append(
                TraceEvent(
                    event="refine",
                    depth=depth,
                    node_id=self._next_node_id(depth),
                    task=task,
                    data={
                        "result": current_result,
                        "confidence": current_confidence,
                        "round": round_index + 1,
                    },
                )
            )
            if current_confidence >= self.config.review_threshold:
                break

        return current_result, current_confidence

    def _serialize_memory(self, memory: str, state: RecursiveMemory) -> str:
        state_text = state.serialize()
        if memory and state_text:
            return f"{memory}\n{state_text}"
        return memory or state_text

    def _next_node_id(self, depth: int) -> str:
        return f"{depth}-{next(self._node_counter)}"
