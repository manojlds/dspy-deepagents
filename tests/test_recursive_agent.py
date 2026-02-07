import dspy

from dspy_deepagents.memory import RecursiveMemory
from dspy_deepagents.recursive_agent import RecursionConfig, RecursiveAgent
from dspy_deepagents.roles import ExecutorAgent, PlannerAgent, ReviewerAgent, SynthesizerAgent


class StaticPlanner(PlannerAgent):
    def forward(self, task: str, context: str, memory: str) -> dspy.Prediction:
        return dspy.Prediction(sub_tasks=["child-task"])


class StaticSynthesizer(SynthesizerAgent):
    def forward(
        self, task: str, child_results: list[str], memory: str
    ) -> dspy.Prediction:
        return dspy.Prediction(result="synth-result", confidence=0.6)


class StaticReviewer(ReviewerAgent):
    def forward(
        self, task: str, draft: str, context: str, memory: str
    ) -> dspy.Prediction:
        return dspy.Prediction(critique="", should_retry="no", confidence=0.9)


class StaticExecutor(ExecutorAgent):
    def forward(
        self, task: str, context: str, memory: str, tool_result: str = ""
    ) -> dspy.Prediction:
        return dspy.Prediction(result="leaf-result", confidence=0.4)


def test_recursive_agent_records_trace() -> None:
    agent = RecursiveAgent(
        config=RecursionConfig(max_depth=1, budget=2, review_threshold=0.5),
        planner=StaticPlanner(),
        executor=StaticExecutor(),
        reviewer=StaticReviewer(),
        synthesizer=StaticSynthesizer(),
    )
    memory = RecursiveMemory()

    result = agent(task="root", context="", state=memory)

    assert result.result == "synth-result"
    assert result.trace
    assert "memory" in result
