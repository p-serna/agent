from dataclasses import dataclass
from enum import Enum
from typing import List

from src.tools import tool_names


class TaskType(Enum):
    STATEMENT_COLLECTION = "Statement collection"
    ENTITY_IDENTIFICATION = "Entity identification"
    TIME_SERIES = "Time series"
    NUMERICAL_COMPARISON = "Numerical comparison"
    COMPLEX_FILTERING = "Complex filtering"


@dataclass
class TaskRequirement:
    source_verification: bool
    temporal_ordering: bool
    numerical_extraction: bool
    entity_resolution: bool
    structured_output: bool


class TaskAnalyzer:
    def analyze_task(self, task_description: str):
        task_type = self._analyze_task_type(task_description)
        task_requirements = self._analyze_task_requirements(task_description)
        return {
            "task_type": task_type,
            "task_requirements": task_requirements,
            "suggested_tools": self._suggest_tools(task_type, task_requirements),
            "validation_steps": self._determine_validation_steps(task_requirements),
        }

    def _analyze_task_type(self, task_description: str):
        # Check for keywords, classify or ask an LLM
        pass

    def _analyze_task_requirements(self, task_description: str):
        # Check with an LLM
        return TaskRequirement(*[False] * 5)

    def _suggest_tools(
        self, task_type, task_requirements: TaskRequirement
    ) -> List[str]:
        # Check for tools that match the requirements (ask LLM?)
        return tool_names

    def _determine_validation_steps(self, requirements: TaskRequirement) -> List[str]:
        # Based on requirements, suggest validation steps
        steps = []
        if requirements.source_verification:
            steps.append("Verify source authenticity")
            steps.append("Cross-reference information")
        if requirements.temporal_ordering:
            steps.append("Verify temporal consistency")
        if requirements.numerical_extraction:
            steps.append("Validate numerical data")
        if requirements.entity_resolution:
            steps.append("Verify entity relationships")
        return steps
