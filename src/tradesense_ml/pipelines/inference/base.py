"""Base abstractions for Teacher Inference pipelines and execution strategies."""

from abc import ABC, abstractmethod
from typing import Any

from tradesense_ml.domain.schemas.coaching import CoachRequest, CoachResponse
from tradesense_ml.domain.schemas.teacher import RenderedPrompt, TeacherResponse
from tradesense_ml.pipelines.base import BasePipeline
from tradesense_ml.teachers.router import TeacherRouter


class BaseInferenceStrategy(ABC):
    """Abstract interface for teacher inference strategies (e.g. single_teacher, multi_teacher, consensus, debate)."""

    @abstractmethod
    def execute(
        self,
        request: CoachRequest,
        rendered_prompt: RenderedPrompt,
        router: TeacherRouter,
        **kwargs: Any,
    ) -> TeacherResponse | list[TeacherResponse]:
        """Execute teacher inference across one or more providers according to the strategy."""
        pass


class BaseInferencePipeline(BasePipeline[CoachRequest, CoachResponse], ABC):
    """Abstract orchestrator interface for Teacher inference pipelines."""

    def __init__(self, pipeline_name: str = "teacher_inference_pipeline") -> None:
        super().__init__(pipeline_name=pipeline_name)

    @abstractmethod
    def run(self, input_data: CoachRequest, **kwargs: Any) -> CoachResponse:
        """Run teacher inference pipeline for a CoachRequest."""
        pass
