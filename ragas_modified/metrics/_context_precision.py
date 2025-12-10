from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ragas_modified.dataset_schema import SingleTurnSample
from ragas_modified.metrics._string import NonLLMStringSimilarity
from ragas_modified.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
    ensembler,
)
from ragas_modified.prompt import PydanticPrompt
from ragas_modified.run_config import RunConfig
from ragas_modified.utils import deprecated

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class QAC(BaseModel):
    question: str = Field(..., description="Question")
    context: str = Field(..., description="Context")
    answer: str = Field(..., description="Answer")


class Verification(BaseModel):
    reason: str = Field(..., description="Reason for verification")
    verdict: int = Field(..., description="Binary (0/1) verdict of verification")


class ContextPrecisionPrompt(PydanticPrompt[QAC, Verification]):
    name: str = "context_precision"
    instruction: str = (
        'Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not with json output.'
    )
    input_model = QAC
    output_model = Verification
    examples = [
        (
            QAC(
                question="How often should children with type 1 diabetes be screened for thyroid disease?",
                context="Offer children and young people with type 1 diabetes monitoring for: • thyroid disease, at diagnosis and then annually until transfer to adult services • moderately increased albuminuria (albumin to creatinine ratio [ACR] 3 mg/mmol to 30 mg/mmol) to detect diabetic kidney disease, annually from 12 years • hypertension, annually from 12 years.",
                answer="Thyroid disease should be screened at diagnosis and then annually until transfer to adult services.",
            ),
            Verification(
                reason="The context directly states the recommended screening frequency for thyroid disease, which matches the frequency in the response provided.",
                verdict=1,
            ),
        ),
        (
            QAC(
                question="When should antihypertensive drug treatment be considered for people aged over 80 with stage 1 hypertension?",
                context="Consider antihypertensive drug treatment in addition to lifestyle advice for adults aged under 60 with stage 1 hypertension and an estimated 10-year risk below 10%. Bear in mind that 10-year cardiovascular risk may underestimate the lifetime probability of developing cardiovascular disease. Consider antihypertensive drug treatment in addition to lifestyle advice for people aged over 80 with stage 1 hypertension if their clinic blood pressure is over 150/90 mmHg. Use clinical judgement for people with frailty or multimorbidity.",
                answer="For people aged over 80 with stage 1 hypertension, antihypertensive drug treatment should be considered if their clinic blood pressure is over 150/90 mmHg.",
            ),
            Verification(
                reason="The context explicitly states that for people aged over 80 with stage 1 hypertension, antihypertensive drug treatment should be considered if their clinic blood pressure is over 150/90 mmHg, which matches the answer provided.",
                verdict=1,
            ),
        ),
        (
            QAC(
                question="Is atomoxetine recommended as the first line pharmacological treatment for children aged 5 years and over with ADHD?",
                context="Offer methylphenidate (either short or long acting) as the first line pharmacological treatment for children aged 5 years and over and young people with ADHD. Consider switching to lisdexamfetamine for children aged 5 years and over and young people who have had a 6-week trial of methylphenidate at an adequate dose and not derived enough benefit in terms of reduced ADHD symptoms and associated impairment.",
                answer="Atomoxetine is recommended as the first line pharmacological treatment for children aged 5 years and over with ADHD.",
            ),
            Verification(
                reason="The context recommends methylphenidate as the first line treatment for children aged 5 years and over with ADHD, not atomoxetine. Therefore, the context does not support the answer provided.",
                verdict=0,
            ),
        ),
    ]


@dataclass
class LLMContextPrecisionWithReference(MetricWithLLM, SingleTurnMetric):
    """
    Average Precision is a metric that evaluates whether all of the
    relevant items selected by the model are ranked higher or not.

    Attributes
    ----------
    name : str
    evaluation_mode: EvaluationMode
    context_precision_prompt: Prompt
    """

    name: str = "llm_context_precision_with_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "retrieved_contexts",
                "reference",
            }
        }
    )
    output_type = MetricOutputType.CONTINUOUS
    context_precision_prompt: PydanticPrompt = field(
        default_factory=ContextPrecisionPrompt
    )
    max_retries: int = 1

    def _get_row_attributes(self, row: t.Dict) -> t.Tuple[str, t.List[str], t.Any]:
        return row["user_input"], row["retrieved_contexts"], row["reference"]

    def _calculate_average_precision(
        self, verifications: t.List[Verification]
    ) -> float:
        score = np.nan

        verdict_list = [1 if ver.verdict else 0 for ver in verifications]
        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator
        if np.isnan(score):
            logger.warning(
                "Invalid response format. Expected a list of dictionaries with keys 'verdict'"
            )
        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(
        self,
        row: t.Dict,
        callbacks: Callbacks,
    ) -> float:
        assert self.llm is not None, "LLM is not set"

        user_input, retrieved_contexts, reference = self._get_row_attributes(row)
        responses = []
        for context in retrieved_contexts:
            verdicts: t.List[Verification] = (
                await self.context_precision_prompt.generate_multiple(
                    data=QAC(
                        question=user_input,
                        context=context,
                        answer=reference,
                    ),
                    llm=self.llm,
                    callbacks=callbacks,
                )
            )

            responses.append([result.model_dump() for result in verdicts])

        answers = []
        for response in responses:
            agg_answer = ensembler.from_discrete([response], "verdict")
            answers.append(Verification(**agg_answer[0]))

        score = self._calculate_average_precision(answers)
        return score


@dataclass
class LLMContextPrecisionWithoutReference(LLMContextPrecisionWithReference):
    name: str = "llm_context_precision_without_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )

    def _get_row_attributes(self, row: t.Dict) -> t.Tuple[str, t.List[str], t.Any]:
        return row["user_input"], row["retrieved_contexts"], row["response"]


@dataclass
class NonLLMContextPrecisionWithReference(SingleTurnMetric):
    name: str = "non_llm_context_precision_with_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_contexts",
                "reference_contexts",
            }
        }
    )
    distance_measure: SingleTurnMetric = field(
        default_factory=lambda: NonLLMStringSimilarity()
    )
    threshold: float = 0.5

    def __post_init__(self):
        if isinstance(self.distance_measure, MetricWithLLM):
            raise ValueError(
                "distance_measure must not be an instance of MetricWithLLM for NonLLMContextPrecisionWithReference"
            )

    def init(self, run_config: RunConfig) -> None: ...

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        sample = SingleTurnSample(**row)
        return await self._single_turn_ascore(sample, callbacks)

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved_contexts = sample.retrieved_contexts
        reference_contexts = sample.reference_contexts
        assert retrieved_contexts is not None, "retrieved_contexts is empty"
        assert reference_contexts is not None, "reference_contexts is empty"

        scores = []
        for rc in retrieved_contexts:
            scores.append(
                max(
                    [
                        await self.distance_measure.single_turn_ascore(
                            SingleTurnSample(reference=rc, response=ref), callbacks
                        )
                        for ref in reference_contexts
                    ]
                )
            )
        scores = [1 if score >= self.threshold else 0 for score in scores]
        return self._calculate_average_precision(scores)

    def _calculate_average_precision(self, verdict_list: t.List[int]) -> float:
        score = np.nan

        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator
        return score


@dataclass
class ContextPrecision(LLMContextPrecisionWithReference):
    name: str = "context_precision"

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        return await super()._single_turn_ascore(sample, callbacks)

    @deprecated(
        since="0.2", removal="0.3", alternative="LLMContextPrecisionWithReference"
    )
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


@dataclass
class ContextUtilization(LLMContextPrecisionWithoutReference):
    name: str = "context_utilization"

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        return await super()._single_turn_ascore(sample, callbacks)

    @deprecated(
        since="0.2", removal="0.3", alternative="LLMContextPrecisionWithoutReference"
    )
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


context_precision = ContextPrecision()
context_utilization = ContextUtilization()
