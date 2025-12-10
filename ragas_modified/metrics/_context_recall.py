from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas_modified.dataset_schema import SingleTurnSample
from ragas_modified.metrics._string import DistanceMeasure, NonLLMStringSimilarity
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


class QCA(BaseModel):
    question: str
    context: str
    answer: str


class ContextRecallClassification(BaseModel):
    statement: str
    reason: str
    attributed: int


class ContextRecallClassifications(BaseModel):
    classifications: t.List[ContextRecallClassification]


class ContextRecallClassificationPrompt(
    PydanticPrompt[QCA, ContextRecallClassifications]
):
    name: str = "context_recall_classification"
    instruction: str = (
        "Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Use only 'Yes' (1) or 'No' (0) as a binary classification. Output json with reason."
    )
    input_model = QCA
    output_model = ContextRecallClassifications
    examples = [
        (
            QCA(
                question="What are risk factors for ADHD?",
                context="People born preterm may have increased prevalence of ADHD compared with the general population. ADHD is thought to be under-recognised in girls and women. Universal screening for ADHD should not be undertaken in schools.",
                answer="Children born preterm are at higher risk for ADHD. Girls may be under-recognised for ADHD. Family history is a major risk factor for ADHD.",
            ),
            ContextRecallClassifications(
                classifications=[
                    ContextRecallClassification(
                        statement="Children born preterm are at higher risk for ADHD.",
                        reason="This statement directly matches the first sentence.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement="Girls may be under-recognised for ADHD.",
                        reason="This statement directly matches the second sentence.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement="Family history is a major risk factor for ADHD.",
                        reason="There is no mention of family history as a risk factor in any sentence in the context.",
                        attributed=0,
                    ),
                ]
            ),
        ),
    ]


@dataclass
class LLMContextRecall(MetricWithLLM, SingleTurnMetric):
    """
    Estimates context recall by estimating TP and FN using annotated answer and
    retrieved context.

    Attributes
    ----------
    name : str
    """

    name: str = "context_recall"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "retrieved_contexts",
                "reference",
            }
        }
    )
    output_type: t.Optional[MetricOutputType] = MetricOutputType.CONTINUOUS
    context_recall_prompt: PydanticPrompt = field(
        default_factory=ContextRecallClassificationPrompt
    )
    max_retries: int = 1

    def _compute_score(self, responses: t.List[ContextRecallClassification]) -> float:
        response = [1 if item.attributed else 0 for item in responses]
        denom = len(response)
        numerator = sum(response)
        score = numerator / denom if denom > 0 else np.nan

        if np.isnan(score):
            logger.warning("The LLM did not return a valid classification.")

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "set LLM before use"

        # run classification
        classifications_list: t.List[ContextRecallClassifications] = (
            await self.context_recall_prompt.generate_multiple(
                data=QCA(
                    question=row["user_input"],
                    context="\n".join(row["retrieved_contexts"]),
                    answer=row["reference"],
                ),
                llm=self.llm,
                callbacks=callbacks,
            )
        )
        classification_dicts = []
        for classification in classifications_list:
            classification_dicts.append(
                [clasif.model_dump() for clasif in classification.classifications]
            )

        ensembled_clasif = ensembler.from_discrete(classification_dicts, "attributed")

        return self._compute_score(
            [ContextRecallClassification(**clasif) for clasif in ensembled_clasif]
        )


@dataclass
class ContextRecall(LLMContextRecall):
    name: str = "context_recall"

    @deprecated(since="0.2", removal="0.3", alternative="LLMContextRecall")
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    @deprecated(since="0.2", removal="0.3", alternative="LLMContextRecall")
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


@dataclass
class NonLLMContextRecall(SingleTurnMetric):
    name: str = "non_llm_context_recall"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_contexts",
                "reference_contexts",
            }
        }
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS
    _distance_measure: SingleTurnMetric = field(
        default_factory=lambda: NonLLMStringSimilarity()
    )
    threshold: float = 0.5

    def init(self, run_config: RunConfig) -> None: ...

    @property
    def distance_measure(self) -> SingleTurnMetric:
        return self._distance_measure

    @distance_measure.setter
    def distance_measure(self, distance_measure: DistanceMeasure) -> None:
        self._distance_measure = NonLLMStringSimilarity(
            distance_measure=distance_measure
        )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved_contexts = sample.retrieved_contexts
        reference_contexts = sample.reference_contexts
        assert retrieved_contexts is not None, "retrieved_contexts is empty"
        assert reference_contexts is not None, "reference_contexts is empty"

        scores = []
        for ref in reference_contexts:
            scores.append(
                max(
                    [
                        await self.distance_measure.single_turn_ascore(
                            SingleTurnSample(reference=rc, response=ref), callbacks
                        )
                        for rc in retrieved_contexts
                    ]
                )
            )
        return self._compute_score(scores)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)

    def _compute_score(self, verdict_list: t.List[float]) -> float:
        response = [1 if score > self.threshold else 0 for score in verdict_list]
        denom = len(response)
        numerator = sum(response)
        score = numerator / denom if denom > 0 else np.nan
        return score


context_recall = ContextRecall()
