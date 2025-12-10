from __future__ import annotations

import asyncio
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas_modified.dataset_schema import SingleTurnSample
from ragas_modified.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas_modified.prompt import PydanticPrompt

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


class ResponseRelevanceOutput(BaseModel):
    question: str
    noncommittal: int


class ResponseRelevanceInput(BaseModel):
    response: str


class ResponseRelevancePrompt(
    PydanticPrompt[ResponseRelevanceInput, ResponseRelevanceOutput]
):
    instruction = """Given an answer based on NICE clinical guidelines, generate a relevant question that this answer addresses. Also, identify if the answer is noncommittal. Set 'noncommittal' to 1 if the answer is vague, evasive, or ambiguous (e.g., "No relevant NICE guidelines were found"), and 0 if the answer is committal."""
    input_model = ResponseRelevanceInput
    output_model = ResponseRelevanceOutput
    examples = [
        (
            ResponseRelevanceInput(
                response="""Offer methylphenidate (either short or long acting) as the first line pharmacological treatment for children aged 5 years and over and young people with ADHD.""",
            ),
            ResponseRelevanceOutput(
                question="What is the first-line pharmacological treatment for children aged 5 years and over with ADHD?",
                noncommittal=0,
            ),
        ),
        (
            ResponseRelevanceInput(
                response="""Advise children and young people with type 1 diabetes who are using capillary blood glucose monitoring (and their families or carers) to routinely perform at least 5 capillary blood glucose tests per day.""",
            ),
            ResponseRelevanceOutput(
                question="How many capillary blood glucose tests per day should children and young people with type 1 diabetes perform?",
                noncommittal=0,
            ),
        ),
        (
            ResponseRelevanceInput(
                response="""No relevant NICE guidelines were found to answer your question.""",
            ),
            ResponseRelevanceOutput(
                question="What is the first-line treatment for hypertension?",
                noncommittal=1,
            ),
        ),
    ]

@dataclass
class ResponseRelevancy(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Attributes
    ----------
    name: string
        The name of the metrics
    strictness: int
        Here indicates the number questions generated per answer.
        Ideal range between 3 to 5.
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
            }
        }
    )
    output_type = MetricOutputType.CONTINUOUS

    question_generation: PydanticPrompt = ResponseRelevancePrompt()
    strictness: int = 1

    def calculate_similarity(self, question: str, generated_questions: list[str]):
        assert (
            self.embeddings is not None
        ), f"Error: '{self.name}' requires embeddings to be set."
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(-1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)
        ).reshape(len(generated_questions), -1)
        return np.dot(gen_question_vec, question_vec)

    def _calculate_score(
        self, answers: t.Sequence[ResponseRelevanceOutput], row: t.Dict
    ) -> float:
        question = row["user_input"]
        gen_questions = [answer.question for answer in answers]
        committal = np.any([answer.noncommittal for answer in answers])
        if all(q == "" for q in gen_questions):
            logger.warning(
                "Invalid JSON response. Expected dictionary with key 'question'"
            )
            score = np.nan
        else:
            dot_prod = self.calculate_similarity(question, gen_questions)
            score = dot_prod.mean() * int(not committal)

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt_input = ResponseRelevanceInput(response=row["response"])
        tasks = [
            self.question_generation.generate(
                data=prompt_input,
                llm=self.llm,
                callbacks=callbacks,
            )
            for _ in range(self.strictness)
        ]
        responses = await asyncio.gather(*tasks)

        return self._calculate_score(responses, row)


class AnswerRelevancy(ResponseRelevancy):
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


answer_relevancy = AnswerRelevancy()
