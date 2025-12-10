from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ragas_modified.dataset_schema import SingleTurnSample
from ragas_modified.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas_modified.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class StatementGeneratorInput(BaseModel):
    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")


class StatementGeneratorOutput(BaseModel):
    statements: t.List[str] = Field(description="The generated statements")

class StatementGeneratorPrompt(
    PydanticPrompt[StatementGeneratorInput, StatementGeneratorOutput]
):
    instruction = (
        "Given a medical question and an answer, break down the answer into fully understandable statements relevant to clinical guidelines. Ensure that no pronouns are used in any statement. Format the outputs in JSON."
    )
    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput
    examples = [
        (
            StatementGeneratorInput(
                question="What are risk factors for ADHD?",
                answer="Children born preterm are at significantly higher risk for developing attention deficit hyperactivity disorder (ADHD) compared to those born at term. Girls may be under-recognised for ADHD due to differences in symptom presentation and referral patterns. Family history, especially having a close relative diagnosed with ADHD, is considered a major risk factor for developing the disorder.",
            ),
            StatementGeneratorOutput(
                statements=[
                    "Children born preterm are at higher risk for ADHD.",
                    "Girls may be under-recognised for ADHD.",
                    "Family history is a major risk factor for ADHD.",
                ]
            ),
        ),
    ]


class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")


class NLIStatementOutput(BaseModel):
    statements: t.List[StatementFaithfulnessAnswer]


class NLIStatementInput(BaseModel):
    context: str = Field(..., description="The context of the question")
    statements: t.List[str] = Field(..., description="The statements to judge")


class NLIStatementPrompt(PydanticPrompt[NLIStatementInput, NLIStatementOutput]):
    instruction = (
        "Your task is to judge the faithfulness of a series of statements based on a given medical context. For each statement, return verdict as 1 if the statement can be directly inferred from the context or 0 if the statement cannot be directly inferred from the context. Output JSON with reasoning."
    )
    input_model = NLIStatementInput
    output_model = NLIStatementOutput
    examples = [
        (
            NLIStatementInput(
                context="""Consider antihypertensive drug treatment in addition to lifestyle advice for people aged over 80 with stage 1 hypertension if their clinic blood pressure is over 150/90 mmHg. Use clinical judgement for people with frailty or multimorbidity.""",
                statements=[
                    "Antihypertensive drug treatment should be considered for people aged over 80 with stage 1 hypertension if their clinic blood pressure is over 150/90 mmHg.",
                    "Antihypertensive drug treatment should be considered for people aged over 80 with stage 1 hypertension regardless of blood pressure.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Antihypertensive drug treatment should be considered for people aged over 80 with stage 1 hypertension if their clinic blood pressure is over 150/90 mmHg.",
                        reason="This statement is directly supported by the context.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Antihypertensive drug treatment should be considered for people aged over 80 with stage 1 hypertension regardless of blood pressure.",
                        reason="The context specifies that treatment should be considered only if clinic blood pressure is over 150/90 mmHg.",
                        verdict=0,
                    ),
                ]
            ),
        ),
        (
            NLIStatementInput(
                context="Offer methylphenidate as the first line pharmacological treatment for children aged 5 years and over and young people with ADHD.",
                statements=[
                    "Methylphenidate is recommended as the first line pharmacological treatment for children aged 5 years and over with ADHD.",
                    "Atomoxetine is recommended as the first line pharmacological treatment for children aged 5 years and over with ADHD.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Methylphenidate is recommended as the first line pharmacological treatment for children aged 5 years and over with ADHD.",
                        reason="This statement is directly supported by the context.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Atomoxetine is recommended as the first line pharmacological treatment for children aged 5 years and over with ADHD.",
                        reason="The context recommends methylphenidate, not atomoxetine, as the first line treatment.",
                        verdict=0,
                    ),
                ]
            ),
        ),
    ]


@dataclass
class Faithfulness(MetricWithLLM, SingleTurnMetric):
    name: str = "faithfulness"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "retrieved_contexts",
            }
        }
    )
    output_type: t.Optional[MetricOutputType] = MetricOutputType.CONTINUOUS
    nli_statements_prompt: PydanticPrompt = field(default_factory=NLIStatementPrompt)
    statement_generator_prompt: PydanticPrompt = field(
        default_factory=StatementGeneratorPrompt
    )
    max_retries: int = 1

    async def _create_verdicts(
        self, row: t.Dict, statements: t.List[str], callbacks: Callbacks
    ) -> NLIStatementOutput:
        assert self.llm is not None, "llm must be set to compute score"

        contexts_str: str = "\n".join(row["retrieved_contexts"])
        verdicts = await self.nli_statements_prompt.generate(
            data=NLIStatementInput(context=contexts_str, statements=statements),
            llm=self.llm,
            callbacks=callbacks,
        )

        return verdicts

    async def _create_statements(
        self, row: t.Dict, callbacks: Callbacks
    ) -> StatementGeneratorOutput:
        assert self.llm is not None, "llm is not set"

        text, question = row["response"], row["user_input"]

        prompt_input = StatementGeneratorInput(question=question, answer=text)
        statements = await self.statement_generator_prompt.generate(
            llm=self.llm,
            data=prompt_input,
            callbacks=callbacks,
        )

        return statements

    def _compute_score(self, answers: NLIStatementOutput):
        # check the verdicts and compute the score
        faithful_statements = sum(
            1 if answer.verdict else 0 for answer in answers.statements
        )
        num_statements = len(answers.statements)
        if num_statements:
            score = faithful_statements / num_statements
        else:
            logger.warning("No statements were generated from the answer.")
            score = np.nan

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        statements = await self._create_statements(row, callbacks)
        statements = statements.statements
        if statements == []:
            return np.nan

        verdicts = await self._create_verdicts(row, statements, callbacks)
        return self._compute_score(verdicts)


@dataclass
class FaithfulnesswithHHEM(Faithfulness):
    name: str = "faithfulness_with_hhem"
    device: str = "cpu"
    batch_size: int = 10

    def __post_init__(self):
        try:
            from transformers import AutoModelForSequenceClassification  # type: ignore
        except ImportError:
            raise ImportError(
                "Huggingface transformers must be installed to use this feature, try `pip install transformers`"
            )
        self.nli_classifier = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True
        )
        self.nli_classifier.to(self.device)
        super().__post_init__()

    def _create_pairs(
        self, row: t.Dict, statements: t.List[str]
    ) -> t.List[t.Tuple[str, str]]:
        """
        create pairs of (question, answer) from the row
        """
        premise = "\n".join(row["retrieved_contexts"])
        pairs = [(premise, statement) for statement in statements]
        return pairs

    def _create_batch(
        self, pairs: t.List[t.Tuple[str, str]]
    ) -> t.Generator[t.List[t.Tuple[str, str]], None, None]:
        length_of_pairs = len(pairs)
        for ndx in range(0, length_of_pairs, self.batch_size):
            yield pairs[ndx : min(ndx + self.batch_size, length_of_pairs)]

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        statements = await self._create_statements(row, callbacks)
        statements = statements.statements
        if statements == []:
            return np.nan

        scores = []
        pairs = self._create_pairs(row, statements)
        for input_pairs in self._create_batch(pairs):  # to avoid OOM
            batch_scores = (
                self.nli_classifier.predict(input_pairs).cpu().detach().round()
            )
            # convert tensor to list of floats
            scores.extend(batch_scores.tolist())

        return sum(scores) / len(scores)


faithfulness = Faithfulness()
