import asyncio
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

logger = logging.getLogger(__name__)

from langchain_core.callbacks import Callbacks


class ResponseRelevanceOutput(BaseModel):
    question: str
    noncommittal: int


class ResponseRelevanceInput(BaseModel):
    response: str


class ResponseRelevancePromptIT(
    PydanticPrompt[ResponseRelevanceInput, ResponseRelevanceOutput]
):
    instruction = """Genera una domanda basata sulla risposta fornita e identifica se la risposta è evasiva. Imposta 'noncommittal' a 1 se la risposta è evasiva e a 0 se è chiara e decisa. Una risposta evasiva è vaga, ambigua o evita di fornire un'informazione precisa. Ad esempio, "Non lo so" o "Non ne sono sicuro" sono risposte evasive."""
    
    input_model = ResponseRelevanceInput
    output_model = ResponseRelevanceOutput

    examples = [
        (
            ResponseRelevanceInput(
                response="""Albert Einstein è nato in Germania.""",
            ),
            ResponseRelevanceOutput(
                question="Dove è nato Albert Einstein?",
                noncommittal=0,
            ),
        ),
        (
            ResponseRelevanceInput(
                response="""Non so quale sia stata la caratteristica rivoluzionaria dello smartphone inventato nel 2023, perché non ho informazioni oltre il 2022.""",
            ),
            ResponseRelevanceOutput(
                question="Qual è stata la caratteristica rivoluzionaria dello smartphone inventato nel 2023?",
                noncommittal=1,
            ),
        ),
    ]



@dataclass
class ResponseRelevancyIT(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
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

    name: str = "answer_relevancy_it"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
            }
        }
    )
    output_type = MetricOutputType.CONTINUOUS

    question_generation: PydanticPrompt = ResponseRelevancePromptIT()
    strictness: int = 1 #3

    def calculate_similarity(self, question: str, generated_questions: list[str]):
        assert (
            self.embeddings is not None
        ), f"Error: '{self.name}' requires embeddings to be set."
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)
        ).reshape(len(generated_questions), -1)
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )

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
            cosine_sim = self.calculate_similarity(question, gen_questions)
            score = cosine_sim.mean() * int(not committal)

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


class AnswerRelevancyIT(ResponseRelevancyIT):
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


answer_relevancy_it = AnswerRelevancyIT()