import json
import os
import re

import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_ollama import ChatOllama

import config
from rag.services.rag import RagService

os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "600")
os.environ.setdefault("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE", "60")


class OllamaModel(DeepEvalBaseLLM):
    """Use a stronger Ollama model for DeepEval judge scoring."""

    def __init__(self, model_name: str | None = None):
        if model_name is None:
            model_name = config.DEEPEVAL_JUDGE_MODEL
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def get_model_name(self) -> str:
        return self.model_name

    def _extract_json(self, content: str):
        """Normalize malformed JSON from Ollama responses into a dict/list."""
        if not content:
            return {}

        text = content.strip()
        markdown_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S | re.I)
        if markdown_match:
            text = markdown_match.group(1)

        if text.startswith("json") and "\n" in text:
            text = text.split("\n", 1)[1].strip()

        # Some local judge models return only a partial object like {"statement": ...}
        # without the required "verdict" field. A strict schema is expected.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: extract the first object-like blob in the answer.
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return {}
            return {}

    def _normalize_contextual_relevancy_schema(self, payload):
        """Repair the verdict schema that some local Ollama models omit."""
        if not isinstance(payload, dict):
            return payload

        verdicts = payload.get("verdicts")
        if not isinstance(verdicts, list):
            return payload

        repaired = []
        for item in verdicts:
            if not isinstance(item, dict):
                continue
            repaired_item = dict(item)
            verdict = str(repaired_item.get("verdict") or "").strip().lower()
            if not verdict:
                reason = str(repaired_item.get("reason") or "").lower()
                statement = str(repaired_item.get("statement") or "").lower()
                negative_markers = (
                    "not relevant",
                    "irrelevant",
                    "off-topic",
                    "not related",
                    "not about",
                    "does not support",
                    "unsupported",
                )
                verdict = "no" if any(marker in reason or marker in statement for marker in negative_markers) else "yes"
            repaired_item["verdict"] = verdict
            repaired.append(repaired_item)

        payload["verdicts"] = repaired
        return payload

    def generate(self, prompt: str) -> str:
        llm = ChatOllama(model=self.model_name, temperature=0, streaming=False)
        return llm.invoke(prompt).content

    def generate_with_schema(self, prompt: str, schema):
        raw = self.generate(prompt)
        payload = self._extract_json(raw)
        if schema.__name__ == "ContextualRelevancyVerdicts":
            payload = self._normalize_contextual_relevancy_schema(payload)
        try:
            return schema(**payload) if isinstance(payload, dict) else payload
        except TypeError:
            return payload

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    async def a_generate_with_schema(self, prompt: str, schema):
        return self.generate_with_schema(prompt, schema)


class TestRAGWithDeepEval:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.rag = RagService()
        self.custom_model = OllamaModel()  # Uses model from config.DEEPEVAL_JUDGE_MODEL
        yield
        self.rag = None

    def _create_test_case(self, question: str) -> LLMTestCase:
        answer, _, _ = self.rag.ask(question)
        docs = self.rag.retriever.invoke(question)
        retrieval_context = [doc.page_content for doc in docs]

        return LLMTestCase(
            input=question,
            actual_output=answer,
            context=retrieval_context,
            retrieval_context=retrieval_context,
        )

    @pytest.mark.slow
    def test_rag_faithfulness(self):
        """Check whether the answer is grounded in the retrieved context."""
        question = "Welche antiseptischen Mittel werden erwähnt?"
        test_case = self._create_test_case(question)

        metric = FaithfulnessMetric(
            threshold=0.5,
            model=self.custom_model,
            async_mode=False,
        )
        metric.measure(test_case)
        print(f"FaithfulnessMetric score: {metric.score}")

        assert metric.score is not None
        assert 0.0 <= metric.score <= 1.0

    @pytest.mark.slow
    def test_rag_relevancy(self):
        """Check whether the answer addresses the user query."""
        question = "Welche antiseptischen Mittel werden erwähnt?"
        test_case = self._create_test_case(question)

        metric = AnswerRelevancyMetric(
            threshold=0.5,
            model=self.custom_model,
            async_mode=False,
        )
        metric.measure(test_case)
        print(f"AnswerRelevancyMetric score: {metric.score}")

        assert metric.score is not None
        assert 0.0 <= metric.score <= 1.0

    @pytest.mark.slow
    def test_rag_contextual_relevancy(self):
        """Check whether the retrieved context is relevant to the question."""
        question = "Wie wird die Wundheilung unterstützt?"
        test_case = self._create_test_case(question)

        metric = ContextualRelevancyMetric(
            threshold=0.5,
            model=self.custom_model,
            async_mode=False,
        )
        metric.measure(test_case)
        print(f"ContextualRelevancyMetric score: {metric.score}")

        assert metric.score is not None
        assert 0.0 <= metric.score <= 1.0

    @pytest.mark.slow
    def test_rag_hallucination(self):
        """Check whether the answer contains unsupported or hallucinated content."""
        question = "Welche Wirkstoffe sind in der Hausapotheke dokumentiert?"
        test_case = self._create_test_case(question)

        metric = HallucinationMetric(
            threshold=0.5,
            model=self.custom_model,
            async_mode=False,
        )
        metric.measure(test_case)
        print(f"HallucinationMetric score: {metric.score}")

        assert metric.score is not None
        assert 0.0 <= metric.score <= 1.0
