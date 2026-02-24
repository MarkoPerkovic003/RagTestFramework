"""
RAGAS-Integration für Qualitätsmetriken.

Berechnet folgende Metriken (Abschnitt 3.2 der Bachelorarbeit):
- Faithfulness:       Faktische Konsistenz der Antwort gegenüber dem Kontext
- Answer Relevancy:   Relevanz der Antwort für die Frage
- Context Precision:  Signal-Rausch-Verhältnis des abgerufenen Kontexts
- Context Recall:     Vollständigkeit des abgerufenen Kontexts

Judge-LLM (Priorität):
  1. JUDGE_AGENT_URL gesetzt → SyntaxChatLLM (GPT 5.2 Chat via Syntax AI Studio)
  2. Fallback → ChatAnthropic (claude-opus-4-6, benötigt ANTHROPIC_API_KEY)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

from rag.wrapper import RAGResult
import config


@dataclass
class RAGASResult:
    """Ergebnis einer RAGAS-Evaluation."""
    faithfulness:      float | None = None
    answer_relevancy:  float | None = None
    context_precision: float | None = None
    context_recall:    float | None = None
    raw:               dict[str, Any] = field(default_factory=dict)

    def passed_gates(self) -> dict[str, bool]:
        """Prüft ob die Metriken die Quality-Gate-Schwellenwerte erfüllen."""
        gates = config.QUALITY_GATES
        results = {}
        for metric, value in self.to_dict().items():
            if metric not in gates or value is None:
                continue
            gate = gates[metric]
            if gate["operator"] == ">=":
                results[metric] = value >= gate["threshold"]
            elif gate["operator"] == "<=":
                results[metric] = value <= gate["threshold"]
        return results

    def to_dict(self) -> dict[str, float | None]:
        return {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }


class RAGASEvaluator:
    """
    Berechnet RAGAS-Metriken für eine Liste von RAG-Ergebnissen.

    Judge-LLM: Syntax AI Studio GPT-Agent (JUDGE_AGENT_URL) oder Claude-Fallback.
    Embeddings: HuggingFace all-MiniLM-L6-v2 (lokal, keine API nötig).
    """

    def __init__(self) -> None:
        if config.JUDGE_AGENT_URL:
            from rag.syntax_agent import SyntaxChatLLM
            judge_llm = SyntaxChatLLM(
                api_key=config.JUDGE_AGENT_API_KEY or config.SYNTAX_AGENT_API_KEY,
                agent_url=config.JUDGE_AGENT_URL,
            )
        else:
            from langchain_anthropic import ChatAnthropic
            judge_llm = ChatAnthropic(
                model=config.JUDGE_MODEL,
                api_key=config.ANTHROPIC_API_KEY,
            )

        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

        self._llm_wrapper = LangchainLLMWrapper(judge_llm)
        self._emb_wrapper = LangchainEmbeddingsWrapper(embeddings)

    def evaluate(
        self,
        rag_results: list[RAGResult],
        ground_truths: list[str | None] | None = None,
    ) -> RAGASResult:
        """
        Berechnet RAGAS-Metriken für eine Liste von RAG-Ergebnissen.

        Args:
            rag_results:   Ausgaben der RAG-Pipeline (question, answer, contexts).
            ground_truths: Optionale Ground-Truth-Antworten (für Context Recall).

        Returns:
            RAGASResult mit allen Metrik-Scores.
        """
        if not rag_results:
            return RAGASResult()

        # Dataset für RAGAS vorbereiten
        data = {
            "question":  [r.question for r in rag_results],
            "answer":    [r.answer for r in rag_results],
            "contexts":  [r.contexts for r in rag_results],
        }

        if ground_truths:
            data["ground_truth"] = [gt or "" for gt in ground_truths]
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        else:
            metrics = [faithfulness, answer_relevancy, context_precision]

        dataset = Dataset.from_dict(data)

        # Metriken mit Judge-LLM berechnen
        for m in metrics:
            m.llm = self._llm_wrapper
            if hasattr(m, "embeddings"):
                m.embeddings = self._emb_wrapper

        result = evaluate(dataset=dataset, metrics=metrics)
        df = result.to_pandas()

        # Durchschnitte berechnen
        return RAGASResult(
            faithfulness=float(df["faithfulness"].mean()) if "faithfulness" in df else None,
            answer_relevancy=float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else None,
            context_precision=float(df["context_precision"].mean()) if "context_precision" in df else None,
            context_recall=float(df["context_recall"].mean()) if "context_recall" in df else None,
            raw=result.to_pandas().to_dict(),
        )

    def evaluate_single(
        self,
        rag_result: RAGResult,
        ground_truth: str | None = None,
    ) -> RAGASResult:
        """Berechnet RAGAS-Metriken für ein einzelnes RAG-Ergebnis."""
        return self.evaluate(
            rag_results=[rag_result],
            ground_truths=[ground_truth] if ground_truth else None,
        )
