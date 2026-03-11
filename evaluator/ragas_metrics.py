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

        # Kontext-Metriken nur wenn mindestens ein nicht-leerer Kontext vorhanden
        has_contexts = any(r.contexts for r in rag_results)

        if ground_truths and has_contexts:
            data["ground_truth"] = [gt or "" for gt in ground_truths]
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        elif has_contexts:
            metrics = [faithfulness, answer_relevancy, context_precision]
        else:
            # Kein Retrieval-Logging verfügbar (z.B. Syntax AI Studio ohne citations)
            # → nur antwort-basierte Metriken
            metrics = [answer_relevancy]

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


def evaluate_with_judge(
    rag_results: list[RAGResult],
    judge: Any,
) -> RAGASResult:
    """
    Berechnet Qualitätsmetriken via LLM-Judge als Fallback für fehlende RAGAS-Werte.

    Normalisiert Judge-Scores (0-10) auf RAGAS-Skala (0-1).
    Gibt RAGASResult mit faithfulness und answer_relevancy zurück.

    Args:
        rag_results: RAG-Ergebnisse (question + answer, contexts optional).
        judge:       Initialisierter LLMJudge.

    Returns:
        RAGASResult mit judge-basierten Scores.
    """
    if not rag_results:
        return RAGASResult()

    faith_scores: list[float] = []
    rel_scores:   list[float] = []

    for rr in rag_results:
        verdict = judge.evaluate_quality(rr)

        if verdict.faithfulness_score is not None:
            faith_scores.append(verdict.faithfulness_score / 10.0)
        if verdict.relevancy_score is not None:
            rel_scores.append(verdict.relevancy_score / 10.0)
        elif verdict.score is not None:
            # Fallback: combined score für beide Dimensionen nutzen
            faith_scores.append(verdict.score / 10.0)
            rel_scores.append(verdict.score / 10.0)

    return RAGASResult(
        faithfulness=round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None,
        answer_relevancy=round(sum(rel_scores) / len(rel_scores), 4) if rel_scores else None,
    )
