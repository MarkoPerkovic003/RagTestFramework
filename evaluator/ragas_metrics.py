"""
RAGAS-Integration für Qualitätsmetriken.

Berechnet folgende Metriken (Abschnitt 3.2 der Bachelorarbeit):
- Faithfulness:     Faktische Konsistenz der Antwort gegenüber dem Kontext
                    (nur berechenbar wenn RAG-System Kontext-Dokumente zurückgibt)
- Answer Relevancy: Relevanz der Antwort für die Frage
                    (immer berechenbar, braucht nur Frage + Antwort)

Judge-LLM (Priorität):
  1. JUDGE_AGENT_URL gesetzt → SyntaxChatLLM (GPT 5.2 Chat via Syntax AI Studio)
  2. Fallback → ChatAnthropic (claude-opus-4-6, benötigt ANTHROPIC_API_KEY)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

from rag.wrapper import RAGResult
import config


@dataclass
class RAGASResult:
    """Ergebnis einer RAGAS-Evaluation."""
    faithfulness:     float | None = None  # Nur wenn contexts vorhanden
    answer_relevancy: float | None = None  # Immer berechenbar
    kb_consistency:   float | None = None  # LLM-Judge mit KB-Referenz (discover-kb)
    raw:              dict[str, Any] = field(default_factory=dict)

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
            "faithfulness":     self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "kb_consistency":   self.kb_consistency,
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
            ground_truths: Nicht verwendet (für Kompatibilität behalten).

        Returns:
            RAGASResult mit Metrik-Scores.
        """
        if not rag_results:
            return RAGASResult()

        # Dataset für RAGAS vorbereiten
        has_contexts = any(r.contexts for r in rag_results)
        data = {
            "question": [r.question for r in rag_results],
            "answer":   [r.answer for r in rag_results],
            "contexts": [r.contexts or [""] for r in rag_results],
        }

        if has_contexts:
            metrics = [faithfulness, answer_relevancy]
        else:
            # Kein Retrieval-Logging verfügbar → nur antwort-basierte Metrik
            metrics = [answer_relevancy]

        dataset = Dataset.from_dict(data)

        for m in metrics:
            m.llm = self._llm_wrapper
            if hasattr(m, "embeddings"):
                m.embeddings = self._emb_wrapper

        result = evaluate(dataset=dataset, metrics=metrics)
        df = result.to_pandas()

        return RAGASResult(
            faithfulness=float(df["faithfulness"].mean()) if "faithfulness" in df else None,
            answer_relevancy=float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else None,
            raw=df.to_dict(),
        )

    def evaluate_single(
        self,
        rag_result: RAGResult,
        ground_truth: str | None = None,
    ) -> RAGASResult:
        """Berechnet RAGAS-Metriken für ein einzelnes RAG-Ergebnis."""
        return self.evaluate(rag_results=[rag_result])


def evaluate_with_judge(
    rag_results: list[RAGResult],
    judge: Any,
    ground_truths: list[str | None] | None = None,
) -> RAGASResult:
    """
    Berechnet Qualitätsmetriken via LLM-Judge als Fallback.

    Wird nur aufgerufen wenn Kontext vorhanden ist aber RAGAS-Metriken fehlen
    (z.B. bei Fehlern in der RAGAS-Berechnung). Normalisiert Judge-Scores
    (0-10) auf RAGAS-Skala (0-1).

    Args:
        rag_results:   RAG-Ergebnisse (question + answer + contexts).
        judge:         Initialisierter LLMJudge.
        ground_truths: Bekannte korrekte Antworten je Testfall (optional).

    Returns:
        RAGASResult mit judge-basierten Scores (nur für Fallback mit Kontext).
    """
    if not rag_results:
        return RAGASResult()

    faith_scores: list[float] = []
    rel_scores:   list[float] = []

    for i, rr in enumerate(rag_results):
        gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
        verdict = judge.evaluate_quality(rr, ground_truth=gt)

        if verdict.faithfulness_score is not None and bool(rr.contexts):
            faith_scores.append(verdict.faithfulness_score / 10.0)

        if verdict.relevancy_score is not None:
            rel_scores.append(verdict.relevancy_score / 10.0)
        elif verdict.score is not None and verdict.faithfulness_score is None:
            rel_scores.append(verdict.score / 10.0)

    return RAGASResult(
        faithfulness=round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None,
        answer_relevancy=round(sum(rel_scores) / len(rel_scores), 4) if rel_scores else None,
    )
