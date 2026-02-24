"""
RAGPipelineWrapper – Abstraktionsschicht für die RAG-Pipeline.

Stellt eine einheitliche Schnittstelle bereit und loggt alle
Zwischenergebnisse (Retrieved Docs, Kontext, Antwort) für die Evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

from rag.pipeline import build_rag_pipeline, SYSTEM_PROMPT
import config


@dataclass
class RAGResult:
    """Enthält alle Zwischenergebnisse einer RAG-Anfrage."""
    question: str
    answer: str
    retrieved_docs: list[Document]
    # Für RAGAS: Liste der Chunk-Texte
    contexts: list[str] = field(default_factory=list)
    # Metadaten für Security-Evaluation
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.contexts:
            self.contexts = [doc.page_content for doc in self.retrieved_docs]


class RAGPipelineWrapper:
    """
    Abstraktionsschicht über die RAG-Pipeline.

    Ermöglicht:
    - Einheitliche query()-Schnittstelle
    - Injection von manipulierten Dokumenten (für adversariale Tests)
    - Vollständiges Logging aller Zwischenergebnisse
    - Austausch des System-Prompts (für verschiedene Test-Szenarien)
    """

    def __init__(
        self,
        system_prompt: str = SYSTEM_PROMPT,
        inject_docs: list[str] | None = None,
    ) -> None:
        """
        Args:
            system_prompt: Prompt-Template für den Generator.
            inject_docs: Optionale manipulierte Dokumente, die der Wissensbasis
                         temporär hinzugefügt werden (für Corpus-Poisoning-Tests).
        """
        self._system_prompt = system_prompt
        self._inject_docs = inject_docs or []
        self._chain: RetrievalQA | None = None

    def _get_chain(self) -> RetrievalQA:
        if self._chain is None:
            self._chain = build_rag_pipeline(system_prompt=self._system_prompt)
        return self._chain

    def query(self, question: str, extra_metadata: dict | None = None) -> RAGResult:
        """
        Führt eine Anfrage durch die RAG-Pipeline aus.

        Args:
            question: Die Nutzerfrage.
            extra_metadata: Optionale Metadaten (z.B. TestCase-ID).

        Returns:
            RAGResult mit Antwort, abgerufenen Dokumenten und Kontext.
        """
        chain = self._get_chain()
        result = chain.invoke({"query": question})

        retrieved = result.get("source_documents", [])

        # Injizierte Dokumente zur Kontext-Liste hinzufügen (simuliert Corpus Poisoning)
        injected_contexts = list(self._inject_docs)

        return RAGResult(
            question=question,
            answer=result["result"],
            retrieved_docs=retrieved,
            contexts=[doc.page_content for doc in retrieved] + injected_contexts,
            metadata={
                "injected_doc_count": len(self._inject_docs),
                **(extra_metadata or {}),
            },
        )

    def reset(self) -> None:
        """Setzt die gecachte Chain zurück (z.B. nach System-Prompt-Wechsel)."""
        self._chain = None

    @classmethod
    def with_injected_docs(
        cls, inject_docs: list[str], system_prompt: str = SYSTEM_PROMPT
    ) -> "RAGPipelineWrapper":
        """Factory für Tests mit injizierten Dokumenten."""
        return cls(system_prompt=system_prompt, inject_docs=inject_docs)
