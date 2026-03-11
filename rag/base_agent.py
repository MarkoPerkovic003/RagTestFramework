"""
Abstrakte Basis-Klasse für alle RAG-Agent-Wrapper.

Jeder Wrapper implementiert:
  - query()    → Anfrage an den Agent stellen
  - from_env() → Instanz aus Umgebungsvariablen erstellen

Beide Methoden müssen in Unterklassen implementiert werden.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.wrapper import RAGResult


class BaseAgentWrapper(ABC):
    """
    Einheitliche Schnittstelle für alle RAG-Agent-Typen.

    Attributes:
        LABEL:      Anzeigename im Dashboard (z.B. "Azure OpenAI").
        AGENT_TYPE: Interner Schlüssel für die Registry (z.B. "azure").
    """

    LABEL:      str = "Unbekannter Agent"
    AGENT_TYPE: str = "unknown"

    @abstractmethod
    def query(
        self,
        question: str,
        extra_metadata: dict | None = None,
    ) -> "RAGResult":
        """
        Stellt eine Anfrage an den RAG-Agent.

        Args:
            question:       Die Frage an den Agent.
            extra_metadata: Optionale Metadaten (z.B. test_case_id).

        Returns:
            RAGResult mit Antwort und ggf. abgerufenen Kontexten.
        """

    @classmethod
    @abstractmethod
    def from_env(cls) -> "BaseAgentWrapper":
        """Erstellt eine Instanz aus Umgebungsvariablen."""

    def test_connection(self, question: str = "Hallo, bist du erreichbar?") -> bool:
        """Testet die Verbindung durch eine einfache Testanfrage."""
        try:
            result = self.query(question)
            return bool(result.answer.strip())
        except Exception:
            return False
