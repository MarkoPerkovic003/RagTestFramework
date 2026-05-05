"""
Basistypen für den Adversarial Test Case Generator.

TestCase ist das zentrale Daten-Objekt. Alle Generatoren geben
Listen von TestCase-Objekten zurück.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
import json
import uuid
from pathlib import Path


class AttackCategory(str, Enum):
    """Die 5 Angriffskategorien aus der Bachelorarbeit (Abschnitt 6.2)."""
    FAITHFULNESS          = "faithfulness"
    CONTEXT_MANIPULATION  = "context_manipulation"
    DIRECT_INJECTION      = "direct_injection"
    CORPUS_POISONING      = "corpus_poisoning"
    DATA_EXFILTRATION     = "data_exfiltration"


class GenerationMethod(str, Enum):
    TEMPLATE = "template"   # Deterministisch, reproduzierbar
    LLM      = "llm"        # Adaptiv, diverser


@dataclass
class TestCase:
    """
    Ein adversarialer Testfall.

    Felder:
        id:                Eindeutige ID (UUID)
        category:          Angriffskategorie
        question:          Die Nutzerfrage (ggf. mit eingebetteter Injection)
        injected_docs:     Manipulierte Dokumente, die dem Retriever zugespielt werden
        attack_payload:    Die eigentliche Angriffs-Instruktion (für Logging)
        expected_behavior: Beschreibung des erwarteten (korrekten) Verhaltens
        ground_truth:      Korrekte Antwort (für RAGAS-Metriken)
        generation_method: Wie der Testfall erstellt wurde
        metadata:          Weitere Informationen (Template-Name, Schwierigkeitsgrad, etc.)
    """
    category:          AttackCategory
    question:          str
    expected_behavior: str
    id:                str               = field(default_factory=lambda: str(uuid.uuid4()))
    injected_docs:     list[str]         = field(default_factory=list)
    attack_payload:    str | None        = None
    ground_truth:      str | None        = None
    generation_method: GenerationMethod  = GenerationMethod.TEMPLATE
    metadata:          dict[str, Any]    = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["generation_method"] = self.generation_method.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TestCase":
        d = dict(d)
        d["category"] = AttackCategory(d["category"])
        d["generation_method"] = GenerationMethod(d["generation_method"])
        return cls(**d)


class BaseTestCaseGenerator:
    """Basisklasse für alle Kategorie-Generatoren."""

    category: AttackCategory

    def generate_templates(self, n: int) -> list[TestCase]:
        """Generiert n template-basierte Testfälle (überschreiben in Unterklasse)."""
        raise NotImplementedError

    def save(self, test_cases: list[TestCase], output_dir: Path) -> Path:
        """Speichert Testfälle als JSON-Datei."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{self.category.value}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([tc.to_dict() for tc in test_cases], f, ensure_ascii=False, indent=2)
        return out_file

    @staticmethod
    def load(json_file: Path) -> list[TestCase]:
        """Lädt Testfälle aus einer JSON-Datei."""
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        return [TestCase.from_dict(d) for d in data]

