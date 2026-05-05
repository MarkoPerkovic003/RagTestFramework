"""
Agent-Registry: Zentrale Verwaltung aller RAG-Agent-Wrapper.

Neue Wrapper werden mit @register() oder register() eingetragen und
stehen danach im Dashboard und CLI automatisch zur Verfügung.

Verwendung:
    from rag.agent_registry import get_wrapper, list_types, labels

    wrapper = get_wrapper()   # Liest RAG_TARGET aus config/Env
    agents  = list_types()    # ["syntax", "azure", "copilot", ...]
    lbls    = labels()        # {"syntax": "Syntax AI Studio", ...}
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.base_agent import BaseAgentWrapper

# ── Interne Registry ──────────────────────────────────────────────────────────

_registry: dict[str, type] = {}


def register(agent_type: str, wrapper_class: type) -> None:
    """Registriert einen Agent-Wrapper-Typ."""
    _registry[agent_type] = wrapper_class


def get_class(agent_type: str) -> type:
    """Gibt die Wrapper-Klasse für einen Agent-Typ zurück."""
    if agent_type not in _registry:
        available = list(_registry.keys())
        raise ValueError(
            f"Unbekannter Agent-Typ: '{agent_type}'. "
            f"Verfügbar: {available}"
        )
    return _registry[agent_type]


def list_types() -> list[str]:
    """Gibt alle registrierten Agent-Typen zurück."""
    return list(_registry.keys())


def labels() -> dict[str, str]:
    """Gibt {agent_type: Anzeigename} zurück."""
    return {
        t: cls.LABEL
        for t, cls in _registry.items()
        if hasattr(cls, "LABEL")
    }


def get_wrapper() -> "BaseAgentWrapper":
    """
    Factory: Erstellt den Agent-Wrapper anhand von RAG_TARGET.

    Liest RAG_TARGET aus der Umgebungsvariable (überschreibt .env).
    Fällt auf config.RAG_TARGET zurück wenn nicht gesetzt.
    """
    import config
    agent_type = os.getenv("RAG_TARGET") or config.RAG_TARGET
    wrapper_class = get_class(agent_type)
    return wrapper_class.from_env()


# ── Auto-Registrierung ────────────────────────────────────────────────────────
# Wird beim ersten Import dieses Moduls ausgeführt.

def _auto_register() -> None:
    """Registriert alle eingebauten Agent-Typen."""
    # Demo RAG (lokale ChromaDB)
    try:
        from rag.wrapper import RAGPipelineWrapper
        register("demo", RAGPipelineWrapper)
    except Exception:
        pass

    # Syntax AI Studio
    try:
        from rag.syntax_agent import SyntaxAgentWrapper
        register("syntax", SyntaxAgentWrapper)
    except Exception:
        pass

    # Generic HTTP
    try:
        from rag.generic_http_agent import GenericHTTPAgentWrapper
        register("generic", GenericHTTPAgentWrapper)
    except Exception:
        pass

    # Azure OpenAI
    try:
        from rag.azure_agent import AzureOpenAIAgentWrapper
        register("azure", AzureOpenAIAgentWrapper)
    except Exception:
        pass

    # Microsoft Copilot Studio
    try:
        from rag.copilot_agent import CopilotStudioAgentWrapper
        register("copilot", CopilotStudioAgentWrapper)
    except Exception:
        pass


_auto_register()

