"""
Generic HTTP Agent Wrapper.

Universeller Wrapper für beliebige REST-Endpunkte.
Konfigurierbar via Umgebungsvariablen:

    AGENT_URL              Endpunkt-URL (POST)
    AGENT_API_KEY          API-Key / Token
    GENERIC_AUTH_HEADER    Header-Name für den Key (Standard: x-api-key)
    GENERIC_ANSWER_FIELD   JSON-Pfad zur Antwort  (Standard: output)
    GENERIC_CONTEXTS_FIELD JSON-Pfad zu Kontexten (Standard: citations)
    GENERIC_REQUEST_TMPL   JSON-Template mit {question} (optional)

Unterstützte Request-Formate (automatische Erkennung falls kein Template):
    {"question": "..."}                          ← Einfach
    {"input": [{"type": "text", "text": "..."}]} ← Syntax AI Studio kompatibel
    {"messages": [{"role": "user", ...}]}        ← OpenAI-kompatibel

Unterstützte Response-Felder (automatische Erkennung):
    output, result, answer, response, content, text,
    choices[0].message.content
"""

from __future__ import annotations
import json
import os
import uuid

import requests

from rag.base_agent import BaseAgentWrapper
from rag.wrapper import RAGResult


def _get_nested(data: dict | list, path: str) -> str | list | None:
    """
    Extrahiert einen Wert aus einem verschachtelten Dict/List via Dot-Notation.
    Unterstützt: "output", "choices.0.message.content", "result.text"
    """
    parts = path.split(".")
    current: dict | list | None = data
    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def _extract_answer_auto(data: dict) -> str:
    """Versucht die Antwort aus bekannten Feldern zu extrahieren."""
    # Direkte Felder (nach Häufigkeit sortiert)
    for field in ("output", "result", "answer", "response", "content", "text"):
        v = data.get(field)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # OpenAI-kompatibel: choices[0].message.content
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        if content:
            return content.strip()
    # Letzter Fallback
    return str(data)


def _extract_contexts_auto(data: dict) -> list[str]:
    """Versucht Kontexte aus bekannten Feldern zu extrahieren."""
    for field in ("citations", "sourceDocuments", "source_documents",
                  "context", "sources", "references", "chunks"):
        val = data.get(field)
        if not val:
            continue
        if isinstance(val, str):
            return [val]
        if isinstance(val, list):
            result = []
            for item in val:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, dict):
                    text = (item.get("content") or item.get("text")
                            or item.get("pageContent") or item.get("page_content"))
                    if text:
                        result.append(str(text))
            if result:
                return result
    return []


class GenericHTTPAgentWrapper(BaseAgentWrapper):
    """
    Universeller HTTP-Wrapper für beliebige RAG-Agent-REST-APIs.

    Deckt ab: Flowise, LangServe, Custom LangChain, FastAPI-basierte Agents,
    n8n-Workflows, Botpress, Rasa und weitere POST-Endpunkte.
    """

    LABEL      = "Generic HTTP Agent"
    AGENT_TYPE = "generic"

    def __init__(
        self,
        url:             str,
        api_key:         str  = "",
        auth_header:     str  = "x-api-key",
        answer_field:    str  = "output",
        contexts_field:  str  = "citations",
        request_template: str | None = None,
        timeout:         int  = 120,
    ) -> None:
        """
        Args:
            url:              POST-Endpunkt.
            api_key:          Authentifizierungs-Key (leer = kein Auth).
            auth_header:      Header-Name für den API-Key.
            answer_field:     Dot-Pfad zur Antwort im Response-JSON.
            contexts_field:   Dot-Pfad zu Kontextdokumenten (optional).
            request_template: JSON-String mit {question}-Platzhalter.
                              Wenn leer, wird automatisch erkannt.
            timeout:          HTTP-Timeout in Sekunden.
        """
        self._url              = url
        self._api_key          = api_key
        self._auth_header      = auth_header
        self._answer_field     = answer_field
        self._contexts_field   = contexts_field
        self._request_template = request_template
        self._timeout          = timeout

    @classmethod
    def from_env(cls) -> "GenericHTTPAgentWrapper":
        return cls(
            url=os.getenv("AGENT_URL", ""),
            api_key=os.getenv("AGENT_API_KEY", ""),
            auth_header=os.getenv("GENERIC_AUTH_HEADER", "x-api-key"),
            answer_field=os.getenv("GENERIC_ANSWER_FIELD", "output"),
            contexts_field=os.getenv("GENERIC_CONTEXTS_FIELD", "citations"),
            request_template=os.getenv("GENERIC_REQUEST_TMPL") or None,
        )

    def _build_request_body(self, question: str) -> dict:
        if self._request_template:
            try:
                filled = self._request_template.replace("{question}", question)
                return json.loads(filled)
            except json.JSONDecodeError:
                pass
        # Sendet beide häufigen Formate gleichzeitig → maximale Kompatibilität
        return {
            "question": question,
            "input": question,
            "session_id": str(uuid.uuid4()),
        }

    def query(self, question: str, extra_metadata: dict | None = None) -> RAGResult:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers[self._auth_header] = self._api_key

        body = self._build_request_body(question)
        response = requests.post(
            self._url, headers=headers, json=body, timeout=self._timeout
        )
        response.raise_for_status()
        data = response.json()

        # Antwort extrahieren
        raw_answer = _get_nested(data, self._answer_field)
        if raw_answer and isinstance(raw_answer, str):
            answer = raw_answer.strip()
        else:
            answer = _extract_answer_auto(data)

        # Kontexte extrahieren
        raw_ctx = _get_nested(data, self._contexts_field)
        if isinstance(raw_ctx, list) and raw_ctx:
            contexts = [str(c) for c in raw_ctx if c]
        else:
            contexts = _extract_contexts_auto(data)

        return RAGResult(
            question=question,
            answer=answer,
            retrieved_docs=[],
            contexts=contexts,
            metadata={"source": "generic_http", **(extra_metadata or {})},
        )
