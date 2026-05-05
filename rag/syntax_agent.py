"""
Syntax AI Studio Agenten-Wrapper.

Enthält zwei Klassen:

SyntaxAgentWrapper
    RAG-Agent-Wrapper: Sendet Fragen an den Syntax AI Studio RAG-Agent und
    gibt RAGResult-Objekte zurück (für das Test-Framework).

SyntaxChatLLM
    LangChain BaseChatModel-Wrapper für den Syntax AI Studio Chat-Agent
    (z.B. GPT 5.2). Kann als Judge-LLM in RAGAS und LLMJudge verwendet
    werden – gleiche API-Struktur wie der RAG-Agent.

API-Endpunkt: POST /api/v1/agents/<agent-id>/invoke
Auth: x-api-key Header

Retrieval-Logik exposieren
──────────────────────────
Viele RAG-Agent-Plattformen (Flowise, LangServe, custom LangChain) geben
Quelldokumente im Response zurück. Die API wird auf folgende Felder geprüft:

  response.sourceDocuments          ← Flowise Standard
  response.source_documents         ← LangChain RetrievalQA
  response.context                  ← direktes Kontext-Feld
  response.intermediate_steps       ← LangChain Agent verbose mode
  response.metadata.sources         ← Custom-Implementierungen

Falls keines verfügbar: debug_raw() aufrufen um den echten Response zu sehen.
"""

from __future__ import annotations
import os
import uuid
import requests
from typing import Any, List, Optional

import config
from rag.base_agent import BaseAgentWrapper
from rag.wrapper import RAGResult


class SyntaxAgentError(Exception):
    """Fehler bei der Kommunikation mit dem Syntax AI Studio Agent."""


# Bekannte Request-Parameter, die Quelldokumente aktivieren können
# (werden als Extra-Felder im Payload mitgeschickt)
SOURCE_ENABLE_PARAMS = {
    "returnSourceDocuments": True,   # Flowise
    "return_source_documents": True, # LangChain
    "includeSourceDocuments": True,  # Variante
}


def _try_extract_json(text: str) -> str:
    """
    Versucht reines JSON aus einer Antwort zu extrahieren.

    Wird verwendet um RAGAS-Kompatibilität herzustellen: Chat-Modelle
    verpacken JSON oft in Markdown-Blöcke oder fügen Erklärungen hinzu.
    RAGAS erwartet reines JSON ohne Formatierung.
    """
    import json as _json

    # ── Markdown-Codeblock ────────────────────────────────────────────────────
    if "```json" in text:
        try:
            return text.split("```json")[1].split("```")[0].strip()
        except IndexError:
            pass
    if "```" in text:
        try:
            candidate = text.split("```")[1].split("```")[0].strip()
            _json.loads(candidate)
            return candidate
        except (IndexError, _json.JSONDecodeError):
            pass

    # ── Erstes JSON-Objekt oder Array im Text suchen ──────────────────────────
    for start_char in ("{", "["):
        idx = text.find(start_char)
        if idx != -1:
            try:
                decoder = _json.JSONDecoder()
                obj, _ = decoder.raw_decode(text[idx:])
                return _json.dumps(obj, ensure_ascii=False)
            except _json.JSONDecodeError:
                continue

    return text


def _extract_text(r: dict) -> str:
    """
    Extrahiert den Antwort-Text aus verschiedenen Response-Strukturen.

    Gemeinsame Hilfsfunktion für SyntaxAgentWrapper und SyntaxChatLLM.
    """
    candidates = [
        lambda r: r.get("output") if isinstance(r.get("output"), str) else None,
        lambda r: r.get("result") if isinstance(r.get("result"), str) else None,
        lambda r: r.get("answer") if isinstance(r.get("answer"), str) else None,
        lambda r: r.get("response") if isinstance(r.get("response"), str) else None,
        lambda r: r.get("content") if isinstance(r.get("content"), str) else None,
        lambda r: r.get("text") if isinstance(r.get("text"), str) else None,
        # Nested: {"output": {"text": "..."}}
        lambda r: r["output"].get("text") if isinstance(r.get("output"), dict) else None,
        # Array: {"output": [{"text": "..."}]}
        lambda r: r["output"][0].get("text") if isinstance(r.get("output"), list) and r["output"] else None,
        # LangChain verbose: intermediate_steps enthalten das finale Ergebnis
        lambda r: r.get("intermediate_steps", [{}])[-1].get("output") if r.get("intermediate_steps") else None,
    ]
    for fn in candidates:
        try:
            v = fn(r)
            if v and isinstance(v, str):
                return v
        except (KeyError, TypeError, IndexError):
            continue
    return str(r)


class SyntaxAgentWrapper(BaseAgentWrapper):
    """
    Wrapper für den Syntax AI Studio GenAI-Agent.

    Implementiert dieselbe query()-Schnittstelle wie RAGPipelineWrapper,
    sodass alle Evaluatoren ohne Änderung gegen den echten Agenten laufen.

    Retrieval-Logik: Der Wrapper versucht automatisch Quelldokumente aus
    der API-Response zu extrahieren. Mit request_sources=True werden
    plattformspezifische Parameter mitgesendet, um die Rückgabe zu aktivieren.
    """

    LABEL      = "Syntax AI Studio"
    AGENT_TYPE = "syntax"

    @classmethod
    def from_env(cls) -> "SyntaxAgentWrapper":
        return cls(
            api_key   = os.getenv("SYNTAX_AGENT_API_KEY") or config.SYNTAX_AGENT_API_KEY,
            agent_url = os.getenv("SYNTAX_AGENT_URL") or os.getenv("AGENT_URL") or config.SYNTAX_AGENT_URL,
        )

    def __init__(
        self,
        api_key: str | None = None,
        agent_url: str | None = None,
        fixed_session_id: str | None = None,
        request_sources: bool = True,
    ) -> None:
        """
        Args:
            api_key:          Syntax AI Studio API Key.
            agent_url:        Agent-Endpunkt-URL.
            fixed_session_id: Feste Session-ID (für Multi-Turn-Tests).
            request_sources:  Sendet Source-Document-Parameter im Request.
        """
        self._api_key          = api_key or config.SYNTAX_AGENT_API_KEY
        self._agent_url        = agent_url or config.SYNTAX_AGENT_URL
        self._fixed_session_id = fixed_session_id
        self._request_sources  = request_sources

        if not self._api_key:
            raise ValueError(
                "SYNTAX_AGENT_API_KEY nicht gesetzt. "
                "Bitte in .env eintragen oder als Parameter uebergeben."
            )

    def query(self, question: str, extra_metadata: dict | None = None) -> RAGResult:
        """Sendet eine Anfrage an den Syntax AI Studio Agenten."""
        session_id  = self._fixed_session_id or str(uuid.uuid4())
        raw_json    = self._call_api(question, session_id)
        answer      = self._extract_answer(raw_json)
        contexts    = self._extract_contexts(raw_json)

        return RAGResult(
            question=question,
            answer=answer,
            retrieved_docs=[],
            contexts=contexts,
            metadata={
                "session_id":     session_id,
                "source":         "syntax_ai_studio",
                "contexts_found": len(contexts) > 0,
                **(extra_metadata or {}),
            },
        )

    def debug_raw(self, question: str = "Was sind die Urlaubsregelungen?") -> dict:
        """
        Gibt die vollständige, ungefilterte API-Response zurück.

        Damit kann man sehen welche Felder die API liefert – insbesondere
        ob Quelldokumente bereits enthalten sind.

        Aufruf:
            from rag.syntax_agent import SyntaxAgentWrapper
            w = SyntaxAgentWrapper()
            import json
            print(json.dumps(w.debug_raw(), indent=2, ensure_ascii=False))

        Oder via CLI:
            uv run python main.py ping --debug
        """
        return self._call_api(question, session_id=str(uuid.uuid4()))

    def _call_api(self, question: str, session_id: str) -> dict:
        """Führt den HTTP-Request aus und gibt den rohen JSON-Body zurück."""
        payload: dict[str, Any] = {
            "input": [{"type": "text", "text": question}],
            "session_id": session_id,
        }
        # Source-Document-Parameter mitschicken (schadet nicht falls nicht unterstützt)
        if self._request_sources:
            payload.update(SOURCE_ENABLE_PARAMS)

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
        }

        import time
        retries = 3
        backoff = 5  # Sekunden
        last_error: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                response = requests.post(
                    self._agent_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                # 502/503/429: transient – kurz warten und nochmal versuchen
                if response.status_code in (429, 502, 503) and attempt < retries:
                    time.sleep(backoff * attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                last_error = SyntaxAgentError(f"Timeout (>60s): {question[:80]}")
                if attempt < retries:
                    time.sleep(backoff * attempt)
            except requests.exceptions.HTTPError as e:
                last_error = SyntaxAgentError(f"HTTP {response.status_code}: {e}")
                if attempt < retries:
                    time.sleep(backoff * attempt)
            except requests.exceptions.RequestException as e:
                raise SyntaxAgentError(f"Verbindungsfehler: {e}")

        raise last_error  # type: ignore[misc]

    def _extract_answer(self, r: dict) -> str:
        """Extrahiert den Antwort-Text aus verschiedenen Response-Strukturen."""
        return _extract_text(r)

    def _extract_contexts(self, r: dict) -> list[str]:
        """
        Versucht Quelldokumente/Kontext aus der API-Response zu extrahieren.

        Probiert alle bekannten Felder in der Reihenfolge ihrer Häufigkeit.
        Gibt leere Liste zurück wenn nichts gefunden – kein Fehler.
        """
        contexts: list[str] = []

        # ── Syntax AI Studio: citations ───────────────────────────────────────
        # Bestätigt durch ping --debug: API gibt {"citations": [...]} zurück.
        # Leer wenn kein relevantes Dokument gefunden, gefüllt bei Treffern.
        citations = r.get("citations")
        if isinstance(citations, list):
            for cite in citations:
                if isinstance(cite, str) and cite:
                    contexts.append(cite)
                elif isinstance(cite, dict):
                    text = (cite.get("content") or cite.get("text")
                            or cite.get("pageContent") or cite.get("passage"))
                    if text:
                        contexts.append(str(text))

        # ── Flowise: sourceDocuments ──────────────────────────────────────────
        if not contexts:
            source_docs = r.get("sourceDocuments") or r.get("source_documents")
            if isinstance(source_docs, list):
                for doc in source_docs:
                    if isinstance(doc, str):
                        contexts.append(doc)
                    elif isinstance(doc, dict):
                        # Flowise-Format: {"pageContent": "...", "metadata": {...}}
                        text = doc.get("pageContent") or doc.get("page_content") or doc.get("content") or doc.get("text")
                        if text:
                            contexts.append(str(text))

        # ── Direktes Kontext-Feld ─────────────────────────────────────────────
        if not contexts:
            context = r.get("context")
            if isinstance(context, str) and context:
                contexts = [context]
            elif isinstance(context, list):
                contexts = [str(c) for c in context if c]

        # ── LangChain intermediate_steps ──────────────────────────────────────
        if not contexts:
            for step in r.get("intermediate_steps", []):
                if isinstance(step, dict):
                    obs = step.get("observation") or step.get("output")
                    if obs and isinstance(obs, str):
                        contexts.append(obs)

        # ── Custom sources/references-Felder ─────────────────────────────────
        if not contexts:
            for field in ("sources", "references", "chunks", "documents", "passages"):
                val = r.get(field)
                if isinstance(val, list) and val:
                    for item in val:
                        text = item if isinstance(item, str) else (
                            item.get("text") or item.get("content") or item.get("pageContent") or str(item)
                            if isinstance(item, dict) else str(item)
                        )
                        if text:
                            contexts.append(text)
                    break

        return contexts

    def test_connection(self) -> bool:
        """Testet ob die API erreichbar ist."""
        try:
            result = self.query("Hallo, bist du erreichbar?")
            return bool(result.answer)
        except SyntaxAgentError:
            return False


class SyntaxChatLLM:
    """
    LangChain BaseChatModel-Wrapper für Syntax AI Studio Chat-Agenten (z.B. GPT 5.2).

    Kann als Judge-LLM für RAGAS (via LangchainLLMWrapper) und den LLMJudge
    verwendet werden. Die API-Struktur ist identisch mit SyntaxAgentWrapper.

    Verwendung:
        from rag.syntax_agent import SyntaxChatLLM
        from ragas.llms import LangchainLLMWrapper
        llm = LangchainLLMWrapper(SyntaxChatLLM(
            api_key=config.JUDGE_AGENT_API_KEY,
            agent_url=config.JUDGE_AGENT_URL,
        ))
    """

    def __new__(cls, api_key: str, agent_url: str, timeout: int = 120) -> "SyntaxChatLLM":  # type: ignore[misc]
        # Lazy import: langchain_core ist nur bei RAGAS-Nutzung erforderlich
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import BaseMessage, AIMessage
        from langchain_core.outputs import ChatResult, ChatGeneration

        class _SyntaxChatLLM(BaseChatModel):
            api_key: str
            agent_url: str
            timeout: int = 120

            @property
            def _llm_type(self) -> str:
                return "syntax_chat_llm"

            def _generate(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> ChatResult:
                text = "\n\n".join(
                    str(m.content) for m in messages if m.content
                )
                # RAGAS-Kompatibilität: JSON-only Instruction anhängen.
                # Chat-Agenten antworten sonst mit Markdown/Erklärungen,
                # was RAGAS's Output-Parser nicht verarbeiten kann.
                text += (
                    "\n\nIMPORTANT: Your response MUST be valid JSON only. "
                    "No markdown formatting, no explanations, no text outside the JSON."
                )
                raw = self._call_syntax_api(text)
                answer = _extract_text(raw)
                answer = _try_extract_json(answer)
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=answer))]
                )

            async def _agenerate(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> ChatResult:
                import asyncio
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self._generate(messages, stop=stop, **kwargs),
                )

            def _call_syntax_api(self, text: str) -> dict:
                payload = {
                    "input": [{"type": "text", "text": text}],
                    "session_id": str(uuid.uuid4()),
                }
                try:
                    response = requests.post(
                        self.agent_url,
                        headers={
                            "Content-Type": "application/json",
                            "x-api-key": self.api_key,
                        },
                        json=payload,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                except requests.exceptions.Timeout:
                    raise SyntaxAgentError(f"Judge-Timeout (>{self.timeout}s)")
                except requests.exceptions.HTTPError as e:
                    raise SyntaxAgentError(f"Judge HTTP {response.status_code}: {e}")
                except requests.exceptions.RequestException as e:
                    raise SyntaxAgentError(f"Judge Verbindungsfehler: {e}")
                return response.json()

        return _SyntaxChatLLM(api_key=api_key, agent_url=agent_url, timeout=timeout)


def get_wrapper() -> "SyntaxAgentWrapper | object":
    """
    Factory-Funktion: Gibt den konfigurierten Wrapper zurück.

    RAG_TARGET="syntax" → SyntaxAgentWrapper
    RAG_TARGET="demo"   → RAGPipelineWrapper (lokale ChromaDB)
    """
    from rag.wrapper import RAGPipelineWrapper

    if config.RAG_TARGET == "syntax":
        return SyntaxAgentWrapper()
    return RAGPipelineWrapper()

