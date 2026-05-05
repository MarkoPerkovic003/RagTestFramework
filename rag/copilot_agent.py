"""
Microsoft Copilot Studio Agent Wrapper.

Kommuniziert mit Copilot Studio Agents über die Direct Line API v3.0.

Ablauf:
  1. Neue Konversation starten (POST /conversations)
  2. Nachricht senden              (POST /conversations/{id}/activities)
  3. Antwort pollen                (GET  /conversations/{id}/activities?watermark=...)

Endpunkt:
    https://directline.botframework.com/v3/directline/

Umgebungsvariablen:
    COPILOT_DIRECT_LINE_SECRET  Direct Line Secret aus Copilot Studio
    COPILOT_BOT_HANDLE          Optionaler Bot-Handle (für Display)
    AGENT_URL                   Override des Direct-Line-Endpunkts (optional)
"""

from __future__ import annotations
import os
import time
import requests

from rag.base_agent import BaseAgentWrapper
from rag.wrapper import RAGResult

_DIRECT_LINE_BASE = "https://directline.botframework.com/v3/directline"
_POLL_INTERVAL    = 1.0   # Sekunden zwischen Polls
_MAX_POLLS        = 30    # max 30 Versuche → 30 s Timeout


class CopilotStudioAgentWrapper(BaseAgentWrapper):
    """
    Wrapper für Microsoft Copilot Studio Agents via Direct Line API.

    Jede query()-Anfrage startet eine neue Konversation, sendet die Frage
    und wartet auf die Bot-Antwort (polling). Stateless – kein Session-Reuse.
    """

    LABEL      = "Microsoft Copilot Studio"
    AGENT_TYPE = "copilot"

    def __init__(
        self,
        direct_line_secret: str,
        base_url: str = _DIRECT_LINE_BASE,
        bot_handle: str = "",
        timeout: int = 60,
    ) -> None:
        self._secret     = direct_line_secret
        self._base_url   = base_url.rstrip("/")
        self._bot_handle = bot_handle
        self._timeout    = timeout

    @classmethod
    def from_env(cls) -> "CopilotStudioAgentWrapper":
        return cls(
            direct_line_secret = os.getenv("COPILOT_DIRECT_LINE_SECRET", ""),
            base_url           = os.getenv("AGENT_URL", _DIRECT_LINE_BASE),
            bot_handle         = os.getenv("COPILOT_BOT_HANDLE", ""),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def query(self, question: str, extra_metadata: dict | None = None) -> RAGResult:
        """Sendet eine Frage an den Copilot Studio Agent und gibt das Ergebnis zurück."""
        if not self._secret:
            raise ValueError(
                "COPILOT_DIRECT_LINE_SECRET nicht gesetzt. "
                "Bitte in .env eintragen oder als Parameter übergeben."
            )

        headers = {
            "Authorization": f"Bearer {self._secret}",
            "Content-Type":  "application/json",
        }

        # 1. Neue Konversation starten
        conv_id, token = self._start_conversation(headers)

        # 2. Frage senden
        watermark = self._send_message(conv_id, question, token or self._secret)

        # 3. Antwort pollen
        answer, citations = self._poll_answer(conv_id, watermark, token or self._secret)

        return RAGResult(
            question=question,
            answer=answer,
            retrieved_docs=[],
            contexts=citations,
            metadata={
                "source":         "copilot_studio",
                "conversation_id": conv_id,
                **(extra_metadata or {}),
            },
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _start_conversation(self, headers: dict) -> tuple[str, str]:
        """
        Startet eine neue Direct Line Konversation.

        Returns:
            (conversation_id, token)  – token kann leer sein wenn nur Secret verwendet wird
        """
        resp = requests.post(
            f"{self._base_url}/conversations",
            headers=headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data  = resp.json()
        return data["conversationId"], data.get("token", "")

    def _send_message(self, conv_id: str, text: str, token: str) -> str:
        """
        Sendet eine Textnachricht an den Bot.

        Returns:
            watermark – wird für das Response-Polling benötigt
        """
        payload = {
            "type": "message",
            "from": {"id": "rag-test-framework", "name": "Tester"},
            "text": text,
        }
        resp = requests.post(
            f"{self._base_url}/conversations/{conv_id}/activities",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
            },
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("id", "")

    def _poll_answer(
        self,
        conv_id: str,
        sent_activity_id: str,
        token: str,
    ) -> tuple[str, list[str]]:
        """
        Pollt den Activity-Stream bis eine Bot-Antwort erscheint.

        Der watermark-Parameter sorgt dafür dass nur neue Activities abgerufen
        werden. Eine Activity mit role='bot' ist die gesuchte Antwort.

        Returns:
            (answer_text, citations_list)
        """
        # Watermark aus der gesendeten Activity-ID ableiten
        # (Direct Line: watermarks sind Zahlen oder Strings)
        watermark: str | None = None

        for _ in range(_MAX_POLLS):
            time.sleep(_POLL_INTERVAL)

            url = f"{self._base_url}/conversations/{conv_id}/activities"
            if watermark:
                url += f"?watermark={watermark}"

            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            watermark = data.get("watermark", watermark)

            for activity in data.get("activities", []):
                # Bot-Antworten haben from.role = "bot"
                frm  = activity.get("from", {})
                role = frm.get("role", "") or frm.get("id", "")
                if role not in ("bot", "channel") and activity.get("type") != "message":
                    continue
                # Eigene Nachricht ausfiltern
                if frm.get("id") == "rag-test-framework":
                    continue
                if activity.get("type") != "message":
                    continue

                text      = activity.get("text", "").strip()
                citations = self._extract_citations(activity)

                if text:
                    return text, citations

        return "", []

    @staticmethod
    def _extract_citations(activity: dict) -> list[str]:
        """
        Extrahiert Zitate / Quelldokumente aus einer Bot-Activity.

        Copilot Studio kann Quellen als:
          - attachments mit contentType 'application/vnd.microsoft.card.adaptive'
          - entities vom Typ 'citation'
          - channelData.references
        zurückgeben.
        """
        citations: list[str] = []

        # ── entities: type=citation ────────────────────────────────────────
        for entity in activity.get("entities", []):
            if entity.get("type") == "citation":
                text = entity.get("content", "") or entity.get("name", "")
                if text:
                    citations.append(str(text))

        # ── channelData.references ────────────────────────────────────────
        channel_data = activity.get("channelData", {})
        for ref in channel_data.get("references", []):
            text = ref.get("content", "") or ref.get("title", "") or str(ref)
            if text:
                citations.append(text)

        # ── suggestedActions als Quellen-Hinweise ─────────────────────────
        # (selten, aber manche Bots schicken Quellen als suggested actions)

        return citations

