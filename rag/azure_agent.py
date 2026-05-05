"""
Azure OpenAI Agent Wrapper.

Unterstützt:
  - Azure OpenAI Chat Completions  (GPT-4o, GPT-4, GPT-3.5-Turbo)
  - Azure AI Foundry / AI Studio   (gleiche API-Struktur)
  - "On Your Data" RAG             (azure_search_endpoint gesetzt)

Endpunkt-Format:
    https://<resource>.openai.azure.com/openai/deployments/<deployment>/
    chat/completions?api-version=<api_version>

Umgebungsvariablen:
    AGENT_URL          Azure OpenAI Resource URL
                       z.B. https://my-resource.openai.azure.com/
    AGENT_API_KEY      Azure OpenAI API Key
    AZURE_DEPLOYMENT   Deployment-Name  (Standard: gpt-4o)
    AZURE_API_VERSION  API-Version      (Standard: 2024-02-01)
    AZURE_SEARCH_URL   Azure AI Search Endpunkt (optional, für "On Your Data")
    AZURE_SEARCH_KEY   Azure AI Search API Key  (optional)
    AZURE_SEARCH_INDEX Suchindex-Name            (optional)
    AZURE_SYSTEM_PROMPT System-Prompt            (optional)
"""

from __future__ import annotations
import os

import requests

from rag.base_agent import BaseAgentWrapper
from rag.wrapper import RAGResult

_DEFAULT_SYSTEM_PROMPT = (
    "Du bist ein hilfreicher Assistent. "
    "Antworte präzise und auf Basis der bereitgestellten Informationen."
)


class AzureOpenAIAgentWrapper(BaseAgentWrapper):
    """
    Wrapper für Azure OpenAI Chat Completions.

    Unterstützt optional "On Your Data" (Azure AI Search), das dem
    Standard-Azure-OpenAI-Aufruf einen RAG-Retrieval-Schritt hinzufügt
    und Quelldokumente im Response zurückgibt.
    """

    LABEL      = "Azure OpenAI"
    AGENT_TYPE = "azure"

    def __init__(
        self,
        resource_url:   str,
        api_key:        str,
        deployment:     str = "gpt-4o",
        api_version:    str = "2024-02-01",
        system_prompt:  str = _DEFAULT_SYSTEM_PROMPT,
        search_url:     str = "",
        search_key:     str = "",
        search_index:   str = "",
        timeout:        int = 120,
    ) -> None:
        self._resource_url  = resource_url.rstrip("/")
        self._api_key       = api_key
        self._deployment    = deployment
        self._api_version   = api_version
        self._system_prompt = system_prompt
        self._search_url    = search_url
        self._search_key    = search_key
        self._search_index  = search_index
        self._timeout       = timeout

    @classmethod
    def from_env(cls) -> "AzureOpenAIAgentWrapper":
        return cls(
            resource_url  = os.getenv("AGENT_URL", ""),
            api_key       = os.getenv("AGENT_API_KEY", ""),
            deployment    = os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
            api_version   = os.getenv("AZURE_API_VERSION", "2024-02-01"),
            system_prompt = os.getenv("AZURE_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT),
            search_url    = os.getenv("AZURE_SEARCH_URL", ""),
            search_key    = os.getenv("AZURE_SEARCH_KEY", ""),
            search_index  = os.getenv("AZURE_SEARCH_INDEX", ""),
        )

    def _build_endpoint(self) -> str:
        return (
            f"{self._resource_url}/openai/deployments/{self._deployment}"
            f"/chat/completions?api-version={self._api_version}"
        )

    def _build_body(self, question: str) -> dict:
        body: dict = {
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user",   "content": question},
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
        }
        # "On Your Data" RAG-Erweiterung
        if self._search_url and self._search_key and self._search_index:
            body["data_sources"] = [{
                "type": "azure_search",
                "parameters": {
                    "endpoint":   self._search_url,
                    "index_name": self._search_index,
                    "authentication": {
                        "type":    "api_key",
                        "api_key": self._search_key,
                    },
                },
            }]
        return body

    def query(self, question: str, extra_metadata: dict | None = None) -> RAGResult:
        endpoint = self._build_endpoint()
        headers  = {
            "Content-Type": "application/json",
            "api-key":      self._api_key,
        }
        response = requests.post(
            endpoint,
            headers=headers,
            json=self._build_body(question),
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()

        # Antwort extrahieren
        choices = data.get("choices", [])
        answer  = ""
        if choices:
            msg = choices[0].get("message", {})
            answer = msg.get("content", "").strip()

        # Kontexte extrahieren ("On Your Data" gibt citations zurück)
        contexts: list[str] = []
        if choices:
            msg      = choices[0].get("message", {})
            ctx_data = msg.get("context", {})
            cits     = ctx_data.get("citations", [])
            for cit in cits:
                text = cit.get("content") or cit.get("title", "")
                if text:
                    contexts.append(text)

        return RAGResult(
            question=question,
            answer=answer,
            retrieved_docs=[],
            contexts=contexts,
            metadata={"source": "azure_openai", **(extra_metadata or {})},
        )

