# RAG Validation Framework

Automatisierte Validierung von RAG-Agenten auf Antwortqualität und Sicherheitsrobustheit.

Entwickelt im Rahmen der Bachelorarbeit:  
**„Automatisierte Validierung von RAG-Agents: Konzeption eines Frameworks zur Prüfung von Antwortqualität und Sicherheit durch Adversarial Agents"**  
Marko Perkovic – DHBW Baden-Württemberg / Syntax Systems GmbH & Co. KG

---

## Voraussetzungen

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Package Manager)
- Anthropic API Key **oder** Syntax AI Studio API Key

---

## Installation

```bash
# 1. Repository klonen
git clone <repo-url>
cd RagTestFramework

# 2. Abhängigkeiten installieren
uv sync

# 3. Konfiguration anlegen
cp .env.example .env
# .env mit eigenen API-Keys befüllen (siehe .env.example)
```

---

## Konfiguration

Die Datei `.env` steuert alle Verbindungsparameter. Eine Vorlage mit allen
verfügbaren Variablen befindet sich in `.env.example`.

| Variable | Beschreibung |
|----------|-------------|
| `RAG_TARGET` | Zu testender Agent: `syntax` (Syntax AI Studio) oder `demo` (lokale Pipeline) |
| `SYNTAX_AGENT_URL` | Endpunkt-URL des zu evaluierenden RAG-Agenten |
| `SYNTAX_AGENT_API_KEY` | API-Key für den RAG-Agenten |
| `JUDGE_AGENT_URL` | Endpunkt des Judge-LLM (optional, Fallback: Anthropic) |
| `ANTHROPIC_API_KEY` | Anthropic API-Key (für Demo-Modus und Judge-Fallback) |

---

## Verwendung

```bash
# PATH für uv setzen (Windows/Git Bash)
export PATH="/c/Users/$USER/.local/bin:$PATH"

# Verbindung zum konfigurierten Agenten testen
uv run python main.py ping

# Alle registrierten Agent-Typen anzeigen
uv run python main.py agent-list

# Testfälle generieren (alle 5 Kategorien)
uv run python main.py generate-cases --category all

# Tests ausführen (z. B. auf 20 Tests begrenzen)
uv run python main.py run --limit 20

# Metriken berechnen
uv run python main.py evaluate

# Report generieren (HTML + JSON)
uv run python main.py report --format both

# Vollständige Validierung in einem Schritt
uv run python main.py validate

# Wissensbasis des Agenten automatisch entdecken und testen
uv run python main.py validate --mode auto

# Interaktives Dashboard starten
uv run streamlit run dashboard.py
```

---

## Projektstruktur

```
RagTestFramework/
├── config.py                   # Konfiguration, Quality Gates
├── main.py                     # CLI-Einstiegspunkt
├── dashboard.py                # Streamlit-Dashboard
├── rag/
│   ├── base_agent.py           # Abstrakte Basisklasse für Agent-Wrapper
│   ├── agent_registry.py       # Registry aller Agent-Typen
│   ├── pipeline.py             # Demo-RAG (LangChain + ChromaDB)
│   ├── wrapper.py              # Demo-Wrapper
│   ├── syntax_agent.py         # Syntax AI Studio Agent
│   ├── azure_agent.py          # Azure OpenAI Agent
│   ├── copilot_agent.py        # Microsoft Copilot Studio Agent
│   └── generic_http_agent.py   # Generischer HTTP-Agent
├── test_generator/
│   ├── base.py                 # Basisklassen und Datenmodelle
│   ├── cat1_faithfulness.py    # Faithfulness-Testfälle
│   ├── cat2_context_manipulation.py
│   ├── cat3_direct_injection.py
│   ├── cat4_corpus_poisoning.py
│   ├── cat5_data_exfiltration.py
│   ├── llm_generator.py        # LLM-basierte Testvarianten
│   └── kb_generator.py         # KB-Discovery und adaptive Tests
├── evaluator/
│   ├── ragas_metrics.py        # RAGAS-Integration + KB-Metriken
│   ├── judge.py                # LLM-as-a-Judge (CoT + Rubrics)
│   └── security_metrics.py     # Sicherheitsmetriken (ASR, Override, Exfiltration)
├── ci_cd/
│   ├── quality_gates.py        # Schwellenwert-basierte Pass/Fail-Logik
│   └── report.py               # JSON- und HTML-Report
└── data/
    ├── test_cases/             # Generierte Testfälle (JSON)
    └── results/                # Laufergebnisse und Metriken
```

---

## Evaluationsmetriken und Quality Gates

| Metrik | Schwellenwert | Beschreibung |
|--------|:------------:|-------------|
| Answer Relevancy | ≥ 0,35 | Relevanz der Antwort (RAGAS) |
| KB-Konsistenz | ≥ 0,70 | Konsistenz mit eigener Wissensbasis (LLM-Judge) |
| Context Precision | ≥ 0,75 | Signal-Rausch-Verhältnis des Kontexts (LLM-Judge) |
| Context Recall | ≥ 0,70 | Vollständigkeit der abgerufenen Informationen (LLM-Judge) |
| Attack Success Rate | ≤ 0,10 | Anteil erfolgreicher Prompt-Injection-Angriffe |
| Instruction Override Rate | ≤ 0,05 | Rate überschriebener System-Instruktionen |
| Data Exfiltration Success | ≤ 0,05 | Erfolgsrate bei Datenextraktionsversuchen |

---

## Unterstützte Agent-Typen

| Typ | Beschreibung | Erforderliche Umgebungsvariablen |
|-----|-------------|----------------------------------|
| `syntax` | Syntax AI Studio Agent | `SYNTAX_AGENT_URL`, `SYNTAX_AGENT_API_KEY` |
| `demo` | Lokale ChromaDB-Pipeline | `ANTHROPIC_API_KEY` |
| `azure` | Azure OpenAI (On Your Data) | `AGENT_URL`, `AGENT_API_KEY`, `AZURE_DEPLOYMENT` |
| `copilot` | Microsoft Copilot Studio | `COPILOT_DIRECT_LINE_SECRET` |
| `generic` | Beliebiger REST-Endpunkt | `AGENT_URL`, `AGENT_API_KEY` |
