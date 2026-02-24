"""
RAG Validation Framework – CLI Entry Point.

Kommandos:
  generate-cases    Testfälle generieren (alle oder eine Kategorie)
  run               RAG-Pipeline gegen Testfälle ausführen
  evaluate          RAGAS + Security-Metriken berechnen
  report            HTML + JSON Report generieren
  validate          Alles in einem Schritt (generate → run → evaluate → report)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

import config

app = typer.Typer(name="rag-validator", help="RAG Validation Framework - Bachelorarbeit Marko Perkovic")
console = Console(highlight=False)


# ── generate-cases ────────────────────────────────────────────────────────────

@app.command("generate-cases")
def generate_cases(
    category: Annotated[
        str,
        typer.Option("--category", "-c", help="all | faithfulness | context_manipulation | direct_injection | corpus_poisoning | data_exfiltration")
    ] = "all",
    llm_variants: Annotated[bool, typer.Option("--llm-variants/--no-llm-variants", help="LLM-generierte Varianten hinzufügen")] = False,
) -> None:
    """Generiert adversariale Testfälle und speichert sie als JSON."""
    from test_generator import (
        FaithfulnessGenerator, ContextManipulationGenerator,
        DirectInjectionGenerator, CorpusPoisoningGenerator,
        DataExfiltrationGenerator,
    )
    from test_generator.llm_generator import LLMTestCaseGenerator

    generators = {
        "faithfulness":         FaithfulnessGenerator(),
        "context_manipulation": ContextManipulationGenerator(),
        "direct_injection":     DirectInjectionGenerator(),
        "corpus_poisoning":     CorpusPoisoningGenerator(),
        "data_exfiltration":    DataExfiltrationGenerator(),
    }

    selected = list(generators.items()) if category == "all" else [(category, generators[category])]

    total = 0
    for cat_name, gen in selected:
        with console.status(f"Generiere {cat_name}..."):
            cases = gen.generate_templates()

            if llm_variants:
                with console.status(f"  LLM-Varianten für {cat_name}..."):
                    llm_gen = LLMTestCaseGenerator()
                    sample = cases[:max(1, len(cases) // 3)]
                    llm_cases = llm_gen.diversify(sample, variants_per_case=2)
                    cases.extend(llm_cases)

            out_path = gen.save(cases, config.TEST_CASES_DIR)
            console.print(f"  [green]OK[/green] {cat_name}: {len(cases)} Testfaelle -> {out_path.name}")
            total += len(cases)

    console.print(Panel(f"[bold green]Gesamt: {total} Testfaelle generiert[/bold green]"))


# ── run ───────────────────────────────────────────────────────────────────────

@app.command("run")
def run_tests(
    category: Annotated[str, typer.Option("--category", "-c", help="all | <kategorie>")] = "all",
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximale Anzahl Tests pro Kategorie (0=alle)")] = 0,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Nur erste 3 Tests pro Kategorie")] = False,
) -> None:
    """Fuehrt den RAG-Agenten gegen Testfaelle aus und speichert Ergebnisse."""
    from rag.wrapper import RAGPipelineWrapper
    from rag.syntax_agent import SyntaxAgentWrapper, get_wrapper
    from test_generator.base import TestCase, BaseTestCaseGenerator

    # API-Key-Validierung je nach Target
    target = config.RAG_TARGET
    if target == "syntax":
        if not config.SYNTAX_AGENT_API_KEY or config.SYNTAX_AGENT_API_KEY.startswith("sk-5-..."):
            console.print("[red]FEHLER: SYNTAX_AGENT_API_KEY nicht gesetzt. Bitte .env befuellen.[/red]")
            raise typer.Exit(1)
        console.print(f"[bold cyan]Target: Syntax AI Studio Agent[/bold cyan]")
    else:
        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY.startswith("sk-ant-..."):
            console.print("[red]FEHLER: ANTHROPIC_API_KEY nicht gesetzt. Bitte .env befuellen.[/red]")
            raise typer.Exit(1)
        console.print(f"[bold cyan]Target: Demo RAG (ChromaDB + Claude)[/bold cyan]")

    test_files = list(config.TEST_CASES_DIR.glob("*.json"))
    if category != "all":
        test_files = [f for f in test_files if category in f.stem]

    if not test_files:
        console.print("[yellow]Keine Testfaelle gefunden. Zuerst 'generate-cases' ausfuehren.[/yellow]")
        raise typer.Exit(1)

    all_results = []
    n_limit = 3 if dry_run else (limit if limit > 0 else None)

    with console.status(f"Initialisiere {target.upper()} Agent..."):
        wrapper = get_wrapper()

    for test_file in test_files:
        cases = BaseTestCaseGenerator.load(test_file)
        if n_limit:
            cases = cases[:n_limit]

        console.print(f"\n[bold]Kategorie:[/bold] {test_file.stem} ({len(cases)} Tests)")

        for i, tc in enumerate(cases, 1):
            with console.status(f"  Test {i}/{len(cases)}: {tc.question[:60]}..."):
                # Syntax Agent: Injections direkt in die Frage einbetten
                # Demo RAG:     Injizierte Dokumente an ChromaDB übergeben
                if target == "syntax" and tc.injected_docs:
                    # Corpus-Poisoning-Simulation: Injizierte Inhalte in die Frage einbetten
                    poisoned_question = tc.question + "\n\n[Kontext aus Wissensbasis]:\n" + "\n".join(tc.injected_docs)
                    rag_result = wrapper.query(poisoned_question, extra_metadata={"test_case_id": tc.id, "poisoned": True})
                elif target == "demo" and tc.injected_docs:
                    w = RAGPipelineWrapper.with_injected_docs(tc.injected_docs)
                    rag_result = w.query(tc.question, extra_metadata={"test_case_id": tc.id})
                else:
                    rag_result = wrapper.query(tc.question, extra_metadata={"test_case_id": tc.id})

                all_results.append({
                    "test_case": tc.to_dict(),
                    "rag_result": {
                        "question": rag_result.question,
                        "answer":   rag_result.answer,
                        "contexts": rag_result.contexts,
                    },
                })

            console.print(f"  [dim]Q: {tc.question[:50]}...[/dim]")
            console.print(f"  [dim]A: {rag_result.answer[:80]}...[/dim]")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = config.RESULTS_DIR / "run_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    console.print(Panel(f"[bold green]{len(all_results)} Tests abgeschlossen → {results_file}[/bold green]"))


# ── evaluate ──────────────────────────────────────────────────────────────────

@app.command("evaluate")
def evaluate(
    ragas: Annotated[bool, typer.Option("--ragas/--no-ragas")] = True,
    security: Annotated[bool, typer.Option("--security/--no-security")] = True,
) -> None:
    """Berechnet RAGAS + Security-Metriken auf Basis der Run-Ergebnisse."""
    from rag.wrapper import RAGResult
    from evaluator.ragas_metrics import RAGASEvaluator, RAGASResult
    from evaluator.security_metrics import SecurityEvaluator, SecurityResult
    from test_generator.base import TestCase

    results_file = config.RESULTS_DIR / "run_results.json"
    if not results_file.exists():
        console.print("[red]Keine Run-Ergebnisse gefunden. Zuerst 'run' ausführen.[/red]")
        raise typer.Exit(1)

    with open(results_file, encoding="utf-8") as f:
        raw = json.load(f)

    test_cases = [TestCase.from_dict(r["test_case"]) for r in raw]
    rag_results = [
        RAGResult(
            question=r["rag_result"]["question"],
            answer=r["rag_result"]["answer"],
            retrieved_docs=[],
            contexts=r["rag_result"]["contexts"],
        )
        for r in raw
    ]

    ragas_result = RAGASResult()
    if ragas:
        with console.status("Berechne RAGAS-Metriken (Claude als Judge)..."):
            evaluator = RAGASEvaluator()
            ground_truths = [tc.ground_truth for tc in test_cases]
            ragas_result = evaluator.evaluate(rag_results, ground_truths)

        table = Table(title="RAGAS-Metriken")
        table.add_column("Metrik")
        table.add_column("Score")
        table.add_column("Schwellenwert")
        for name, val in ragas_result.to_dict().items():
            gate = config.QUALITY_GATES.get(name, {})
            thr  = gate.get("threshold", "-")
            op   = gate.get("operator", ">=")
            passed = (val >= thr) if val and isinstance(thr, float) else None
            status = "[green]✓[/green]" if passed else "[red]✗[/red]" if passed is False else "-"
            table.add_row(name, f"{val:.3f}" if val else "N/A", f"{status} {op}{thr}")
        console.print(table)

    security_result = SecurityResult()
    if security:
        with console.status("Berechne Security-Metriken..."):
            sec_eval = SecurityEvaluator()
            security_result = sec_eval.evaluate_batch(test_cases, rag_results)

        sec_table = Table(title="Security-Metriken")
        sec_table.add_column("Metrik")
        sec_table.add_column("Wert")
        sec_table.add_column("Schwellenwert")
        for label, val, key in [
            ("Attack Success Rate (ASR)", security_result.asr, "asr"),
            ("Instruction Override Rate", security_result.instruction_override_rate, "instruction_override_rate"),
            ("Data Exfiltration Success", security_result.data_exfiltration_success, "data_exfiltration_success"),
        ]:
            gate   = config.QUALITY_GATES.get(key, {})
            thr    = gate.get("threshold", 0.10)
            passed = val <= thr
            status = "[green]✓[/green]" if passed else "[red]✗[/red]"
            sec_table.add_row(label, f"{val:.3f}", f"{status} <={thr}")
        console.print(sec_table)

    metrics_file = config.RESULTS_DIR / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "ragas":    ragas_result.to_dict(),
            "security": security_result.to_dict(),
        }, f, ensure_ascii=False, indent=2)

    console.print(f"\n[green]Metriken gespeichert: {metrics_file}[/green]")


# ── report ────────────────────────────────────────────────────────────────────

@app.command("report")
def report(
    fmt: Annotated[str, typer.Option("--format", "-f", help="html | json | both")] = "both",
) -> None:
    """Generiert HTML + JSON Report aus den Evaluationsergebnissen."""
    from evaluator.ragas_metrics import RAGASResult
    from evaluator.security_metrics import SecurityResult
    from ci_cd.quality_gates import QualityGateEvaluator
    from ci_cd.report import ReportGenerator

    metrics_file = config.RESULTS_DIR / "metrics.json"
    if not metrics_file.exists():
        console.print("[red]Keine Metriken gefunden. Zuerst 'evaluate' ausführen.[/red]")
        raise typer.Exit(1)

    with open(metrics_file, encoding="utf-8") as f:
        data = json.load(f)

    ragas_data    = data.get("ragas", {})
    security_data = data.get("security", {})

    ragas_result = RAGASResult(
        faithfulness=ragas_data.get("faithfulness"),
        answer_relevancy=ragas_data.get("answer_relevancy"),
        context_precision=ragas_data.get("context_precision"),
        context_recall=ragas_data.get("context_recall"),
    )
    security_result = SecurityResult()
    security_result.asr                       = security_data.get("asr", 0.0)
    security_result.instruction_override_rate = security_data.get("instruction_override_rate", 0.0)
    security_result.data_exfiltration_success = security_data.get("data_exfiltration_success", 0.0)

    all_metrics = {**ragas_result.to_dict(), **{
        "asr": security_result.asr,
        "instruction_override_rate": security_result.instruction_override_rate,
        "data_exfiltration_success": security_result.data_exfiltration_success,
    }}
    gate_result = QualityGateEvaluator().evaluate(all_metrics)

    console.print(Panel(gate_result.summary(), title="Quality Gate Ergebnis"))

    gen = ReportGenerator()
    paths = gen.generate(
        ragas_result=ragas_result,
        security_result=security_result,
        gate_result=gate_result,
        total_tests=security_data.get("total_tests", 0),
    )

    console.print(f"[green]JSON Report:[/green] {paths['json']}")
    console.print(f"[green]HTML Report:[/green] {paths['html']}")


# ── validate (all-in-one) ────────────────────────────────────────────────────

@app.command("validate")
def validate_all(
    category: Annotated[str, typer.Option("--category", "-c")] = "all",
    limit: Annotated[int, typer.Option("--limit", "-n")] = 0,
) -> None:
    """Vollständige Validierung: generate-cases → run → evaluate → report."""
    console.print(Panel("[bold]RAG Validation Framework – Vollständige Validierung[/bold]"))

    console.rule("Schritt 1: Testfälle generieren")
    generate_cases(category=category, llm_variants=False)

    console.rule("Schritt 2: RAG-Pipeline ausführen")
    run_tests(category=category, limit=limit, dry_run=False)

    console.rule("Schritt 3: Metriken berechnen")
    evaluate(ragas=True, security=True)

    console.rule("Schritt 4: Report generieren")
    report(fmt="both")


# ── ping ─────────────────────────────────────────────────────────────────────

@app.command("ping")
def ping(
    debug: Annotated[bool, typer.Option("--debug", help="Zeigt die vollstaendige rohe API-Response (wichtig fuer Retrieval-Analyse)")] = False,
    question: Annotated[str, typer.Option("--question", "-q", help="Test-Frage")] = "Was sind die Urlaubsregelungen?",
) -> None:
    """Testet die Verbindung und zeigt optional die rohe API-Response."""
    target = config.RAG_TARGET
    console.print(f"Teste Verbindung zu: [bold]{target.upper()}[/bold]")

    if target == "syntax":
        from rag.syntax_agent import SyntaxAgentWrapper
        if not config.SYNTAX_AGENT_API_KEY:
            console.print("[red]SYNTAX_AGENT_API_KEY nicht gesetzt.[/red]")
            raise typer.Exit(1)
        wrapper = SyntaxAgentWrapper()

        if debug:
            # ── Debug-Modus: rohe API-Response anzeigen ────────────────────
            console.print(f"\n[bold]Frage:[/bold] {question}")
            console.print("[bold]Rohe API-Response (vollstaendig):[/bold]\n")
            with console.status("Sende Anfrage..."):
                try:
                    raw = wrapper.debug_raw(question)
                except Exception as e:
                    console.print(f"[red]Fehler: {e}[/red]")
                    raise typer.Exit(1)

            console.print(json.dumps(raw, indent=2, ensure_ascii=False))
            console.print("\n[bold]Analyse der Response-Felder:[/bold]")
            console.print(f"  Top-Level Keys: {list(raw.keys())}")

            # Zeige ob Kontext-Felder vorhanden sind
            context_fields = ["citations", "sourceDocuments", "source_documents",
                              "context", "intermediate_steps", "sources",
                              "references", "chunks", "documents", "passages"]
            found = [f for f in context_fields if f in raw]
            if found:
                console.print(f"  [green]Kontext-Felder gefunden: {found}[/green]")
                # citations separat prüfen (Syntax AI Studio eigenes Format)
                if "citations" in raw:
                    cites = raw["citations"]
                    if cites:
                        console.print(f"  [green]citations: {len(cites)} Eintraege -> RAGAS Context-Metriken berechenbar![/green]")
                    else:
                        console.print("  [yellow]citations: leer (Agent hat keine passenden Dokumente gefunden)[/yellow]")
                        console.print("  -> Frage zu einem Thema stellen das der Agent kennt, um citations zu sehen")
                other = [f for f in found if f != "citations"]
                if other:
                    console.print(f"  -> Weitere Kontext-Felder: {other}")
            else:
                console.print(f"  [yellow]Keine Kontext-Felder in Response.[/yellow]")
                console.print("  -> Siehe Optionen in rag/syntax_agent.py Docstring")
        else:
            # ── Normaler Verbindungstest ───────────────────────────────────
            with console.status("Sende Test-Anfrage an Syntax AI Studio..."):
                try:
                    result = wrapper.query(question)
                    console.print(f"[green]Verbindung OK![/green]")
                    console.print(f"Antwort: {result.answer[:300]}")
                    if result.contexts:
                        console.print(f"[green]Kontext abgerufen: {len(result.contexts)} Chunk(s)[/green]")
                        console.print(f"  Erster Chunk (100 Zeichen): {result.contexts[0][:100]}...")
                    else:
                        console.print("[yellow]Kein Kontext in Response. Fuer Details: --debug[/yellow]")
                except Exception as e:
                    console.print(f"[red]Verbindungsfehler: {e}[/red]")
                    raise typer.Exit(1)
    else:
        console.print("[yellow]Target ist 'demo' (lokale Pipeline). Kein Remote-Ping noetig.[/yellow]")
        console.print("Setze RAG_TARGET=syntax in .env um den echten Agenten zu testen.")


if __name__ == "__main__":
    app()
