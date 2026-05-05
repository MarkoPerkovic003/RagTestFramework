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
    variants_per_case: Annotated[int, typer.Option("--variants-per-case", help="LLM-Varianten je Template (Standard: 2)")] = 2,
    sample_ratio: Annotated[float, typer.Option("--sample-ratio", help="Anteil der Templates fuer LLM-Varianten (0.0–1.0, Standard: 0.5)")] = 0.5,
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
                with console.status(f"  LLM-Varianten fuer {cat_name} ({variants_per_case} je Template, {sample_ratio:.0%} Sample)..."):
                    llm_gen = LLMTestCaseGenerator()
                    n_sample = max(1, int(len(cases) * sample_ratio))
                    sample = cases[:n_sample]
                    llm_cases = llm_gen.diversify(sample, variants_per_case=variants_per_case)
                    cases.extend(llm_cases)

            out_path = gen.save(cases, config.TEST_CASES_DIR)
            console.print(f"  [green]OK[/green] {cat_name}: {len(cases)} Testfaelle -> {out_path.name}")
            total += len(cases)

    console.print(Panel(f"[bold green]Gesamt: {total} Testfaelle generiert[/bold green]"))


# ── run ───────────────────────────────────────────────────────────────────────

@app.command("agent-list")
def agent_list() -> None:
    """Listet alle registrierten Agent-Typen auf."""
    from rag.agent_registry import list_types, labels as registry_labels

    table = Table(title="Registrierte Agent-Typen")
    table.add_column("Typ",          style="bold cyan")
    table.add_column("Anzeigename")
    table.add_column("Env-Variable")

    env_hints = {
        "syntax":  "SYNTAX_AGENT_URL + SYNTAX_AGENT_API_KEY",
        "demo":    "ANTHROPIC_API_KEY",
        "azure":   "AGENT_URL + AGENT_API_KEY + AZURE_DEPLOYMENT",
        "copilot": "COPILOT_DIRECT_LINE_SECRET",
        "generic": "AGENT_URL + AGENT_API_KEY",
    }

    lbls = registry_labels()
    for t in list_types():
        table.add_row(t, lbls.get(t, "–"), env_hints.get(t, "–"))

    console.print(table)
    console.print(Panel(f"[dim]Aktuell aktiv: [bold]{config.RAG_TARGET}[/bold]  (RAG_TARGET)[/dim]"))


@app.command("run")
def run_tests(
    category: Annotated[str, typer.Option("--category", "-c", help="all | <kategorie>")] = "all",
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximale Anzahl Tests pro Kategorie (0=alle)")] = 0,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Nur erste 3 Tests pro Kategorie")] = False,
) -> None:
    """Fuehrt den RAG-Agenten gegen Testfaelle aus und speichert Ergebnisse."""
    from rag.wrapper import RAGPipelineWrapper
    from rag.agent_registry import get_wrapper as _get_wrapper
    from test_generator.base import TestCase, BaseTestCaseGenerator

    # API-Key-Validierung je nach Target
    target = config.RAG_TARGET
    if target == "demo":
        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY.startswith("sk-ant-..."):
            console.print("[red]FEHLER: ANTHROPIC_API_KEY nicht gesetzt. Bitte .env befuellen.[/red]")
            raise typer.Exit(1)

    from rag.agent_registry import labels as _labels
    agent_label = _labels().get(target, target)
    console.print(f"[bold cyan]Target: {agent_label}[/bold cyan]")

    test_files = list(config.TEST_CASES_DIR.glob("*.json"))
    if category != "all":
        test_files = [f for f in test_files if category in f.stem]

    if not test_files:
        console.print("[yellow]Keine Testfaelle gefunden. Zuerst 'generate-cases' ausfuehren.[/yellow]")
        raise typer.Exit(1)

    all_results = []
    n_limit = 3 if dry_run else (limit if limit > 0 else None)

    with console.status(f"Initialisiere {target.upper()} Agent..."):
        wrapper = _get_wrapper()

    consecutive_errors = 0
    aborted = False

    for test_file in test_files:
        if aborted:
            break

        cases = BaseTestCaseGenerator.load(test_file)
        if n_limit:
            cases = cases[:n_limit]

        console.print(f"\n[bold]Kategorie:[/bold] {test_file.stem} ({len(cases)} Tests)")

        for i, tc in enumerate(cases, 1):
            q_safe = tc.question.encode("ascii", errors="replace").decode("ascii")
            with console.status(f"  Test {i}/{len(cases)}: {q_safe[:60]}..."):
                try:
                    # Injizierte Dokumente (Corpus-Poisoning-Simulation):
                    # Demo RAG:     Injizierte Dokumente an ChromaDB übergeben
                    # Alle anderen: Inhalte direkt in die Frage einbetten
                    if target == "demo" and tc.injected_docs:
                        w = RAGPipelineWrapper.with_injected_docs(tc.injected_docs)
                        rag_result = w.query(tc.question, extra_metadata={"test_case_id": tc.id})
                    elif target != "demo" and tc.injected_docs:
                        poisoned_question = tc.question + "\n\n[Kontext aus Wissensbasis]:\n" + "\n".join(tc.injected_docs)
                        rag_result = wrapper.query(poisoned_question, extra_metadata={"test_case_id": tc.id, "poisoned": True})
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
                    a_safe = rag_result.answer.encode("ascii", errors="replace").decode("ascii")
                    console.print(f"  [dim]Q: {q_safe[:50]}...[/dim]")
                    console.print(f"  [dim]A: {a_safe[:80]}...[/dim]")
                    consecutive_errors = 0  # Reset bei Erfolg
                except Exception as e:
                    consecutive_errors += 1
                    console.print(f"  [yellow]SKIP Test {i} ({tc.id}): {e}[/yellow]")
                    # Abbruch nach 3 aufeinanderfolgenden Fehlern (Server wahrscheinlich down)
                    if consecutive_errors >= 3:
                        console.print("[red]3 Fehler hintereinander – Run abgebrochen. Bisherige Ergebnisse werden gespeichert.[/red]")
                        aborted = True
                        break

    import shutil
    from datetime import datetime

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_file = config.RESULTS_DIR / f"run_results_{ts}.json"
    latest_file      = config.RESULTS_DIR / "run_results.json"

    with open(timestamped_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    shutil.copy2(timestamped_file, latest_file)

    console.print(Panel(
        f"[bold green]{len(all_results)} Tests abgeschlossen[/bold green]\n"
        f"Timestamped: {timestamped_file.name}\n"
        f"Latest:      {latest_file.name}"
    ))


# ── evaluate ──────────────────────────────────────────────────────────────────

@app.command("evaluate")
def evaluate(
    ragas: Annotated[bool, typer.Option("--ragas/--no-ragas")] = True,
    security: Annotated[bool, typer.Option("--security/--no-security")] = True,
    judge_quality: Annotated[bool, typer.Option(
        "--judge-quality/--no-judge-quality",
        help="Fehlende Qualitaetsmetriken via LLM-Judge berechnen (Fallback wenn kein Kontext)"
    )] = True,
    ragas_category: Annotated[str, typer.Option(
        "--ragas-category",
        help="Kategorie fuer RAGAS-Evaluation (default: faithfulness | 'all' fuer alle)"
    )] = "faithfulness",
    results_input: Annotated[str, typer.Option(
        "--results-file", "-r",
        help="Run-Ergebnis-Datei (default: run_results.json)"
    )] = "run_results.json",
    kb_file: Annotated[str | None, typer.Option(
        "--kb-file",
        help="KB-Dokument(e): Pfad zu .txt/.md Datei oder Verzeichnis. "
             "Aktiviert KB-basierte Context Precision + Recall via LLM-Judge."
    )] = None,
) -> None:
    """Berechnet RAGAS + Security-Metriken auf Basis der Run-Ergebnisse."""
    from rag.wrapper import RAGResult
    from evaluator.ragas_metrics import RAGASEvaluator, RAGASResult
    from evaluator.security_metrics import SecurityEvaluator, SecurityResult
    from test_generator.base import TestCase

    results_file = config.RESULTS_DIR / results_input
    if not results_file.exists():
        console.print(f"[red]Keine Run-Ergebnisse gefunden: {results_file}[/red]")
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
        # Nur Qualitäts-Tests für RAGAS (Standard: faithfulness).
        # Security-Tests lehnen Angriffe korrekt ab → RAGAS würde das als
        # "irrelevant" werten und einen künstlich niedrigen Score erzeugen.
        if ragas_category == "all":
            ragas_pairs = list(zip(test_cases, rag_results))
        else:
            ragas_pairs = [
                (tc, rr) for tc, rr in zip(test_cases, rag_results)
                if tc.category.value == ragas_category
            ]

        if not ragas_pairs:
            console.print(
                f"[yellow]Warnung: Keine Tests der Kategorie '{ragas_category}' in den "
                f"Ergebnissen. RAGAS wird übersprungen.[/yellow]"
            )
        else:
            ragas_tcs, ragas_rrs = zip(*ragas_pairs)
            console.print(
                f"[dim]RAGAS: {len(ragas_rrs)} Tests (Kategorie: {ragas_category})[/dim]"
            )
            with console.status("Berechne RAGAS-Metriken..."):
                evaluator = RAGASEvaluator()
                ground_truths = [tc.ground_truth for tc in ragas_tcs]
                ragas_result = evaluator.evaluate(list(ragas_rrs), ground_truths)

        table = Table(title="RAGAS-Metriken")
        table.add_column("Metrik")
        table.add_column("Score")
        table.add_column("Schwellenwert")
        for name, val in ragas_result.to_dict().items():
            gate = config.QUALITY_GATES.get(name, {})
            thr  = gate.get("threshold", "-")
            op   = gate.get("operator", ">=")
            passed = (val >= thr) if val and isinstance(thr, float) else None
            status = "[green]OK[/green]" if passed else "[red]FAIL[/red]" if passed is False else "-"
            table.add_row(name, f"{val:.3f}" if val else "N/A", f"{status} {op}{thr}")
        console.print(table)

    # ── KB-Qualität via LLM-Judge (KB-entdeckte Testfälle mit ground_truth) ────
    # Läuft IMMER wenn KB-Discovery-Tests vorhanden sind (metadata.source="discover_kb").
    # Nutzt ground_truth (echte Agenten-Antwort aus KB-Discovery) als Referenz.
    # Ergibt kb_consistency: Wie konsistent antwortet der Agent gegenüber seiner KB?
    if judge_quality:
        _kb_pairs = [
            (tc, rr) for tc, rr in zip(test_cases, rag_results)
            if tc.category.value == "faithfulness"
            and isinstance(getattr(tc, "metadata", None), dict)
            and tc.metadata.get("source") == "discover_kb"
            and tc.ground_truth
        ]
        if _kb_pairs:
            _kb_tcs, _kb_rrs = zip(*_kb_pairs)
            console.print(f"[dim]KB-Judge: {len(_kb_tcs)} KB-Testfaelle mit Referenz[/dim]")
            with console.status("Berechne KB-Konsistenz via LLM-Judge..."):
                from evaluator.judge import LLMJudge
                _judge = LLMJudge()
                _faith_sc: list[float] = []
                _rel_sc:   list[float] = []
                for _tc, _rr in zip(_kb_tcs, _kb_rrs):
                    _v = _judge.evaluate_quality(_rr, ground_truth=_tc.ground_truth)
                    if _v.faithfulness_score is not None:
                        _faith_sc.append(_v.faithfulness_score / 10.0)
                    if _v.relevancy_score is not None:
                        _rel_sc.append(_v.relevancy_score / 10.0)
            if _faith_sc:
                ragas_result.kb_consistency = round(sum(_faith_sc) / len(_faith_sc), 4)
            if _rel_sc and ragas_result.answer_relevancy is None:
                ragas_result.answer_relevancy = round(sum(_rel_sc) / len(_rel_sc), 4)
            _kb_table = Table(title="KB-Qualitaet (LLM-Judge mit KB-Referenz)")
            _kb_table.add_column("Metrik")
            _kb_table.add_column("Score")
            _kb_table.add_column("Schwellenwert")
            for _mn, _mv, _mk in [
                ("KB-Konsistenz",    ragas_result.kb_consistency,   "kb_consistency"),
                ("Answer Relevancy", ragas_result.answer_relevancy, "answer_relevancy"),
            ]:
                if _mv is None:
                    continue
                _gate = config.QUALITY_GATES.get(_mk, {})
                _thr  = _gate.get("threshold", "-")
                _op   = _gate.get("operator", ">=")
                _pass = (_mv >= _thr) if isinstance(_thr, float) else None
                _st   = "[green]OK[/green]" if _pass else "[red]FAIL[/red]" if _pass is False else "-"
                _kb_table.add_row(_mn, f"{_mv:.3f}", f"{_st} {_op}{_thr}")
            console.print(_kb_table)

    # ── KB-basierte Context Precision + Recall via LLM-Judge ─────────────────
    # Läuft wenn --kb-file angegeben UND Faithfulness-Tests vorhanden.
    # Der LLM-Judge bekommt die vollständige Wissensbasis und bewertet:
    #   - Context Precision: Basiert die Antwort auf relevanten Docs?
    #   - Context Recall:    Deckt die Antwort alle relevanten Infos ab?
    # Funktioniert auch für Black-Box-Agents ohne Retrieval-Logs.
    if kb_file and judge_quality:
        _kb_path = Path(kb_file)
        if not _kb_path.exists():
            console.print(f"[yellow]Warnung: --kb-file nicht gefunden: {kb_file} – Context Precision/Recall wird übersprungen.[/yellow]")
        else:
            # KB-Dokumente laden
            _kb_docs: list[str] = []
            if _kb_path.is_dir():
                for _fp in sorted(list(_kb_path.glob("*.txt")) + list(_kb_path.glob("*.md"))):
                    _kb_docs.append(_fp.read_text(encoding="utf-8", errors="replace"))
            else:
                _kb_docs.append(_kb_path.read_text(encoding="utf-8", errors="replace"))
            console.print(f"[dim]KB Context Precision/Recall: {len(_kb_docs)} Dokument(e) geladen[/dim]")

            # Nur Faithfulness-Tests verwenden (Qualitätstests, keine Security-Tests)
            _faith_pairs = [
                (tc, rr) for tc, rr in zip(test_cases, rag_results)
                if tc.category.value == "faithfulness"
            ]

            if not _faith_pairs:
                console.print("[yellow]Keine Faithfulness-Tests gefunden – Context Precision/Recall wird übersprungen.[/yellow]")
            else:
                _f_tcs, _f_rrs = zip(*_faith_pairs)
                # Jedes RAG-Ergebnis bekommt dieselben KB-Docs (globale Wissensbasis)
                _kb_docs_per_result = [_kb_docs] * len(_f_rrs)
                console.print(f"[dim]Berechne Context Precision/Recall fuer {len(_f_rrs)} Tests...[/dim]")
                with console.status("Berechne KB-basierte Context Precision + Recall via LLM-Judge..."):
                    from evaluator.judge import LLMJudge
                    from evaluator.ragas_metrics import evaluate_kb_context_metrics
                    _pr_judge = LLMJudge()
                    _pr_result = evaluate_kb_context_metrics(
                        rag_results=list(_f_rrs),
                        kb_documents_per_result=_kb_docs_per_result,
                        judge=_pr_judge,
                    )
                ragas_result.context_precision = _pr_result.context_precision
                ragas_result.context_recall    = _pr_result.context_recall

                _pr_table = Table(title="KB-basierte Context Metriken (LLM-Judge)")
                _pr_table.add_column("Metrik")
                _pr_table.add_column("Score")
                _pr_table.add_column("Schwellenwert")
                for _mn, _mv, _mk in [
                    ("Context Precision", ragas_result.context_precision, "context_precision"),
                    ("Context Recall",    ragas_result.context_recall,    "context_recall"),
                ]:
                    if _mv is None:
                        continue
                    _gate = config.QUALITY_GATES.get(_mk, {})
                    _thr  = _gate.get("threshold", "-")
                    _op   = _gate.get("operator", ">=")
                    _pass = (_mv >= _thr) if isinstance(_thr, float) else None
                    _st   = "[green]OK[/green]" if _pass else "[red]FAIL[/red]" if _pass is False else "-"
                    _pr_table.add_row(_mn, f"{_mv:.3f}", f"{_st} {_op}{_thr}")
                console.print(_pr_table)

    # ── Judge-Quality Evaluation (nur Fallback bei vorhandenem Kontext) ────────
    # Judge wird NUR ausgeführt wenn:
    #   - Kontext vorhanden ist (Demo RAG mit ChromaDB)
    #   - UND RAGAS-Metriken fehlen (None) → z.B. nach einem RAGAS-Fehler
    #
    # Für Black-Box-Agents ohne Kontext-Logging (z.B. Syntax AI Studio) läuft
    # der Judge NICHT – RAGAS-Werte (answer_relevancy) bleiben unverändert.
    if judge_quality:
        if ragas_category == "all":
            jq_pairs = list(zip(test_cases, rag_results))
        else:
            jq_pairs = [
                (tc, rr) for tc, rr in zip(test_cases, rag_results)
                if tc.category.value == ragas_category
            ]

        if jq_pairs:
            jq_tcs, jq_rrs = zip(*jq_pairs)
            has_any_context = any(rr.contexts for rr in jq_rrs)

            # Judge nur als Fallback wenn Kontext vorhanden aber RAGAS Werte fehlen
            missing = [k for k, v in ragas_result.to_dict().items()
                       if v is None and k in ("faithfulness", "answer_relevancy")]

            if has_any_context and missing:
                ground_truths = [tc.ground_truth for tc in jq_tcs]
                console.print(f"[dim]Judge-Quality Fallback: {len(missing)} fehlende Metriken[/dim]")
                with console.status("Berechne Qualitaetsmetriken via LLM-Judge (Fallback)..."):
                    from evaluator.judge import LLMJudge
                    from evaluator.ragas_metrics import evaluate_with_judge
                    judge_inst = LLMJudge()
                    judge_result = evaluate_with_judge(
                        list(jq_rrs),
                        judge_inst,
                        ground_truths=ground_truths,
                    )

                # Nur fehlende RAGAS-Werte auffuellen
                if ragas_result.faithfulness is None:
                    ragas_result.faithfulness = judge_result.faithfulness
                if ragas_result.answer_relevancy is None:
                    ragas_result.answer_relevancy = judge_result.answer_relevancy

                jq_table = Table(title="Qualitaetsmetriken (LLM-Judge Fallback)")
                jq_table.add_column("Metrik")
                jq_table.add_column("Score")
                jq_table.add_column("Schwellenwert")
                for name, val in [
                    ("faithfulness",     judge_result.faithfulness),
                    ("answer_relevancy", judge_result.answer_relevancy),
                ]:
                    if val is None:
                        continue
                    gate   = config.QUALITY_GATES.get(name, {})
                    thr    = gate.get("threshold", "-")
                    op     = gate.get("operator", ">=")
                    passed = (val >= thr) if isinstance(thr, float) else None
                    status = "[green]OK[/green]" if passed else "[red]FAIL[/red]" if passed is False else "-"
                    jq_table.add_row(name, f"{val:.3f}", f"{status} {op}{thr}")
                console.print(jq_table)

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
            status = "[green]OK[/green]" if passed else "[red]FAIL[/red]"
            sec_table.add_row(label, f"{val:.3f}", f"{status} <={thr}")
        console.print(sec_table)

    metrics_file = config.RESULTS_DIR / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "ragas":          ragas_result.to_dict(),
            "ragas_category": ragas_category,
            "security":       security_result.to_dict(),
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
        kb_consistency=ragas_data.get("kb_consistency"),
        context_precision=ragas_data.get("context_precision"),
        context_recall=ragas_data.get("context_recall"),
    )
    security_result = SecurityResult()
    security_result.asr                       = security_data.get("asr", 0.0)
    security_result.instruction_override_rate = security_data.get("instruction_override_rate", 0.0)
    security_result.data_exfiltration_success = security_data.get("data_exfiltration_success", 0.0)
    security_result.per_category              = security_data.get("per_category", {})

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
    mode: Annotated[str, typer.Option(
        "--mode", "-m",
        help="predefined: eigene Testfälle nutzen | auto: Agent-KB entdecken und Tests generieren"
    )] = "predefined",
) -> None:
    """Vollstaendige Validierung: generate-cases -> run -> evaluate -> report.

    Modi:
      predefined  Eigene Template-Testfaelle generieren und ausfuehren (Standard)
      auto        Agent-Wissensbasis automatisch entdecken, dann Tests generieren und ausfuehren
    """
    console.print(Panel(
        f"[bold]RAG Validation Framework – Vollständige Validierung[/bold]\n"
        f"[dim]Modus: {'Eigene Tests (predefined)' if mode == 'predefined' else 'KB-Discovery (auto)'}[/dim]"
    ))

    if mode == "auto":
        console.rule("Schritt 1: Agent-Wissensbasis entdecken & Testfälle generieren")
        discover_kb()

        console.rule("Schritt 2: RAG-Pipeline ausführen")
        run_tests(category=category, limit=limit, dry_run=False)
    else:
        console.rule("Schritt 1: Testfälle generieren")
        generate_cases(category=category, llm_variants=False)

        console.rule("Schritt 2: RAG-Pipeline ausführen")
        run_tests(category=category, limit=limit, dry_run=False)

    console.rule("Schritt 3: Metriken berechnen")
    evaluate(ragas=True, security=True)

    console.rule("Schritt 4: Report generieren")
    report(fmt="both")


# ── discover-kb ───────────────────────────────────────────────────────────────

@app.command("discover-kb")
def discover_kb(
    kb_file: str | None = typer.Option(
        None, "--kb-file",
        help="KB-Dokument(e): Pfad zu einer Textdatei ODER einem Verzeichnis mit .txt/.md Dateien"
    ),
    probe_file: str | None = typer.Option(
        None, "--probe-file",
        help="Textdatei mit eigenen Probe-Fragen (eine Frage pro Zeile)"
    ),
    n_probes: int = typer.Option(
        20, "--n-probes",
        help="Anzahl Standard-Probe-Fragen im klassischen Modus (Standard: 20)"
    ),
    adaptive: bool = typer.Option(
        True, "--adaptive/--no-adaptive",
        help="Adaptiver Modus: erst Domäne erkennen, dann gezielt fragen (Standard: an)"
    ),
) -> None:
    """Entdeckt die KB des Agents und generiert KB-spezifische Testfaelle.

    Modi (Priorität):
      --kb-file    KB-Dokumente direkt übergeben → stärkste ground_truth (aus echten Dokumenten)
      --adaptive   Agent mit Meta-Fragen befragen, LLM leitet Domäne ab (Standard)
      --no-adaptive 20 feste HR/Unternehmens-Fragen (Fallback)
    """
    from rag.agent_registry import get_wrapper as _get_wrapper, labels as _labels
    from test_generator.kb_generator import AgentKBDiscovery

    agent_type = config.RAG_TARGET
    agent_label = _labels().get(agent_type, agent_type)

    mode_label = "[cyan]Adaptiv[/cyan]" if adaptive else "[dim]Klassisch[/dim]"
    console.print(f"[bold cyan]Agent Knowledge Discovery[/bold cyan]  {mode_label}")
    console.print(f"Target: [bold]{agent_label}[/bold]  (RAG_TARGET={agent_type})")

    # ── Eigene Probe-Fragen aus Datei laden ───────────────────────────────────
    extra_questions: list[str] | None = None
    if probe_file:
        probe_path = Path(probe_file)
        if not probe_path.exists():
            console.print(f"[red]Probe-Datei nicht gefunden: {probe_file}[/red]")
            raise typer.Exit(1)
        lines = probe_path.read_text(encoding="utf-8").splitlines()
        extra_questions = [l.strip() for l in lines if l.strip()]
        console.print(f"[dim]Zusaetzliche Fragen aus Datei: {len(extra_questions)}[/dim]")

    # ── KB-Dokumente laden (falls angegeben) ─────────────────────────────────
    kb_documents: list[str] = []
    if kb_file:
        kb_path = Path(kb_file)
        if not kb_path.exists():
            console.print(f"[red]KB-Datei/-Verzeichnis nicht gefunden: {kb_file}[/red]")
            raise typer.Exit(1)
        if kb_path.is_dir():
            doc_files = list(kb_path.glob("*.txt")) + list(kb_path.glob("*.md"))
            for fp in sorted(doc_files):
                kb_documents.append(fp.read_text(encoding="utf-8", errors="replace"))
            console.print(f"[dim]KB-Dokumente geladen: {len(kb_documents)} Dateien aus {kb_path.name}/[/dim]")
        else:
            kb_documents.append(kb_path.read_text(encoding="utf-8", errors="replace"))
            console.print(f"[dim]KB-Dokument geladen: {kb_path.name} ({len(kb_documents[0])} Zeichen)[/dim]")

    # ── Agent initialisieren ──────────────────────────────────────────────────
    with console.status(f"Initialisiere {agent_type.upper()} Agent..."):
        try:
            wrapper = _get_wrapper()
        except Exception as e:
            console.print(f"[red]Fehler beim Initialisieren des Agents: {e}[/red]")
            raise typer.Exit(1)

    # ── Discovery ─────────────────────────────────────────────────────────────
    discovery = AgentKBDiscovery()

    if kb_documents:
        # Modus 1: KB-Dokumente direkt → stärkste ground_truth
        console.print(f"\n[bold]Schritt 1: Q&A-Paare aus {len(kb_documents)} KB-Dokument(en) extrahieren[/bold]")
        with console.status("LLM extrahiert Fragen und Antworten aus Dokumenten..."):
            discoveries = discovery.kb_document_discover(kb_documents)
        console.print(f"[dim]{len(discoveries)} Q&A-Paare aus Dokumenten extrahiert[/dim]")
        if discoveries:
            doc_table = Table(title="Extrahierte Q&A-Paare (aus KB-Dokumenten)")
            doc_table.add_column("Frage",   style="dim", max_width=50)
            doc_table.add_column("Antwort (ground_truth)", max_width=70)
            for d in discoveries:
                q_s = d["question"].encode("ascii", errors="replace").decode("ascii")
                a_s = d["answer"][:70].encode("ascii", errors="replace").decode("ascii")
                doc_table.add_row(q_s[:50], a_s)
            console.print(doc_table)
    elif adaptive:
        console.print(f"\n[bold]Schritt 1: Domäne erkennen (5 Meta-Fragen) + gezielte Folgefragen[/bold]")
        with console.status("Phase 1: Meta-Fragen senden..."):
            discoveries = discovery.adaptive_discover(
                wrapper,
                extra_questions=extra_questions,
            )
    else:
        console.print(f"\n[bold]Schritt 1: {n_probes} Standard-Probe-Fragen senden[/bold]")
        from test_generator.kb_generator import DEFAULT_PROBE_QUESTIONS, IGNORE_PATTERNS
        questions = DEFAULT_PROBE_QUESTIONS[:n_probes]
        if extra_questions:
            questions = questions + extra_questions

        discoveries = []
        table = Table(title="Probe-Ergebnisse")
        table.add_column("Frage",   style="dim", max_width=45)
        table.add_column("Antwort", max_width=60)
        table.add_column("Status",  style="bold")
        for q in questions:
            q_safe = q.encode("ascii", errors="replace").decode("ascii")
            with console.status(f"  Frage: {q_safe[:60]}..."):
                try:
                    result = wrapper.query(q)
                    answer = result.answer.strip()
                    is_meaningless = any(pat in answer.lower() for pat in IGNORE_PATTERNS)
                    if answer and not is_meaningless:
                        discoveries.append({"question": q, "answer": answer})
                        a_safe = answer.encode("ascii", errors="replace").decode("ascii")
                        table.add_row(q_safe[:45], a_safe[:60], "[green]OK[/green]")
                    else:
                        a_safe = answer.encode("ascii", errors="replace").decode("ascii") if answer else "(leer)"
                        table.add_row(q_safe[:45], a_safe[:60], "[yellow]Gefiltert[/yellow]")
                except Exception as e:
                    table.add_row(q_safe[:45], f"Fehler: {str(e)[:40]}", "[red]Fehler[/red]")
        console.print(table)

    console.print(f"\n[dim]{len(discoveries)} aussagekraeftige Antworten gesammelt[/dim]")

    if adaptive and not kb_documents and discoveries:
        # Gefundene Antworten anzeigen (nur bei Agent-Probing, nicht bei kb-file)
        disc_table = Table(title="Entdeckte KB-Inhalte")
        disc_table.add_column("Frage",   style="dim", max_width=45)
        disc_table.add_column("Antwort", max_width=70)
        for d in discoveries:
            q_s = d["question"].encode("ascii", errors="replace").decode("ascii")
            a_s = d["answer"][:70].encode("ascii", errors="replace").decode("ascii")
            disc_table.add_row(q_s[:45], a_s)
        console.print(disc_table)

    if not discoveries:
        console.print(
            "[yellow]Keine aussagekraeftigen Antworten erhalten.\n"
            "Moegliche Ursachen:\n"
            "  1. Agent hat keine passenden Dokumente in seiner KB\n"
            "  2. Verbindungsproblem (RAG_TARGET, API-Key pruefen)\n"
            "  3. Eigene Probe-Fragen mit --probe-file angeben[/yellow]"
        )
        raise typer.Exit(1)

    # ── Testfälle generieren ──────────────────────────────────────────────────
    console.print(f"\n[bold]Schritt 2: Testfaelle aus {len(discoveries)} Entdeckungen generieren[/bold]")

    with console.status(f"LLM generiert Testfaelle ({len(discoveries) * 2} Aufrufe)..."):
        cases_by_category = discovery.generate_cases_from_discoveries(discoveries, agent_type)

    n_faith  = len(cases_by_category.get("faithfulness", []))
    n_poison = len(cases_by_category.get("corpus_poisoning", []))
    console.print(f"  Faithfulness:    [green]{n_faith}[/green] Testfaelle")
    console.print(f"  Corpus-Poisoning: [green]{n_poison}[/green] Testfaelle")

    if n_faith + n_poison == 0:
        console.print("[yellow]Keine Testfaelle generiert. LLM-Backend pruefen.[/yellow]")
        raise typer.Exit(1)

    # ── Speichern ─────────────────────────────────────────────────────────────
    saved = discovery.save_cases(cases_by_category, agent_type, config.TEST_CASES_DIR)

    console.print(Panel(
        f"[bold green]{len(discoveries)} Entdeckungen -> {n_faith + n_poison} Testfaelle gespeichert[/bold green]\n"
        + "\n".join(
            f"  {cat}: [cyan]{path.name}[/cyan]"
            for cat, path in saved.items()
        )
        + "\n\n[dim]Naechste Schritte:[/dim]\n"
        f"  uv run python main.py run --category kb_{agent_type}_faithfulness --limit 20\n"
        f"  uv run python main.py evaluate --ragas-category faithfulness"
    ))


# ── ping ─────────────────────────────────────────────────────────────────────

@app.command("ping")
def ping(
    debug: Annotated[bool, typer.Option("--debug", help="Zeigt die vollstaendige rohe API-Response (wichtig fuer Retrieval-Analyse)")] = False,
    question: Annotated[str, typer.Option("--question", "-q", help="Test-Frage")] = "Was sind die Urlaubsregelungen?",
) -> None:
    """Testet die Verbindung und zeigt optional die rohe API-Response."""
    from rag.agent_registry import get_wrapper as _get_wrapper, labels as _labels
    target = config.RAG_TARGET
    agent_label = _labels().get(target, target)
    console.print(f"Teste Verbindung zu: [bold]{agent_label}[/bold]  (RAG_TARGET={target})")

    if target == "demo":
        console.print("[yellow]Target ist 'demo' (lokale Pipeline). Kein Remote-Ping noetig.[/yellow]")
        console.print("Setze RAG_TARGET=syntax/azure/copilot/generic in .env um einen echten Agenten zu testen.")
        return

    try:
        wrapper = _get_wrapper()
    except Exception as e:
        console.print(f"[red]Fehler beim Initialisieren des Wrappers: {e}[/red]")
        raise typer.Exit(1)

    # ── Debug: raw Response (nur für Syntax AI Studio verfügbar) ──────────────
    if debug and target == "syntax":
        from rag.syntax_agent import SyntaxAgentWrapper
        if not isinstance(wrapper, SyntaxAgentWrapper):
            console.print("[yellow]--debug nur fuer target=syntax verfuegbar.[/yellow]")
        else:
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

            context_fields = ["citations", "sourceDocuments", "source_documents",
                              "context", "intermediate_steps", "sources",
                              "references", "chunks", "documents", "passages"]
            found = [f for f in context_fields if f in raw]
            if found:
                console.print(f"  [green]Kontext-Felder gefunden: {found}[/green]")
                if "citations" in raw:
                    cites = raw["citations"]
                    if cites:
                        console.print(f"  [green]citations: {len(cites)} Eintraege -> RAGAS Context-Metriken berechenbar![/green]")
                    else:
                        console.print("  [yellow]citations: leer (Agent hat keine passenden Dokumente gefunden)[/yellow]")
                other = [f for f in found if f != "citations"]
                if other:
                    console.print(f"  -> Weitere Kontext-Felder: {other}")
            else:
                console.print("[yellow]Keine Kontext-Felder in Response.[/yellow]")
            return

    # ── Normaler Verbindungstest ───────────────────────────────────────────────
    with console.status(f"Sende Test-Anfrage an {agent_label}..."):
        try:
            result = wrapper.query(question)
            console.print("[green]Verbindung OK![/green]")
            console.print(f"Antwort: {result.answer[:300]}")
            if result.contexts:
                console.print(f"[green]Kontext abgerufen: {len(result.contexts)} Chunk(s)[/green]")
                console.print(f"  Erster Chunk (100 Zeichen): {result.contexts[0][:100]}...")
            else:
                console.print("[yellow]Kein Kontext in Response.[/yellow]")
        except Exception as e:
            console.print(f"[red]Verbindungsfehler: {e}[/red]")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()

