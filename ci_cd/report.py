"""
Report-Generierung für das Validierungsframework.

Erzeugt:
- JSON-Report (maschinenlesbar, für CI-Systeme)
- HTML-Report (für Bachelorarbeit-Screenshots und manuelle Auswertung)
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

from evaluator.ragas_metrics import RAGASResult
from evaluator.security_metrics import SecurityResult
from ci_cd.quality_gates import GateResult
import config


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>RAG Validation Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }}
  h2 {{ color: #16213e; margin-top: 30px; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .metric {{ display: flex; justify-content: space-between; padding: 8px 0;
             border-bottom: 1px solid #eee; }}
  .metric:last-child {{ border-bottom: none; }}
  .pass {{ color: #27ae60; font-weight: bold; }}
  .fail {{ color: #e74c3c; font-weight: bold; }}
  .partial {{ color: #f39c12; font-weight: bold; }}
  .badge {{ padding: 4px 12px; border-radius: 20px; font-size: 14px; }}
  .badge-pass {{ background: #d5f5e3; color: #27ae60; }}
  .badge-fail {{ background: #fadbd8; color: #e74c3c; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #16213e; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .timestamp {{ color: #888; font-size: 12px; }}
</style>
</head>
<body>
<h1>RAG Validation Framework – {agent_name}</h1>
<p style="color:#555; margin-top:-8px;">{agent_description}</p>
<p class="timestamp">Generiert: {timestamp} | Modell: {generator_model} | Judge: {judge_model}</p>

<div class="card">
  <h2>Gesamtergebnis</h2>
  <div class="metric">
    <span><strong>Quality Gates</strong></span>
    <span class="badge {gate_badge_class}">{gate_status}</span>
  </div>
  <div class="metric">
    <span>Anzahl ausgeführter Tests</span>
    <span>{total_tests}</span>
  </div>
</div>

<div class="card">
  <h2>Qualitätsmetriken</h2>
  <table>
    <tr><th>Metrik</th><th>Score</th><th>Schwellenwert</th><th>Status</th></tr>
    {ragas_rows}
  </table>
</div>

<div class="card">
  <h2>Sicherheitsmetriken</h2>
  <table>
    <tr><th>Metrik</th><th>Wert</th><th>Schwellenwert</th><th>Status</th></tr>
    {security_rows}
  </table>
</div>

{per_category_section}

{failures_section}

</body>
</html>"""


def _per_category_table(per_category: dict) -> str:
    """Rendert eine HTML-Tabelle mit ASR-Aufschlüsselung je Angriffskategorie."""
    if not per_category:
        return ""
    rows = ""
    for cat, stats in sorted(per_category.items()):
        asr   = stats.get("asr", 0.0)
        total = stats.get("total", 0)
        succ  = stats.get("successes", 0)
        color = "fail" if asr > 0.10 else "pass"
        rows += (
            f"<tr><td>{cat}</td><td>{total}</td><td>{succ}</td>"
            f"<td><span class='{color}'>{asr:.1%}</span></td></tr>"
        )
    return (
        '<div class="card">'
        "<h2>Sicherheitsmetriken nach Angriffskategorie</h2>"
        "<table>"
        "<tr><th>Kategorie</th><th>Tests</th><th>Erfolgreich</th><th>ASR</th></tr>"
        f"{rows}"
        "</table></div>"
    )


def _metric_row(name: str, value: float | None, gate: dict | None, higher_is_better: bool = True) -> str:
    if value is None:
        return f"<tr><td>{name}</td><td>N/A</td><td>-</td><td>-</td></tr>"

    threshold = gate["threshold"] if gate else None
    if gate:
        op = gate["operator"]
        passed = (value >= threshold) if op == ">=" else (value <= threshold)
        status = '<span class="pass">✓ PASS</span>' if passed else '<span class="fail">✗ FAIL</span>'
        threshold_str = f"{op} {threshold}"
    else:
        status = "-"
        threshold_str = "-"

    return f"<tr><td>{name}</td><td>{value:.3f}</td><td>{threshold_str}</td><td>{status}</td></tr>"


class ReportGenerator:
    """Erstellt JSON- und HTML-Reports aus Evaluationsergebnissen."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self._output_dir = output_dir or config.RESULTS_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        ragas_result: RAGASResult,
        security_result: SecurityResult,
        gate_result: GateResult,
        total_tests: int = 0,
        run_id: str | None = None,
        per_category: dict | None = None,
    ) -> dict[str, Path]:
        """
        Erstellt JSON- und HTML-Reports.

        Returns:
            Dict mit Pfaden: {"json": Path, "html": Path}
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = run_id or ts
        timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

        report_data = {
            "run_id":    run_id,
            "timestamp": timestamp,
            "generator_model": config.GENERATOR_MODEL,
            "judge_model":     config.JUDGE_MODEL,
            "total_tests":     total_tests,
            "quality_gates": {
                "passed":   gate_result.passed,
                "failures": [str(f) for f in gate_result.failures],
            },
            "ragas_metrics":    ragas_result.to_dict(),
            "security_metrics": security_result.to_dict(),
        }

        # JSON
        json_path = self._output_dir / f"report_{run_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        # HTML
        html_path = self._output_dir / f"report_{run_id}.html"
        gates = config.QUALITY_GATES

        ragas_rows = "".join([
            _metric_row("Faithfulness (N/A bei Black-Box-Agent)",        ragas_result.faithfulness,    None),
            _metric_row("Answer Relevancy (RAGAS)",                       ragas_result.answer_relevancy, gates.get("answer_relevancy")),
            _metric_row("KB-Konsistenz (LLM-Judge, nur KB-Discovery)",   ragas_result.kb_consistency,  gates.get("kb_consistency")),
        ])

        security_rows = "".join([
            _metric_row("Attack Success Rate",          security_result.asr,                       gates.get("asr"), False),
            _metric_row("Instruction Override Rate",    security_result.instruction_override_rate, gates.get("instruction_override_rate"), False),
            _metric_row("Data Exfiltration Success",    security_result.data_exfiltration_success, gates.get("data_exfiltration_success"), False),
        ])

        if gate_result.failures:
            failure_rows = "".join(f"<li class='fail'>{f}</li>" for f in gate_result.failures)
            failures_section = f'<div class="card"><h2>Fehlgeschlagene Gates</h2><ul>{failure_rows}</ul></div>'
        else:
            failures_section = ""

        per_cat_data = per_category or getattr(security_result, "per_category", {})
        per_category_section = _per_category_table(per_cat_data)

        html = HTML_TEMPLATE.format(
            agent_name=config.TARGET_AGENT_NAME,
            agent_description=config.TARGET_AGENT_DESCRIPTION,
            timestamp=timestamp,
            generator_model=config.GENERATOR_MODEL,
            judge_model=config.JUDGE_MODEL,
            gate_badge_class="badge-pass" if gate_result.passed else "badge-fail",
            gate_status="PASSED" if gate_result.passed else "FAILED",
            total_tests=total_tests,
            ragas_rows=ragas_rows,
            security_rows=security_rows,
            per_category_section=per_category_section,
            failures_section=failures_section,
        )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return {"json": json_path, "html": html_path}

