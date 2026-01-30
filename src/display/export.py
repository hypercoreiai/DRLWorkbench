# Export: PDF/HTML/CSV reports (V3 — PROJECT_OUTLINE Section 6.4)

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def export_csv_metrics(results_dict: Dict[str, Any], path: str) -> None:
    """
    Export metrics to CSV.

    Args:
        results_dict: Strategy name -> results dict or DataFrame.
        path: Output file or directory path.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if path_obj.suffix.lower() != ".csv":
        path_obj = path_obj / "metrics.csv"
    rows = []
    for name, res in results_dict.items():
        if isinstance(res, pd.DataFrame):
            res = res.to_dict(orient="records")
            for r in res:
                r["strategy"] = name
                rows.append(r)
        elif isinstance(res, dict):
            row = {"strategy": name, **res}
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(path_obj, index=False)


def export_backtest_report(
    results_dict: Dict[str, Any],
    path: str,
    format: str = "html",
) -> None:
    """
    Export backtest report (PDF/HTML).

    Args:
        results_dict: Strategy results.
        path: Output path.
        format: 'pdf' or 'html'.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if format == "html":
        html = "<html><body><h1>Backtest Report</h1><pre>"
        for name, res in results_dict.items():
            html += f"\n{name}\n{res}\n"
        html += "</pre></body></html>"
        out = path_obj if path_obj.suffix.lower() == ".html" else path_obj / "report.html"
        out.write_text(html, encoding="utf-8")
    # PDF can be added via weasyprint or reportlab


def export_results(
    results_dict: Dict[str, Any],
    output_dir: str,
    formats: List[str] = None
) -> None:
    """
    Export results in multiple formats.
    
    Args:
        results_dict: Dictionary of results to export.
        output_dir: Output directory path.
        formats: List of formats ('csv', 'html', 'pdf'). Default: ['csv']
    """
    if formats is None:
        formats = ['csv']
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if 'csv' in formats:
        export_csv_metrics(results_dict, str(output_path / "results.csv"))
    
    if 'html' in formats:
        export_backtest_report(results_dict, str(output_path / "report.html"), format='html')
    
    if 'pdf' in formats:
        export_backtest_report(results_dict, str(output_path / "report.pdf"), format='pdf')
