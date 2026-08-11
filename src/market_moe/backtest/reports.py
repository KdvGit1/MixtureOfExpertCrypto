"""Portable JSON, Parquet and standalone HTML reports."""

from __future__ import annotations

import json
from pathlib import Path

from market_moe.backtest.engine import BacktestResult


def write_backtest_report(result: BacktestResult, output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    result.equity.to_parquet(output / "equity.parquet")
    result.trades.to_parquet(output / "trades.parquet", index=False)
    (output / "metrics.json").write_text(
        json.dumps({"metrics": result.metrics, "warnings": result.warnings}, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = "".join(
        f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in result.metrics.items()
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>MarketMoE Backtest</title><style>body{{font:16px system-ui;max-width:900px;margin:2rem auto}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}}</style>
</head><body><h1>MarketMoE Backtest Report</h1><p>Simulated research result; not investment advice.</p>
<table>{rows}</table></body></html>"""
    (output / "report.html").write_text(html, encoding="utf-8")
    return {
        name: output / name
        for name in ("metrics.json", "equity.parquet", "trades.parquet", "report.html")
    }
