"""Fetch a reproducible free-data health sample or every configured instrument."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from market_moe.data.universe import discover_universes, load_universe
from market_moe.services.data_service import DataService
from market_moe.settings import get_settings

REGIONAL_SAMPLE = {
    "crypto_major": "BTCUSDT",
    "bist30": "THYAO",
    "us_large_cap": "AAPL",
    "europe_large_cap": "SAP",
    "asia_large_cap": "7203",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="check every default universe member")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--output", default="artifacts/reports/data-health.json")
    args = parser.parse_args()
    settings = get_settings()
    service = DataService(settings)
    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    results = []
    for universe_id, path in discover_universes(settings.config_dir).items():
        universe = load_universe(path)
        instruments = universe.instruments
        if not args.all:
            instruments = tuple(
                item for item in instruments if item.symbol == REGIONAL_SAMPLE[universe_id]
            )
        for instrument in instruments:
            try:
                if instrument.asset_class.value == "crypto":
                    frame, quality = service.fetch(instrument, "1d", start, end)
                    provider = "ccxt_binance"
                else:
                    frame, quality, provider = service.fetch_equity_with_fallback(
                        instrument, "1d", start, end
                    )
                results.append(
                    {
                        "universe": universe_id,
                        "instrument_id": instrument.instrument_id,
                        "provider": provider,
                        "status": "ok",
                        "rows": len(frame),
                        "quality": quality.model_dump(mode="json"),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "universe": universe_id,
                        "instrument_id": instrument.instrument_id,
                        "status": "error",
                        "error": str(exc),
                    }
                )
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "all" if args.all else "regional_sample",
        "paid_api_used": False,
        "results": results,
        "summary": {
            "total": len(results),
            "ok": sum(item["status"] == "ok" for item in results),
            "error": sum(item["status"] == "error" for item in results),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 0 if payload["summary"]["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
