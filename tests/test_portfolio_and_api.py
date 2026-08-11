from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from market_moe.api.app import create_app
from market_moe.portfolio.store import PaperPortfolioStore
from market_moe.settings import Settings


def test_manual_portfolio_crud_valuation_and_export(tmp_path) -> None:
    store = PaperPortfolioStore(tmp_path / "portfolio.sqlite", initial_cash=10_000)
    trade_id = store.open_trade(
        instrument_id="equity:XNAS:AAPL",
        currency="USD",
        side="long",
        quantity=10,
        entry_price=100,
        entry_time_utc=datetime(2025, 1, 1, tzinfo=UTC),
        prediction_snapshot={"model": "test", "probability_up": 0.7},
    )
    valuation = store.valuation({"equity:XNAS:AAPL": 110})
    assert valuation["equity"] == 10_100
    store.close_trade(trade_id, exit_price=120, exit_time_utc=datetime(2025, 2, 1, tzinfo=UTC))
    assert store.valuation({})["realized_pnl"] == 200
    assert store.attribution()["direction_match_rate"] == 1
    assert store.export(tmp_path / "trades.json").exists()
    assert store.export(tmp_path / "trades.csv").exists()


def test_api_health_pages_and_manual_trade(tmp_path) -> None:
    project_root = tmp_path / "project"
    config = project_root / "configs" / "universes"
    config.mkdir(parents=True)
    (config / "empty.yaml").write_text(
        "universe_id: empty\ndisplay_name: Empty\nas_of: '2026-01-01'\nsource: test\ninstruments: []\n",
        encoding="utf-8",
    )
    settings = Settings(
        project_root=project_root,
        data_dir=project_root / "data",
        artifacts_dir=project_root / "artifacts",
        config_dir=project_root / "configs",
    )
    client = TestClient(create_app(settings))
    assert client.get("/").status_code == 200
    health = client.get("/api/health")
    assert health.status_code == 200 and health.json()["healthy"]
    assert client.get("/pages/scanner").status_code == 200
    response = client.post(
        "/api/portfolio/trades",
        json={
            "instrument_id": "crypto:BINANCE:BTC/USDT",
            "currency": "USD",
            "side": "long",
            "quantity": 0.1,
            "entry_price": 50000,
            "entry_time_utc": "2026-01-01T00:00:00Z",
        },
    )
    assert response.status_code == 201
    assert len(client.get("/api/portfolio/trades").json()) == 1
    assert "*" not in settings.allowed_origins
