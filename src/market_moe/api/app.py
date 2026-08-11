"""Local-only web and JSON API."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from market_moe import __version__
from market_moe.backtest.engine import BacktestConfig, run_backtest
from market_moe.data.universe import discover_universes, load_universe
from market_moe.diagnostics import diagnostics_payload
from market_moe.models.registry import ModelRegistry
from market_moe.portfolio.store import PaperPortfolioStore
from market_moe.services.scanner import ScannerService
from market_moe.settings import Settings, get_settings

PACKAGE_DIR = Path(__file__).resolve().parents[1]


class OpenTradeRequest(BaseModel):
    instrument_id: str
    currency: str
    side: str
    quantity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    entry_time_utc: datetime
    entry_fx_rate: float = Field(default=1.0, gt=0)
    fees: float = Field(default=0.0, ge=0)
    note: str = ""
    prediction_snapshot: dict[str, object] | None = None


class CloseTradeRequest(BaseModel):
    exit_price: float = Field(gt=0)
    exit_time_utc: datetime
    exit_fx_rate: float = Field(default=1.0, gt=0)
    additional_fees: float = Field(default=0.0, ge=0)


class BacktestRequest(BaseModel):
    bars: list[dict[str, Any]]
    signals: list[float]


def create_app(settings: Settings | None = None) -> FastAPI:
    active = settings or get_settings()
    app = FastAPI(
        title="MarketMoE",
        version=__version__,
        description="Local-first market research; never sends real orders.",
    )
    app.state.settings = active
    app.state.portfolio = PaperPortfolioStore(
        active.data_dir / "paper_portfolio.sqlite", base_currency=active.default_base_currency
    )
    app.state.registry = ModelRegistry(active.model_dir)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(active.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        health = diagnostics_payload(active)
        universes = [load_universe(path) for path in discover_universes(active.config_dir).values()]
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={"health": health, "universes": universes, "version": __version__},
        )

    page_names = {
        "scanner": "Market Scanner",
        "instrument": "Enstrüman Detayı",
        "models": "Model Laboratuvarı",
        "backtests": "Backtest V2",
        "data": "Veri Yöneticisi",
        "portfolio": "Manuel Paper Portfolio",
        "health": "Sistem Sağlığı",
    }

    @app.get("/pages/{page_name}", response_class=HTMLResponse)
    def page(request: Request, page_name: str):
        if page_name not in page_names:
            raise HTTPException(404, "page not found")
        return templates.TemplateResponse(
            request=request,
            name="page.html",
            context={
                "page_name": page_name,
                "title": page_names[page_name],
                "version": __version__,
            },
        )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return diagnostics_payload(active)

    @app.get("/api/data/universes")
    def universes() -> list[dict[str, object]]:
        return [
            load_universe(path).model_dump(mode="json")
            for path in discover_universes(active.config_dir).values()
        ]

    @app.get("/api/data/catalog")
    def data_catalog() -> list[dict[str, object]]:
        from market_moe.data.catalog import DataCatalog

        return DataCatalog(active.catalog_path).list_datasets()

    @app.get("/api/instruments")
    def instrument_detail(instrument_id: str, timeframe: str = "1d") -> dict[str, object]:
        from market_moe.data.catalog import DataCatalog

        rows = DataCatalog(active.catalog_path).list_datasets()
        matches = [
            row
            for row in rows
            if row["instrument_id"] == instrument_id and row["timeframe"] == timeframe
        ]
        if not matches:
            raise HTTPException(404, "cached instrument data not found")
        path = Path(str(matches[0]["path"]))
        if not path.exists() or active.data_dir.resolve() not in path.resolve().parents:
            raise HTTPException(404, "catalog data path is unavailable")
        frame = pd.read_parquet(path).tail(200).copy()
        for column in ("open_time_utc", "close_time_utc", "ingested_at_utc"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True).map(
                    lambda value: value.isoformat()
                )
        return {
            "instrument_id": instrument_id,
            "timeframe": timeframe,
            "dataset": {key: value for key, value in matches[0].items() if key != "path"},
            "bars": frame.to_dict(orient="records"),
        }

    @app.get("/api/models")
    def models() -> list[dict[str, object]]:
        return app.state.registry.list()

    @app.get("/api/scanner")
    def scanner(
        universe_id: str | None = Query(default=None), timeframe: str = Query(default="1d")
    ) -> dict[str, object]:
        universes = discover_universes(active.config_dir)
        if universe_id is None:
            return {
                "results": [],
                "available_universes": sorted(universes),
                "message": "Tarama için universe_id seçin. Yalnız cache ve production bundle kullanılır.",
                "automated_trading": False,
            }
        if universe_id not in universes:
            raise HTTPException(404, "universe not found")
        return ScannerService(active).scan(universes[universe_id], timeframe)

    @app.post("/api/backtests")
    def backtest(payload: BacktestRequest) -> dict[str, object]:
        if len(payload.bars) != len(payload.signals):
            raise HTTPException(422, "bars and signals must have equal length")
        bars = pd.DataFrame(payload.bars)
        signals = pd.Series(payload.signals, index=pd.to_datetime(bars["open_time_utc"], utc=True))
        try:
            result = run_backtest(bars, signals, BacktestConfig())
        except Exception as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "metrics": result.metrics,
            "warnings": result.warnings,
            "trades": result.trades.to_dict("records"),
        }

    @app.get("/api/portfolio/trades")
    def list_trades() -> list[dict[str, object]]:
        return app.state.portfolio.list_trades()

    @app.get("/api/portfolio/attribution")
    def portfolio_attribution() -> dict[str, object]:
        return app.state.portfolio.attribution()

    @app.post("/api/portfolio/trades", status_code=201)
    def open_trade(payload: OpenTradeRequest) -> dict[str, str]:
        trade_id = app.state.portfolio.open_trade(**payload.model_dump())
        return {"trade_id": trade_id}

    @app.post("/api/portfolio/trades/{trade_id}/close")
    def close_trade(trade_id: str, payload: CloseTradeRequest) -> dict[str, bool]:
        try:
            app.state.portfolio.close_trade(trade_id, **payload.model_dump())
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"closed": True}

    @app.delete("/api/portfolio/trades/{trade_id}")
    def delete_trade(trade_id: str) -> dict[str, bool]:
        try:
            app.state.portfolio.delete_trade(trade_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"deleted": True}

    return app
