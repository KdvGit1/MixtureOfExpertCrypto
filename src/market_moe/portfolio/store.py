"""SQLite-backed manual paper-trade ledger."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"expected numeric ledger value, got {type(value).__name__}")
    return float(value)


class PaperPortfolioStore:
    def __init__(
        self, path: Path, *, base_currency: str = "USD", initial_cash: float = 100_000
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.base_currency = base_currency.upper()
        self.initial_cash = initial_cash
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY, instrument_id TEXT NOT NULL, currency TEXT NOT NULL,
                side TEXT NOT NULL CHECK(side IN ('long','short')), quantity REAL NOT NULL,
                entry_time_utc TEXT NOT NULL, entry_price REAL NOT NULL, entry_fx_rate REAL NOT NULL,
                exit_time_utc TEXT, exit_price REAL, exit_fx_rate REAL,
                fees REAL NOT NULL DEFAULT 0, note TEXT NOT NULL DEFAULT '',
                prediction_snapshot TEXT, created_at_utc TEXT NOT NULL)"""
            )

    def open_trade(
        self,
        *,
        instrument_id: str,
        currency: str,
        side: str,
        quantity: float,
        entry_price: float,
        entry_time_utc: datetime,
        entry_fx_rate: float = 1.0,
        fees: float = 0.0,
        note: str = "",
        prediction_snapshot: dict[str, object] | None = None,
    ) -> str:
        if side not in {"long", "short"} or quantity <= 0 or entry_price <= 0 or entry_fx_rate <= 0:
            raise ValueError("invalid manual trade values")
        if entry_time_utc.tzinfo is None:
            raise ValueError("entry time must be timezone-aware")
        trade_id = str(uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?)",
                (
                    trade_id,
                    instrument_id,
                    currency.upper(),
                    side,
                    quantity,
                    entry_time_utc.isoformat(),
                    entry_price,
                    entry_fx_rate,
                    fees,
                    note,
                    json.dumps(prediction_snapshot) if prediction_snapshot else None,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return trade_id

    def close_trade(
        self,
        trade_id: str,
        *,
        exit_price: float,
        exit_time_utc: datetime,
        exit_fx_rate: float = 1.0,
        additional_fees: float = 0.0,
    ) -> None:
        if exit_price <= 0 or exit_fx_rate <= 0 or exit_time_utc.tzinfo is None:
            raise ValueError("invalid close values")
        with self._connect() as connection:
            result = connection.execute(
                """UPDATE trades SET exit_time_utc=?, exit_price=?, exit_fx_rate=?, fees=fees+?
                WHERE trade_id=? AND exit_time_utc IS NULL""",
                (exit_time_utc.isoformat(), exit_price, exit_fx_rate, additional_fees, trade_id),
            )
        if result.rowcount != 1:
            raise KeyError("open trade not found")

    def delete_trade(self, trade_id: str) -> None:
        with self._connect() as connection:
            result = connection.execute("DELETE FROM trades WHERE trade_id=?", (trade_id,))
        if result.rowcount != 1:
            raise KeyError(trade_id)

    def list_trades(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM trades ORDER BY entry_time_utc").fetchall()
        return [dict(row) for row in rows]

    def valuation(
        self, prices: dict[str, float], fx_rates: dict[str, float] | None = None
    ) -> dict[str, float]:
        fx_rates = {self.base_currency: 1.0, **(fx_rates or {})}
        cash = self.initial_cash
        market_value = 0.0
        realized = 0.0
        unrealized = 0.0
        for trade in self.list_trades():
            direction = 1.0 if trade["side"] == "long" else -1.0
            entry_notional = (
                _as_float(trade["quantity"])
                * _as_float(trade["entry_price"])
                * _as_float(trade["entry_fx_rate"])
            )
            cash -= direction * entry_notional + _as_float(trade["fees"])
            if trade["exit_price"] is not None:
                exit_notional = (
                    _as_float(trade["quantity"])
                    * _as_float(trade["exit_price"])
                    * _as_float(trade["exit_fx_rate"])
                )
                cash += direction * exit_notional
                realized += direction * (exit_notional - entry_notional) - _as_float(trade["fees"])
            else:
                instrument_id = str(trade["instrument_id"])
                if instrument_id not in prices:
                    raise ValueError(f"missing valuation price: {instrument_id}")
                currency = str(trade["currency"])
                if currency not in fx_rates:
                    raise ValueError(f"missing FX rate: {currency}/{self.base_currency}")
                current = _as_float(trade["quantity"]) * prices[instrument_id] * fx_rates[currency]
                market_value += direction * current
                unrealized += direction * (current - entry_notional) - _as_float(trade["fees"])
        return {
            "cash": cash,
            "market_value": market_value,
            "equity": cash + market_value,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
        }

    def attribution(self) -> dict[str, object]:
        """Compare stored prediction direction with manually closed-trade outcomes."""

        rows = []
        for trade in self.list_trades():
            if trade["exit_price"] is None or not trade["prediction_snapshot"]:
                continue
            snapshot = json.loads(str(trade["prediction_snapshot"]))
            probability_up = snapshot.get("probability_up")
            if not isinstance(probability_up, (int, float)):
                continue
            direction = 1.0 if trade["side"] == "long" else -1.0
            realized_return = direction * (
                _as_float(trade["exit_price"]) / _as_float(trade["entry_price"]) - 1
            )
            predicted_positive = probability_up >= 0.5 if direction > 0 else probability_up < 0.5
            rows.append(
                {
                    "trade_id": trade["trade_id"],
                    "probability_up": probability_up,
                    "realized_return": realized_return,
                    "direction_match": predicted_positive == (realized_return > 0),
                }
            )
        return {
            "matched_trades": len(rows),
            "direction_match_rate": (
                sum(bool(row["direction_match"]) for row in rows) / len(rows) if rows else None
            ),
            "rows": rows,
        }

    def export(self, path: Path) -> Path:
        rows = self.list_trades()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        elif path.suffix.lower() == ".csv":
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["trade_id"])
                writer.writeheader()
                writer.writerows(rows)
        else:
            raise ValueError("export extension must be .json or .csv")
        return path
