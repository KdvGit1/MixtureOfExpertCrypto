"""Exchange-calendar helpers and session-relative feature calculation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from market_moe.domain.instruments import AssetClass, Instrument


@dataclass(frozen=True, slots=True)
class SessionContext:
    session_progress: float
    minutes_since_open: float
    minutes_to_close: float
    is_opening_window: bool
    is_closing_window: bool
    is_regular_session: bool


class MarketCalendarService:
    def __init__(self) -> None:
        self._calendars: dict[str, object] = {}

    def _get_calendar(self, name: str):
        if name not in self._calendars:
            import exchange_calendars as xcals

            self._calendars[name] = xcals.get_calendar(name)
        return self._calendars[name]

    def context(self, instrument: Instrument, timestamp: pd.Timestamp) -> SessionContext:
        ts = pd.Timestamp(timestamp)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

        if instrument.asset_class == AssetClass.CRYPTO or instrument.calendar == "24/7":
            minute_of_day = ts.hour * 60 + ts.minute
            progress = minute_of_day / 1440.0
            return SessionContext(
                session_progress=progress,
                minutes_since_open=float(minute_of_day),
                minutes_to_close=float(1440 - minute_of_day),
                is_opening_window=minute_of_day < 30,
                is_closing_window=minute_of_day >= 1410,
                is_regular_session=True,
            )

        calendar = self._get_calendar(instrument.calendar)
        minute = ts.floor("min")
        try:
            session = calendar.minute_to_session(minute, direction="none")
            session_open = calendar.session_open(session)
            session_close = calendar.session_close(session)
            since_open = max(0.0, (minute - session_open).total_seconds() / 60.0)
            to_close = max(0.0, (session_close - minute).total_seconds() / 60.0)
            total = max(1.0, (session_close - session_open).total_seconds() / 60.0)
            return SessionContext(
                session_progress=min(1.0, since_open / total),
                minutes_since_open=since_open,
                minutes_to_close=to_close,
                is_opening_window=since_open <= 30,
                is_closing_window=to_close <= 30,
                is_regular_session=True,
            )
        except (ValueError, KeyError):
            return SessionContext(0.0, 0.0, 0.0, False, False, False)
