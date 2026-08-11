"""Single feature pipeline used by training, inference, scanner and backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_moe.data.calendars import MarketCalendarService
from market_moe.data.quality import canonicalize_bar_frame
from market_moe.domain.instruments import AssetClass, Instrument
from market_moe.features.indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rolling_slope,
    rsi,
    sma,
)
from market_moe.features.schema import FeatureSchema

COMMON_FEATURES = (
    "log_return_1",
    "log_return_3",
    "log_return_5",
    "rsi_14",
    "dist_sma_50",
    "dist_ema_200",
    "bb_pctb",
    "bb_width",
    "macd_norm",
    "atr_pct",
    "vol_ratio",
    "vol_spike",
    "realized_vol_20",
    "downside_vol_20",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "distance_high_20",
    "distance_low_20",
    "trend_slope_20",
)

CRYPTO_FEATURES = (
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_weekend",
)

EQUITY_FEATURES = (
    "session_progress",
    "is_opening_window",
    "is_closing_window",
    "overnight_gap",
    "dollar_volume_log",
    "amihud_illiquidity",
)


@dataclass(frozen=True, slots=True)
class FeatureResult:
    frame: pd.DataFrame
    schema: FeatureSchema
    dropped_rows: int


class FeaturePipeline:
    schema_version = "2.0.0"

    def feature_names(self, instrument: Instrument) -> tuple[str, ...]:
        domain = CRYPTO_FEATURES if instrument.asset_class == AssetClass.CRYPTO else EQUITY_FEATURES
        return COMMON_FEATURES + domain

    def transform(
        self,
        bars: pd.DataFrame,
        instrument: Instrument,
        timeframe: str,
        *,
        horizon_bars: int = 1,
        include_targets: bool = False,
        drop_incomplete: bool = True,
    ) -> FeatureResult:
        canonical = canonicalize_bar_frame(bars)
        if canonical.empty:
            schema = FeatureSchema(
                self.schema_version,
                instrument.asset_class.value,
                timeframe,
                self.feature_names(instrument),
            )
            return FeatureResult(pd.DataFrame(columns=schema.feature_names), schema, 0)

        canonical = canonical.set_index("open_time_utc", drop=False)
        open_price = canonical["open"].astype(float)
        high = canonical["high"].astype(float)
        low = canonical["low"].astype(float)
        close = canonical["close"].astype(float)
        volume = canonical["volume"].astype(float).fillna(0.0)
        features = pd.DataFrame(index=canonical.index)

        log_return = pd.Series(
            np.log((close / close.shift(1)).to_numpy()), index=close.index, dtype=float
        )
        features["log_return_1"] = log_return
        features["log_return_3"] = pd.Series(
            np.log((close / close.shift(3)).to_numpy()), index=close.index
        )
        features["log_return_5"] = pd.Series(
            np.log((close / close.shift(5)).to_numpy()), index=close.index
        )
        features["rsi_14"] = rsi(close, 14) / 100.0

        sma_50 = sma(close, 50)
        ema_200 = ema(close, 200)
        features["dist_sma_50"] = (close - sma_50) / sma_50
        features["dist_ema_200"] = (close - ema_200) / ema_200

        bb_upper, bb_middle, bb_lower = bollinger_bands(close, 20)
        bb_range = (bb_upper - bb_lower).replace(0.0, np.nan)
        features["bb_pctb"] = (close - bb_lower) / bb_range
        features["bb_width"] = bb_range / bb_middle.replace(0.0, np.nan)

        macd_line, _, _ = macd(close)
        features["macd_norm"] = macd_line / close
        features["atr_pct"] = atr(high, low, close, 14) / close

        volume_average = volume.rolling(20, min_periods=20).mean()
        features["vol_ratio"] = volume / volume_average.replace(0.0, np.nan)
        features["vol_spike"] = (features["vol_ratio"] > 2.0).astype(float)
        features["realized_vol_20"] = log_return.rolling(20, min_periods=20).std(ddof=0)
        features["downside_vol_20"] = (
            log_return.clip(upper=0.0).rolling(20, min_periods=20).std(ddof=0)
        )

        candle_range = (high - low).replace(0.0, np.nan)
        body_high = pd.concat([open_price, close], axis=1).max(axis=1)
        body_low = pd.concat([open_price, close], axis=1).min(axis=1)
        features["body_ratio"] = (close - open_price).abs() / candle_range
        features["upper_wick_ratio"] = (high - body_high) / candle_range
        features["lower_wick_ratio"] = (body_low - low) / candle_range

        rolling_high = high.rolling(20, min_periods=20).max()
        rolling_low = low.rolling(20, min_periods=20).min()
        features["distance_high_20"] = (rolling_high - close) / close
        features["distance_low_20"] = (close - rolling_low) / close
        log_close = pd.Series(np.log(close.to_numpy()), index=close.index)
        features["trend_slope_20"] = rolling_slope(log_close, 20)

        if instrument.asset_class == AssetClass.CRYPTO:
            utc_index = pd.DatetimeIndex(features.index).tz_convert("UTC")
            utc_hours = utc_index.hour + utc_index.minute / 60.0
            features["hour_sin"] = np.sin(2 * np.pi * utc_hours / 24.0)
            features["hour_cos"] = np.cos(2 * np.pi * utc_hours / 24.0)
            features["day_sin"] = np.sin(2 * np.pi * utc_index.dayofweek / 7.0)
            features["day_cos"] = np.cos(2 * np.pi * utc_index.dayofweek / 7.0)
            features["is_weekend"] = (utc_index.dayofweek >= 5).astype(float)
        else:
            if timeframe == "1d":
                features["session_progress"] = 1.0
                features["is_opening_window"] = 0.0
                features["is_closing_window"] = 1.0
            else:
                calendar_service = MarketCalendarService()
                contexts = [calendar_service.context(instrument, stamp) for stamp in features.index]
                features["session_progress"] = [context.session_progress for context in contexts]
                features["is_opening_window"] = [
                    float(context.is_opening_window) for context in contexts
                ]
                features["is_closing_window"] = [
                    float(context.is_closing_window) for context in contexts
                ]
            features["overnight_gap"] = open_price / close.shift(1) - 1.0
            dollar_volume = (close * volume).clip(lower=0.0)
            features["dollar_volume_log"] = np.log1p(dollar_volume)
            features["amihud_illiquidity"] = log_return.abs() / dollar_volume.replace(0.0, np.nan)

        if include_targets:
            future_log_return = pd.Series(
                np.log((close.shift(-horizon_bars) / close).to_numpy()), index=close.index
            )
            features["target_log_return"] = future_log_return
            features["target_direction"] = (future_log_return > 0.0).astype(float)
            features["target_volatility"] = (
                log_return.shift(-1)
                .rolling(horizon_bars, min_periods=horizon_bars)
                .std(ddof=0)
                .shift(-(horizon_bars - 1))
                .fillna(future_log_return.abs())
            )

        feature_names = self.feature_names(instrument)
        schema = FeatureSchema(
            version=self.schema_version,
            asset_class=instrument.asset_class.value,
            timeframe=timeframe,
            feature_names=feature_names,
        )
        features = features.replace([np.inf, -np.inf], np.nan)
        before = len(features)
        if drop_incomplete:
            required = list(feature_names)
            if include_targets:
                required.extend(schema.target_names)
            features = features.dropna(subset=required)
        return FeatureResult(features, schema, before - len(features))
