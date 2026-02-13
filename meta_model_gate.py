"""
================================================================================
🚪 META-MODEL GATE — Trade Filtresi
================================================================================
Trading bot'un import edeceği lightweight gate modülü.
Meta-modeli yükler ve her trade kararında karlılık olasılığını hesaplar.

Kullanım (bot içinde):
    from meta_model_gate import MetaModelGate
    gate = MetaModelGate()
    gate.load()
    allow, prob = gate.should_allow_trade(prediction_dict, df_display, candle_idx)
================================================================================
"""

import json
import logging
from pathlib import Path
from collections import deque

import numpy as np

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent

# Feature names (must match trainer — v2 sadeleştirilmiş 18 feature)
FEATURE_NAMES = [
    # Model Quality (4)
    "pred_pct", "confidence", "pred_conf_product",
    "branch_agreement",
    # Price Action / Stop Hunt (3)
    "wick_up_ratio", "wick_down_ratio", "body_ratio",
    # Regime (2)
    "atr_rank", "bb_squeeze",
    # Market Structure (2)
    "chop_score", "adx_value",
    # Cost Edge (1)
    "net_edge_after_fee",
    # Context — Streak (2)
    "recent_stop_streak", "recent_win_streak",
    # Stop Hunt — Liquidity Sweep (2)
    "sweep_low", "sweep_high",
    # Stop Hunt — Swing Distance (2)
    "dist_to_swing_low", "dist_to_swing_high",
]

FEE_PCT = 0.002  # 0.2% round-trip fee


class MetaModelGate:
    """
    Trade gate that predicts if a trade will be profitable.
    Loads a trained XGBoost model and computes features on-the-fly.
    """

    def __init__(self, threshold: float = 0.60):
        self.model = None
        self.config = None
        self.threshold = threshold
        self.loaded = False
        self.recent_labels = deque(maxlen=20)  # Streak tracking: 1=win, 0=loss

    def load(self, model_path: Path = None, config_path: Path = None) -> bool:
        """Load meta-model and config."""
        if not HAS_XGBOOST:
            logger.warning("⚠️ xgboost kütüphanesi yüklü değil, meta-model devre dışı")
            return False

        model_path = model_path or PROJECT_ROOT / 'meta_model.json'
        config_path = config_path or PROJECT_ROOT / 'meta_model_config.json'

        if not model_path.exists():
            logger.warning(f"⚠️ Meta-model bulunamadı: {model_path}")
            return False

        try:
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))

            if config_path.exists():
                with open(config_path) as f:
                    self.config = json.load(f)
                self.threshold = self.config.get('threshold', self.threshold)

            self.loaded = True
            logger.info(f"✅ Meta-model yüklendi (threshold: {self.threshold:.2f})")
            return True

        except Exception as e:
            logger.error(f"❌ Meta-model yükleme hatası: {e}")
            return False

    def update_trade_result(self, is_profitable: bool):
        """Update recent trade results for streak tracking."""
        self.recent_labels.append(1 if is_profitable else 0)

    def extract_features(self, prediction_dict: dict,
                          df_display, candle_idx: int) -> dict:
        """
        Extract all 18 features from prediction dict and market data (v2).
        """
        pred_pct = prediction_dict.get('prediction_pct', 0)
        confidence = prediction_dict.get('confidence', 33)
        pred_cnn = prediction_dict.get('pred_cnn', pred_pct)
        pred_lstm = prediction_dict.get('pred_lstm', pred_pct)
        pred_tr = prediction_dict.get('pred_tr', pred_pct)

        branches = [pred_cnn, pred_lstm, pred_tr]
        main_sign = 1 if pred_pct > 0 else -1
        signs = [1 if b > 0 else -1 for b in branches]
        agreement = sum(1 for s in signs if s == main_sign)

        features = {
            "pred_pct": pred_pct,
            "confidence": confidence,
            "pred_conf_product": pred_pct * confidence / 100.0,
            "branch_agreement": agreement,
        }

        # === Price Action (3) ===
        row = df_display.iloc[candle_idx]
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        hl_range = h - l

        if hl_range > 1e-10:
            features["wick_up_ratio"] = (h - max(o, c)) / hl_range
            features["wick_down_ratio"] = (min(o, c) - l) / hl_range
            features["body_ratio"] = abs(c - o) / hl_range
        else:
            features["wick_up_ratio"] = 0.0
            features["wick_down_ratio"] = 0.0
            features["body_ratio"] = 0.0

        # === Regime (2) ===
        lookback = min(100, candle_idx)
        close_arr = df_display['Close'].values[candle_idx - lookback:candle_idx + 1]
        high_arr = df_display['High'].values[candle_idx - lookback:candle_idx + 1]
        low_arr = df_display['Low'].values[candle_idx - lookback:candle_idx + 1]

        # ATR rank
        if len(close_arr) > 14:
            tr_vals = np.maximum(high_arr[1:] - low_arr[1:],
                                 np.maximum(np.abs(high_arr[1:] - close_arr[:-1]),
                                            np.abs(low_arr[1:] - close_arr[:-1])))
            atrs = np.convolve(tr_vals, np.ones(14)/14, mode='valid')
            if len(atrs) > 0:
                features["atr_rank"] = float(np.mean(atrs <= atrs[-1]))
            else:
                features["atr_rank"] = 0.5
        else:
            features["atr_rank"] = 0.5

        # BB squeeze
        if len(close_arr) >= 20:
            sma20 = np.mean(close_arr[-20:])
            std20 = np.std(close_arr[-20:])
            bb_width = (4 * std20) / sma20 if sma20 > 0 else 0
            bb_widths = []
            for j in range(20, len(close_arr)):
                s = np.mean(close_arr[j-20:j])
                st = np.std(close_arr[j-20:j])
                if s > 0:
                    bb_widths.append((4 * st) / s)
            features["bb_squeeze"] = 1.0 - np.mean(np.array(bb_widths) <= bb_width) if bb_widths else 0.5
        else:
            features["bb_squeeze"] = 0.5

        # === Market Structure (2) ===
        # Choppiness
        if len(close_arr) > 14:
            tr_vals = np.maximum(high_arr[1:] - low_arr[1:],
                                 np.maximum(np.abs(high_arr[1:] - close_arr[:-1]),
                                            np.abs(low_arr[1:] - close_arr[:-1])))
            atr_sum = np.sum(tr_vals[-14:])
            highest = np.max(high_arr[-14:])
            lowest = np.min(low_arr[-14:])
            hl = highest - lowest
            if hl > 0 and atr_sum > 0:
                features["chop_score"] = float(np.clip(100 * np.log10(atr_sum / hl) / np.log10(14), 0, 100))
            else:
                features["chop_score"] = 50.0
        else:
            features["chop_score"] = 50.0

        # ADX
        if candle_idx >= 28:
            h_arr = df_display['High'].values[candle_idx-27:candle_idx+1]
            l_arr = df_display['Low'].values[candle_idx-27:candle_idx+1]
            c_arr = df_display['Close'].values[candle_idx-27:candle_idx+1]
            plus_dm = np.maximum(h_arr[1:] - h_arr[:-1], 0)
            minus_dm = np.maximum(l_arr[:-1] - l_arr[1:], 0)
            mask = plus_dm > minus_dm
            plus_dm[~mask] = 0
            minus_dm[mask] = 0
            tr = np.maximum(h_arr[1:] - l_arr[1:],
                            np.maximum(np.abs(h_arr[1:] - c_arr[:-1]),
                                       np.abs(l_arr[1:] - c_arr[:-1])))
            atr14 = np.mean(tr[-14:])
            plus_di = np.mean(plus_dm[-14:]) / atr14 * 100 if atr14 > 0 else 0
            minus_di = np.mean(minus_dm[-14:]) / atr14 * 100 if atr14 > 0 else 0
            di_sum = plus_di + minus_di
            features["adx_value"] = abs(plus_di - minus_di) / di_sum * 100 if di_sum > 0 else 0
        else:
            features["adx_value"] = 25.0

        # === Cost Edge (1) ===
        features["net_edge_after_fee"] = abs(pred_pct) - FEE_PCT * 100

        # === Context — Streak (2) ===
        if len(self.recent_labels) > 0:
            labels_list = list(self.recent_labels)
            stop_streak = 0
            for lbl in reversed(labels_list):
                if lbl == 0:
                    stop_streak += 1
                else:
                    break
            win_streak = 0
            for lbl in reversed(labels_list):
                if lbl == 1:
                    win_streak += 1
                else:
                    break
            features["recent_stop_streak"] = float(stop_streak)
            features["recent_win_streak"] = float(win_streak)
        else:
            features["recent_stop_streak"] = 0.0
            features["recent_win_streak"] = 0.0

        # === Stop Hunt — Liquidity Sweep (2) ===
        if candle_idx >= 1:
            prev_row = df_display.iloc[candle_idx - 1]
            prev_low, prev_high = prev_row['Low'], prev_row['High']
            features["sweep_low"] = 1.0 if (l < prev_low and c > prev_low) else 0.0
            features["sweep_high"] = 1.0 if (h > prev_high and c < prev_high) else 0.0
        else:
            features["sweep_low"] = 0.0
            features["sweep_high"] = 0.0

        # === Stop Hunt — Swing Distance (2) ===
        swing_lookback = min(20, candle_idx)
        if swing_lookback >= 5:
            lows = df_display['Low'].values[candle_idx - swing_lookback:candle_idx]
            highs = df_display['High'].values[candle_idx - swing_lookback:candle_idx]
            swing_low = np.min(lows)
            swing_high = np.max(highs)
            features["dist_to_swing_low"] = (c - swing_low) / c * 100 if c > 0 else 0.0
            features["dist_to_swing_high"] = (swing_high - c) / c * 100 if c > 0 else 0.0
        else:
            features["dist_to_swing_low"] = 0.0
            features["dist_to_swing_high"] = 0.0

        return features

    def should_allow_trade(self, prediction_dict: dict,
                            df_display=None, candle_idx: int = -1) -> tuple:
        """
        Check if a trade should be allowed.

        Returns:
            (allow: bool, probability: float)
            - allow: True if meta-model says trade is likely profitable
            - probability: estimated probability of profitability (0-1)
        """
        if not self.loaded or self.model is None:
            return True, 1.0  # If model not loaded, allow all

        try:
            if df_display is not None and candle_idx >= 0:
                features = self.extract_features(prediction_dict, df_display, candle_idx)
            else:
                # Fallback: use only model-based features
                features = self._extract_model_only_features(prediction_dict)

            # Build feature vector in correct order
            feat_vector = [features.get(f, 0.0) for f in FEATURE_NAMES]
            dmatrix = xgb.DMatrix(
                np.array([feat_vector]),
                feature_names=FEATURE_NAMES
            )

            probability = float(self.model.predict(dmatrix)[0])
            allow = probability >= self.threshold

            return allow, probability

        except Exception as e:
            logger.error(f"Meta-model tahmin hatası: {e}")
            return True, 1.0  # On error, allow trade

    def _extract_model_only_features(self, prediction_dict: dict) -> dict:
        """Fallback: extract only model-based features when df_display is not available."""
        pred_pct = prediction_dict.get('prediction_pct', 0)
        confidence = prediction_dict.get('confidence', 33)
        pred_cnn = prediction_dict.get('pred_cnn', pred_pct)
        pred_lstm = prediction_dict.get('pred_lstm', pred_pct)
        pred_tr = prediction_dict.get('pred_tr', pred_pct)

        branches = [pred_cnn, pred_lstm, pred_tr]
        main_sign = 1 if pred_pct > 0 else -1
        signs = [1 if b > 0 else -1 for b in branches]

        # Streak from recent_labels
        if len(self.recent_labels) > 0:
            labels_list = list(self.recent_labels)
            stop_streak = 0
            for lbl in reversed(labels_list):
                if lbl == 0:
                    stop_streak += 1
                else:
                    break
            win_streak = 0
            for lbl in reversed(labels_list):
                if lbl == 1:
                    win_streak += 1
                else:
                    break
        else:
            stop_streak = 0
            win_streak = 0

        return {
            "pred_pct": pred_pct,
            "confidence": confidence,
            "pred_conf_product": pred_pct * confidence / 100.0,
            "branch_agreement": sum(1 for s in signs if s == main_sign),
            "wick_up_ratio": 0.0,
            "wick_down_ratio": 0.0,
            "body_ratio": 0.0,
            "atr_rank": 0.5,
            "bb_squeeze": 0.5,
            "chop_score": 50.0,
            "adx_value": 25.0,
            "net_edge_after_fee": abs(pred_pct) - FEE_PCT * 100,
            "recent_stop_streak": float(stop_streak),
            "recent_win_streak": float(win_streak),
            "sweep_low": 0.0,
            "sweep_high": 0.0,
            "dist_to_swing_low": 0.0,
            "dist_to_swing_high": 0.0,
        }

    def get_status(self) -> dict:
        """Get current meta-model status for Telegram /meta command."""
        return {
            'loaded': self.loaded,
            'threshold': self.threshold,
            'recent_trades': len(self.recent_labels),
            'recent_win_rate': sum(self.recent_labels) / len(self.recent_labels) if self.recent_labels else 0.0,
            'config': self.config,
        }
