"""
================================================================================
🧠 META-MODEL TRAINER — Trade Gate (Karlı/Zararlı Tahmin)
================================================================================
Mevcut MultiBranchModel tahminlerini analiz ederek her trade'in
karlı mı zararlı mı olacağını tahmin eden bir üst model (XGBoost) eğitir.

Kullanım:
    python meta_model_trainer.py
    python meta_model_trainer.py --coins BTC LTC ADA --months 6
    python meta_model_trainer.py --months 3 --threshold 0.60

Çıktılar:
    - meta_model.json          → Eğitilmiş XGBoost model
    - meta_model_config.json   → Feature listesi + threshold + stats
    - meta_training_data.csv   → Üretilen eğitim verisi
    - meta_reports/            → Grafikler ve raporlar klasörü
================================================================================
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import optuna
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

warnings.filterwarnings('ignore')

# ============================================
# PATHS & CONFIG
# ============================================
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes
from train_models.ai_engine_improved import MultiBranchModel, CNN_FEATURES, LSTM_FEATURES, TR_FEATURES

KAGGLE_OUTPUTS_DIR = PROJECT_ROOT / 'kaggle_outputs'
REPORTS_DIR = PROJECT_ROOT / 'meta_reports'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Window sizes (matching main model)
CNN_WINDOW = 12
LSTM_WINDOW = 120
TR_WINDOW = 120
MAX_WINDOW = max(CNN_WINDOW, LSTM_WINDOW, TR_WINDOW)

# Coins to use
DEFAULT_COINS = ["BTC", "LTC", "ADA"]
TIMEFRAMES = ["15m"]

# Feature names (18 features — v2 sadeleştirilmiş)
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
FUTURE_CANDLES = 32  # Label: sonraki 32 mum (8 saat)
LABEL_TP_PCT = 0.01   # %1 TP hedefi
LABEL_SL_PCT = 0.02   # %2 SL hedefi
MIN_PREDICTION_PCT = 10.0  # Bot'taki min_prediction_pct (%10 üstü trade açılır)


# ============================================
# MODEL LOADING
# ============================================
def load_model(coin: str, timeframe: str):
    """Load a trained MultiBranchModel from kaggle_outputs."""
    model_path = KAGGLE_OUTPUTS_DIR / f"{coin}_{timeframe}_model.pth"
    params_path = KAGGLE_OUTPUTS_DIR / f"{coin}_{timeframe}_params.json"
    stats_path = KAGGLE_OUTPUTS_DIR / f"{coin}_{timeframe}_stats.json"

    if not model_path.exists():
        print(f"  ❌ Model bulunamadı: {model_path}")
        return None, None

    with open(params_path) as f:
        params = json.load(f)
    with open(stats_path) as f:
        stats = json.load(f)

    embed_dim = params.get('embed_dim', 96)
    dropout = params.get('dropout', 0.15)

    model = MultiBranchModel(embed_dim=embed_dim, dropout=dropout).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()

    return model, stats


# ============================================
# FEATURE COMPUTATION
# ============================================
def compute_price_action_features(df_display: pd.DataFrame, idx: int) -> dict:
    """Compute price action features from OHLCV data."""
    row = df_display.iloc[idx]
    o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
    hl_range = h - l

    if hl_range < 1e-10:
        return {"wick_up_ratio": 0.0, "wick_down_ratio": 0.0, "body_ratio": 0.0}

    wick_up = (h - max(o, c)) / hl_range
    wick_down = (min(o, c) - l) / hl_range
    body = abs(c - o) / hl_range

    return {
        "wick_up_ratio": wick_up,
        "wick_down_ratio": wick_down,
        "body_ratio": body,
    }


def compute_regime_features(df_display: pd.DataFrame, idx: int) -> dict:
    """Compute atr_rank and bb_squeeze."""
    lookback = min(100, idx)
    if lookback < 20:
        return {"atr_rank": 0.5, "bb_squeeze": 0.5}

    close_arr = df_display['Close'].values[idx - lookback:idx + 1]
    high_arr = df_display['High'].values[idx - lookback:idx + 1]
    low_arr = df_display['Low'].values[idx - lookback:idx + 1]

    # ATR rank: current ATR vs rolling percentile
    tr_values = np.maximum(high_arr[1:] - low_arr[1:],
                           np.maximum(np.abs(high_arr[1:] - close_arr[:-1]),
                                      np.abs(low_arr[1:] - close_arr[:-1])))
    if len(tr_values) > 14:
        atrs = np.convolve(tr_values, np.ones(14)/14, mode='valid')
        current_atr = atrs[-1]
        atr_rank = np.mean(atrs <= current_atr)
    else:
        atr_rank = 0.5

    # BB squeeze: Bollinger Band width relative to recent history
    if len(close_arr) >= 20:
        sma20 = np.mean(close_arr[-20:])
        std20 = np.std(close_arr[-20:])
        bb_width = (2 * std20 * 2) / sma20 if sma20 > 0 else 0
        bb_widths = []
        for j in range(20, len(close_arr)):
            s = np.mean(close_arr[j-20:j])
            st = np.std(close_arr[j-20:j])
            if s > 0:
                bb_widths.append((2 * st * 2) / s)
        if bb_widths:
            bb_squeeze = 1.0 - np.mean(np.array(bb_widths) <= bb_width)
        else:
            bb_squeeze = 0.5
    else:
        bb_squeeze = 0.5

    return {
        "atr_rank": atr_rank,
        "bb_squeeze": bb_squeeze,
    }


def compute_market_structure_features(df_display: pd.DataFrame, idx: int) -> dict:
    """Compute chop_score and adx_value."""
    lookback = min(50, idx)
    if lookback < 14:
        return {"chop_score": 50.0, "adx_value": 25.0}

    close_arr = df_display['Close'].values[idx - lookback:idx + 1]
    high_arr = df_display['High'].values[idx - lookback:idx + 1]
    low_arr = df_display['Low'].values[idx - lookback:idx + 1]

    # Choppiness Index (14-period)
    tr_values = np.maximum(high_arr[1:] - low_arr[1:],
                           np.maximum(np.abs(high_arr[1:] - close_arr[:-1]),
                                      np.abs(low_arr[1:] - close_arr[:-1])))
    if len(tr_values) >= 14:
        atr_sum = np.sum(tr_values[-14:])
        highest = np.max(high_arr[-14:])
        lowest = np.min(low_arr[-14:])
        hl_range = highest - lowest
        if hl_range > 0 and atr_sum > 0:
            chop_score = 100 * np.log10(atr_sum / hl_range) / np.log10(14)
            chop_score = np.clip(chop_score, 0, 100)
        else:
            chop_score = 50.0
    else:
        chop_score = 50.0

    # ADX (14-period)
    if lookback >= 28:
        h28 = high_arr[-28:]
        l28 = low_arr[-28:]
        c28 = close_arr[-28:]
        plus_dm = np.maximum(h28[1:] - h28[:-1], 0)
        minus_dm = np.maximum(l28[:-1] - l28[1:], 0)
        mask = plus_dm > minus_dm
        plus_dm[~mask] = 0
        minus_dm[mask] = 0
        tr = np.maximum(h28[1:] - l28[1:],
                        np.maximum(np.abs(h28[1:] - c28[:-1]),
                                   np.abs(l28[1:] - c28[:-1])))
        atr14 = np.mean(tr[-14:])
        plus_di = np.mean(plus_dm[-14:]) / atr14 * 100 if atr14 > 0 else 0
        minus_di = np.mean(minus_dm[-14:]) / atr14 * 100 if atr14 > 0 else 0
        di_sum = plus_di + minus_di
        adx_value = abs(plus_di - minus_di) / di_sum * 100 if di_sum > 0 else 0
    else:
        adx_value = 25.0

    return {
        "chop_score": chop_score,
        "adx_value": adx_value,
    }


def compute_all_features(
    pred_main: float,
    pred_cnn: float,
    pred_lstm: float,
    pred_tr: float,
    df_display: pd.DataFrame,
    candle_idx: int,
    recent_labels: deque,
) -> dict:
    """Compute all 18 features for a single prediction point (v2)."""

    # === Model Quality features (4) ===
    branches = [pred_cnn, pred_lstm, pred_tr]
    signs = [1 if b > 0 else -1 for b in branches]
    main_sign = 1 if pred_main > 0 else -1

    confidence = abs(sum(signs)) / 3.0 * 100
    agreement_count = sum(1 for s in signs if s == main_sign)

    features = {
        "pred_pct": pred_main,
        "confidence": confidence,
        "pred_conf_product": pred_main * confidence / 100.0,
        "branch_agreement": agreement_count,
    }

    # === Price Action features (3) ===
    features.update(compute_price_action_features(df_display, candle_idx))

    # === Regime features (2) — atr_rank, bb_squeeze ===
    features.update(compute_regime_features(df_display, candle_idx))

    # === Market Structure features (2) — chop_score, adx_value ===
    features.update(compute_market_structure_features(df_display, candle_idx))

    # === Cost Edge (1) ===
    features["net_edge_after_fee"] = abs(pred_main) - FEE_PCT * 100

    # === Context — Streak features (2) ===
    # recent_labels deque: 1 = karlı (TP hit), 0 = zararlı (SL hit)
    if len(recent_labels) > 0:
        labels_list = list(recent_labels)
        # recent_stop_streak: consecutive losses from latest
        stop_streak = 0
        for lbl in reversed(labels_list):
            if lbl == 0:
                stop_streak += 1
            else:
                break
        # recent_win_streak: consecutive wins from latest
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
    # sweep_low = low pierced previous low but close recovered above it
    # sweep_high = high pierced previous high but close dropped below it
    if candle_idx >= 1:
        row = df_display.iloc[candle_idx]
        prev_row = df_display.iloc[candle_idx - 1]
        cur_low, cur_high, cur_close = row['Low'], row['High'], row['Close']
        prev_low, prev_high = prev_row['Low'], prev_row['High']

        features["sweep_low"] = 1.0 if (cur_low < prev_low and cur_close > prev_low) else 0.0
        features["sweep_high"] = 1.0 if (cur_high > prev_high and cur_close < prev_high) else 0.0
    else:
        features["sweep_low"] = 0.0
        features["sweep_high"] = 0.0

    # === Stop Hunt — Swing Level Distance (2) ===
    # Look back 20 candles to find swing low/high, measure distance as %
    swing_lookback = min(20, candle_idx)
    if swing_lookback >= 5:
        cur_close = df_display.iloc[candle_idx]['Close']
        lows = df_display['Low'].values[candle_idx - swing_lookback:candle_idx]
        highs = df_display['High'].values[candle_idx - swing_lookback:candle_idx]
        swing_low = np.min(lows)
        swing_high = np.max(highs)

        features["dist_to_swing_low"] = (cur_close - swing_low) / cur_close * 100 if cur_close > 0 else 0.0
        features["dist_to_swing_high"] = (swing_high - cur_close) / cur_close * 100 if cur_close > 0 else 0.0
    else:
        features["dist_to_swing_low"] = 0.0
        features["dist_to_swing_high"] = 0.0

    return features


# ============================================
# DATA GENERATION
# ============================================
def generate_training_data(coins: list, months: int, verbose: bool = True) -> pd.DataFrame:
    """
    Generate training data by iterating over historical candles.
    For each candle: run model → compute features → compute label (4-candle future).
    """
    all_samples = []

    for coin in coins:
        for tf in TIMEFRAMES:
            print(f"\n{'='*60}")
            print(f"📊 {coin}/{tf} — Veri üretimi başlıyor...")
            print(f"{'='*60}")

            # Load model
            model, stats = load_model(coin, tf)
            if model is None:
                continue

            # Fetch historical data
            tf_minutes = {'15m': 15, '1h': 60}.get(tf, 15)
            months_needed = max(months, (500 * tf_minutes / (30 * 24 * 60)) * 1.2)

            try:
                df_raw = get_crypto_history(
                    symbol=f"{coin}/USDT",
                    timeframe=tf,
                    months_back=months_needed,
                    exchange_name="binance"
                )
            except Exception as e:
                print(f"  ❌ Veri çekme hatası: {e}")
                continue

            if len(df_raw) < MAX_WINDOW + 100:
                print(f"  ❌ Yetersiz veri: {len(df_raw)} mum")
                continue

            # Prepare data
            df_display, df_ai = prepare_dual_dataframes(df_raw)

            # Normalize
            mean = pd.Series(stats['mean'])
            std = pd.Series(stats['std'])
            std[std == 0] = 1.0
            df_normalized = (df_ai - mean) / std

            # Get column indices
            cnn_cols = [df_ai.columns.get_loc(c) for c in CNN_FEATURES if c in df_ai.columns]
            lstm_cols = [df_ai.columns.get_loc(c) for c in LSTM_FEATURES if c in df_ai.columns]
            tr_cols = [df_ai.columns.get_loc(c) for c in TR_FEATURES if c in df_ai.columns]

            data = df_normalized.values
            total = len(data)
            samples_this = 0

            # Streak buffer — tracks actual TP/SL outcomes for streak features
            recent_labels = deque(maxlen=20)

            # Iterate candles
            for i in range(MAX_WINDOW, total - FUTURE_CANDLES):
                t = i + 1  # end index for data slice

                # Prepare tensors
                x_cnn = torch.tensor(data[t-CNN_WINDOW:t, cnn_cols],
                                     dtype=torch.float32).unsqueeze(0).to(DEVICE)
                x_lstm = torch.tensor(data[t-LSTM_WINDOW:t, lstm_cols],
                                      dtype=torch.float32).unsqueeze(0).to(DEVICE)
                x_tr = torch.tensor(data[t-TR_WINDOW:t, tr_cols],
                                    dtype=torch.float32).unsqueeze(0).to(DEVICE)

                # Run model
                with torch.no_grad():
                    pred_main, pred_aux_cnn, pred_aux_lstm, pred_aux_tr = model(x_cnn, x_lstm, x_tr)

                pm = pred_main.item() / 100.0
                pc = pred_aux_cnn.item() / 100.0
                pl = pred_aux_lstm.item() / 100.0
                pt = pred_aux_tr.item() / 100.0

                # === Bot filtresi: sadece %10 üstü tahminlere trade açılır ===
                pred_pct_abs = abs(pm) * 100  # pm=0.15 → 15%
                if pred_pct_abs < MIN_PREDICTION_PCT:
                    continue  # Bot bu tahminde trade açmaz, skip

                # === Bot filtresi: confidence kontrolü ===
                branches = [pc, pl, pt]
                signs = [1 if b > 0 else -1 for b in branches]
                conf = abs(sum(signs)) / 3.0 * 100  # 33, 66, veya 100
                
                if pm > 0 and conf < 66.0:
                    continue  # Bot zayıf LONG açmaz, skip

                # Map df_ai index to df_display index
                display_idx = df_display.index.get_loc(df_ai.index[i])

                # === Compute label: TP/SL simülasyonu (32 mum = 8 saat) ===
                entry_price = df_display.iloc[display_idx]['Close']
                
                label = 0  # Default: zararlı
                if (display_idx + FUTURE_CANDLES) < len(df_display):
                    if pm > 0:  # LONG tahmin
                        tp_price = entry_price * (1 + LABEL_TP_PCT)   # +%1
                        sl_price = entry_price * (1 - LABEL_SL_PCT)   # -%2
                    else:       # SHORT tahmin
                        tp_price = entry_price * (1 - LABEL_TP_PCT)   # -%1
                        sl_price = entry_price * (1 + LABEL_SL_PCT)   # +%2
                    
                    for k in range(1, FUTURE_CANDLES + 1):
                        future_row = df_display.iloc[display_idx + k]
                        if pm > 0:  # LONG
                            if future_row['Low'] <= sl_price:
                                label = 0
                                break
                            if future_row['High'] >= tp_price:
                                label = 1
                                break
                        else:       # SHORT
                            if future_row['High'] >= sl_price:
                                label = 0
                                break
                            if future_row['Low'] <= tp_price:
                                label = 1
                                break

                # === Compute features (BEFORE adding label to streak) ===
                features = compute_all_features(
                    pred_main=pm * 100,  # back to %
                    pred_cnn=pc * 100,
                    pred_lstm=pl * 100,
                    pred_tr=pt * 100,
                    df_display=df_display,
                    candle_idx=display_idx,
                    recent_labels=recent_labels,
                )
                features['label'] = label
                features['coin'] = coin
                features['timeframe'] = tf
                features['timestamp'] = str(df_ai.index[i])

                # Update streak buffer AFTER feature computation
                recent_labels.append(label)

                all_samples.append(features)
                samples_this += 1

                # Progress
                if verbose and samples_this % 2000 == 0:
                    progress = (i - MAX_WINDOW) / (total - MAX_WINDOW - FUTURE_CANDLES) * 100
                    print(f"  📈 İlerleme: {progress:.1f}% — {samples_this} sample")

            print(f"  ✅ {coin}/{tf}: {samples_this} sample üretildi")

    if not all_samples:
        print("❌ Hiç sample üretilemedi!")
        return pd.DataFrame()

    df = pd.DataFrame(all_samples)
    print(f"\n🎯 Toplam: {len(df)} sample")
    print(f"   Karlı: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"   Zararlı: {(df['label']==0).sum()} ({(1-df['label'].mean())*100:.1f}%)")

    return df


# ============================================
# TRAINING
# ============================================
def train_meta_model(df: pd.DataFrame, threshold: float = 0.60):
    """Train XGBoost meta-model with time-based split and extended reporting."""

    REPORTS_DIR.mkdir(exist_ok=True)

    # Sort by timestamp (preserve time order)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Feature matrix
    X = df[FEATURE_NAMES].values
    y = df['label'].values

    # Time-based split: first 80% train, last 20% val
    split_idx = int(len(df) * 0.80)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # === UNDERSAMPLING: Zararılı sınıfı karlı sayısına düşür ===
    # XGBoost'ta sıra önemli değil, eşit sayıda örnek çok daha iyi öğrenir
    karli_mask = y_train == 1
    zarali_mask = y_train == 0
    n_karli = karli_mask.sum()
    n_zarali = zarali_mask.sum()
    
    if n_karli > 0 and n_zarali > n_karli:
        # Rastgele zararılı örnekler seç (karlı sayısı kadar)
        np.random.seed(42)
        zarali_indices = np.where(zarali_mask)[0]
        selected_zarali = np.random.choice(zarali_indices, size=n_karli, replace=False)
        karli_indices = np.where(karli_mask)[0]
        balanced_indices = np.concatenate([karli_indices, selected_zarali])
        np.random.shuffle(balanced_indices)
        
        X_train = X_train[balanced_indices]
        y_train = y_train[balanced_indices]
        print(f"  ⚖️ Undersampled: {n_zarali} zararılı → {n_karli} (karlı ile eşit)")

    print(f"\n{'='*60}")
    print(f"🧠 XGBoost Eğitimi Başlıyor")
    print(f"{'='*60}")
    print(f"  Train: {len(X_train)} sample (Karlı: {y_train.sum()}/{len(y_train)})")
    print(f"  Val:   {len(X_val)} sample (Karlı: {y_val.sum()}/{len(y_val)})")

    # Class balance
    pos_ratio = y_train.sum() / len(y_train)
    neg_ratio = 1 - pos_ratio
    scale_pos = neg_ratio / pos_ratio if pos_ratio > 0 else 1.0

    # XGBoost DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)

    # ============================================
    # OPTUNA HYPERPARAMETER OPTIMIZATION
    # ============================================
    print(f"\n{'='*60}")
    print(f"🔍 Optuna Hyperparameter Optimization (50 trial)")
    print(f"{'='*60}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'scale_pos_weight': scale_pos,
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'seed': 42,
            'verbosity': 0,
        }
        n_rounds = trial.suggest_int('n_rounds', 50, 500)

        mdl = xgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            evals=[(dval, 'val')],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        y_prob = mdl.predict(dval)
        from sklearn.metrics import average_precision_score
        return average_precision_score(y_val, y_prob)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best = study.best_params
    print(f"\n  ✅ En iyi skor (avg_precision): {study.best_value:.4f}")
    print(f"  🎯 En iyi parametreler:")
    for k, v in best.items():
        print(f"     {k}: {v}")

    # ============================================
    # FINAL TRAINING with best params
    # ============================================
    print(f"\n{'='*60}")
    print(f"🧠 XGBoost Eğitimi (optimize edilmiş parametrelerle)")
    print(f"{'='*60}")
    print(f"  Train: {len(X_train)} sample (Karlı: {y_train.sum()}/{len(y_train)})")
    print(f"  Val:   {len(X_val)} sample (Karlı: {y_val.sum()}/{len(y_val)})")

    n_rounds_best = best.pop('n_rounds')
    best_params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'error', 'auc'],
        'scale_pos_weight': scale_pos,
        'seed': 42,
        'verbosity': 0,
        **best,
    }

    evals_result = {}
    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=n_rounds_best,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=50,
        evals_result=evals_result,
    )

    # ============================================
    # PREDICTIONS
    # ============================================
    y_pred_proba = model.predict(dval)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # ============================================
    # REPORT 1: Classification Report
    # ============================================
    print(f"\n{'='*60}")
    print(f"📊 CLASSIFICATION REPORT (threshold={threshold:.2f})")
    print(f"{'='*60}")
    report = classification_report(y_val, y_pred, target_names=['Zararlı', 'Karlı'], output_dict=True)
    print(classification_report(y_val, y_pred, target_names=['Zararlı', 'Karlı']))

    # ============================================
    # REPORT 2: Confusion Matrix
    # ============================================
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Zararlı (Pred)', 'Karlı (Pred)'],
                yticklabels=['Zararlı (Gerçek)', 'Karlı (Gerçek)'],
                ax=ax)
    ax.set_title(f'Meta-Model Confusion Matrix\n(Threshold: {threshold:.2f})', fontsize=14)
    ax.set_ylabel('Gerçek', fontsize=12)
    ax.set_xlabel('Tahmin', fontsize=12)

    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'confusion_matrix.png', dpi=150)
    plt.close()
    print(f"  💾 Confusion matrix → {REPORTS_DIR / 'confusion_matrix.png'}")

    # ============================================
    # REPORT 3: Train/Val Loss Curves
    # ============================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Log Loss
    axes[0].plot(evals_result['train']['logloss'], label='Train', color='#2196F3', linewidth=1.5)
    axes[0].plot(evals_result['val']['logloss'], label='Val', color='#FF5722', linewidth=1.5)
    axes[0].set_title('Log Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Boosting Round')
    axes[0].set_ylabel('Log Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error Rate
    axes[1].plot(evals_result['train']['error'], label='Train', color='#2196F3', linewidth=1.5)
    axes[1].plot(evals_result['val']['error'], label='Val', color='#FF5722', linewidth=1.5)
    axes[1].set_title('Error Rate', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Boosting Round')
    axes[1].set_ylabel('Error Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC
    axes[2].plot(evals_result['train']['auc'], label='Train', color='#2196F3', linewidth=1.5)
    axes[2].plot(evals_result['val']['auc'], label='Val', color='#FF5722', linewidth=1.5)
    axes[2].set_title('AUC Score', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Boosting Round')
    axes[2].set_ylabel('AUC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Meta-Model Training Curves', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Training curves → {REPORTS_DIR / 'training_curves.png'}")

    # ============================================
    # REPORT 4: Feature Importance
    # ============================================
    importance = model.get_score(importance_type='gain')
    # Make sure all features have a score
    feat_imp = {f: importance.get(f, 0.0) for f in FEATURE_NAMES}
    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    feat_names = [f[0] for f in sorted_feats]
    feat_vals = [f[1] for f in sorted_feats]

    # Color by category
    category_colors = {
        'Model': '#2196F3',
        'Price Action': '#4CAF50',
        'Regime': '#FF9800',
        'Market Structure': '#9C27B0',
        'Cost Aware': '#F44336',
        'Time': '#607D8B',
    }
    feature_categories = {
        "pred_pct": "Model", "confidence": "Model", "pred_conf_product": "Model",
        "branch_agreement": "Model", "rolling_hit_rate": "Model",
        "pred_cnn": "Model", "pred_lstm": "Model", "pred_tr": "Model", "branch_std": "Model",
        "wick_up_ratio": "Price Action", "wick_down_ratio": "Price Action", "body_ratio": "Price Action",
        "atr_rank": "Regime", "bb_squeeze": "Regime", "vol_spike_strength": "Regime",
        "chop_score": "Market Structure", "trend_strength": "Market Structure",
        "net_edge_after_fee": "Cost Aware",
        "hour_sin": "Time", "hour_cos": "Time",
    }
    colors = [category_colors.get(feature_categories.get(f, 'Model'), '#999') for f in feat_names]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    bars = ax.barh(range(len(feat_names)), feat_vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=10)
    ax.set_xlabel('Gain (Importance)', fontsize=12)
    ax.set_title('Meta-Model Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=cat) for cat, c in category_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'feature_importance.png', dpi=150)
    plt.close()
    print(f"  💾 Feature importance → {REPORTS_DIR / 'feature_importance.png'}")

    # ============================================
    # REPORT 5: ROC Curve & Precision-Recall Curve
    # ============================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='#2196F3', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Mark optimal threshold on ROC
    optimal_idx = np.argmax(tpr - fpr)
    axes[0].scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, zorder=5,
                    label=f'Optimal: {thresholds_roc[optimal_idx]:.2f}')
    axes[0].legend(fontsize=11)

    # Precision-Recall Curve
    prec, rec, thresholds_pr = precision_recall_curve(y_val, y_pred_proba)
    avg_prec = average_precision_score(y_val, y_pred_proba)
    axes[1].plot(rec, prec, color='#4CAF50', linewidth=2, label=f'PR (AP = {avg_prec:.3f})')
    axes[1].fill_between(rec, prec, alpha=0.1, color='#4CAF50')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Baseline
    baseline = y_val.mean()
    axes[1].axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label=f'Baseline: {baseline:.2f}')
    axes[1].legend(fontsize=11)

    fig.suptitle('Meta-Model Evaluation Curves', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'roc_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 ROC & PR curves → {REPORTS_DIR / 'roc_pr_curves.png'}")

    # ============================================
    # REPORT 6: Threshold Analysis
    # ============================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    thresholds_test = np.arange(0.30, 0.85, 0.05)
    accuracies = []
    precisions = []
    recalls = []
    trade_rates = []

    for t in thresholds_test:
        y_t = (y_pred_proba >= t).astype(int)
        if y_t.sum() == 0:
            accuracies.append(0)
            precisions.append(0)
            recalls.append(0)
            trade_rates.append(0)
            continue

        correct = (y_t == y_val).sum()
        accuracies.append(correct / len(y_val))

        # Precision for "karlı" class
        tp = ((y_t == 1) & (y_val == 1)).sum()
        fp = ((y_t == 1) & (y_val == 0)).sum()
        fn = ((y_t == 0) & (y_val == 1)).sum()
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        trade_rates.append(y_t.mean())

    ax.plot(thresholds_test, accuracies, 'o-', color='#2196F3', label='Accuracy', linewidth=2)
    ax.plot(thresholds_test, precisions, 's-', color='#4CAF50', label='Precision (Karlı)', linewidth=2)
    ax.plot(thresholds_test, recalls, '^-', color='#FF9800', label='Recall (Karlı)', linewidth=2)
    ax.plot(thresholds_test, trade_rates, 'D-', color='#9C27B0', label='Trade Oranı', linewidth=2)
    ax.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, label=f'Seçilen Threshold: {threshold:.2f}')

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Threshold Analizi — Trade Filtresi Etkisi', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.25, 0.90)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'threshold_analysis.png', dpi=150)
    plt.close()
    print(f"  💾 Threshold analysis → {REPORTS_DIR / 'threshold_analysis.png'}")

    # ============================================
    # REPORT 7: Label Distribution per Coin/TF
    # ============================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By coin
    coin_stats = df.groupby('coin')['label'].agg(['mean', 'count']).reset_index()
    colors_coin = ['#2196F3', '#4CAF50', '#FF9800'][:len(coin_stats)]
    bars = axes[0].bar(coin_stats['coin'], coin_stats['mean'] * 100, color=colors_coin, edgecolor='white')
    for bar, count in zip(bars, coin_stats['count']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'n={count}', ha='center', fontsize=10)
    axes[0].set_ylabel('Karlı Trade Oranı (%)', fontsize=12)
    axes[0].set_title('Coin Bazında Karlı Oran', fontsize=13, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)

    # By timeframe
    tf_stats = df.groupby('timeframe')['label'].agg(['mean', 'count']).reset_index()
    colors_tf = ['#9C27B0', '#F44336'][:len(tf_stats)]
    bars = axes[1].bar(tf_stats['timeframe'], tf_stats['mean'] * 100, color=colors_tf, edgecolor='white')
    for bar, count in zip(bars, tf_stats['count']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'n={count}', ha='center', fontsize=10)
    axes[1].set_ylabel('Karlı Trade Oranı (%)', fontsize=12)
    axes[1].set_title('Timeframe Bazında Karlı Oran', fontsize=13, fontweight='bold')
    axes[1].grid(True, axis='y', alpha=0.3)

    fig.suptitle('Label Dağılımı Analizi', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'label_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Label distribution → {REPORTS_DIR / 'label_distribution.png'}")

    # ============================================
    # SAVE MODEL & CONFIG
    # ============================================
    model_path = PROJECT_ROOT / 'meta_model.json'
    model.save_model(str(model_path))
    print(f"\n💾 Model kaydedildi → {model_path}")

    # Optimal threshold from ROC
    optimal_threshold = float(thresholds_roc[optimal_idx])

    config = {
        'feature_names': FEATURE_NAMES,
        'threshold': threshold,
        'optimal_threshold_roc': optimal_threshold,
        'fee_pct': FEE_PCT,
        'future_candles': FUTURE_CANDLES,
        'coins_used': df['coin'].unique().tolist(),
        'timeframes_used': TIMEFRAMES,
        'training_date': datetime.now().isoformat(),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'val_metrics': {
            'accuracy': float(report.get('accuracy', 0)),
            'precision_karli': float(report.get('Karlı', {}).get('precision', 0)),
            'recall_karli': float(report.get('Karlı', {}).get('recall', 0)),
            'f1_karli': float(report.get('Karlı', {}).get('f1-score', 0)),
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_prec),
        },
        'best_iteration': int(model.best_iteration) if hasattr(model, 'best_iteration') else 0,
    }

    config_path = PROJECT_ROOT / 'meta_model_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Config kaydedildi → {config_path}")

    # ============================================
    # FINAL SUMMARY
    # ============================================
    print(f"\n{'='*60}")
    print(f"🏆 EĞİTİM TAMAMLANDI")
    print(f"{'='*60}")
    print(f"  📊 Val Accuracy: {report.get('accuracy', 0)*100:.1f}%")
    print(f"  🎯 Karlı Precision: {report.get('Karlı', {}).get('precision', 0)*100:.1f}%")
    print(f"  📈 Karlı Recall: {report.get('Karlı', {}).get('recall', 0)*100:.1f}%")
    print(f"  🔄 Karlı F1: {report.get('Karlı', {}).get('f1-score', 0)*100:.1f}%")
    print(f"  📉 ROC AUC: {roc_auc:.3f}")
    print(f"  📊 Average Precision: {avg_prec:.3f}")
    print(f"  🎯 Seçilen Threshold: {threshold:.2f}")
    print(f"  🎯 Optimal Threshold (ROC): {optimal_threshold:.2f}")
    print(f"  📁 Raporlar: {REPORTS_DIR}/")

    return model


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Meta-Model Trainer — Trade Gate')
    parser.add_argument('--coins', nargs='+', default=DEFAULT_COINS,
                        help='Coinler (default: BTC LTC ADA)')
    parser.add_argument('--months', type=int, default=18,
                        help='Kaç aylık veri (default: 18)')
    parser.add_argument('--threshold', type=float, default=0.60,
                        help='Karlılık threshold (default: 0.60)')
    args = parser.parse_args()

    print(f"""
{'='*60}
🧠 META-MODEL TRAINER — Trade Gate
{'='*60}
📌 Coinler: {', '.join(args.coins)}
📌 Timeframe: {', '.join(TIMEFRAMES)}
📌 Veri Süresi: {args.months} ay
📌 Threshold: {args.threshold}
📌 Label: Sonraki {FUTURE_CANDLES} mum kârlılığı
📌 Features: {len(FEATURE_NAMES)} adet
{'='*60}
""")

    # Step 1: Generate training data
    df = generate_training_data(args.coins, args.months)
    if df.empty:
        return

    # Save training data
    csv_path = PROJECT_ROOT / 'meta_training_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"💾 Eğitim verisi → {csv_path}")

    # Step 2: Train model
    train_meta_model(df, threshold=args.threshold)


if __name__ == "__main__":
    main()
