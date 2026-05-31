"""
================================================================================
🧠 ENHANCED META-MODEL TRAINER — Trade Gate with MoE features
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

warnings.filterwarnings('ignore')

# Setup paths to import from enhanced and parent directories
ENHANCED_ROOT = Path(__file__).parent
PARENT_ROOT = ENHANCED_ROOT.parent

sys.path.insert(0, str(PARENT_ROOT))
sys.path.insert(0, str(ENHANCED_ROOT))

from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes
from train_models.ai_engine_enhanced import MultiBranchModel, CNN_FEATURES, LSTM_FEATURES, TR_FEATURES

# Dynamic path resolution
KAGGLE_OUTPUTS_DIR = PARENT_ROOT / 'kaggle_outputs'
if not KAGGLE_OUTPUTS_DIR.exists():
    KAGGLE_OUTPUTS_DIR = ENHANCED_ROOT / 'kaggle_outputs'

REPORTS_DIR = ENHANCED_ROOT / 'meta_reports'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CNN_WINDOW = 12
LSTM_WINDOW = 120
TR_WINDOW = 120
MAX_WINDOW = max(CNN_WINDOW, LSTM_WINDOW, TR_WINDOW)

DEFAULT_COINS = ["BTC", "LTC", "ADA"]
TIMEFRAMES = ["15m", "1h"]

# Expanded feature list (21 features - includes MoE weights)
FEATURE_NAMES = [
    # Model Quality (4)
    "pred_pct", "confidence", "pred_conf_product",
    "branch_agreement",
    # MoE Router Weights (3) - NEW!
    "g_cnn", "g_lstm", "g_tr",
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

FEE_PCT = 0.002
FUTURE_CANDLES = 32
LABEL_TP_PCT = 0.01
LABEL_SL_PCT = 0.02
MIN_PREDICTION_PCT = 10.0

def load_model(coin: str, timeframe: str):
    """Load a trained MultiBranchModel from kaggle_outputs."""
    model_path = KAGGLE_OUTPUTS_DIR / f"{coin}_{timeframe}_model.pth"
    params_path = KAGGLE_OUTPUTS_DIR / f"{coin}_{timeframe}_params.json"
    stats_path = KAGGLE_OUTPUTS_DIR / f"{coin}_{timeframe}_stats.json"

    if not model_path.exists():
        # Fallback local
        model_path = ENHANCED_ROOT / 'trained_models' / f"{coin}_{timeframe}_model.pth"
        params_path = ENHANCED_ROOT / 'trained_models' / f"{coin}_{timeframe}_params.json"
        stats_path = ENHANCED_ROOT / 'trained_models' / f"{coin}_{timeframe}_stats.json"

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
    
    # Load strict=False to handle potential missing weights of router when loading original models
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

    return model, stats

def compute_price_action_features(df_display: pd.DataFrame, idx: int) -> dict:
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
    lookback = min(100, idx)
    if lookback < 20:
        return {"atr_rank": 0.5, "bb_squeeze": 0.5}

    close_arr = df_display['Close'].values[idx - lookback:idx + 1]
    high_arr = df_display['High'].values[idx - lookback:idx + 1]
    low_arr = df_display['Low'].values[idx - lookback:idx + 1]

    tr_values = np.maximum(high_arr[1:] - low_arr[1:],
                           np.maximum(np.abs(high_arr[1:] - close_arr[:-1]),
                                      np.abs(low_arr[1:] - close_arr[:-1])))
    if len(tr_values) > 14:
        atrs = np.convolve(tr_values, np.ones(14)/14, mode='valid')
        current_atr = atrs[-1]
        atr_rank = np.mean(atrs <= current_atr)
    else:
        atr_rank = 0.5

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
    lookback = min(50, idx)
    if lookback < 14:
        return {"chop_score": 50.0, "adx_value": 25.0}

    close_arr = df_display['Close'].values[idx - lookback:idx + 1]
    high_arr = df_display['High'].values[idx - lookback:idx + 1]
    low_arr = df_display['Low'].values[idx - lookback:idx + 1]

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
    g_cnn: float,
    g_lstm: float,
    g_tr: float,
    df_display: pd.DataFrame,
    candle_idx: int,
    recent_labels: deque,
) -> dict:
    """Compute all 21 features for a single prediction point (includes MoE weights)."""

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
        # MoE Gating weights
        "g_cnn": g_cnn,
        "g_lstm": g_lstm,
        "g_tr": g_tr,
    }

    # Price Action
    features.update(compute_price_action_features(df_display, candle_idx))

    # Regime
    features.update(compute_regime_features(df_display, candle_idx))

    # Market Structure
    features.update(compute_market_structure_features(df_display, candle_idx))

    # Cost Edge
    features["net_edge_after_fee"] = abs(pred_main) - FEE_PCT * 100

    # Streak
    if len(recent_labels) > 0:
        labels_list = list(recent_labels)
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

    # Liquidity Sweep
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

    # Swing Distance
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

def generate_training_data(coins: list, months: int, verbose: bool = True) -> pd.DataFrame:
    all_samples = []

    for coin in coins:
        for tf in TIMEFRAMES:
            print(f"\n{'='*60}")
            print(f"📊 {coin}/{tf} — Entegre MoE modeliyle veri üretimi başlıyor...")
            print(f"{'='*60}")

            model, stats = load_model(coin, tf)
            if model is None:
                continue

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

            df_display, df_ai = prepare_dual_dataframes(df_raw)

            mean = pd.Series(stats['mean'])
            std = pd.Series(stats['std'])
            std[std == 0] = 1.0
            df_normalized = (df_ai - mean) / std

            cnn_cols = [df_ai.columns.get_loc(c) for c in CNN_FEATURES if c in df_ai.columns]
            lstm_cols = [df_ai.columns.get_loc(c) for c in LSTM_FEATURES if c in df_ai.columns]
            tr_cols = [df_ai.columns.get_loc(c) for c in TR_FEATURES if c in df_ai.columns]

            data = df_normalized.values
            total = len(data)
            samples_this = 0

            recent_labels = deque(maxlen=20)

            for i in range(MAX_WINDOW, total - FUTURE_CANDLES):
                t = i + 1

                x_cnn = torch.tensor(data[t-CNN_WINDOW:t, cnn_cols], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                x_lstm = torch.tensor(data[t-LSTM_WINDOW:t, lstm_cols], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                x_tr = torch.tensor(data[t-TR_WINDOW:t, tr_cols], dtype=torch.float32).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    pred_main, pred_aux_cnn, pred_aux_lstm, pred_aux_tr, g_weights = model(x_cnn, x_lstm, x_tr)

                pm = pred_main.item() / 100.0
                pc = pred_aux_cnn.item() / 100.0
                pl = pred_aux_lstm.item() / 100.0
                pt = pred_aux_tr.item() / 100.0
                
                # Dynamic MoE weights
                g_c = g_weights[0, 0].item()
                g_l = g_weights[0, 1].item()
                g_t = g_weights[0, 2].item()

                pred_pct_abs = abs(pm) * 100
                if pred_pct_abs < MIN_PREDICTION_PCT:
                    continue

                branches = [pc, pl, pt]
                signs = [1 if b > 0 else -1 for b in branches]
                conf = abs(sum(signs)) / 3.0 * 100
                
                if pm > 0 and conf < 66.0:
                    continue

                display_idx = df_display.index.get_loc(df_ai.index[i])
                entry_price = df_display.iloc[display_idx]['Close']
                
                label = 0
                if (display_idx + FUTURE_CANDLES) < len(df_display):
                    if pm > 0:
                        tp_price = entry_price * (1 + LABEL_TP_PCT)
                        sl_price = entry_price * (1 - LABEL_SL_PCT)
                    else:
                        tp_price = entry_price * (1 - LABEL_TP_PCT)
                        sl_price = entry_price * (1 + LABEL_SL_PCT)
                    
                    for k in range(1, FUTURE_CANDLES + 1):
                        future_row = df_display.iloc[display_idx + k]
                        if pm > 0:
                            if future_row['Low'] <= sl_price:
                                label = 0
                                break
                            if future_row['High'] >= tp_price:
                                label = 1
                                break
                        else:
                            if future_row['High'] >= sl_price:
                                label = 0
                                break
                            if future_row['Low'] <= tp_price:
                                label = 1
                                break

                features = compute_all_features(
                    pred_main=pm * 100,
                    pred_cnn=pc * 100,
                    pred_lstm=pl * 100,
                    pred_tr=pt * 100,
                    g_cnn=g_c,
                    g_lstm=g_l,
                    g_tr=g_t,
                    df_display=df_display,
                    candle_idx=display_idx,
                    recent_labels=recent_labels,
                )
                features['label'] = label
                features['coin'] = coin
                features['timeframe'] = tf
                features['timestamp'] = str(df_ai.index[i])

                recent_labels.append(label)
                all_samples.append(features)
                samples_this += 1

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

def train_meta_model(df: pd.DataFrame, threshold: float = 0.60):
    """Train XGBoost meta-model incorporating new MoE weights as features."""
    REPORTS_DIR.mkdir(exist_ok=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    X = df[FEATURE_NAMES].values
    y = df['label'].values

    split_idx = int(len(df) * 0.80)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Balanced class weights
    karli_mask = y_train == 1
    zarali_mask = y_train == 0
    n_karli = karli_mask.sum()
    n_zarali = zarali_mask.sum()
    
    if n_karli > 0 and n_zarali > n_karli:
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
    print(f"🧠 XGBoost MoE-Entegre Eğitim Başlıyor")
    print(f"{'='*60}")
    print(f"  Train: {len(X_train)} sample (Karlı: {y_train.sum()}/{len(y_train)})")
    print(f"  Val:   {len(X_val)} sample (Karlı: {y_val.sum()}/{len(y_val)})")

    pos_ratio = y_train.sum() / len(y_train)
    neg_ratio = 1 - pos_ratio
    scale_pos = neg_ratio / pos_ratio if pos_ratio > 0 else 1.0

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)

    # Optuna tuning
    print(f"\n🔍 Optuna Optimization (25 trial)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'scale_pos_weight': scale_pos,
            'verbosity': 0,
        }
        n_rounds = trial.suggest_int('n_rounds', 50, 400)

        mdl = xgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            evals=[(dval, 'val')],
            early_stopping_rounds=25,
            verbose_eval=False,
        )

        y_prob = mdl.predict(dval)
        return average_precision_score(y_val, y_prob)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25)

    best = study.best_params
    print(f"  ✅ En iyi skor (AP): {study.best_value:.4f}")

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
        best_params, dtrain,
        num_boost_round=n_rounds_best,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=40,
        verbose_eval=50,
        evals_result=evals_result,
    )

    # Save outputs
    model.save_model(str(ENHANCED_ROOT / 'meta_model_enhanced.json'))
    
    # Save meta config
    config = {
        "features": FEATURE_NAMES,
        "threshold": threshold,
        "train_samples": len(df),
        "val_auc": float(evals_result['val']['auc'][-1]),
        "timestamp": datetime.now().isoformat()
    }
    with open(ENHANCED_ROOT / 'meta_model_config_enhanced.json', 'w') as f:
        json.dump(config, f, indent=4)

    y_pred_proba = model.predict(dval)
    y_pred = (y_pred_proba >= threshold).astype(int)

    print(classification_report(y_val, y_pred, target_names=['Zararlı', 'Karlı']))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'confusion_matrix_enhanced.png', dpi=150)
    plt.close()

    # Feature Importance plotting with Router categories
    importance = model.get_score(importance_type='gain')
    feat_imp = {f: importance.get(f, 0.0) for f in FEATURE_NAMES}
    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    feat_names = [f[0] for f in sorted_feats]
    feat_vals = [f[1] for f in sorted_feats]

    category_colors = {
        'Model': '#2196F3',
        'Router': '#9C27B0',  # Purple for MoE router weights
        'Price Action': '#4CAF50',
        'Regime': '#FF9800',
        'Market Structure': '#FF5722',
        'Cost Aware': '#F44336',
    }
    feature_categories = {
        "pred_pct": "Model", "confidence": "Model", "pred_conf_product": "Model", "branch_agreement": "Model",
        "g_cnn": "Router", "g_lstm": "Router", "g_tr": "Router",
        "wick_up_ratio": "Price Action", "wick_down_ratio": "Price Action", "body_ratio": "Price Action",
        "atr_rank": "Regime", "bb_squeeze": "Regime",
        "chop_score": "Market Structure", "adx_value": "Market Structure",
        "net_edge_after_fee": "Cost Aware",
    }
    colors = [category_colors.get(feature_categories.get(f, 'Model'), '#999') for f in feat_names]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.barh(range(len(feat_names)), feat_vals, color=colors)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'feature_importance_enhanced.png', dpi=150)
    plt.close()

    print(f"✅ Raporlar {REPORTS_DIR} altına kaydedildi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=DEFAULT_COINS)
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    df_data = generate_training_data(args.coins, args.months)
    if not df_data.empty:
        df_data.to_csv(ENHANCED_ROOT / "meta_training_data_enhanced.csv", index=False)
        train_meta_model(df_data, args.threshold)
