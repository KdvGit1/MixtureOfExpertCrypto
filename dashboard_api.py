"""
================================================================================
🚀 CRYPTO AI MODEL DASHBOARD - API Backend
================================================================================
40 eğitilmiş model için Web API - Tahminler, Grafikler, Backtest Sonuçları

Endpoints:
    GET /                           → Dashboard ana sayfa
    GET /api/models                 → Tüm modeller listesi (accuracy sıralı)
    GET /api/models/{coin}/{tf}     → Model detayları
    GET /api/predict/{coin}/{tf}    → Anlık tahmin
    GET /api/chart/{coin}/{tf}      → Grafik verisi
    GET /api/backtest/{coin}/{tf}   → Backtest sonuçları
================================================================================
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "train_models"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import torch

# Import existing modules
from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes
from train_models.ai_engine_improved import MultiBranchModel, MultiBranchCryptoDataset, CNN_FEATURES, LSTM_FEATURES, TR_FEATURES

# ============================================
# CONFIGURATION
# ============================================
KAGGLE_OUTPUTS = PROJECT_ROOT / "kaggle_outputs"
ANALYSIS_RESULTS = PROJECT_ROOT / "train_models" / "analysis_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Timeframe configs (same as batch_model_analyzer.py)
TIMEFRAME_CONFIGS = {
    "15m": {
        "cnn_window": 16,
        "lstm_window": 96,
        "tr_window": 96,
        "months_back": 2
    },
    "1h": {
        "cnn_window": 12,
        "lstm_window": 120,
        "tr_window": 120,
        "months_back": 6
    }
}

# Model cache
_model_cache: Dict[str, MultiBranchModel] = {}
_stats_cache: Dict[str, dict] = {}

# ============================================
# PREDICTION CACHE SYSTEM
# Performance: Tahminler timeframe'e göre cache'lenir
# 15m modeller -> 15 dakikada bir güncellenir
# 1h modeller -> 1 saatte bir güncellenir
# ============================================
import threading
import time

# Global prediction cache
_prediction_cache: Dict[str, dict] = {}  # model_key -> prediction data
_prediction_cache_time: Dict[str, datetime] = {}  # model_key -> last update time
_chart_cache: Dict[str, dict] = {}  # model_key -> chart data
_chart_cache_time: Dict[str, datetime] = {}
_live_predictions_cache: List[dict] = []  # Sorted predictions list
_cache_updating = False
_cache_lock = threading.Lock()

# Cache TTL in minutes
CACHE_TTL = {
    "15m": 15,  # 15 dakika
    "1h": 60    # 1 saat
}

def is_cache_valid(model_key: str, timeframe: str) -> bool:
    """Cache hala geçerli mi kontrol et."""
    if model_key not in _prediction_cache_time:
        return False
    
    age = (datetime.now() - _prediction_cache_time[model_key]).total_seconds() / 60
    return age < CACHE_TTL.get(timeframe, 15)


def update_single_prediction(coin: str, timeframe: str) -> Optional[dict]:
    """Tek bir model için tahmin güncelle ve cache'e yaz."""
    try:
        model, stats = load_model(coin, timeframe)
        config = TIMEFRAME_CONFIGS[timeframe]
        
        symbol = f"{coin}/USDT"
        df_raw = get_crypto_history(symbol, timeframe, config['months_back'])
        df_display, df_ai = prepare_dual_dataframes(df_raw)
        
        last_row = df_display.iloc[-1]
        current_price = float(last_row['Close'])
        current_rsi = float(last_row['RSI'])
        
        mean = pd.Series(stats.get('mean', {}))
        std = pd.Series(stats.get('std', {}))
        
        cnn_w = config['cnn_window']
        lstm_w = config['lstm_window']
        tr_w = config['tr_window']
        
        df_norm = (df_ai - mean) / std
        df_norm = df_norm.fillna(0)
        
        cnn_indices = [df_norm.columns.get_loc(c) for c in CNN_FEATURES if c in df_norm.columns]
        lstm_indices = [df_norm.columns.get_loc(c) for c in LSTM_FEATURES if c in df_norm.columns]
        tr_indices = [df_norm.columns.get_loc(c) for c in TR_FEATURES if c in df_norm.columns]
        
        data_matrix = torch.tensor(df_norm.values, dtype=torch.float32)
        t = len(df_norm) - 1
        
        x_cnn = data_matrix[t - cnn_w:t, cnn_indices].unsqueeze(0).to(DEVICE)
        x_lstm = data_matrix[t - lstm_w:t, lstm_indices].unsqueeze(0).to(DEVICE)
        x_tr = data_matrix[t - tr_w:t, tr_indices].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred, _, _, _ = model(x_cnn, x_lstm, x_tr)
            prediction_pct = float(pred.cpu().numpy())
        
        if prediction_pct > 0.1:
            signal = "LONG"
        elif prediction_pct < -0.1:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"
        
        model_key = f"{coin}_{timeframe}"
        result = {
            "coin": coin,
            "timeframe": timeframe,
            "model_key": model_key,
            "prediction_pct": round(prediction_pct, 4),
            "signal": signal,
            "current_price": current_price,
            "rsi": round(current_rsi, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        with _cache_lock:
            _prediction_cache[model_key] = result
            _prediction_cache_time[model_key] = datetime.now()
        
        return result
    except Exception as e:
        print(f"❌ Prediction error {coin}_{timeframe}: {e}")
        return None


def update_all_predictions():
    """Tüm modellerin tahminlerini güncelle."""
    global _live_predictions_cache, _cache_updating
    
    if _cache_updating:
        return
    
    _cache_updating = True
    print(f"🔄 Updating predictions... {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        models = get_all_models()
        predictions = []
        
        for model in models:
            coin = model['coin']
            tf = model['timeframe']
            model_key = model['model_key']
            
            # Cache geçerli mi?
            if is_cache_valid(model_key, tf):
                pred = _prediction_cache.get(model_key)
                if pred:
                    pred['logo_url'] = model.get('logo_url', '')
                    pred['accuracy'] = model.get('accuracy', 0)
                    predictions.append(pred)
                    continue
            
            # Cache geçersiz, yeni tahmin al
            pred = update_single_prediction(coin, tf)
            if pred:
                pred['logo_url'] = model.get('logo_url', '')
                pred['accuracy'] = model.get('accuracy', 0)
                predictions.append(pred)
        
        # Tahmine göre sırala
        predictions.sort(key=lambda x: abs(x.get('prediction_pct', 0)), reverse=True)
        
        with _cache_lock:
            _live_predictions_cache = predictions
        
        print(f"✅ Updated {len(predictions)} predictions")
    
    except Exception as e:
        print(f"❌ Cache update error: {e}")
    finally:
        _cache_updating = False


def background_cache_updater():
    """Arka planda cache güncelleyici thread."""
    while True:
        try:
            update_all_predictions()
        except Exception as e:
            print(f"❌ Background updater error: {e}")
        
        # 5 dakikada bir kontrol et
        time.sleep(300)

# Backtest cache directory (daily)
BACKTEST_CACHE_DIR = PROJECT_ROOT / "backtest_cache"
BACKTEST_CACHE_DIR.mkdir(exist_ok=True)

# Coin logo URLs (CryptoCompare)
COIN_LOGOS = {
    "BTC": "https://assets.coingecko.com/coins/images/1/small/bitcoin.png",
    "ETH": "https://assets.coingecko.com/coins/images/279/small/ethereum.png",
    "BNB": "https://assets.coingecko.com/coins/images/825/small/bnb-icon2_2x.png",
    "XRP": "https://assets.coingecko.com/coins/images/44/small/xrp-symbol-white-128.png",
    "SOL": "https://assets.coingecko.com/coins/images/4128/small/solana.png",
    "ADA": "https://assets.coingecko.com/coins/images/975/small/cardano.png",
    "DOGE": "https://assets.coingecko.com/coins/images/5/small/dogecoin.png",
    "TRX": "https://assets.coingecko.com/coins/images/1094/small/tron-logo.png",
    "LTC": "https://assets.coingecko.com/coins/images/2/small/litecoin.png",
    "DOT": "https://assets.coingecko.com/coins/images/12171/small/polkadot.png",
    "LINK": "https://assets.coingecko.com/coins/images/877/small/chainlink-new-logo.png",
    "UNI": "https://assets.coingecko.com/coins/images/12504/small/uni.jpg",
    "AVAX": "https://assets.coingecko.com/coins/images/12559/small/Avalanche_Circle_RedWhite_Trans.png",
    "MATIC": "https://assets.coingecko.com/coins/images/4713/small/polygon.png",
    "ATOM": "https://assets.coingecko.com/coins/images/1481/small/cosmos_hub.png",
    "FIL": "https://assets.coingecko.com/coins/images/12817/small/filecoin.png",
    "APT": "https://assets.coingecko.com/coins/images/26455/small/aptos_round.png",
    "ARB": "https://assets.coingecko.com/coins/images/16547/small/photo_2023-03-29_21.47.00.jpeg",
    "OP": "https://assets.coingecko.com/coins/images/25244/small/Optimism.png",
    "INJ": "https://assets.coingecko.com/coins/images/12882/small/Secondary_Symbol.png"
}

def get_coin_logo(coin: str) -> str:
    """Coin logosunu getir."""
    return COIN_LOGOS.get(coin.upper(), "https://assets.coingecko.com/coins/images/1/small/bitcoin.png")


def get_cached_backtest(coin: str, timeframe: str) -> Optional[dict]:
    """Bugünün backtest cache'ini getir, yoksa None döndür."""
    today = datetime.now().strftime("%Y-%m-%d")
    cache_file = BACKTEST_CACHE_DIR / f"{coin}_{timeframe}_{today}.json"
    
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def save_backtest_cache(coin: str, timeframe: str, data: dict):
    """Backtest sonucunu cache'e kaydet."""
    today = datetime.now().strftime("%Y-%m-%d")
    cache_file = BACKTEST_CACHE_DIR / f"{coin}_{timeframe}_{today}.json"
    
    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=2)


def run_simple_backtest(coin: str, timeframe: str, months: int = 3) -> dict:
    """Basit backtest çalıştır: son N ay için tahmin vs gerçek karşılaştırması."""
    try:
        model, stats = load_model(coin, timeframe)
        config = TIMEFRAME_CONFIGS[timeframe]
        
        symbol = f"{coin}/USDT"
        df_raw = get_crypto_history(symbol, timeframe, months)
        df_display, df_ai = prepare_dual_dataframes(df_raw)
        
        # Normalizasyon
        mean = pd.Series(stats.get('mean', {}))
        std = pd.Series(stats.get('std', {}))
        
        cnn_w = config['cnn_window']
        lstm_w = config['lstm_window']
        tr_w = config['tr_window']
        max_w = max(cnn_w, lstm_w, tr_w)
        
        df_norm = (df_ai - mean) / std
        df_norm = df_norm.fillna(0)
        
        cnn_indices = [df_norm.columns.get_loc(c) for c in CNN_FEATURES if c in df_norm.columns]
        lstm_indices = [df_norm.columns.get_loc(c) for c in LSTM_FEATURES if c in df_norm.columns]
        tr_indices = [df_norm.columns.get_loc(c) for c in TR_FEATURES if c in df_norm.columns]
        target_idx = df_norm.columns.get_loc("Log_Ret")
        
        data_matrix = torch.tensor(df_norm.values, dtype=torch.float32)
        
        # Backtest: her 10 mum için tahmin yap
        predictions = []
        actuals = []
        correct_direction = 0
        total_predictions = 0
        
        step = 10 if timeframe == "15m" else 4  # 15m için 10 mum, 1h için 4 mum
        
        for t in range(max_w, len(df_norm) - 1, step):
            x_cnn = data_matrix[t - cnn_w:t, cnn_indices].unsqueeze(0).to(DEVICE)
            x_lstm = data_matrix[t - lstm_w:t, lstm_indices].unsqueeze(0).to(DEVICE)
            x_tr = data_matrix[t - tr_w:t, tr_indices].unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                pred, _, _, _ = model(x_cnn, x_lstm, x_tr)
                pred_val = float(pred.cpu().numpy())
            
            actual_val = float(data_matrix[t, target_idx].numpy())
            
            predictions.append(pred_val)
            actuals.append(actual_val)
            
            # Yön doğruluğu
            if (pred_val > 0 and actual_val > 0) or (pred_val < 0 and actual_val < 0):
                correct_direction += 1
            total_predictions += 1
        
        accuracy = (correct_direction / total_predictions * 100) if total_predictions > 0 else 0
        
        # Kümülatif getiri hesapla (basit simülasyon)
        # Model ve actual değerler log_ret * 100 olarak eğitildi, zaten % cinsinde
        cumulative_return = 0
        for pred, actual in zip(predictions, actuals):
            if pred > 0:  # Long pozisyon
                cumulative_return += actual  # Zaten % değer
            else:  # Short pozisyon
                cumulative_return -= actual  # Zaten % değer
        
        result = {
            "coin": coin,
            "timeframe": timeframe,
            "months_tested": months,
            "total_predictions": total_predictions,
            "correct_direction": correct_direction,
            "accuracy": round(accuracy, 2),
            "cumulative_return": round(cumulative_return, 2),
            "avg_prediction": round(np.mean(np.abs(predictions)), 4),  # Zaten % değer
            "tested_at": datetime.now().isoformat()
        }
        
        # Cache'e kaydet
        save_backtest_cache(coin, timeframe, result)
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "coin": coin,
            "timeframe": timeframe
        }

# ============================================
# MODEL MANAGEMENT
# ============================================

def get_all_models() -> List[dict]:
    """SUMMARY_ALL_MODELS.csv dosyasından tüm modelleri yükle."""
    summary_path = ANALYSIS_RESULTS / "SUMMARY_ALL_MODELS.csv"
    if not summary_path.exists():
        return []
    
    df = pd.read_csv(summary_path)
    models = []
    
    for _, row in df.iterrows():
        model_key = f"{row['Coin']}_{row['Timeframe']}"
        
        # Analysis dosyasından ek bilgi al
        analysis_path = ANALYSIS_RESULTS / f"{model_key}_analysis.json"
        analysis_data = {}
        if analysis_path.exists():
            with open(analysis_path) as f:
                analysis_data = json.load(f)
        
        # History dosyasından training curve
        history_path = KAGGLE_OUTPUTS / f"{model_key}_history.csv"
        has_history = history_path.exists()
        
        models.append({
            "coin": row["Coin"],
            "timeframe": row["Timeframe"],
            "model_key": model_key,
            "logo_url": get_coin_logo(row["Coin"]),
            "accuracy": round(row["Accuracy (%)"], 2),
            "cnn_impact": round(row["CNN Impact"], 2),
            "lstm_impact": round(row["LSTM Impact"], 2),
            "tr_impact": round(row["TR Impact"], 2),
            "has_analysis": bool(analysis_data),
            "has_history": has_history,
            "test_samples": analysis_data.get("test_samples", 0),
            "analyzed_at": analysis_data.get("analyzed_at", "")
        })
    
    return models


def load_model(coin: str, timeframe: str) -> tuple:
    """Model ve istatistikleri yükle (cache'li)."""
    model_key = f"{coin}_{timeframe}"
    
    if model_key in _model_cache:
        return _model_cache[model_key], _stats_cache[model_key]
    
    model_path = KAGGLE_OUTPUTS / f"{model_key}_model.pth"
    params_path = KAGGLE_OUTPUTS / f"{model_key}_params.json"
    stats_path = KAGGLE_OUTPUTS / f"{model_key}_stats.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_key}")
    
    # Load params
    with open(params_path) as f:
        params = json.load(f)
    
    # Load model
    model = MultiBranchModel(
        embed_dim=int(params['embed_dim']),
        dropout=float(params['dropout'])
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Load stats
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    
    # Cache
    _model_cache[model_key] = model
    _stats_cache[model_key] = stats
    
    return model, stats


def get_prediction(coin: str, timeframe: str) -> dict:
    """Anlık tahmin yap."""
    try:
        model, stats = load_model(coin, timeframe)
        config = TIMEFRAME_CONFIGS[timeframe]
        
        # Veriyi çek
        symbol = f"{coin}/USDT"
        df_raw = get_crypto_history(symbol, timeframe, config['months_back'])
        df_display, df_ai = prepare_dual_dataframes(df_raw)
        
        # Son mum bilgisi
        last_row = df_display.iloc[-1]
        current_price = float(last_row['Close'])
        current_rsi = float(last_row['RSI'])
        
        # Normalizasyon
        mean = pd.Series(stats.get('mean', {}))
        std = pd.Series(stats.get('std', {}))
        
        # Son window'u al
        cnn_w = config['cnn_window']
        lstm_w = config['lstm_window']
        tr_w = config['tr_window']
        
        df_norm = (df_ai - mean) / std
        df_norm = df_norm.fillna(0)
        
        # Feature indices
        cnn_indices = [df_norm.columns.get_loc(c) for c in CNN_FEATURES if c in df_norm.columns]
        lstm_indices = [df_norm.columns.get_loc(c) for c in LSTM_FEATURES if c in df_norm.columns]
        tr_indices = [df_norm.columns.get_loc(c) for c in TR_FEATURES if c in df_norm.columns]
        
        data_matrix = torch.tensor(df_norm.values, dtype=torch.float32)
        t = len(df_norm) - 1
        
        x_cnn = data_matrix[t - cnn_w:t, cnn_indices].unsqueeze(0).to(DEVICE)
        x_lstm = data_matrix[t - lstm_w:t, lstm_indices].unsqueeze(0).to(DEVICE)
        x_tr = data_matrix[t - tr_w:t, tr_indices].unsqueeze(0).to(DEVICE)
        
        # Tahmin - Model log_ret * 100 olarak eğitildi, çıktı zaten % cinsinde
        with torch.no_grad():
            pred, _, _, _ = model(x_cnn, x_lstm, x_tr)
            prediction_pct = float(pred.cpu().numpy())  # Zaten % değer
        
        # Signal
        if prediction_pct > 0.1:
            signal = "LONG"
            signal_color = "#22c55e"
        elif prediction_pct < -0.1:
            signal = "SHORT"
            signal_color = "#ef4444"
        else:
            signal = "NEUTRAL"
            signal_color = "#6b7280"
        
        return {
            "coin": coin,
            "timeframe": timeframe,
            "prediction_pct": round(prediction_pct, 4),
            "signal": signal,
            "signal_color": signal_color,
            "current_price": current_price,
            "rsi": round(current_rsi, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "coin": coin,
            "timeframe": timeframe
        }


def get_chart_data(coin: str, timeframe: str, limit: int = 200) -> dict:
    """Grafik verisi al."""
    try:
        config = TIMEFRAME_CONFIGS[timeframe]
        symbol = f"{coin}/USDT"
        df_raw = get_crypto_history(symbol, timeframe, config['months_back'])
        df_display, _ = prepare_dual_dataframes(df_raw)
        
        # Son N mum
        df_chart = df_display.tail(limit)
        
        return {
            "coin": coin,
            "timeframe": timeframe,
            "candles": [
                {
                    "time": str(idx),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": float(row['Volume'])
                }
                for idx, row in df_chart.iterrows()
            ],
            "indicators": {
                "rsi": df_chart['RSI'].tolist(),
                "sma50": df_chart['SMA_50_Val'].tolist(),
                "ema200": df_chart['EMA_200_Val'].tolist()
            }
        }
        
    except Exception as e:
        return {"error": str(e)}


def get_model_analysis(coin: str, timeframe: str) -> dict:
    """Model analiz verilerini al."""
    model_key = f"{coin}_{timeframe}"
    analysis_path = ANALYSIS_RESULTS / f"{model_key}_analysis.json"
    history_path = KAGGLE_OUTPUTS / f"{model_key}_history.csv"
    
    result = {
        "coin": coin,
        "timeframe": timeframe,
        "model_key": model_key
    }
    
    # Analysis data
    if analysis_path.exists():
        with open(analysis_path) as f:
            result["analysis"] = json.load(f)
    
    # Training history
    if history_path.exists():
        df_history = pd.read_csv(history_path)
        result["training_history"] = {
            "epochs": df_history['epoch'].tolist(),
            "train_loss": df_history['train_loss'].tolist(),
            "val_loss": df_history['val_loss'].tolist(),
            "val_acc": df_history['val_acc'].tolist()
        }
    
    return result


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="Crypto AI Model Dashboard",
    description="40 AI model ile kripto tahminleri",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Templates
static_dir = PROJECT_ROOT / "static"
templates_dir = PROJECT_ROOT / "templates"
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates = Jinja2Templates(directory=str(templates_dir))

# ============================================
# STARTUP / SHUTDOWN EVENTS
# ============================================
_background_thread = None

@app.on_event("startup")
async def startup_event():
    """Sunucu başlatıldığında cache'i doldur ve background thread başlat."""
    global _background_thread
    
    print("🚀 Initializing prediction cache...")
    
    # İlk yükleme - senkron olarak tahminleri al
    update_all_predictions()
    
    # Background thread başlat
    _background_thread = threading.Thread(target=background_cache_updater, daemon=True)
    _background_thread.start()
    
    print("✅ Cache initialized and background updater started")


@app.on_event("shutdown")
async def shutdown_event():
    """Sunucu kapatıldığında temizlik yap."""
    print("🛑 Shutting down...")

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Dashboard ana sayfası."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/models")
async def api_get_models():
    """Tüm modelleri listele (accuracy sıralı)."""
    models = get_all_models()
    return {
        "models": models,
        "total": len(models),
        "updated_at": datetime.now().isoformat()
    }


@app.get("/api/models/{coin}/{timeframe}")
async def api_get_model_detail(coin: str, timeframe: str):
    """Tek model detayları."""
    coin = coin.upper()
    if timeframe not in TIMEFRAME_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}")
    
    analysis = get_model_analysis(coin, timeframe)
    if "analysis" not in analysis:
        raise HTTPException(status_code=404, detail=f"Model not found: {coin}_{timeframe}")
    
    return analysis


@app.get("/api/predict/{coin}/{timeframe}")
async def api_get_prediction(coin: str, timeframe: str):
    """Cache'den tahmin döndür."""
    coin = coin.upper()
    model_key = f"{coin}_{timeframe}"
    
    if timeframe not in TIMEFRAME_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}")
    
    # Cache'den oku
    with _cache_lock:
        if model_key in _prediction_cache:
            return _prediction_cache[model_key]
    
    # Cache'de yoksa fallback (ilk yüklemede olabilir)
    prediction = get_prediction(coin, timeframe)
    if "error" in prediction:
        raise HTTPException(status_code=500, detail=prediction["error"])
    
    return prediction


@app.get("/api/live-predictions")
async def api_get_all_predictions():
    """Cache'den tüm tahminleri döndür (5 dakikada bir güncellenir)."""
    with _cache_lock:
        predictions = _live_predictions_cache.copy()
    
    return {
        "predictions": predictions,
        "total": len(predictions),
        "cached": True,
        "updated_at": datetime.now().isoformat()
    }


@app.get("/api/backtest-rankings")
async def api_get_backtest_rankings():
    """Tüm modellerin backtest sonuçlarını döndür (return'e göre sıralı)."""
    rankings = []
    models = get_all_models()
    
    for model in models:
        coin = model['coin']
        tf = model['timeframe']
        cache_file = BACKTEST_CACHE_DIR / f"{coin}_{tf}_{datetime.now().strftime('%Y%m%d')}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    bt = json.load(f)
                    rankings.append({
                        "coin": coin,
                        "timeframe": tf,
                        "model_key": model['model_key'],
                        "logo_url": model.get('logo_url', ''),
                        "accuracy": bt.get('accuracy', 0),
                        "cumulative_return": bt.get('cumulative_return', 0),
                        "total_predictions": bt.get('total_predictions', 0),
                        "months_tested": bt.get('months_tested', 3)
                    })
            except:
                pass
    
    # Cumulative return'e göre sırala
    rankings.sort(key=lambda x: x.get('cumulative_return', 0), reverse=True)
    
    return {
        "rankings": rankings,
        "total": len(rankings),
        "updated_at": datetime.now().isoformat()
    }


@app.get("/api/chart/{coin}/{timeframe}")
async def api_get_chart(coin: str, timeframe: str, limit: int = 200):
    """Grafik verisi."""
    coin = coin.upper()
    if timeframe not in TIMEFRAME_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}")
    
    chart_data = get_chart_data(coin, timeframe, limit)
    if "error" in chart_data:
        raise HTTPException(status_code=500, detail=chart_data["error"])
    
    return chart_data


@app.get("/api/backtest/{coin}/{timeframe}")
async def api_get_backtest(coin: str, timeframe: str, months: int = 3):
    """Backtest sonuçları (günlük cache'li)."""
    coin = coin.upper()
    if timeframe not in TIMEFRAME_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}")
    
    # Önce cache kontrol et
    cached = get_cached_backtest(coin, timeframe)
    if cached:
        cached["from_cache"] = True
        return cached
    
    # Cache yoksa yeni backtest çalıştır
    backtest_result = run_simple_backtest(coin, timeframe, months)
    if "error" in backtest_result:
        raise HTTPException(status_code=500, detail=backtest_result["error"])
    
    backtest_result["from_cache"] = False
    return backtest_result


@app.get("/api/disclaimer")
async def get_disclaimer():
    """Yasal uyarı metni."""
    return {
        "disclaimer": {
            "en": "This information is for educational purposes only and is not financial advice. Do your own research (DYOR). Past performance does not guarantee future results. Cryptocurrency trading involves substantial risk of loss.",
            "tr": "Bu bilgiler sadece eğitim amaçlıdır ve yatırım tavsiyesi değildir. Kendi araştırmanızı yapın (DYOR). Geçmiş performans gelecekteki sonuçları garanti etmez. Kripto para ticareti önemli kayıp riski içerir."
        },
        "risk_warning": "⚠️ Never invest more than you can afford to lose."
    }


@app.get("/api/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "models_loaded": len(_model_cache),
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Crypto AI Model Dashboard...")
    print(f"📂 Models directory: {KAGGLE_OUTPUTS}")
    print(f"📊 Analysis directory: {ANALYSIS_RESULTS}")
    uvicorn.run(app, host="0.0.0.0", port=8080)
