"""
Crypto Market Scanner - FastAPI Backend
Scans Binance Futures coins using MoE (Mixture of Experts) models
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
import ccxt
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Import from existing modules
from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes
from train_models.ai_engine_improved import MultiBranchModel, CNN_FEATURES, LSTM_FEATURES, TR_FEATURES

# ============================================
# CONFIGURATION
# ============================================
MODEL_MAP = {
    '15m': PROJECT_ROOT / 'train_models' / 'finalized_models' / '3BranchApproach' / '6try' / 'BEST_MODEL_FINAL.pth',
    '1h': PROJECT_ROOT / 'train_models' / 'finalized_models' / '3BranchApproach' / '7try_1h' / 'BEST_MODEL_FINAL.pth'
}

# Model parameters loaded from best_params JSON files
MODEL_PARAMS = {
    '15m': {'embed_dim': 96, 'dropout': 0.31},
    '1h': {'embed_dim': 128, 'dropout': 0.32}
}

# Load params from JSON files if they exist
for tf in MODEL_MAP.keys():
    params_file = PROJECT_ROOT / 'train_models' / 'CryptoMoeApp' / f'best_params_{tf}.json'
    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
            MODEL_PARAMS[tf] = {
                'embed_dim': params.get('embed_dim', 128),
                'dropout': params.get('dropout', 0.15)
            }

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CACHE: Dict[str, MultiBranchModel] = {}

# Global scan state
scan_state = {
    "is_scanning": False,
    "current_pair": "",
    "total_pairs": 0,
    "scanned_count": 0,
    "timeframe": "",
    "results": {},
    "errors": []
}

# ============================================
# MODEL LOADING
# ============================================
def load_model(timeframe: str) -> Optional[MultiBranchModel]:
    """Load and cache the MoE model for a given timeframe."""
    if timeframe in MODEL_CACHE:
        return MODEL_CACHE[timeframe]
    
    if timeframe not in MODEL_MAP:
        print(f"❌ No model defined for timeframe: {timeframe}")
        return None
    
    model_path = MODEL_MAP[timeframe]
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Get model parameters for this timeframe
    params = MODEL_PARAMS.get(timeframe, {'embed_dim': 128, 'dropout': 0.15})
    embed_dim = params['embed_dim']
    dropout = params['dropout']
    
    print(f"🧠 Loading MoE model for {timeframe}: {model_path}")
    print(f"   Parameters: embed_dim={embed_dim}, dropout={dropout:.2f}")
    
    try:
        model = MultiBranchModel(embed_dim=embed_dim, dropout=dropout).to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        
        # Handle DataParallel prefix if present
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
        model.eval()
        
        MODEL_CACHE[timeframe] = model
        print(f"✅ Model loaded successfully for {timeframe}")
        return model
    except Exception as e:
        print(f"❌ Error loading model for {timeframe}: {e}")
        return None

# ============================================
# DATA PROCESSING
# ============================================
def calculate_months_needed(timeframe: str, candle_count: int = 500) -> float:
    """Calculate months of data needed for the given number of candles."""
    tf_minutes = {'1h': 60, '15m': 15, '5m': 5}.get(timeframe, 60)
    total_minutes = candle_count * tf_minutes
    minutes_in_month = 30 * 24 * 60
    return (total_minutes / minutes_in_month) * 1.1  # 10% safety margin

def get_all_futures_pairs(exchange_name: str = "binance") -> List[str]:
    """Fetch all USDT futures pairs from the exchange."""
    try:
        exchange = getattr(ccxt, exchange_name.lower())({
            'enableRateLimit': True,
            'options': {'defaultType': 'futures'}
        })
        exchange.load_markets()
        # Get all USDT perpetual pairs (format: BTC/USDT:USDT)
        pairs = [s for s in exchange.symbols if '/USDT:USDT' in s]
        print(f"📊 Found {len(pairs)} USDT perpetual pairs")
        return sorted(pairs)
    except Exception as e:
        print(f"❌ Error fetching pairs: {e}")
        return []

def prepare_model_input(df_ai: pd.DataFrame, cnn_window: int = 12, lstm_window: int = 120, tr_window: int = 120):
    """Prepare the input tensors for the MultiBranchModel."""
    # Get column indices for each branch
    cnn_cols = [df_ai.columns.get_loc(c) for c in CNN_FEATURES if c in df_ai.columns]
    lstm_cols = [df_ai.columns.get_loc(c) for c in LSTM_FEATURES if c in df_ai.columns]
    tr_cols = [df_ai.columns.get_loc(c) for c in TR_FEATURES if c in df_ai.columns]
    
    # Normalize the data
    mean = df_ai.mean()
    std = df_ai.std()
    if 'Log_Ret' in mean:
        mean['Log_Ret'] = 0.0
    std[std == 0] = 1.0
    df_normalized = (df_ai - mean) / std
    
    data = df_normalized.values
    max_window = max(cnn_window, lstm_window, tr_window)
    
    if len(data) < max_window:
        return None, None, None
    
    # Get the last window of data
    t = len(data)
    x_cnn = data[t - cnn_window:t, cnn_cols]
    x_lstm = data[t - lstm_window:t, lstm_cols]
    x_tr = data[t - tr_window:t, tr_cols]
    
    # Convert to tensors and add batch dimension
    x_cnn = torch.tensor(x_cnn, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x_lstm = torch.tensor(x_lstm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x_tr = torch.tensor(x_tr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    return x_cnn, x_lstm, x_tr

# ============================================
# SCANNING LOGIC
# ============================================
async def scan_single_pair(pair: str, timeframe: str, model: MultiBranchModel) -> dict:
    """Scan a single trading pair and return prediction."""
    # Need 500 candles to account for EMA 200 warmup period
    months_needed = calculate_months_needed(timeframe, candle_count=500)
    
    try:
        # Fetch historical data
        df = get_crypto_history(
            symbol=pair,
            timeframe=timeframe,
            months_back=months_needed,
            exchange_name="binance"
        )
        
        if len(df) < 120:  # Minimum required for LSTM/Transformer window
            return {"error": "Insufficient data", "candles": len(df)}
        
        # Prepare data
        df_display, df_ai = prepare_dual_dataframes(df)
        
        # Check for NaN/Inf
        if df_ai.isnull().values.any() or np.isinf(df_ai.values).any():
            return {"error": "Data contains NaN/Inf values"}
        
        # Prepare model input
        x_cnn, x_lstm, x_tr = prepare_model_input(df_ai)
        if x_cnn is None:
            return {"error": "Insufficient data for model input"}
        
        # Get prediction from all branches
        with torch.no_grad():
            pred_main, pred_cnn, pred_lstm, pred_tr = model(x_cnn, x_lstm, x_tr)
            
            # Reverse training scaling: model was trained with log_return * 100
            prediction = pred_main.item() / 100.0
            aux_cnn = pred_cnn.item() / 100.0
            aux_lstm = pred_lstm.item() / 100.0
            aux_tr = pred_tr.item() / 100.0
            
            # Calculate confidence based on branch agreement
            # If all branches agree on direction (all positive or all negative), high confidence
            # If branches disagree, lower confidence
            branches = [aux_cnn, aux_lstm, aux_tr]
            signs = [1 if b > 0 else -1 for b in branches]
            direction_agreement = abs(sum(signs)) / 3.0  # 1.0 = all agree, 0.33 = split
            
            # Also factor in strength consistency (std dev of predictions)
            branch_std = np.std(branches)
            strength_consistency = max(0, 1 - branch_std * 10)  # Penalize high variance
            
            # Combined confidence: 60% direction agreement, 40% strength consistency
            confidence = (direction_agreement * 0.6 + strength_consistency * 0.4) * 100
            confidence = round(min(100, max(0, confidence)), 1)  # Clamp to 0-100%
        
        # Get last candle info
        last_candle = df_display.iloc[-1]
        
        return {
            "prediction": round(prediction * 100, 4),  # Display as percentage (e.g., 0.5%)
            "prediction_raw": prediction,  # Actual log return value
            "confidence": confidence,  # AI confidence score (0-100%)
            "price": float(last_candle['Close']),
            "rsi": float(last_candle['RSI']) if 'RSI' in last_candle else None,
            "volume": float(last_candle['Volume']),
            "updated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    except Exception as e:
        return {"error": str(e)}

async def run_market_scan(timeframe: str):
    """Run a full market scan for all Binance Futures pairs."""
    global scan_state
    
    scan_state["is_scanning"] = True
    scan_state["timeframe"] = timeframe
    scan_state["results"] = {}
    scan_state["errors"] = []
    scan_state["scanned_count"] = 0
    
    # Load model
    model = load_model(timeframe)
    if model is None:
        scan_state["is_scanning"] = False
        scan_state["errors"].append(f"Failed to load model for {timeframe}")
        return
    
    # Get all pairs
    pairs = get_all_futures_pairs("binance")
    scan_state["total_pairs"] = len(pairs)
    
    print(f"🚀 Starting scan for {len(pairs)} pairs on {timeframe} timeframe")
    
    for i, pair in enumerate(pairs):
        if not scan_state["is_scanning"]:
            break
        
        scan_state["current_pair"] = pair
        scan_state["scanned_count"] = i + 1
        
        result = await scan_single_pair(pair, timeframe, model)
        
        if "error" in result:
            scan_state["errors"].append(f"{pair}: {result['error']}")
        else:
            scan_state["results"][pair] = result
        
        # Small delay to avoid rate limiting
        await asyncio.sleep(0.1)
    
    scan_state["is_scanning"] = False
    scan_state["current_pair"] = ""
    print(f"✅ Scan complete! {len(scan_state['results'])} pairs analyzed.")

# ============================================
# FASTAPI APP
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload models
    print("🚀 Starting Crypto Scanner API...")
    for tf in MODEL_MAP.keys():
        load_model(tf)
    yield
    # Shutdown
    print("👋 Shutting down Crypto Scanner API...")

app = FastAPI(
    title="Crypto MoE Scanner",
    description="Scan Binance Futures coins using Mixture of Experts AI models",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
static_dir = PROJECT_ROOT / "static"
templates_dir = PROJECT_ROOT / "templates"
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ============================================
# API ENDPOINTS
# ============================================
class ScanRequest(BaseModel):
    timeframe: str

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    index_file = templates_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Crypto MoE Scanner</h1><p>UI files not found. Please create templates/index.html</p>")

@app.get("/api/pairs")
async def get_pairs():
    """Get all available Binance Futures pairs."""
    pairs = get_all_futures_pairs("binance")
    return {"pairs": pairs, "count": len(pairs)}

@app.get("/api/timeframes")
async def get_timeframes():
    """Get available timeframes with loaded models."""
    return {
        "timeframes": list(MODEL_MAP.keys()),
        "loaded": list(MODEL_CACHE.keys())
    }

@app.post("/api/scan/start")
async def start_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    """Start a market scan for the given timeframe."""
    if scan_state["is_scanning"]:
        raise HTTPException(status_code=400, detail="A scan is already in progress")
    
    if request.timeframe not in MODEL_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe. Available: {list(MODEL_MAP.keys())}")
    
    background_tasks.add_task(run_market_scan, request.timeframe)
    return {"status": "started", "timeframe": request.timeframe}

@app.post("/api/scan/stop")
async def stop_scan():
    """Stop the current scan."""
    global scan_state
    scan_state["is_scanning"] = False
    return {"status": "stopped"}

@app.get("/api/scan/status")
async def get_scan_status():
    """Get the current scan status."""
    return {
        "is_scanning": scan_state["is_scanning"],
        "current_pair": scan_state["current_pair"],
        "total_pairs": scan_state["total_pairs"],
        "scanned_count": scan_state["scanned_count"],
        "timeframe": scan_state["timeframe"],
        "progress": round(scan_state["scanned_count"] / max(scan_state["total_pairs"], 1) * 100, 1),
        "results_count": len(scan_state["results"]),
        "errors_count": len(scan_state["errors"])
    }

@app.get("/api/scan/results")
async def get_scan_results():
    """Get the scan results."""
    # Sort by absolute prediction value (strongest signals first, both bullish and bearish)
    sorted_results = dict(
        sorted(
            scan_state["results"].items(),
            key=lambda x: abs(x[1].get("prediction", 0)),
            reverse=True
        )
    )
    return {
        "results": sorted_results,
        "timeframe": scan_state["timeframe"],
        "total": len(sorted_results)
    }

@app.get("/api/scan/errors")
async def get_scan_errors():
    """Get scan errors."""
    return {"errors": scan_state["errors"]}

@app.post("/api/analyze/{pair}")
async def analyze_single_pair(pair: str, timeframe: str = "1h"):
    """Analyze a single trading pair."""
    # Format pair if needed
    if "/" not in pair:
        pair = f"{pair.upper()}/USDT:USDT"
    
    model = load_model(timeframe)
    if model is None:
        raise HTTPException(status_code=500, detail=f"Failed to load model for {timeframe}")
    
    result = await scan_single_pair(pair, timeframe, model)
    return {"pair": pair, "timeframe": timeframe, **result}

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
