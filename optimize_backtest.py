"""
================================================================================
🔬 BACKTEST PARAMETER OPTIMIZER
================================================================================
Futures işlemlerini etkileyen parametrelerin grid search ile optimize edilmesi.
Mevcut visual_backtest.py'deki RealAIBacktester'ı kullanır ~ eski koda dokunmaz.

Kullanım:
    # Varsayılan: BTC, son 30 gün, tüm parametreler
    python optimize_backtest.py --coin BTC --days 30

    # Sadece SL/TP optimizasyonu (hızlı)
    python optimize_backtest.py --coin BTC --days 30 --params sl_tp

    # Sadece trailing SL optimizasyonu
    python optimize_backtest.py --coin BTC --days 30 --params trailing

    # Sadece entry threshold optimizasyonu
    python optimize_backtest.py --coin BTC --days 30 --params entry

    # Tüm parametreler (uzun sürer)
    python optimize_backtest.py --coin BTC --days 30 --params all

    # Fine-tune: en iyi sonuç etrafında daha dar aralık
    python optimize_backtest.py --coin BTC --days 30 --fine-tune results.json

    # Birden fazla coin test et
    python optimize_backtest.py --coin BTC,ETH,SOL --days 30 --params sl_tp
================================================================================
"""

import os
import sys
import json
import copy
import time
import argparse
import itertools
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# ========== CUDA SETUP ==========
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    # Performance optimizations for Ampere+ GPUs
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    print(f"🚀 CUDA aktif: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ CUDA bulunamadı, CPU kullanılacak")

# ========== PATH SETUP ==========
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch visual_backtest DEVICE before any model loading
import visual_backtest as _vb
_vb.DEVICE = DEVICE

from visual_backtest import RealAIBacktester, SimConfig

# Import feature lists for batch prediction
try:
    from visual_backtest import CNN_FEATURES, LSTM_FEATURES, TR_FEATURES
except ImportError:
    CNN_FEATURES = LSTM_FEATURES = TR_FEATURES = None


# ============================================
# BATCH PREDICTION (GPU-optimized)
# ============================================

def batch_precompute_predictions(bt: RealAIBacktester, batch_size: int = 64) -> int:
    """
    Pre-compute ALL predictions using GPU-friendly batching.
    
    Instead of calling get_prediction() one-by-one (each creating tensors + GPU transfer),
    this batches multiple candles together for a single model forward pass.
    
    Returns: number of predictions computed
    """
    import pandas as pd
    
    if CNN_FEATURES is None:
        # Fallback to serial prediction
        print("⚠️ Feature lists not importable, falling back to serial prediction")
        return _serial_precompute(bt)
    
    total_frames = bt.end_idx - bt.start_idx
    valid_indices = []
    
    # Find all valid indices (idx >= 120)
    for frame in range(total_frames):
        idx = bt.start_idx + frame
        if idx >= 120:
            valid_indices.append(idx)
    
    if not valid_indices:
        return 0
    
    try:
        # Get column indices
        cnn_cols = [bt.df_ai.columns.get_loc(c) for c in CNN_FEATURES if c in bt.df_ai.columns]
        lstm_cols = [bt.df_ai.columns.get_loc(c) for c in LSTM_FEATURES if c in bt.df_ai.columns]
        tr_cols = [bt.df_ai.columns.get_loc(c) for c in TR_FEATURES if c in bt.df_ai.columns]
        
        # Normalize data once
        mean = pd.Series(bt.model_stats['mean'])
        std = pd.Series(bt.model_stats['std'])
        std[std == 0] = 1.0
        df_normalized = (bt.df_ai - mean) / std
        data = df_normalized.values
        
        computed = 0
        
        # Process in batches
        for batch_start in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[batch_start:batch_start + batch_size]
            
            # Build batch tensors
            cnn_batch = []
            lstm_batch = []
            tr_batch = []
            valid_batch_indices = []
            
            for idx in batch_indices:
                try:
                    pos = bt.df_ai.index.get_loc(bt.df_display.index[idx])
                    if pos < 120:
                        continue
                    
                    cnn_batch.append(data[pos-12:pos, cnn_cols])
                    lstm_batch.append(data[pos-120:pos, lstm_cols])
                    tr_batch.append(data[pos-120:pos, tr_cols])
                    valid_batch_indices.append(idx)
                except Exception:
                    continue
            
            if not valid_batch_indices:
                continue
            
            # Stack into batch tensors and move to GPU
            x_cnn = torch.tensor(np.array(cnn_batch), dtype=torch.float32).to(DEVICE)
            x_lstm = torch.tensor(np.array(lstm_batch), dtype=torch.float32).to(DEVICE)
            x_tr = torch.tensor(np.array(tr_batch), dtype=torch.float32).to(DEVICE)
            
            # Batch forward pass
            with torch.no_grad():
                pred_main, pred_cnn, pred_lstm, pred_tr = bt.model(x_cnn, x_lstm, x_tr)
            
            # Store results
            for i, idx in enumerate(valid_batch_indices):
                prediction = pred_main[i].item() / 100.0
                p_cnn = pred_cnn[i].item() / 100.0
                p_lstm = pred_lstm[i].item() / 100.0
                p_tr = pred_tr[i].item() / 100.0
                
                signs = [1 if b > 0 else -1 for b in [p_cnn, p_lstm, p_tr]]
                confidence = abs(sum(signs)) / 3.0 * 100
                
                current_price = float(bt.df_display.iloc[idx]['Close'])
                current_time = bt.df_display.iloc[idx]['datetime']
                hour = bt.df_display.iloc[idx]['hour']
                
                bt.predictions[idx] = {
                    'prediction': prediction * bt.config.prediction_scale,
                    'prediction_pct': prediction * bt.config.prediction_scale * 100,
                    'confidence': confidence,
                    'price': current_price,
                    'timestamp': current_time,
                    'hour': hour,
                    'pred_cnn': p_cnn,
                    'pred_lstm': p_lstm,
                    'pred_tr': p_tr,
                    'df_display': bt.df_display,
                    'candle_idx': idx,
                }
                computed += 1
            
            # Progress
            done = min(batch_start + batch_size, len(valid_indices))
            if done % (batch_size * 5) == 0 or done == len(valid_indices):
                print(f"  {done}/{len(valid_indices)} tahmin...", end="\r")
        
        print()  # newline after progress
        return computed
        
    except Exception as e:
        print(f"⚠️ Batch prediction hatası: {e}, serial fallback...")
        return _serial_precompute(bt)


def _serial_precompute(bt: RealAIBacktester) -> int:
    """Fallback: serial prediction one-by-one."""
    total_frames = bt.end_idx - bt.start_idx
    computed = 0
    for frame in range(total_frames):
        idx = bt.start_idx + frame
        pred = bt.get_prediction(idx)
        if pred is not None:
            computed += 1
    return computed


# ============================================
# PARAMETER SEARCH SPACES
# ============================================

# SL/TP parameters
SL_TP_PARAMS = {
    'futures_sl_pct': [-3.0, -5.0, -7.0, -10.0, -15.0],
    'futures_tp_pct': [5.0, 10.0, 15.0, 20.0, 25.0],
}

# Entry threshold parameters
ENTRY_PARAMS = {
    'prediction_threshold': [0.001, 0.003, 0.005, 0.008, 0.01],
    'prediction_scale': [0.3, 0.5, 0.65, 0.8, 1.0],
    'min_prediction_pct': [5.0, 8.0, 10.0, 15.0, 20.0],
}

# Trailing SL parameters
TRAILING_PARAMS = {
    'trailing_activation_pct': [1.0, 2.0, 3.0, 4.0, 5.0],
    'trailing_distance_pct': [0.5, 1.0, 1.5, 2.0],
}

# Exit parameters
EXIT_PARAMS = {
    'min_profit_to_exit': [0.005, 0.01, 0.02, 0.03],
    'tp_multiplier': [0.5, 0.65, 0.80, 1.0],
    'meta_threshold': [0.30, 0.40, 0.50, 0.60, 0.70],
}

# Grouped search presets
PARAM_GROUPS = {
    'sl_tp': SL_TP_PARAMS,
    'entry': ENTRY_PARAMS,
    'trailing': TRAILING_PARAMS,
    'exit': EXIT_PARAMS,
}

# Sequential optimization order for 'all' mode
# Each phase carries forward the best params from previous phases
ALL_PHASES = ['sl_tp', 'entry', 'trailing', 'exit']


# ============================================
# OPTIMIZER
# ============================================

@dataclass
class OptimizationResult:
    """Single optimization run result."""
    params: Dict
    total_pnl_pct: float
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float
    final_balance: float
    sharpe_ratio: float
    profit_factor: float
    sl_count: int
    tp_count: int


def calculate_sharpe(equity_curve: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return float((np.mean(returns) - risk_free_rate) / np.std(returns) * np.sqrt(252 * 4))  # Annualized for 15m


def calculate_profit_factor(trades) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    gross_profit = sum(t.pnl_pct for t in trades if t.exit_time and t.pnl_pct > 0)
    gross_loss = abs(sum(t.pnl_pct for t in trades if t.exit_time and t.pnl_pct < 0))
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def run_single_backtest(coin: str, start_date: datetime, end_date: datetime,
                        base_config: SimConfig, param_overrides: Dict,
                        preloaded_bt: Optional[RealAIBacktester] = None) -> Optional[OptimizationResult]:
    """
    Run a single backtest with given parameter overrides.
    
    If preloaded_bt is provided, reuses its model and data (much faster).
    """
    # Create config with overrides
    config = copy.deepcopy(base_config)
    config.show_live = False
    config.trading_mode = "futures"
    
    # Separate meta_threshold (constructor param, not SimConfig attr)
    meta_threshold = param_overrides.pop('meta_threshold', 0.50)
    
    for key, value in param_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Put it back so it appears in results
    param_overrides['meta_threshold'] = meta_threshold
    
    try:
        bt = RealAIBacktester(coin, start_date, end_date, config,
                              use_meta_model=True, meta_threshold=meta_threshold)
        
        # Reuse model and data from preloaded backtester
        if preloaded_bt is not None:
            bt.model = preloaded_bt.model
            bt.model_stats = preloaded_bt.model_stats
            bt.model_params = preloaded_bt.model_params
            bt.df_display = preloaded_bt.df_display.copy()
            bt.df_ai = preloaded_bt.df_ai.copy()
            bt.start_idx = preloaded_bt.start_idx
            bt.end_idx = preloaded_bt.end_idx
            bt.predictions = dict(preloaded_bt.predictions)  # Reuse predictions cache
            # Reuse meta model
            if preloaded_bt.meta_gate is not None:
                bt.meta_gate = preloaded_bt.meta_gate
                bt.use_meta_model = True
        else:
            if not bt.load_model():
                return None
            if not bt.fetch_data():
                return None
        
        # Run simulation (headless, no animation, no plotting)
        total_frames = bt.end_idx - bt.start_idx
        for frame in range(total_frames):
            idx = bt.start_idx + frame
            bt.current_idx = idx
            prediction = bt.get_prediction(idx)
            
            if prediction is None:
                continue
            
            action, reason = bt.should_trade(prediction)
            if action != 'HOLD':
                bt.execute_action(action, prediction, reason, idx)
            
            # Update equity
            current_equity = bt.balance
            if bt.position.side != "FLAT":
                unrealized_pnl = bt.position.pnl_pct(prediction['price']) * config.get_effective_leverage()
                current_equity = bt.balance * (1 + unrealized_pnl / 100)
            bt.equity_curve.append(current_equity)
        
        # Calculate results
        final_balance = bt.equity_curve[-1] if bt.equity_curve else bt.balance
        total_pnl = ((final_balance - config.initial_balance) / config.initial_balance) * 100
        
        winning = [t for t in bt.trades if t.exit_time and t.pnl_pct > 0]
        losing = [t for t in bt.trades if t.exit_time and t.pnl_pct <= 0]
        total_trades = len(bt.trades)
        
        win_rate = len(winning) / total_trades * 100 if total_trades > 0 else 0
        avg_win = sum(t.pnl_pct for t in winning) / len(winning) if winning else 0
        avg_loss = sum(t.pnl_pct for t in losing) / len(losing) if losing else 0
        
        # Max drawdown
        if bt.equity_curve:
            peak = bt.equity_curve[0]
            max_dd = 0
            for eq in bt.equity_curve:
                if eq > peak:
                    peak = eq
                dd = (eq - peak) / peak * 100
                if dd < max_dd:
                    max_dd = dd
        else:
            max_dd = 0
        
        sharpe = calculate_sharpe(bt.equity_curve)
        pf = calculate_profit_factor(bt.trades)
        
        sl_count = len([t for t in bt.trades if 'Stop-Loss' in (t.exit_reason or '')])
        tp_count = len([t for t in bt.trades if 'Take-Profit' in (t.exit_reason or '') or 'Hedef' in (t.exit_reason or '')])
        
        return OptimizationResult(
            params=param_overrides,
            total_pnl_pct=round(total_pnl, 2),
            total_trades=total_trades,
            win_rate=round(win_rate, 1),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            max_drawdown_pct=round(max_dd, 2),
            final_balance=round(final_balance, 2),
            sharpe_ratio=round(sharpe, 2),
            profit_factor=round(pf, 2),
            sl_count=sl_count,
            tp_count=tp_count,
        )
        
    except Exception as e:
        print(f"  ⚠️ Hata: {e}")
        return None


def generate_param_combinations(search_space: Dict) -> List[Dict]:
    """Generate all parameter combinations from search space."""
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]


def run_optimization(coin: str, start_date: datetime, end_date: datetime,
                     param_group: str = 'sl_tp', leverage: int = 5,
                     initial_balance: float = 100.0) -> List[OptimizationResult]:
    """
    Run grid search optimization for given parameter group.
    
    Loads model and data ONCE, then reuses for all parameter combinations.
    """
    search_space = PARAM_GROUPS.get(param_group)
    if search_space is None:
        print(f"❌ Bilinmeyen parametre grubu: {param_group}")
        print(f"   Mevcut gruplar: {', '.join(PARAM_GROUPS.keys())}")
        return []
    
    combinations = generate_param_combinations(search_space)
    total = len(combinations)
    
    print(f"\n{'='*70}")
    print(f"🔬 PARAMETRE OPTİMİZASYONU")
    print(f"{'='*70}")
    print(f"  Coin: {coin}")
    print(f"  Tarih: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"  Leverage: {leverage}x")
    print(f"  Parametre grubu: {param_group}")
    print(f"  Parametre sayısı: {len(search_space)}")
    print(f"  Kombinasyon sayısı: {total}")
    print(f"  Taranacak parametreler:")
    for k, v in search_space.items():
        print(f"    • {k}: {v}")
    print(f"{'='*70}\n")
    
    # Create base config
    base_config = SimConfig(
        trading_mode="futures",
        leverage=leverage,
        initial_balance=initial_balance,
        show_live=False,
        hour_filter_enabled=False,  # Disable for fair comparison
    )
    
    # === PRELOAD: Load model and data ONCE ===
    print("📦 Model ve veri yükleniyor (tek seferlik)...")
    preload_bt = RealAIBacktester(coin, start_date, end_date, base_config,
                                  use_meta_model=True, meta_threshold=0.50)
    if not preload_bt.load_model():
        print("❌ Model yüklenemedi!")
        return []
    if not preload_bt.fetch_data():
        print("❌ Veri çekilemedi!")
        return []
    preload_bt._load_meta_model()  # Load meta model once
    
    # Pre-compute ALL predictions — GPU-batched for speed
    print("🧠 Tüm tahminler hesaplanıyor (batch GPU)...")
    pred_start = time.time()
    computed = batch_precompute_predictions(preload_bt)
    pred_time = time.time() - pred_start
    print(f"✅ {computed} tahmin hesaplandı ({pred_time:.1f}s)")
    
    # === RUN GRID SEARCH ===
    results: List[OptimizationResult] = []
    start_time = time.time()
    
    for i, params in enumerate(combinations, 1):
        # Progress
        elapsed = time.time() - start_time
        avg_per_run = elapsed / i if i > 1 else 0
        eta = avg_per_run * (total - i)
        
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"  [{i}/{total}] {param_str}", end="")
        
        result = run_single_backtest(coin, start_date, end_date, base_config, params, preload_bt)
        
        if result is not None:
            results.append(result)
            emoji = "🟢" if result.total_pnl_pct > 0 else "🔴"
            print(f"  → {emoji} {result.total_pnl_pct:+.1f}% | {result.total_trades}T | WR:{result.win_rate:.0f}% | DD:{result.max_drawdown_pct:.1f}%", end="")
        else:
            print(f"  → ⚠️ HATA", end="")
        
        if i > 1:
            print(f"  (ETA: {eta:.0f}s)")
        else:
            print()
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Toplam süre: {total_time:.1f}s ({total_time/total:.2f}s/test)")
    
    # Sort by total P&L
    results.sort(key=lambda r: r.total_pnl_pct, reverse=True)
    
    return results


def run_fine_tune(coin: str, start_date: datetime, end_date: datetime,
                  best_params: Dict, leverage: int = 5,
                  initial_balance: float = 100.0) -> List[OptimizationResult]:
    """
    Fine-tune around the best parameters found in coarse search.
    Creates a narrow search space around each best parameter value.
    """
    fine_space = {}
    
    # Generate fine grid around each best parameter
    fine_steps = {
        'futures_sl_pct': 1.0,   # ±2 with step 1
        'futures_tp_pct': 2.0,   # ±4 with step 2
        'prediction_threshold': 0.001,
        'prediction_scale': 0.05,
        'trailing_activation_pct': 0.5,
        'trailing_distance_pct': 0.25,
        'min_prediction_pct': 2.0,
        'min_profit_to_exit': 0.005,
        'tp_multiplier': 0.1,
    }
    
    for param, best_val in best_params.items():
        step = fine_steps.get(param, abs(best_val) * 0.1 if best_val != 0 else 0.1)
        
        # Generate 5 values around best
        values = [round(best_val + step * offset, 6) for offset in [-2, -1, 0, 1, 2]]
        
        # Apply parameter-specific constraints
        if param == 'futures_sl_pct':
            values = [v for v in values if v < 0]  # SL must be negative
        elif param in ['futures_tp_pct', 'prediction_scale', 'trailing_activation_pct',
                        'trailing_distance_pct', 'min_prediction_pct', 'min_profit_to_exit',
                        'tp_multiplier']:
            values = [v for v in values if v > 0]  # Must be positive
        elif param == 'prediction_threshold':
            values = [v for v in values if v > 0]
        
        # Remove duplicates and sort
        values = sorted(set(values))
        fine_space[param] = values
    
    print(f"\n🔍 Fine-Tune Aralıkları:")
    for k, v in fine_space.items():
        print(f"  • {k}: {v}")
    
    # Run as 'all' group but with fine space
    # Temporarily override PARAM_GROUPS
    original = PARAM_GROUPS.get('fine_tune')
    PARAM_GROUPS['fine_tune'] = fine_space
    results = run_optimization(coin, start_date, end_date, 'fine_tune', leverage, initial_balance)
    if original is None:
        del PARAM_GROUPS['fine_tune']
    else:
        PARAM_GROUPS['fine_tune'] = original
    
    return results


def print_results_table(results: List[OptimizationResult], top_n: int = 15):
    """Print top results in a formatted table."""
    if not results:
        print("⚠️ Sonuç bulunamadı!")
        return
    
    top = results[:top_n]
    
    print(f"\n{'='*110}")
    print(f"🏆 EN İYİ {min(top_n, len(top))} SONUÇ (toplam {len(results)} test)")
    print(f"{'='*110}")
    
    # Header
    print(f"{'#':>3} {'P&L%':>8} {'Trades':>7} {'Win%':>6} {'AvgW%':>7} {'AvgL%':>7} {'DD%':>7} {'PF':>6} {'SL':>4} {'TP':>4} | Parametreler")
    print(f"{'─'*110}")
    
    for i, r in enumerate(top, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f" {i}."
        param_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
        pf_str = f"{r.profit_factor:.1f}" if r.profit_factor < 100 else "∞"
        
        print(f"{emoji:>3} {r.total_pnl_pct:>+7.1f}% {r.total_trades:>6} {r.win_rate:>5.0f}% "
              f"{r.avg_win_pct:>+6.1f}% {r.avg_loss_pct:>+6.1f}% {r.max_drawdown_pct:>+6.1f}% "
              f"{pf_str:>5} {r.sl_count:>4} {r.tp_count:>4} | {param_str}")
    
    print(f"{'='*110}")
    
    # Print worst for comparison
    if len(results) > 5:
        worst = results[-3:]
        print(f"\n📉 EN KÖTÜ 3:")
        for r in worst:
            param_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
            print(f"   🔴 {r.total_pnl_pct:+.1f}% | {r.total_trades}T | WR:{r.win_rate:.0f}% | {param_str}")
    
    # Print best config as ready-to-use
    best = results[0]
    print(f"\n{'='*70}")
    print(f"⭐ EN İYİ PARAMETRELER (.env format):")
    print(f"{'='*70}")
    
    param_to_env = {
        'futures_sl_pct': 'FUTURES_SL_PCT',
        'futures_tp_pct': 'FUTURES_TP_PCT',
        'prediction_threshold': 'PREDICTION_THRESHOLD',
        'prediction_scale': 'PREDICTION_SCALE',
        'trailing_activation_pct': 'TRAILING_ACTIVATION_PCT',
        'trailing_distance_pct': 'TRAILING_DISTANCE_PCT',
        'min_prediction_pct': 'MIN_PREDICTION_PCT',
        'min_profit_to_exit': 'MIN_PROFIT_TO_EXIT',
        'tp_multiplier': 'TP_MULTIPLIER',
        'meta_threshold': 'META_MODEL_THRESHOLD',
    }
    
    for param, value in best.params.items():
        env_key = param_to_env.get(param, param.upper())
        print(f"  {env_key}={value}")
    
    print(f"\n  📊 Sonuç: {best.total_pnl_pct:+.2f}% | {best.total_trades} trade | WR: {best.win_rate:.0f}% | DD: {best.max_drawdown_pct:.1f}%")


def save_results(results: List[OptimizationResult], coin: str, param_group: str,
                 start_date: datetime, end_date: datetime) -> Path:
    """Save optimization results to JSON."""
    results_dir = PROJECT_ROOT / "optimization_results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"opt_{coin}_{param_group}_{timestamp}.json"
    filepath = results_dir / filename
    
    data = {
        'meta': {
            'coin': coin,
            'param_group': param_group,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_combinations': len(results),
            'timestamp': datetime.now().isoformat(),
        },
        'results': [
            {
                'rank': i + 1,
                **{k: v for k, v in r.__dict__.items()},
            }
            for i, r in enumerate(results)
        ]
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Sonuçlar kaydedildi: {filepath}")
    return filepath


def run_sequential_optimization(coin: str, start_date: datetime, end_date: datetime,
                                leverage: int = 5, initial_balance: float = 100.0) -> Tuple[List[OptimizationResult], Dict]:
    """
    Sequential phase optimization for 'all' mode.
    
    Runs each parameter group in order (sl_tp → entry → trailing → exit),
    carrying forward the best parameters from each phase.
    
    Total combinations: ~25 + 125 + 20 + 16 = ~186 (vs 1,000,000 for full grid)
    """
    print(f"\n{'='*70}")
    print(f"🔬 SEQUENTIAL PARAMETRE OPTİMİZASYONU")
    print(f"{'='*70}")
    print(f"  Coin: {coin}")
    print(f"  Tarih: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"  Leverage: {leverage}x")
    print(f"  Fazlar: {' → '.join(ALL_PHASES)}")
    
    total_combos = sum(len(generate_param_combinations(PARAM_GROUPS[p])) for p in ALL_PHASES)
    print(f"  Toplam kombinasyon: ~{total_combos}")
    print(f"{'='*70}\n")
    
    best_cumulative_params = {}
    all_phase_results = []
    
    for phase_num, phase in enumerate(ALL_PHASES, 1):
        print(f"\n{'─'*70}")
        print(f"📊 FAZ {phase_num}/{len(ALL_PHASES)}: {phase.upper()}")
        if best_cumulative_params:
            print(f"   Önceki fazlardan taşınan: {best_cumulative_params}")
        print(f"{'─'*70}")
        
        # Get search space for this phase
        search_space = PARAM_GROUPS[phase]
        combinations = generate_param_combinations(search_space)
        
        # Merge with best params from previous phases
        merged_combinations = []
        for combo in combinations:
            merged = {**best_cumulative_params, **combo}
            merged_combinations.append(merged)
        
        # Create base config
        base_config = SimConfig(
            trading_mode="futures",
            leverage=leverage,
            initial_balance=initial_balance,
            show_live=False,
            hour_filter_enabled=False,
        )
        
        # Load model/data for this phase (or reuse if possible)
        if phase_num == 1:
            print("📦 Model ve veri yükleniyor (tek seferlik)...")
            preload_bt = RealAIBacktester(coin, start_date, end_date, base_config,
                                          use_meta_model=True, meta_threshold=0.50)
            if not preload_bt.load_model():
                print("❌ Model yüklenemedi!")
                return [], {}
            if not preload_bt.fetch_data():
                print("❌ Veri çekilemedi!")
                return [], {}
            preload_bt._load_meta_model()  # Load meta model once
            
            # Pre-compute ALL predictions — GPU-batched for speed
            print("🧠 Tüm tahminler hesaplanıyor (batch GPU)...")
            pred_start = time.time()
            computed = batch_precompute_predictions(preload_bt)
            pred_time = time.time() - pred_start
            print(f"✅ {computed} tahmin hesaplandı ({pred_time:.1f}s)")
        
        # Run this phase
        results: List[OptimizationResult] = []
        start_time = time.time()
        total = len(merged_combinations)
        
        for i, params in enumerate(merged_combinations, 1):
            # Only show the phase-specific params in progress
            phase_params = {k: v for k, v in params.items() if k in search_space}
            param_str = ", ".join(f"{k}={v}" for k, v in phase_params.items())
            print(f"  [{i}/{total}] {param_str}", end="")
            
            result = run_single_backtest(coin, start_date, end_date, base_config, params, preload_bt)
            
            if result is not None:
                results.append(result)
                emoji = "🟢" if result.total_pnl_pct > 0 else "🔴"
                print(f"  → {emoji} {result.total_pnl_pct:+.1f}%", end="")
            else:
                print(f"  → ⚠️ HATA", end="")
            print()
        
        phase_time = time.time() - start_time
        
        if results:
            results.sort(key=lambda r: r.total_pnl_pct, reverse=True)
            best = results[0]
            
            # Extract only the phase-specific best params
            for key in search_space.keys():
                best_cumulative_params[key] = best.params[key]
            
            print(f"\n  ✅ Faz {phase_num} tamamlandı ({phase_time:.1f}s)")
            print(f"  🏆 En iyi: {best.total_pnl_pct:+.1f}% | {best.total_trades}T | WR:{best.win_rate:.0f}%")
            print(f"  📌 Seçilen: {', '.join(f'{k}={v}' for k, v in search_space.items() if k in best.params for v in [best.params[k]])}")
            
            all_phase_results.extend(results)
        else:
            print(f"\n  ⚠️ Faz {phase_num} sonuçsuz!")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"⭐ TÜM FAZLAR TAMAMLANDI")
    print(f"{'='*70}")
    print(f"  En iyi kümülatif parametreler:")
    for k, v in best_cumulative_params.items():
        print(f"    • {k} = {v}")
    
    # Run one final backtest with all best params combined
    print(f"\n  🔄 Final doğrulama testi...")
    base_config = SimConfig(
        trading_mode="futures",
        leverage=leverage,
        initial_balance=initial_balance,
        show_live=False,
        hour_filter_enabled=False,
    )
    final_result = run_single_backtest(coin, start_date, end_date, base_config, best_cumulative_params, preload_bt)
    if final_result:
        print(f"  ✅ Final: {final_result.total_pnl_pct:+.1f}% | {final_result.total_trades}T | WR:{final_result.win_rate:.0f}% | DD:{final_result.max_drawdown_pct:.1f}%")
        # Put the final result at the top
        all_phase_results.insert(0, final_result)
    
    # Deduplicate and sort
    seen = set()
    unique_results = []
    for r in all_phase_results:
        key = json.dumps(r.params, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    unique_results.sort(key=lambda r: r.total_pnl_pct, reverse=True)
    
    return unique_results, best_cumulative_params


def main():
    parser = argparse.ArgumentParser(description='🔬 Backtest Parameter Optimizer')
    parser.add_argument('--coin', type=str, default='BTC',
                        help='Coin(s) to optimize. Comma-separated for multiple: BTC,ETH,SOL')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to backtest (default: 30)')
    parser.add_argument('--params', type=str, default='sl_tp',
                        choices=['sl_tp', 'entry', 'trailing', 'exit', 'all'],
                        help='Parameter group to optimize (default: sl_tp)')
    parser.add_argument('--leverage', type=int, default=5,
                        help='Leverage (default: 5)')
    parser.add_argument('--balance', type=float, default=100.0,
                        help='Initial balance (default: 100)')
    parser.add_argument('--fine-tune', type=str, default=None, metavar='RESULTS_JSON',
                        help='Fine-tune around best params from a previous run')
    parser.add_argument('--top', type=int, default=15,
                        help='Number of top results to show (default: 15)')
    
    args = parser.parse_args()
    
    coins = [c.strip().upper() for c in args.coin.split(',')]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    print(f"\n{'='*70}")
    print(f"🔬 BACKTEST PARAMETER OPTIMIZER")
    print(f"{'='*70}")
    print(f"  Coin(s): {', '.join(coins)}")
    print(f"  Tarih: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')} ({args.days} gün)")
    print(f"  Parametre grubu: {args.params}")
    print(f"  Leverage: {args.leverage}x")
    print(f"  Bakiye: ${args.balance}")
    
    if args.fine_tune:
        # Fine-tune mode
        fine_path = Path(args.fine_tune)
        if not fine_path.exists():
            print(f"❌ Dosya bulunamadı: {fine_path}")
            return
        
        with open(fine_path) as f:
            prev_data = json.load(f)
        
        best_params = prev_data['results'][0]['params']
        print(f"\n  🔍 Fine-tune base parametreler: {best_params}")
        
        for coin in coins:
            results = run_fine_tune(coin, start_date, end_date, best_params,
                                   args.leverage, args.balance)
            if results:
                print_results_table(results, args.top)
                save_results(results, coin, f"fine_{args.params}", start_date, end_date)
    else:
        # Normal optimization
        for coin in coins:
            if args.params == 'all':
                # Sequential phase optimization (~186 combinations)
                results, best_params = run_sequential_optimization(
                    coin, start_date, end_date, args.leverage, args.balance
                )
            else:
                # Single parameter group optimization
                results = run_optimization(coin, start_date, end_date, args.params,
                                           args.leverage, args.balance)
            if results:
                print_results_table(results, args.top)
                save_results(results, coin, args.params, start_date, end_date)


if __name__ == "__main__":
    main()
