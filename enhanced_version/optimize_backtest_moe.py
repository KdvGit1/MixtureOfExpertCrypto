import sys
import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Setup paths
ENHANCED_ROOT = Path(__file__).parent
PARENT_ROOT = ENHANCED_ROOT.parent

sys.path.insert(0, str(PARENT_ROOT))
sys.path.insert(0, str(ENHANCED_ROOT))

from train_models.ai_engine_enhanced import MultiBranchCryptoDataset, MultiBranchModel
from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CNN_WINDOW = 12
LSTM_WINDOW = 120
TR_WINDOW = 120
MAX_WINDOW = max(CNN_WINDOW, LSTM_WINDOW, TR_WINDOW)

def run_simulation(df_display, predictions, initial_balance=10000.0, leverage=5, 
                   threshold=0.001, fee_rate=0.0005, mmr=0.005, 
                   min_hold_bars=12, use_sl_tp=True, sl_pct=0.02, tp_pct=0.04):
    """
    Highly optimized leveraged simulation with threshold filtering, min holding cooldown, and SL/TP limits.
    """
    balance = initial_balance
    position = "NONE"  # "LONG", "SHORT", "NONE"
    entry_price = 0.0
    liq_price = 0.0
    bars_held = 0
    
    trades = 0
    wins = 0
    liquidations = 0
    equity_curve = []
    
    for i in range(len(predictions)):
        pred = predictions[i]
        candle_idx = i + MAX_WINDOW
        
        if candle_idx >= len(df_display):
            break
            
        current_candle = df_display.iloc[candle_idx]
        prev_candle = df_display.iloc[candle_idx - 1]
        
        current_close = current_candle['Close']
        current_high = current_candle['High']
        current_low = current_candle['Low']
        
        # Increment hold counter if in position
        if position != "NONE":
            bars_held += 1
            
        # 1. Risk Management Checks (Liquidation, SL, TP) - Active at all times
        if position == "LONG":
            # Check Liquidation
            if current_low <= liq_price:
                balance = 0.0
                trades += 1
                liquidations += 1
                position = "NONE"
                equity_curve.append(0.0)
                break
                
            # Check SL / TP
            elif use_sl_tp and current_low <= entry_price * (1.0 - sl_pct):
                sl_price = entry_price * (1.0 - sl_pct)
                gross_pnl = leverage * (sl_price - entry_price) / entry_price
                open_fee = leverage * fee_rate
                close_fee = leverage * fee_rate * (sl_price / entry_price)
                net_pnl = gross_pnl - (open_fee + close_fee)
                balance = max(0.0, balance * (1.0 + net_pnl))
                trades += 1
                if net_pnl > 0: wins += 1
                position = "NONE"
            elif use_sl_tp and current_high >= entry_price * (1.0 + tp_pct):
                tp_price = entry_price * (1.0 + tp_pct)
                gross_pnl = leverage * (tp_price - entry_price) / entry_price
                open_fee = leverage * fee_rate
                close_fee = leverage * fee_rate * (tp_price / entry_price)
                net_pnl = gross_pnl - (open_fee + close_fee)
                balance = max(0.0, balance * (1.0 + net_pnl))
                trades += 1
                if net_pnl > 0: wins += 1
                position = "NONE"
                
        elif position == "SHORT":
            # Check Liquidation
            if current_high >= liq_price:
                balance = 0.0
                trades += 1
                liquidations += 1
                position = "NONE"
                equity_curve.append(0.0)
                break
                
            # Check SL / TP
            elif use_sl_tp and current_high >= entry_price * (1.0 + sl_pct):
                sl_price = entry_price * (1.0 + sl_pct)
                gross_pnl = leverage * (entry_price - sl_price) / entry_price
                open_fee = leverage * fee_rate
                close_fee = leverage * fee_rate * (sl_price / entry_price)
                net_pnl = gross_pnl - (open_fee + close_fee)
                balance = max(0.0, balance * (1.0 + net_pnl))
                trades += 1
                if net_pnl > 0: wins += 1
                position = "NONE"
            elif use_sl_tp and current_low <= entry_price * (1.0 - tp_pct):
                tp_price = entry_price * (1.0 - tp_pct)
                gross_pnl = leverage * (entry_price - tp_price) / entry_price
                open_fee = leverage * fee_rate
                close_fee = leverage * fee_rate * (tp_price / entry_price)
                net_pnl = gross_pnl - (open_fee + close_fee)
                balance = max(0.0, balance * (1.0 + net_pnl))
                trades += 1
                if net_pnl > 0: wins += 1
                position = "NONE"

        # 2. Signal Processing (Requires hold cooldown to have expired)
        if pred > threshold:
            signal = "BUY"
        elif pred < -threshold:
            signal = "SELL"
        else:
            signal = "NEUTRAL"
            
        if position == "NONE":
            if signal == "BUY" and balance > 1.0:
                position = "LONG"
                entry_price = current_close
                liq_price = entry_price * (1.0 - 1.0 / leverage + mmr)
                bars_held = 0
            elif signal == "SELL" and balance > 1.0:
                position = "SHORT"
                entry_price = current_close
                liq_price = entry_price * (1.0 + 1.0 / leverage - mmr)
                bars_held = 0
                
        else:
            # We are holding a position. We can only exit/reverse if min_hold_bars have passed!
            if bars_held >= min_hold_bars:
                if position == "LONG" and (signal == "SELL" or signal == "NEUTRAL"):
                    # Close Long
                    gross_pnl = leverage * (current_close - entry_price) / entry_price
                    open_fee = leverage * fee_rate
                    close_fee = leverage * fee_rate * (current_close / entry_price)
                    net_pnl = gross_pnl - (open_fee + close_fee)
                    balance = max(0.0, balance * (1.0 + net_pnl))
                    trades += 1
                    if net_pnl > 0: wins += 1
                    
                    # Reverse if signal is Sell
                    if signal == "SELL" and balance > 1.0:
                        position = "SHORT"
                        entry_price = current_close
                        liq_price = entry_price * (1.0 + 1.0 / leverage - mmr)
                        bars_held = 0
                    else:
                        position = "NONE"
                        
                elif position == "SHORT" and (signal == "BUY" or signal == "NEUTRAL"):
                    # Close Short
                    gross_pnl = leverage * (entry_price - current_close) / entry_price
                    open_fee = leverage * fee_rate
                    close_fee = leverage * fee_rate * (current_close / entry_price)
                    net_pnl = gross_pnl - (open_fee + close_fee)
                    balance = max(0.0, balance * (1.0 + net_pnl))
                    trades += 1
                    if net_pnl > 0: wins += 1
                    
                    # Reverse if signal is Buy
                    if signal == "BUY" and balance > 1.0:
                        position = "LONG"
                        entry_price = current_close
                        liq_price = entry_price * (1.0 - 1.0 / leverage + mmr)
                        bars_held = 0
                    else:
                        position = "NONE"
                        
        # Track continuous equity
        if position == "LONG":
            gross_pnl = leverage * (current_close - entry_price) / entry_price
            open_fee = leverage * fee_rate
            close_fee = leverage * fee_rate * (current_close / entry_price)
            net_pnl = gross_pnl - (open_fee + close_fee)
            current_equity = balance * (1.0 + net_pnl)
        elif position == "SHORT":
            gross_pnl = leverage * (entry_price - current_close) / entry_price
            open_fee = leverage * fee_rate
            close_fee = leverage * fee_rate * (current_close / entry_price)
            net_pnl = gross_pnl - (open_fee + close_fee)
            current_equity = balance * (1.0 + net_pnl)
        else:
            current_equity = balance
            
        equity_curve.append(current_equity)
        
    return equity_curve, trades, wins, liquidations

def main(args):
    coin = args.coin.upper()
    tf = args.timeframe
    months = args.months
    leverage = args.leverage
    fee_rate = args.fee
    split = args.split.lower()
    
    print("\n" + "=" * 60)
    print(f"🔍 MOE HYPERPARAMETER GRID OPTIMIZER: {coin} - {tf} ({split.upper()})")
    print(f"   Searching for optimal: Threshold, Cooldown (Bars), SL/TP combos")
    print("=" * 60)
    
    # 1. Load model and normalization stats
    model_dir = ENHANCED_ROOT / "trained_models"
    model_path = model_dir / f"{coin}_{tf}_model.pth"
    params_path = model_dir / f"{coin}_{tf}_params.json"
    stats_path = model_dir / f"{coin}_{tf}_stats.json"
    
    with open(params_path, "r") as f:
        hyperparams = json.load(f)
    with open(stats_path, "r") as f:
        stats_dict = json.load(f)
        
    model = MultiBranchModel(embed_dim=hyperparams['embed_dim'], dropout=hyperparams['dropout']).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # 2. Fetch data
    symbol = f"{coin}/USDT"
    df_raw = get_crypto_history(symbol, tf, months)
    df_display, df_ai = prepare_dual_dataframes(df_raw)
    
    total_len = len(df_ai)
    train_end = int(0.70 * total_len)
    val_end = int(0.85 * total_len)
    
    if split == "train":
        df_ai_slice = df_ai.iloc[:train_end]
        df_display_slice = df_display.iloc[:train_end]
    elif split == "val":
        df_ai_slice = df_ai.iloc[train_end:val_end]
        df_display_slice = df_display.iloc[train_end:val_end]
    elif split == "test":
        df_ai_slice = df_ai.iloc[val_end:]
        df_display_slice = df_display.iloc[val_end:]
    else:
        df_ai_slice = df_ai
        df_display_slice = df_display
        
    stats_mean = pd.Series(stats_dict["mean"])
    stats_std = pd.Series(stats_dict["std"])
    
    test_ds = MultiBranchCryptoDataset(
        df_ai_slice, mean=stats_mean, std=stats_std, cnn_window=CNN_WINDOW, lstm_window=LSTM_WINDOW, tr_window=TR_WINDOW
    )
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            x_cnn = batch["x_cnn"].to(DEVICE)
            x_lstm = batch["x_lstm"].to(DEVICE)
            x_tr = batch["x_tr"].to(DEVICE)
            p_main, _, _, _, _ = model(x_cnn, x_lstm, x_tr)
            predictions.extend(p_main.cpu().numpy() / 100.0)
    predictions = np.array(predictions) * stats_dict["std"]["Log_Ret"]
    
    # 3. Define Grid Search Space
    thresholds = [0.0, 0.00005, 0.0001, 0.00015, 0.0002, 0.0003]
    cooldowns = [8, 16, 24, 48]  # Min hold in bars
    sl_tp_combos = [
        {"use": True, "sl": 0.01, "tp": 0.02},  # Tight 1:2
        {"use": True, "sl": 0.02, "tp": 0.04},  # Medium 1:2
        {"use": True, "sl": 0.02, "tp": 0.06},  # Medium 1:3
        {"use": True, "sl": 0.03, "tp": 0.06},  # Standard 1:2
        {"use": False, "sl": 0.0, "tp": 0.0}    # Pure Signal-driven (No SL/TP)
    ]
    
    results = []
    initial_balance = 10000.0
    
    print("\n🔍 Running parameter grid search...")
    for th in thresholds:
        for cd in cooldowns:
            for combo in sl_tp_combos:
                eq, tr, win, liq = run_simulation(
                    df_display_slice, predictions, initial_balance=initial_balance, leverage=leverage,
                    threshold=th, fee_rate=fee_rate, min_hold_bars=cd,
                    use_sl_tp=combo["use"], sl_pct=combo["sl"], tp_pct=combo["tp"]
                )
                
                final_eq = eq[-1] if len(eq) > 0 else 0.0
                pnl_pct = ((final_eq - initial_balance) / initial_balance) * 100.0
                win_rate = (win / tr * 100.0) if tr > 0 else 0.0
                
                # Drawdown
                eq_arr = np.array(eq)
                run_max = np.maximum.accumulate(eq_arr)
                dd = (run_max - eq_arr) / (run_max + 1e-9) * 100.0
                max_dd = np.max(dd) if len(dd) > 0 else 0.0
                
                results.append({
                    "threshold": th,
                    "cooldown": cd,
                    "use_sl_tp": combo["use"],
                    "sl": combo["sl"],
                    "tp": combo["tp"],
                    "final_balance": final_eq,
                    "pnl_pct": pnl_pct,
                    "trades": tr,
                    "win_rate": win_rate,
                    "max_dd": max_dd,
                    "liquidations": liq
                })
                
    # Sort results by final balance descending
    results.sort(key=lambda x: x["final_balance"], reverse=True)
    
    print("\n" + "=" * 60)
    print("🏆 TOP 10 PARAMETER CONFIGURATIONS FOUND")
    print("=" * 60)
    print(f"{'Rank':<5}{'Thresh':<8}{'Hold':<6}{'SL/TP':<12}{'Trades':<8}{'Win%':<8}{'MaxDD%':<8}{'Final Balance':<15}")
    print("-" * 60)
    
    for r_idx, r in enumerate(results[:10]):
        sl_tp_str = f"{r['sl']*100:.0f}%/{r['tp']*100:.0f}%" if r["use_sl_tp"] else "Signal-Only"
        print(f"#{r_idx+1:<4}{r['threshold']:<8.4f}{r['cooldown']:<6}{sl_tp_str:<12}{r['trades']:<8}{r['win_rate']:<8.2f}{r['max_dd']:<8.2f}${r['final_balance']:<14.2f}")
        
    print("=" * 60)
    
    # Save search results to file
    out_path = ENHANCED_ROOT / "backtest_results" / f"{coin}_{tf}_{split}_grid_search.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"💾 Full grid search results saved to {out_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search Optimization Engine for MoE Model")
    parser.add_argument("--coin", type=str, default="ETH", help="Coin to optimize")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe to optimize")
    parser.add_argument("--months", type=int, default=180, help="Months of historical data")
    parser.add_argument("--leverage", type=int, default=5, help="Simulation leverage multiplier")
    parser.add_argument("--fee", type=float, default=0.0005, help="Maker/taker transaction fee rate")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"], help="Dataset split to evaluate")
    
    args = parser.parse_args()
    main(args)
