import sys
import os
import argparse
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Setup sys path to import from enhanced and parent directories
ENHANCED_ROOT = Path(__file__).parent
PARENT_ROOT = ENHANCED_ROOT.parent

sys.path.insert(0, str(PARENT_ROOT))
sys.path.insert(0, str(ENHANCED_ROOT))

# Imports from existing modules
from train_models.ai_engine_enhanced import (
    MultiBranchCryptoDataset,
    MultiBranchModel
)
from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = ENHANCED_ROOT / "backtest_results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Window settings
CNN_WINDOW = 12
LSTM_WINDOW = 120
TR_WINDOW = 120
MAX_WINDOW = max(CNN_WINDOW, LSTM_WINDOW, TR_WINDOW)

def simulate_trading(df_display, predictions, initial_balance=10000.0, leverage=5, threshold=0.0080, fee_rate=0.0005, mmr=0.005, min_hold_bars=48, use_sl_tp=True, sl_pct=0.03, tp_pct=0.06):
    """
    Simulates leveraged long/short trading with threshold filtering, min holding cooldown (bars), and SL/TP.
    """
    balance = initial_balance
    equity_curve = []
    dates = []
    position = "NONE"  # "LONG", "SHORT", "NONE"
    entry_price = 0.0
    liq_price = 0.0
    bars_held = 0
    trades_log = []
    
    for i in range(len(predictions)):
        pred = predictions[i]
        candle_idx = i + MAX_WINDOW
        
        if candle_idx >= len(df_display):
            break
            
        current_candle = df_display.iloc[candle_idx]
        prev_candle = df_display.iloc[candle_idx - 1]
        
        current_date = df_display.index[candle_idx]
        current_close = current_candle['Close']
        current_high = current_candle['High']
        current_low = current_candle['Low']
        
        if position != "NONE":
            bars_held += 1
            
        # 1. Risk Management Checks (Liquidation, SL, TP) - Active at all times
        if position == "LONG":
            if current_low <= liq_price:
                # Wiped out
                balance = 0.0
                trades_log.append({
                    "type": "LIQ",
                    "entry_date": prev_candle.name,
                    "exit_date": current_date,
                    "entry_price": entry_price,
                    "exit_price": liq_price,
                    "pnl_pct": -100.0,
                    "balance": 0.0,
                    "side": "LONG"
                })
                position = "NONE"
                equity_curve.append(0.0)
                dates.append(current_date)
                break
                
            # Check SL / TP
            elif use_sl_tp and current_low <= entry_price * (1.0 - sl_pct):
                sl_price = entry_price * (1.0 - sl_pct)
                gross_pnl = leverage * (sl_price - entry_price) / entry_price
                open_fee = leverage * fee_rate
                close_fee = leverage * fee_rate * (sl_price / entry_price)
                net_pnl = gross_pnl - (open_fee + close_fee)
                balance = max(0.0, balance * (1.0 + net_pnl))
                trades_log.append({
                    "type": "SL",
                    "entry_date": prev_candle.name,
                    "exit_date": current_date,
                    "entry_price": entry_price,
                    "exit_price": sl_price,
                    "pnl_pct": net_pnl * 100.0,
                    "balance": balance,
                    "side": "LONG"
                })
                position = "NONE"
            elif use_sl_tp and current_high >= entry_price * (1.0 + tp_pct):
                tp_price = entry_price * (1.0 + tp_pct)
                gross_pnl = leverage * (tp_price - entry_price) / entry_price
                open_fee = leverage * fee_rate
                close_fee = leverage * fee_rate * (tp_price / entry_price)
                net_pnl = gross_pnl - (open_fee + close_fee)
                balance = max(0.0, balance * (1.0 + net_pnl))
                trades_log.append({
                    "type": "TP",
                    "entry_date": prev_candle.name,
                    "exit_date": current_date,
                    "entry_price": entry_price,
                    "exit_price": tp_price,
                    "pnl_pct": net_pnl * 100.0,
                    "balance": balance,
                    "side": "LONG"
                })
                position = "NONE"
                
        elif position == "SHORT":
            if current_high >= liq_price:
                # Wiped out
                balance = 0.0
                trades_log.append({
                    "type": "LIQ",
                    "entry_date": prev_candle.name,
                    "exit_date": current_date,
                    "entry_price": entry_price,
                    "exit_price": liq_price,
                    "pnl_pct": -100.0,
                    "balance": 0.0,
                    "side": "SHORT"
                })
                position = "NONE"
                equity_curve.append(0.0)
                dates.append(current_date)
                break
                
            # Check SL / TP
            elif use_sl_tp and current_high >= entry_price * (1.0 + sl_pct):
                sl_price = entry_price * (1.0 + sl_pct)
                gross_pnl = leverage * (entry_price - sl_price) / entry_price
                open_fee = leverage * fee_rate
                close_fee = leverage * fee_rate * (sl_price / entry_price)
                net_pnl = gross_pnl - (open_fee + close_fee)
                balance = max(0.0, balance * (1.0 + net_pnl))
                trades_log.append({
                    "type": "SL",
                    "entry_date": prev_candle.name,
                    "exit_date": current_date,
                    "entry_price": entry_price,
                    "exit_price": sl_price,
                    "pnl_pct": net_pnl * 100.0,
                    "balance": balance,
                    "side": "SHORT"
                })
                position = "NONE"
            elif use_sl_tp and current_low <= entry_price * (1.0 - tp_pct):
                tp_price = entry_price * (1.0 - tp_pct)
                gross_pnl = leverage * (entry_price - tp_price) / entry_price
                open_fee = leverage * fee_rate
                close_fee = leverage * fee_rate * (tp_price / entry_price)
                net_pnl = gross_pnl - (open_fee + close_fee)
                balance = max(0.0, balance * (1.0 + net_pnl))
                trades_log.append({
                    "type": "TP",
                    "entry_date": prev_candle.name,
                    "exit_date": current_date,
                    "entry_price": entry_price,
                    "exit_price": tp_price,
                    "pnl_pct": net_pnl * 100.0,
                    "balance": balance,
                    "side": "SHORT"
                })
                position = "NONE"

        # 2. Process signals for entries/reversals/exits (requires min hold bars limit to have passed)
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
            if bars_held >= min_hold_bars:
                if position == "LONG" and (signal == "SELL" or signal == "NEUTRAL"):
                    # Close Long
                    gross_pnl = leverage * (current_close - entry_price) / entry_price
                    open_fee = leverage * fee_rate
                    close_fee = leverage * fee_rate * (current_close / entry_price)
                    net_pnl = gross_pnl - (open_fee + close_fee)
                    balance = max(0.0, balance * (1.0 + net_pnl))
                    trades_log.append({
                        "type": "EXIT",
                        "entry_date": prev_candle.name,
                        "exit_date": current_date,
                        "entry_price": entry_price,
                        "exit_price": current_close,
                        "pnl_pct": net_pnl * 100.0,
                        "balance": balance,
                        "side": "LONG"
                    })
                    
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
                    trades_log.append({
                        "type": "EXIT",
                        "entry_date": prev_candle.name,
                        "exit_date": current_date,
                        "entry_price": entry_price,
                        "exit_price": current_close,
                        "pnl_pct": net_pnl * 100.0,
                        "balance": balance,
                        "side": "SHORT"
                    })
                    
                    if signal == "BUY" and balance > 1.0:
                        position = "LONG"
                        entry_price = current_close
                        liq_price = entry_price * (1.0 - 1.0 / leverage + mmr)
                        bars_held = 0
                    else:
                        position = "NONE"
                
        # 3. Calculate current equity at the end of candle for visual curve
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
        dates.append(current_date)
        
    return equity_curve, dates, trades_log

def simulate_hodl(df_display, initial_balance=10000.0):
    """
    Simulates spot buy and hold (HODL) strategy.
    """
    start_price = df_display.iloc[MAX_WINDOW]['Close']
    amount = initial_balance / start_price
    equity_curve = []
    dates = []
    
    for i in range(MAX_WINDOW, len(df_display)):
        current_price = df_display.iloc[i]['Close']
        equity_curve.append(amount * current_price)
        dates.append(df_display.index[i])
        
    return equity_curve, dates

def calculate_metrics(equity_curve, trades_log, initial_balance):
    """
    Calculates key performance metrics for a trading strategy.
    """
    final_equity = equity_curve[-1] if len(equity_curve) > 0 else 0.0
    total_return = ((final_equity - initial_balance) / initial_balance) * 100.0
    
    # Drawdown calculation
    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (running_max - equity_arr) / (running_max + 1e-9) * 100.0
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    # Trade statistics
    num_trades = len(trades_log)
    winning_trades = [t for t in trades_log if t["pnl_pct"] > 0]
    num_wins = len(winning_trades)
    win_rate = (num_wins / num_trades * 100.0) if num_trades > 0 else 0.0
    
    # Liquidations
    liquidations = [t for t in trades_log if t["type"] == "LIQ"]
    num_liq = len(liquidations)
    
    # Sharpe Ratio (annualized approximation)
    returns = np.diff(equity_curve) / (equity_arr[:-1] + 1e-9)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(365 * 24 * 4)
    else:
        sharpe = 0.0
        
    # Profit Factor
    gains = sum([t["pnl_pct"] for t in trades_log if t["pnl_pct"] > 0])
    losses = abs(sum([t["pnl_pct"] for t in trades_log if t["pnl_pct"] < 0]))
    profit_factor = gains / losses if losses > 0 else float('inf') if gains > 0 else 0.0
    
    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "max_dd": max_dd,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "num_liq": num_liq,
        "sharpe": sharpe,
        "profit_factor": profit_factor
    }

def main(args):
    coin = args.coin.upper()
    tf = args.timeframe
    months = args.months
    leverage = args.leverage
    threshold = args.threshold
    fee_rate = args.fee
    min_hold_bars = args.cooldown
    use_sl_tp = args.use_sl_tp
    sl_pct = args.sl
    tp_pct = args.tp
    split = args.split.lower()
    
    print("\n" + "=" * 60)
    print(f"📈 OPTIMIZED MOE LEVERAGED BACKTEST ENGINE: {coin} - {tf}")
    print(f"   Leverage: {leverage}x | Threshold: {threshold} | Cooldown: {min_hold_bars} bars")
    print(f"   SL/TP: {sl_pct*100:.1f}%/{tp_pct*100:.1f}% | Fee: {fee_rate*100:.3f}%")
    print(f"   Data split: {split.upper()} | Period: {months} months")
    print("=" * 60)
    
    # 1. Load Model Files
    model_dir = ENHANCED_ROOT / "trained_models"
    model_path = model_dir / f"{coin}_{tf}_model.pth"
    params_path = model_dir / f"{coin}_{tf}_params.json"
    stats_path = model_dir / f"{coin}_{tf}_stats.json"
    
    if not (model_path.exists() and params_path.exists() and stats_path.exists()):
        print(f"❌ Trained model files not found for {coin} - {tf} in {model_dir}.")
        sys.exit(1)
        
    with open(params_path, "r") as f:
        hyperparams = json.load(f)
        
    with open(stats_path, "r") as f:
        stats_dict = json.load(f)
        
    model = MultiBranchModel(embed_dim=hyperparams['embed_dim'], dropout=hyperparams['dropout']).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # 2. Fetch Historical Data
    symbol = f"{coin}/USDT"
    print(f"🔌 Loading price data from Binance Futures...")
    df_raw = get_crypto_history(symbol, tf, months)
    
    if len(df_raw) < MAX_WINDOW + 200:
        print(f"❌ Insufficient candle data.")
        sys.exit(1)
        
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
        
    print(f"📊 Dataset slice length: {len(df_ai_slice)} candles")
    
    # Create dataset using normalization stats
    stats_mean = pd.Series(stats_dict["mean"])
    stats_std = pd.Series(stats_dict["std"])
    
    test_ds = MultiBranchCryptoDataset(
        df_ai_slice, mean=stats_mean, std=stats_std, cnn_window=CNN_WINDOW, lstm_window=LSTM_WINDOW, tr_window=TR_WINDOW
    )
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    # 3. Batch predictions
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            x_cnn = batch["x_cnn"].to(DEVICE)
            x_lstm = batch["x_lstm"].to(DEVICE)
            x_tr = batch["x_tr"].to(DEVICE)
            p_main, _, _, _, _ = model(x_cnn, x_lstm, x_tr)
            predictions.extend(p_main.cpu().numpy() / 100.0)
    predictions = np.array(predictions)
    
    # 4. Strategy Simulations
    initial_balance = 10000.0
    print("📊 Simulating strategies...")
    
    # Strategy A: Leveraged MoE Long/Short (with optimal parameters)
    eq_moe_5x, dates_5x, trades_5x = simulate_trading(
        df_display_slice, predictions, initial_balance, leverage=leverage, 
        threshold=threshold, fee_rate=fee_rate, min_hold_bars=min_hold_bars,
        use_sl_tp=use_sl_tp, sl_pct=sl_pct, tp_pct=tp_pct
    )
    
    # Strategy B: Non-leveraged MoE Long/Short (1x)
    eq_moe_1x, dates_1x, trades_1x = simulate_trading(
        df_display_slice, predictions, initial_balance, leverage=1, 
        threshold=threshold, fee_rate=fee_rate, min_hold_bars=min_hold_bars,
        use_sl_tp=use_sl_tp, sl_pct=sl_pct, tp_pct=tp_pct
    )
    
    # Strategy C: Spot HODL
    eq_hodl, dates_hodl = simulate_hodl(df_display_slice, initial_balance)
    
    # 5. Evaluate Metrics
    metrics_5x = calculate_metrics(eq_moe_5x, trades_5x, initial_balance)
    metrics_1x = calculate_metrics(eq_moe_1x, trades_1x, initial_balance)
    metrics_hodl = calculate_metrics(eq_hodl, [], initial_balance)
    
    print("\n" + "=" * 60)
    print("🏆 BACKTEST RESULTS SUMMARY (OPTIMIZED)")
    print("=" * 60)
    print(f"🚀 Strategy A: MoE Model {leverage}x Leverage Long/Short")
    print(f"   Final Balance: ${metrics_5x['final_equity']:.2f}")
    print(f"   Total Return: {metrics_5x['total_return']:.2f}%")
    print(f"   Max Drawdown: {metrics_5x['max_dd']:.2f}%")
    print(f"   Sharpe Ratio: {metrics_5x['sharpe']:.4f}")
    print(f"   Profit Factor: {metrics_5x['profit_factor']:.4f}")
    print(f"   Total Trades: {metrics_5x['num_trades']} (Win Rate: {metrics_5x['win_rate']:.2f}%)")
    print(f"   Liquidations: {metrics_5x['num_liq']}")
    print("-" * 60)
    print(f"📈 Strategy B: MoE Model 1x Leverage Long/Short")
    print(f"   Final Balance: ${metrics_1x['final_equity']:.2f}")
    print(f"   Total Return: {metrics_1x['total_return']:.2f}%")
    print(f"   Max Drawdown: {metrics_1x['max_dd']:.2f}%")
    print(f"   Sharpe Ratio: {metrics_1x['sharpe']:.4f}")
    print(f"   Profit Factor: {metrics_1x['profit_factor']:.4f}")
    print(f"   Total Trades: {metrics_1x['num_trades']} (Win Rate: {metrics_1x['win_rate']:.2f}%)")
    print("-" * 60)
    print(f"💎 Strategy C: Spot Buy & Hold (HODL)")
    print(f"   Final Balance: ${metrics_hodl['final_equity']:.2f}")
    print(f"   Total Return: {metrics_hodl['total_return']:.2f}%")
    print(f"   Max Drawdown: {metrics_hodl['max_dd']:.2f}%")
    print(f"   Sharpe Ratio: {metrics_hodl['sharpe']:.4f}")
    print("=" * 60)
    
    # 6. Generate Plots
    print("📊 Plotting results...")
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 1, 1)
    plt.plot(dates_5x, eq_moe_5x, label=f"MoE {leverage}x Leverage (PnL: {metrics_5x['total_return']:.1f}%)", color="cyan", linewidth=2)
    plt.plot(dates_1x, eq_moe_1x, label=f"MoE 1x Leverage (PnL: {metrics_1x['total_return']:.1f}%)", color="magenta", linewidth=1.5)
    plt.plot(dates_hodl, eq_hodl, label=f"Buy & Hold (HODL) (PnL: {metrics_hodl['total_return']:.1f}%)", color="yellow", linestyle="--", linewidth=1.5)
    plt.title(f"Mixture of Experts (MoE) - Optimized Strategy Equity Curve ({coin}/{tf} - {split.upper()})", fontsize=14, color="white")
    plt.xlabel("Date", fontsize=11, color="white")
    plt.ylabel("Equity ($)", fontsize=11, color="white")
    plt.legend(facecolor="#1e1e1e", edgecolor="gray", labelcolor="white")
    plt.grid(True, alpha=0.15)
    
    plt.gca().set_facecolor("#121212")
    plt.gcf().set_facecolor("#121212")
    plt.tick_params(colors="white")
    
    plt.subplot(2, 1, 2)
    dd_5x = (np.maximum.accumulate(eq_moe_5x) - np.array(eq_moe_5x)) / (np.maximum.accumulate(eq_moe_5x) + 1e-9) * 100.0
    dd_1x = (np.maximum.accumulate(eq_moe_1x) - np.array(eq_moe_1x)) / (np.maximum.accumulate(eq_moe_1x) + 1e-9) * 100.0
    dd_hodl = (np.maximum.accumulate(eq_hodl) - np.array(eq_hodl)) / (np.maximum.accumulate(eq_hodl) + 1e-9) * 100.0
    
    plt.fill_between(dates_5x, dd_5x, 0, label=f"MoE {leverage}x DD (Max: {metrics_5x['max_dd']:.1f}%)", color="cyan", alpha=0.3)
    plt.fill_between(dates_1x, dd_1x, 0, label=f"MoE 1x DD (Max: {metrics_1x['max_dd']:.1f}%)", color="magenta", alpha=0.2)
    plt.fill_between(dates_hodl, dd_hodl, 0, label=f"HODL DD (Max: {metrics_hodl['max_dd']:.1f}%)", color="yellow", alpha=0.1, linestyle="--")
    
    plt.title("Strategy Drawdown Profiles (%)", fontsize=14, color="white")
    plt.xlabel("Date", fontsize=11, color="white")
    plt.ylabel("Drawdown %", fontsize=11, color="white")
    plt.gca().invert_yaxis()
    plt.legend(facecolor="#1e1e1e", edgecolor="gray", labelcolor="white")
    plt.grid(True, alpha=0.15)
    
    plt.gca().set_facecolor("#121212")
    plt.tick_params(colors="white")
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"{coin}_{tf}_{split}_backtest_comparison.png"
    plt.savefig(plot_path, facecolor="#121212")
    plt.close()
    
    print(f"✅ Comparison plot saved to {plot_path}")
    
    log_path = OUTPUT_DIR / f"{coin}_{tf}_{split}_trades_log.json"
    with open(log_path, "w") as f:
        json.dump(trades_5x, f, indent=4, default=str)
    print(f"✅ Detailed trade log saved to {log_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Branch MoE Leveraged Backtest Engine")
    parser.add_argument("--coin", type=str, default="ETH", help="Coin to backtest")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe to backtest")
    parser.add_argument("--months", type=int, default=180, help="Months of history to fetch")
    parser.add_argument("--leverage", type=int, default=5, help="Simulation leverage multiplier")
    parser.add_argument("--fee", type=float, default=0.0005, help="Transaction fee rate")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"], help="Dataset split to evaluate")
    
    # Optimized default values derived from the grid search
    parser.add_argument("--threshold", type=float, default=0.80, help="Log return signal trigger threshold")
    parser.add_argument("--cooldown", type=int, default=48, help="Cooldown min hold period in bars")
    parser.add_argument("--use-sl-tp", type=bool, default=True, help="Enable Stop-Loss and Take-Profit limits")
    parser.add_argument("--sl", type=float, default=0.03, help="Stop-Loss percentage (e.g. 0.03 = 3%)")
    parser.add_argument("--tp", type=float, default=0.06, help="Take-Profit percentage (e.g. 0.06 = 6%)")
    
    args = parser.parse_args()
    main(args)
