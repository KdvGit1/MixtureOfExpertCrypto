import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ENHANCED_ROOT = Path(__file__).parent.parent
PARENT_ROOT = ENHANCED_ROOT.parent

sys.path.insert(0, str(PARENT_ROOT))
sys.path.insert(0, str(ENHANCED_ROOT))

from train_models.ai_engine_enhanced import MultiBranchCryptoDataset, MultiBranchModel
from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes

# Load data and model once
coin = "ETH"
tf = "15m"
months = 180
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

symbol = f"{coin}/USDT"
df_raw = get_crypto_history(symbol, tf, months)
df_display, df_ai = prepare_dual_dataframes(df_raw)

total_len = len(df_ai)
val_end = int(0.85 * total_len)
df_ai_slice = df_ai.iloc[val_end:]
df_display_slice = df_display.iloc[val_end:]

stats_mean = pd.Series(stats_dict["mean"])
stats_std = pd.Series(stats_dict["std"])

test_ds = MultiBranchCryptoDataset(df_ai_slice, mean=stats_mean, std=stats_std, cnn_window=16, lstm_window=96, tr_window=96)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

predictions = []
with torch.no_grad():
    for batch in test_loader:
        x_cnn = batch["x_cnn"].to(DEVICE)
        x_lstm = batch["x_lstm"].to(DEVICE)
        x_tr = batch["x_tr"].to(DEVICE)
        p_main, _, _, _, _ = model(x_cnn, x_lstm, x_tr)
        predictions.extend(p_main.cpu().numpy() / 100.0)
predictions = np.array(predictions)

# Simulate function
def test_threshold(thresh):
    balance = 10000.0
    position = "NONE"
    entry_price = 0.0
    liq_price = 0.0
    fee_rate = 0.0005
    leverage = 5
    mmr = 0.005
    trades = 0
    wins = 0
    
    max_window = 120
    
    for i in range(len(predictions)):
        pred = predictions[i]
        candle_idx = i + max_window
        if candle_idx >= len(df_display_slice): break
        
        current_candle = df_display_slice.iloc[candle_idx]
        current_close = current_candle['Close']
        current_high = current_candle['High']
        current_low = current_candle['Low']
        
        if position == "LONG":
            if current_low <= liq_price:
                balance = 0.0
                trades += 1
                position = "NONE"
                break
        elif position == "SHORT":
            if current_high >= liq_price:
                balance = 0.0
                trades += 1
                position = "NONE"
                break
                
        # Signal
        if pred > thresh:
            signal = "BUY"
        elif pred < -thresh:
            signal = "SELL"
        else:
            signal = "NEUTRAL"
            
        if position == "NONE":
            if signal == "BUY" and balance > 1.0:
                position = "LONG"
                entry_price = current_close
                liq_price = entry_price * (1.0 - 1.0 / leverage + mmr)
            elif signal == "SELL" and balance > 1.0:
                position = "SHORT"
                entry_price = current_close
                liq_price = entry_price * (1.0 + 1.0 / leverage - mmr)
        elif position == "LONG":
            if signal == "SELL" or signal == "NEUTRAL":
                gross_pnl = leverage * (current_close - entry_price) / entry_price
                open_fee = leverage * fee_rate
                close_fee = leverage * fee_rate * (current_close / entry_price)
                net_pnl = gross_pnl - (open_fee + close_fee)
                balance = max(0.0, balance * (1.0 + net_pnl))
                trades += 1
                if net_pnl > 0: wins += 1
                
                if signal == "SELL" and balance > 1.0:
                    position = "SHORT"
                    entry_price = current_close
                    liq_price = entry_price * (1.0 + 1.0 / leverage - mmr)
                else:
                    position = "NONE"
        elif position == "SHORT":
            if signal == "BUY" or signal == "NEUTRAL":
                gross_pnl = leverage * (entry_price - current_close) / entry_price
                open_fee = leverage * fee_rate
                close_fee = leverage * fee_rate * (current_close / entry_price)
                net_pnl = gross_pnl - (open_fee + close_fee)
                balance = max(0.0, balance * (1.0 + net_pnl))
                trades += 1
                if net_pnl > 0: wins += 1
                
                if signal == "BUY" and balance > 1.0:
                    position = "LONG"
                    entry_price = current_close
                    liq_price = entry_price * (1.0 - 1.0 / leverage + mmr)
                else:
                    position = "NONE"
                    
    return balance, trades, wins

print("🔍 Optimizing threshold...")
for th in [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02]:
    bal, tr, wn = test_threshold(th)
    wr = (wn / tr * 100) if tr > 0 else 0
    print(f"Threshold: {th:.4f} | Final Balance: ${bal:.2f} | Trades: {tr} | Win Rate: {wr:.2f}%")
