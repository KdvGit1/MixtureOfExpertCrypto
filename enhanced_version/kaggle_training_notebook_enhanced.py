"""
================================================================================
🚀 ENHANCED KAGGLE CRYPTO MODEL TRAINER - MoE & CONTRASTIVE PRE-TRAINING
================================================================================
Bu script Kaggle notebook'a yapıştırılarak kullanılır.
20 Popüler Coin için 15m ve 1h timeframe'lerinde 2-Aşamalı (Contrastive Pre-training
ve Supervised MoE Gated Fine-tuning) eğitim yapar.
================================================================================
"""

# ============================= 1. IMPORTS =====================================
import os
import gc
import json
import math
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Kaggle için ccxt kurulumu (ilk çalıştırmada)
try:
    import ccxt
except ImportError:
    os.system("pip install ccxt -q")
    import ccxt

# TA-Lib kurulumu
try:
    import talib
except ImportError:
    os.system("pip install TA-Lib -q")
    try:
        import talib
    except:
        print("⚠️ TA-Lib kurulamadı. Aşağıdaki komutu çalıştırın:")
        print("!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz")
        print("!tar -xzf ta-lib-0.4.0-src.tar.gz")
        print("!cd ta-lib/ && ./configure --prefix=/usr && make && make install")
        print("!pip install TA-Lib")
        raise ImportError("TA-Lib kurulumu gerekli")

warnings.filterwarnings("ignore")

# ============================= 2. CONFIG ======================================

COINS = [
    "BTC", "ETH", "BNB", "XRP", "SOL",
    "ADA", "DOGE", "TRX", "LTC", "DOT",
    "LINK", "UNI", "AVAX", "MATIC", "ATOM",
    "FIL", "APT", "ARB", "OP", "INJ"
]

TIMEFRAME_CONFIGS = {
    "15m": {
        "timeframe": "15m",
        "months_back": 60,  # 5 yıl
        "cnn_window": 16,
        "lstm_window": 96,
        "tr_window": 96,
        "mask_threshold": 0.15
    },
    "1h": {
        "timeframe": "1h",
        "months_back": 180,  # 15 yıl
        "cnn_window": 12,
        "lstm_window": 120,
        "tr_window": 120,
        "mask_threshold": 0.11
    }
}

BEST_PARAMS = {
    "15m": {
        "embed_dim": 96,
        "learning_rate": 0.00012011972393497386,
        "batch_size": 512,
        "dropout": 0.3107166113032219
    },
    "1h": {
        "embed_dim": 128,
        "learning_rate": 0.0001752754276249831,
        "batch_size": 1024,
        "dropout": 0.3216365190144383
    }
}

PRETRAIN_EPOCHS = 15     # Kontrastif Ön Eğitim Epoch Sayısı
FINAL_EPOCHS = 100        # Denetimli İnce Ayar Epoch Sayısı

CHECKPOINT_READ_PATH = "/kaggle/input/checkpoint/training_checkpoint_enhanced.json"
CHECKPOINT_WRITE_PATH = "/kaggle/working/training_checkpoint_enhanced.json"
OUTPUT_DIR = "/kaggle/working/trained_models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# ============================= 3. DATA FETCHING ===============================

_EXCHANGE_CACHE = {}

def get_exchange_instance(exchange_name="binance"):
    exchange_name = exchange_name.lower()
    if exchange_name in _EXCHANGE_CACHE:
        return _EXCHANGE_CACHE[exchange_name]
    
    print(f"🔌 {exchange_name.upper()} bağlantısı kuruluyor...")
    if exchange_name == "binance":
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'futures'}
        })
    else:
        raise ValueError(f"{exchange_name} desteklenmiyor")
    
    exchange.load_markets()
    _EXCHANGE_CACHE[exchange_name] = exchange
    return exchange


def get_crypto_history(symbol, timeframe, months_back, exchange_name="binance"):
    exchange = get_exchange_instance(exchange_name)
    now = datetime.now()
    start_date = now - timedelta(days=30 * months_back)
    since = int(start_date.timestamp() * 1000)
    
    print(f"🚀 Veri çekiliyor: {symbol} - {timeframe}")
    all_candles = []
    
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not candles:
                break
            
            all_candles += candles
            last_candle_time = candles[-1][0]
            since = last_candle_time + 1
            
            if len(all_candles) % 5000 == 0:
                print(f"📦 Çekilen: {len(all_candles)} mum...")
            
            if last_candle_time >= exchange.milliseconds():
                break
                
        except Exception as e:
            print(f"❌ Hata: {e}")
            time.sleep(1)
            continue
    
    print(f"✅ Toplam: {len(all_candles)} mum çekildi")
    
    df = pd.DataFrame(all_candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df.drop(columns=['Timestamp'], inplace=True)
    return df


def add_smart_indicators(df):
    df = df.copy()
    
    df['SMA_50_Val'] = talib.SMA(df['Close'], timeperiod=50)
    df['EMA_200_Val'] = talib.EMA(df['Close'], timeperiod=200)
    
    upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20)
    df['BB_Upper_Val'] = upper
    df['BB_Middle_Val'] = middle
    df['BB_Lower_Val'] = lower
    
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['ATR_Val'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    macd, macdsignal, macdhist = talib.MACD(df['Close'])
    df['MACD_Val'] = macd
    df['MACD_Signal_Val'] = macdsignal
    df['MACD_Hist_Val'] = macdhist
    
    df['Dist_SMA_50'] = (df['Close'] - df['SMA_50_Val']) / df['SMA_50_Val']
    df['Dist_EMA_200'] = (df['Close'] - df['EMA_200_Val']) / df['EMA_200_Val']
    df['BB_PctB'] = (df['Close'] - lower) / (upper - lower)
    df['BB_Width'] = (upper - lower) / middle
    df['MACD_Norm'] = df['MACD_Val'] / df['Close']
    df['ATR_Pct'] = df['ATR_Val'] / df['Close']
    
    df['Vol_SMA_20'] = talib.SMA(df['Volume'], timeperiod=20)
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA_20']
    df['Vol_Spike'] = (df['Vol_Ratio'] > 2.0).astype(int)
    
    df['Hour_Sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['Day_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    return df


def prepare_ai_dataframe(df):
    df_calculated = add_smart_indicators(df)
    df_calculated = df_calculated.replace([np.inf, -np.inf], np.nan)
    df_clean = df_calculated.dropna()
    
    ai_cols = [
        'Log_Ret', 'RSI', 'Dist_SMA_50', 'Dist_EMA_200',
        'BB_PctB', 'BB_Width', 'MACD_Norm', 'ATR_Pct',
        'Vol_Ratio', 'Vol_Spike',
        'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
    ]
    
    df_ai = df_clean[ai_cols].copy()
    df_ai['RSI'] = df_ai['RSI'] / 100.0
    
    print(f"📊 AI verisi hazır: {len(df_ai)} satır")
    return df_ai


# ============================= 4. AUGMENTATIONS & CONTRASTIVE MODULES =========

def apply_jitter(x, std=0.02):
    if std <= 0 or not x.is_floating_point():
        return x
    return x + torch.randn_like(x) * std

def apply_scale(x, std=0.05):
    if std <= 0:
        return x
    factor = 1.0 + torch.randn(x.size(0), 1, 1, device=x.device) * std
    return x * factor

def apply_mask(x, mask_ratio=0.10):
    if mask_ratio <= 0:
        return x
    batch_size, seq_len, _ = x.size()
    mask = torch.rand(batch_size, seq_len, 1, device=x.device) > mask_ratio
    return x * mask.float()

def augment_sequence(x, jitter_std=0.02, scale_std=0.05, mask_ratio=0.10):
    x = apply_jitter(x, jitter_std)
    x = apply_scale(x, scale_std)
    x = apply_mask(x, mask_ratio)
    return x


class TimeSeriesContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        similarity_matrix.masked_fill_(mask, -1e9)
        
        return F.cross_entropy(similarity_matrix, labels)


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)


# ============================= 5. DATASET & ENHANCED MODEL ====================

CNN_FEATURES = ["Log_Ret", "RSI", "BB_PctB", "Vol_Spike"]
LSTM_FEATURES = ["Log_Ret", "Dist_SMA_50", "Dist_EMA_200", "Vol_Ratio"]
TR_FEATURES = ["Log_Ret", "MACD_Norm", "BB_Width", "Dist_EMA_200", "Vol_Ratio", "ATR_Pct", 
               "Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos"]


class MultiBranchCryptoDataset(Dataset):
    def __init__(self, dataframe, mean=None, std=None, cnn_window=16, lstm_window=48, tr_window=96):
        self.df = dataframe.copy()
        
        if mean is None or std is None:
            self.mean = self.df.mean()
            self.std = self.df.std()
            if 'Log_Ret' in self.mean:
                self.mean['Log_Ret'] = 0.0
            self.std[self.std == 0] = 1.0
        else:
            self.mean = mean
            self.std = std
        
        self.df = (self.df - self.mean) / self.std
        
        self.cnn_indices = [self.df.columns.get_loc(c) for c in CNN_FEATURES if c in self.df.columns]
        self.lstm_indices = [self.df.columns.get_loc(c) for c in LSTM_FEATURES if c in self.df.columns]
        self.tr_indices = [self.df.columns.get_loc(c) for c in TR_FEATURES if c in self.df.columns]
        self.target_idx = self.df.columns.get_loc("Log_Ret")
        
        self.data_matrix = torch.tensor(self.df.values, dtype=torch.float32)
        
        self.cnn_window = cnn_window
        self.lstm_window = lstm_window
        self.tr_window = tr_window
        self.max_window = max(cnn_window, lstm_window, tr_window)
    
    def __len__(self):
        return len(self.df) - self.max_window
    
    def __getitem__(self, idx):
        t = idx + self.max_window
        x_cnn = self.data_matrix[t - self.cnn_window : t, self.cnn_indices]
        x_lstm = self.data_matrix[t - self.lstm_window : t, self.lstm_indices]
        x_tr = self.data_matrix[t - self.tr_window : t, self.tr_indices]
        y = self.data_matrix[t, self.target_idx]
        return {"x_cnn": x_cnn, "x_lstm": x_lstm, "x_tr": x_tr, "y": y}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class MoERouter(nn.Module):
    def __init__(self, context_dim=6, num_experts=3, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
    def forward(self, context):
        return torch.softmax(self.net(context), dim=1)


class MultiBranchModel(nn.Module):
    def __init__(self, embed_dim=128, dropout=0.15):
        super().__init__()
        
        # CNN Branch
        cnn_in = len(CNN_FEATURES)
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(cnn_in, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim), nn.LeakyReLU(0.1), nn.MaxPool1d(2),
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim*2), nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )
        self.cnn_out_dim = embed_dim * 2
        
        # LSTM Branch
        lstm_in = len(LSTM_FEATURES)
        self.lstm = nn.LSTM(lstm_in, embed_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm_out_dim = embed_dim
        
        # Transformer Branch
        tr_in = len(TR_FEATURES)
        self.tr_proj = nn.Linear(tr_in, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, 
                                                    dim_feedforward=embed_dim*4, 
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.tr_out_dim = embed_dim
        
        # Auxiliary Heads
        self.aux_cnn = nn.Linear(self.cnn_out_dim, 1)
        self.aux_lstm = nn.Linear(self.lstm_out_dim, 1)
        self.aux_tr = nn.Linear(self.tr_out_dim, 1)
        
        # MoE Router
        self.context_features = ["Log_Ret", "MACD_Norm", "BB_Width", "Dist_EMA_200", "Vol_Ratio", "ATR_Pct"]
        self.context_indices = [TR_FEATURES.index(f) for f in self.context_features if f in TR_FEATURES]
        self.router = MoERouter(context_dim=len(self.context_indices), num_experts=3, hidden_dim=32)
        
        # Initialize Router to zeros for backward compatibility
        nn.init.zeros_(self.router.net[-1].weight)
        nn.init.zeros_(self.router.net[-1].bias)
        
        # Fusion Head
        fusion_dim = self.cnn_out_dim + self.lstm_out_dim + self.tr_out_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self.current_epoch = 0
    
    def forward(self, x_cnn, x_lstm, x_tr):
        # CNN
        x_c = x_cnn.transpose(1, 2)
        out_cnn = self.cnn_branch(x_c)
        
        # LSTM
        lstm_out_seq, _ = self.lstm(x_lstm)
        out_lstm = lstm_out_seq[:, -1, :]
        
        # Transformer
        x_t = self.tr_proj(x_tr)
        x_t = self.pos_enc(x_t)
        out_tr = self.transformer(x_t)
        out_tr = out_tr.mean(dim=1)
        
        # Auxiliary predictions
        pred_aux_cnn = self.aux_cnn(out_cnn)
        pred_aux_lstm = self.aux_lstm(out_lstm)
        pred_aux_tr = self.aux_tr(out_tr)
        
        # MoE Router Context
        context = x_tr[:, -1, self.context_indices]
        g_weights = self.router(context)
        
        # Gated scale
        out_cnn_scaled = out_cnn * g_weights[:, 0:1]
        out_lstm_scaled = out_lstm * g_weights[:, 1:2]
        out_tr_scaled = out_tr * g_weights[:, 2:3]
        
        # Branch Dropout
        if self.training and self.current_epoch >= 3:
            if torch.rand(1).item() < 0.50:
                r = torch.rand(1).item()
                if r < 0.60:
                    out_cnn_scaled = torch.zeros_like(out_cnn_scaled)
                elif r < 0.80:
                    out_lstm_scaled = torch.zeros_like(out_lstm_scaled)
                else:
                    out_tr_scaled = torch.zeros_like(out_tr_scaled)
        
        # Fusion
        fused = torch.cat([out_cnn_scaled, out_lstm_scaled, out_tr_scaled], dim=1)
        pred_main = self.head(fused)
        
        return pred_main.squeeze(), pred_aux_cnn.squeeze(), pred_aux_lstm.squeeze(), pred_aux_tr.squeeze(), g_weights


def calc_ogm_loss(loss_main, loss_cnn, loss_lstm, loss_tr):
    ratio_cnn = loss_main.detach() / (loss_cnn.detach() + 1e-6)
    ratio_lstm = loss_main.detach() / (loss_lstm.detach() + 1e-6)
    ratio_tr = loss_main.detach() / (loss_tr.detach() + 1e-6)
    
    coeff_cnn = math.exp(1.0 - ratio_cnn)
    coeff_lstm = math.exp(1.0 - ratio_lstm)
    coeff_tr = math.exp(1.0 - ratio_tr)
    
    weighted_loss_cnn = loss_cnn * coeff_cnn
    weighted_loss_lstm = loss_lstm * coeff_lstm
    weighted_loss_tr = loss_tr * coeff_tr
    
    total_loss = loss_main + 0.1 * (weighted_loss_cnn + weighted_loss_lstm + weighted_loss_tr)
    
    stats = {'k_cnn': coeff_cnn, 'k_lstm': coeff_lstm, 'k_tr': coeff_tr}
    return total_loss, stats


# ============================= 6. CHECKPOINT SYSTEM ===========================

def save_checkpoint(coin_idx, tf_idx, completed_models, current_state=None):
    checkpoint = {
        "coin_idx": coin_idx,
        "tf_idx": tf_idx,
        "completed_models": completed_models,
        "current_state": current_state,
        "timestamp": datetime.now().isoformat()
    }
    with open(CHECKPOINT_WRITE_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"💾 Checkpoint kaydedildi: Coin {coin_idx}, TF {tf_idx}")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_WRITE_PATH):
        with open(CHECKPOINT_WRITE_PATH, "r") as f:
            checkpoint = json.load(f)
        print(f"📂 Checkpoint yüklendi (working): {checkpoint['timestamp']}")
        return checkpoint
    
    if os.path.exists(CHECKPOINT_READ_PATH):
        with open(CHECKPOINT_READ_PATH, "r") as f:
            checkpoint = json.load(f)
        print(f"📂 Checkpoint yüklendi (input): {checkpoint['timestamp']}")
        return checkpoint
    
    return None


def save_model_files(coin, timeframe, model, best_params, train_ds, history):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = f"{coin}_{timeframe}"
    
    model_path = os.path.join(OUTPUT_DIR, f"{base_name}_model.pth")
    torch.save(model.state_dict(), model_path)
    
    params_path = os.path.join(OUTPUT_DIR, f"{base_name}_params.json")
    params_with_info = {
        "coin": coin,
        "timeframe": timeframe,
        **best_params
    }
    with open(params_path, "w") as f:
        json.dump(params_with_info, f, indent=2)
    
    stats_path = os.path.join(OUTPUT_DIR, f"{base_name}_stats.json")
    stats = {
        "mean": train_ds.mean.to_dict(),
        "std": train_ds.std.to_dict()
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    history_path = os.path.join(OUTPUT_DIR, f"{base_name}_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    
    print(f"✅ Model kaydedildi: {base_name}")
    return model_path


# ============================= 7. TWO-STAGE MAIN TRAINING LOOP ===============

def train_single_model(coin, tf_config, tf_name):
    print(f"\n{'='*60}")
    print(f"🎯 EĞİTİM (MoE & CONTRASTIVE): {coin} - {tf_name}")
    print(f"{'='*60}")
    
    symbol = f"{coin}/USDT"
    df_raw = get_crypto_history(symbol, tf_config["timeframe"], tf_config["months_back"])
    
    if len(df_raw) < 1000:
        print(f"⚠️ {coin} için yeterli veri yok, atlanıyor...")
        return None
    
    df_ai = prepare_ai_dataframe(df_raw)
    
    if "date" in df_ai.columns.str.lower().tolist():
        date_cols = [c for c in df_ai.columns if "date" in c.lower()]
        df_ai.drop(columns=date_cols, inplace=True)
    df_ai = df_ai.select_dtypes(include=[np.number]).fillna(0)
    
    train_end = int(0.70 * len(df_ai))
    val_end = int(0.85 * len(df_ai))
    
    cnn_w = tf_config["cnn_window"]
    lstm_w = tf_config["lstm_window"]
    tr_w = tf_config["tr_window"]
    
    train_ds = MultiBranchCryptoDataset(df_ai.iloc[:train_end], 
                                         cnn_window=cnn_w, lstm_window=lstm_w, tr_window=tr_w)
    val_ds = MultiBranchCryptoDataset(df_ai.iloc[train_end:val_end], 
                                       mean=train_ds.mean, std=train_ds.std,
                                       cnn_window=cnn_w, lstm_window=lstm_w, tr_window=tr_w)
    
    print(f"📊 Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    best_params = BEST_PARAMS[tf_name]
    print(f"✨ Kullanılan parametreler: {best_params}")
    
    train_loader = DataLoader(train_ds, batch_size=best_params["batch_size"], 
                               shuffle=True, drop_last=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=best_params["batch_size"], 
                             shuffle=False, drop_last=True, pin_memory=True, num_workers=0)
    
    model = MultiBranchModel(embed_dim=best_params["embed_dim"], 
                              dropout=best_params["dropout"]).to(DEVICE)
    
    # ----------------------------------------------------
    # STAGE 1: CONTRASTIVE PRE-TRAINING
    # ----------------------------------------------------
    print(f"\n🏆 AŞAMA 1: KONTRASTİF ÖN EĞİTİM ({PRETRAIN_EPOCHS} epoch)...")
    proj_cnn = ContrastiveProjectionHead(in_dim=model.cnn_out_dim, out_dim=128).to(DEVICE)
    proj_lstm = ContrastiveProjectionHead(in_dim=model.lstm_out_dim, out_dim=128).to(DEVICE)
    proj_tr = ContrastiveProjectionHead(in_dim=model.tr_out_dim, out_dim=128).to(DEVICE)
    
    contrastive_criterion = TimeSeriesContrastiveLoss(temperature=0.07)
    
    pretrain_params = (
        list(model.cnn_branch.parameters()) + 
        list(model.lstm.parameters()) + 
        list(model.tr_proj.parameters()) + 
        list(model.transformer.parameters()) + 
        list(proj_cnn.parameters()) + 
        list(proj_lstm.parameters()) + 
        list(proj_tr.parameters())
    )
    pretrain_optimizer = optim.AdamW(pretrain_params, lr=best_params["learning_rate"])
    is_cuda = (DEVICE.type == 'cuda')
    scaler = torch.amp.GradScaler(device_type="cuda", enabled=is_cuda)
    
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        proj_cnn.train()
        proj_lstm.train()
        proj_tr.train()
        
        loss_acc = 0.0
        batches = 0
        
        for batch in train_loader:
            x_cnn = batch["x_cnn"].to(DEVICE)
            x_lstm = batch["x_lstm"].to(DEVICE)
            x_tr = batch["x_tr"].to(DEVICE)
            
            x_cnn_1 = augment_sequence(x_cnn)
            x_lstm_1 = augment_sequence(x_lstm)
            x_tr_1 = augment_sequence(x_tr)
            
            x_cnn_2 = augment_sequence(x_cnn)
            x_lstm_2 = augment_sequence(x_lstm)
            x_tr_2 = augment_sequence(x_tr)
            
            pretrain_optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=is_cuda):
                # 1. Görünüm omurga çıktıları
                out_cnn_1 = model.cnn_branch(x_cnn_1.transpose(1, 2))
                lstm_out_seq_1, _ = model.lstm(x_lstm_1)
                out_lstm_1 = lstm_out_seq_1[:, -1, :]
                out_tr_1 = model.transformer(model.pos_enc(model.tr_proj(x_tr_1))).mean(dim=1)
                
                # 2. Görünüm omurga çıktıları
                out_cnn_2 = model.cnn_branch(x_cnn_2.transpose(1, 2))
                lstm_out_seq_2, _ = model.lstm(x_lstm_2)
                out_lstm_2 = lstm_out_seq_2[:, -1, :]
                out_tr_2 = model.transformer(model.pos_enc(model.tr_proj(x_tr_2))).mean(dim=1)
                
                # Projeksiyon
                z_cnn_1 = proj_cnn(out_cnn_1)
                z_cnn_2 = proj_cnn(out_cnn_2)
                z_lstm_1 = proj_lstm(out_lstm_1)
                z_lstm_2 = proj_lstm(out_lstm_2)
                z_tr_1 = proj_tr(out_tr_1)
                z_tr_2 = proj_tr(out_tr_2)
                
                loss_c = contrastive_criterion(z_cnn_1, z_cnn_2)
                loss_l = contrastive_criterion(z_lstm_1, z_lstm_2)
                loss_t = contrastive_criterion(z_tr_1, z_tr_2)
                
                total_loss = loss_c + loss_l + loss_t
                
            scaler.scale(total_loss).backward()
            scaler.step(pretrain_optimizer)
            scaler.update()
            
            loss_acc += total_loss.item()
            batches += 1
            
        print(f"  Pre-train Ep {epoch+1}/{PRETRAIN_EPOCHS} | InfoNCE Loss: {loss_acc/batches:.5f}")
        
    # Bellek temizliği
    del proj_cnn, proj_lstm, proj_tr, pretrain_optimizer, pretrain_params
    gc.collect()
    torch.cuda.empty_cache()
    
    # ----------------------------------------------------
    # STAGE 2: SUPERVISED MoE FINE-TUNING
    # ----------------------------------------------------
    print(f"\n🏆 AŞAMA 2: DENETİMLİ MoE İNCE AYAR ({FINAL_EPOCHS} epoch)...")
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=best_params["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=7, factor=0.7)
    
    best_val = float("inf")
    best_model_state = None
    history = []
    mask_threshold = tf_config["mask_threshold"]
    
    for epoch in range(FINAL_EPOCHS):
        model.train()
        model.current_epoch = epoch
        
        train_loss_acc = 0
        ogm_stats_acc = {'k_cnn': 0, 'k_lstm': 0, 'k_tr': 0}
        moe_weights_acc = torch.zeros(3, device=DEVICE)
        batch_count = 0
        
        for batch in train_loader:
            x_cnn = batch["x_cnn"].to(DEVICE)
            x_lstm = batch["x_lstm"].to(DEVICE)
            x_tr = batch["x_tr"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=is_cuda):
                p_main, p_cnn, p_lstm, p_tr, g_weights = model(x_cnn, x_lstm, x_tr)
                y_s = y * 100.0
                loss, stats = calc_ogm_loss(
                    criterion(p_main, y_s), 
                    criterion(p_cnn, y_s), 
                    criterion(p_lstm, y_s), 
                    criterion(p_tr, y_s)
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_acc += loss.item()
            for k, v in stats.items():
                ogm_stats_acc[k] += v
            moe_weights_acc += g_weights.mean(dim=0)
            batch_count += 1
        
        avg_train_loss = train_loss_acc / batch_count
        avg_ogm = {k: v / batch_count for k, v in ogm_stats_acc.items()}
        avg_moe = moe_weights_acc / batch_count
        
        # Validation
        model.eval()
        val_loss_acc = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_cnn = batch["x_cnn"].to(DEVICE)
                x_lstm = batch["x_lstm"].to(DEVICE)
                x_tr = batch["x_tr"].to(DEVICE)
                y = batch["y"].to(DEVICE)
                
                with torch.amp.autocast(device_type="cuda", enabled=is_cuda):
                    p_main, _, _, _, _ = model(x_cnn, x_lstm, x_tr)
                    val_loss_acc += criterion(p_main, y * 100.0).item()
                
                mask = torch.abs(y) > mask_threshold
                correct += ((torch.sign(p_main) == torch.sign(y)) & mask).sum().item()
                total += mask.sum().item()
        
        avg_val_loss = val_loss_acc / len(val_loader)
        acc = (correct / max(total, 1)) * 100
        scheduler.step(avg_val_loss)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': acc,
            'avg_k_cnn': avg_ogm['k_cnn'],
            'avg_k_lstm': avg_ogm['k_lstm'],
            'avg_k_tr': avg_ogm['k_tr'],
            'avg_g_cnn': avg_moe[0].item(),
            'avg_g_lstm': avg_moe[1].item(),
            'avg_g_tr': avg_moe[2].item(),
        })
        
        if avg_val_loss < best_val and epoch > 3:
            best_val = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"Ep {epoch+1}: Loss {avg_val_loss:.5f} | Acc %{acc:.2f} | MoE-Weights: [C: {avg_moe[0]:.2f}, L: {avg_moe[1]:.2f}, T: {avg_moe[2]:.2f}] 💾")
        elif epoch % 10 == 0:
            print(f"Ep {epoch+1}: Loss {avg_val_loss:.5f} | Acc %{acc:.2f} | MoE-Weights: [C: {avg_moe[0]:.2f}, L: {avg_moe[1]:.2f}, T: {avg_moe[2]:.2f}]")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model_path = save_model_files(coin, tf_name, model, best_params, train_ds, history)
    
    del model, train_ds, val_ds, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return model_path


def main():
    print("\n" + "="*60)
    print("🚀 KAGGLE CRYPTO MODEL TRAINER (ENHANCED MoE & CONTRASTIVE)")
    print(f"📊 {len(COINS)} coin × 2 timeframe = {len(COINS)*2} model")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint = load_checkpoint()
    start_coin_idx = 0
    start_tf_idx = 0
    completed_models = []
    
    if checkpoint:
        start_coin_idx = checkpoint.get("coin_idx", 0)
        start_tf_idx = checkpoint.get("tf_idx", 0)
        completed_models = checkpoint.get("completed_models", [])
        print(f"📌 Kaldığı yerden devam: Coin {start_coin_idx}, TF {start_tf_idx}")
    
    timeframes = list(TIMEFRAME_CONFIGS.keys())
    total_models = len(COINS) * len(timeframes)
    current_model = len(completed_models)
    
    for coin_idx in range(start_coin_idx, len(COINS)):
        coin = COINS[coin_idx]
        tf_start = start_tf_idx if coin_idx == start_coin_idx else 0
        
        for tf_idx in range(tf_start, len(timeframes)):
            tf_name = timeframes[tf_idx]
            tf_config = TIMEFRAME_CONFIGS[tf_name]
            
            current_model += 1
            print(f"\n📈 İlerleme: {current_model}/{total_models}")
            
            try:
                model_path = train_single_model(coin, tf_config, tf_name)
                if model_path:
                    completed_models.append({
                        "coin": coin,
                        "timeframe": tf_name,
                        "model_path": model_path,
                        "timestamp": datetime.now().isoformat()
                    })
                
            except Exception as e:
                print(f"❌ HATA ({coin}-{tf_name}): {e}")
                save_checkpoint(coin_idx, tf_idx, completed_models)
                raise e
            
            next_tf_idx = tf_idx + 1
            next_coin_idx = coin_idx
            
            if next_tf_idx >= len(timeframes):
                next_tf_idx = 0
                next_coin_idx = coin_idx + 1
            
            save_checkpoint(next_coin_idx, next_tf_idx, completed_models)
    
    print("\n" + "="*60)
    print("🎉 TÜM MODELLER EĞİTİLDİ!")
    print(f"📁 Modeller: {OUTPUT_DIR}/")
    print("="*60)
    
    summary = pd.DataFrame(completed_models)
    summary.to_csv(os.path.join(OUTPUT_DIR, "training_summary_enhanced.csv"), index=False)
    print(summary)
    
    if os.path.exists(CHECKPOINT_WRITE_PATH):
        os.remove(CHECKPOINT_WRITE_PATH)
        print("🧹 Checkpoint temizlendi")


if __name__ == "__main__":
    main()
