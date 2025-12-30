import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math

# 1. CNN (ANLIK TEPKİ):
CNN_FEATURES = [
    "Log_Ret",      # Fiyatın kendisi (Mecbur)
    "RSI",          # Aşırı Alım/Satım
    "BB_PctB",      # Volatilite patlaması
    "Vol_Spike",    # Hacim patlaması
]

# 2. LSTM (HİKAYE ANLATICISI):
LSTM_FEATURES = [
    "Log_Ret",      # <-- KRİTİK EKLEME: Fiyatın akışını görmeli
    "Dist_SMA_50",  # Trend yönü
    "Dist_EMA_200", # Ana trend
    "Vol_Ratio",    # Hacim desteği
]

# 3. TRANSFORMER (BÜYÜK RESİM):
TR_FEATURES = [
    "Log_Ret",      # <-- EKLEME
    "MACD_Norm",
    "BB_Width",
    "Dist_EMA_200",
    "Vol_Ratio",
    "ATR_Pct",
    "Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos"
]

# ==========================================
# 2. DATASET (AYNI KALDI)
# ==========================================
class MultiBranchCryptoDataset(Dataset):
    def __init__(self, dataframe, mean=None, std=None, cnn_window=16, lstm_window=48, tr_window=96):
        self.df = dataframe.copy()

        if mean is None or std is None:
            self.mean = self.df.mean()
            self.std = self.df.std()
            if 'Log_Ret' in self.mean: self.mean['Log_Ret'] = 0.0
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

# ==========================================
# 3. MODEL (AUXILIARY HEADS + BRANCH DROPOUT)
# ==========================================
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

class MultiBranchModel(nn.Module):
    def __init__(self, embed_dim=128, dropout=0.15):
        super().__init__()

        # --- ANA YOLLAR (BACKBONES) ---
        # CNN
        cnn_in = len(CNN_FEATURES)
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(cnn_in, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim), nn.LeakyReLU(0.1), nn.MaxPool1d(2),
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim*2), nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )
        self.cnn_out_dim = embed_dim * 2

        # LSTM
        lstm_in = len(LSTM_FEATURES)
        self.lstm = nn.LSTM(lstm_in, embed_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm_out_dim = embed_dim

        # Transformer
        tr_in = len(TR_FEATURES)
        self.tr_proj = nn.Linear(tr_in, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.tr_out_dim = embed_dim

        # --- YENİ: AUXILIARY HEADS (YARDIMCI KAFALAR) ---
        # Her kolun kendi bağımsız tahminini yapmasını sağlar.
        self.aux_cnn = nn.Linear(self.cnn_out_dim, 1)
        self.aux_lstm = nn.Linear(self.lstm_out_dim, 1)
        self.aux_tr = nn.Linear(self.tr_out_dim, 1)

        # --- FUSION HEAD (ANA KAFA) ---
        fusion_dim = self.cnn_out_dim + self.lstm_out_dim + self.tr_out_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

        self.current_epoch = 0

    def forward(self, x_cnn, x_lstm, x_tr):
        # 1. Öznitelik Çıkarımı (Feature Extraction)
        x_c = x_cnn.transpose(1, 2)
        out_cnn = self.cnn_branch(x_c)

        lstm_out_seq, _ = self.lstm(x_lstm)
        out_lstm = lstm_out_seq[:, -1, :]

        x_t = self.tr_proj(x_tr)
        x_t = self.pos_enc(x_t)
        out_tr = self.transformer(x_t)
        out_tr = out_tr.mean(dim=1)

        # 2. AUXILIARY PREDICTIONS (HAM VECTORS)
        # KRİTİK NOKTA: Dropout uygulamadan önce tahmin alıyoruz!
        # Böylece OGM-GE için temiz veri elde ediyoruz.
        pred_aux_cnn = self.aux_cnn(out_cnn)
        pred_aux_lstm = self.aux_lstm(out_lstm)
        pred_aux_tr = self.aux_tr(out_tr)

        # 3. BRANCH DROPOUT (FUSION İÇİN SABOTAJ)
        if self.training and self.current_epoch >= 3:
            if torch.rand(1).item() < 0.50:
                r = torch.rand(1).item()
                if r < 0.60:   out_cnn = torch.zeros_like(out_cnn) # CNN Sustur
                elif r < 0.80: out_lstm = torch.zeros_like(out_lstm)
                else:          out_tr = torch.zeros_like(out_tr)

        # 4. FUSION
        fused = torch.cat([out_cnn, out_lstm, out_tr], dim=1)
        pred_main = self.head(fused)

        # Hepsini döndür (Main + 3 Aux)
        return pred_main.squeeze(), pred_aux_cnn.squeeze(), pred_aux_lstm.squeeze(), pred_aux_tr.squeeze()