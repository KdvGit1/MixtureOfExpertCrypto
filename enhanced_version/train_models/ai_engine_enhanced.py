import torch
import torch.nn as nn
import torch.nn.functional as F
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
    "Log_Ret",      # Fiyatın akışı
    "Dist_SMA_50",  # Trend yönü
    "Dist_EMA_200", # Ana trend
    "Vol_Ratio",    # Hacim desteği
]

# 3. TRANSFORMER (BÜYÜK RESİM):
TR_FEATURES = [
    "Log_Ret",      
    "MACD_Norm",
    "BB_Width",
    "Dist_EMA_200",
    "Vol_Ratio",
    "ATR_Pct",
    "Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos"
]

# =========================================================
# TIME SERIES AUGMENTATIONS (KONTRASTİF ÖĞRENME İÇİN)
# =========================================================
def apply_jitter(x, std=0.02):
    """Time series girdilerine küçük Gauss gürültüsü ekler."""
    if std <= 0 or not x.is_floating_point():
        return x
    noise = torch.randn_like(x) * std
    return x + noise

def apply_scale(x, std=0.05):
    """Zaman serisi penceresini rastgele bir çarpanla ölçeklendirir."""
    if std <= 0:
        return x
    # Her batch örneği için farklı bir katsayı üretilir
    factor = 1.0 + torch.randn(x.size(0), 1, 1, device=x.device) * std
    return x * factor

def apply_mask(x, mask_ratio=0.10):
    """Zaman serisinde bazı zaman adımlarını rastgele maskeler (sıfırlar)."""
    if mask_ratio <= 0:
        return x
    batch_size, seq_len, _ = x.size()
    # Zaman adımlarını maskele
    mask = torch.rand(batch_size, seq_len, 1, device=x.device) > mask_ratio
    return x * mask.float()

def augment_sequence(x, jitter_std=0.02, scale_std=0.05, mask_ratio=0.10):
    """Zaman serisi penceresine tüm artırımları uygular."""
    x = apply_jitter(x, jitter_std)
    x = apply_scale(x, scale_std)
    x = apply_mask(x, mask_ratio)
    return x

# =========================================================
# CONTRASTIVE LOSS (InfoNCE)
# =========================================================
class TimeSeriesContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        
        # Özellik vektörleri normalize edilir (cosine similarity için)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # İki görünüm birleştirilir
        representations = torch.cat([z1, z2], dim=0) # [2*B, D]
        
        # Tüm örnekler arası benzerlik matrisi hesaplanır
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature # [2*B, 2*B]
        
        # Pozitif çiftlerin indeksleri belirlenir
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0) # [2*B]
        
        # Kendisiyle olan benzerlik (köşegen) maskelenir
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        similarity_matrix.masked_fill_(mask, -1e9)
        
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

class ContrastiveProjectionHead(nn.Module):
    """Kontrastif öğrenme için özellikleri ortak projeksiyon uzayına taşır."""
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

# =========================================================
# MOE ROUTER
# =========================================================
class MoERouter(nn.Module):
    """Anlık volatilite, hacim ve trende bakarak uzman ağırlıklarını belirler."""
    def __init__(self, context_dim=6, num_experts=3, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
    def forward(self, context):
        # Softmax ile ağırlıklar toplamı 1.0 olacak şekilde atanır
        return torch.softmax(self.net(context), dim=1)

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
# 3. ENHANCED MODEL (MoE ROUTED + BACKWARD COMPATIBLE)
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

        # --- AUXILIARY HEADS (YARDIMCI KAFALAR) ---
        self.aux_cnn = nn.Linear(self.cnn_out_dim, 1)
        self.aux_lstm = nn.Linear(self.lstm_out_dim, 1)
        self.aux_tr = nn.Linear(self.tr_out_dim, 1)

        # --- MOE ROUTER (YENİ) ---
        # Kapı ağı girdileri: Log_Ret, MACD_Norm, BB_Width, Dist_EMA_200, Vol_Ratio, ATR_Pct
        self.context_features = ["Log_Ret", "MACD_Norm", "BB_Width", "Dist_EMA_200", "Vol_Ratio", "ATR_Pct"]
        self.context_indices = [TR_FEATURES.index(f) for f in self.context_features if f in TR_FEATURES]
        self.router = MoERouter(context_dim=len(self.context_indices), num_experts=3, hidden_dim=32)
        
        # Backward compatibility için final katmanı sıfırla başlat (başlangıçta tüm dallara eşit %33.3 ağırlık verir)
        nn.init.zeros_(self.router.net[-1].weight)
        nn.init.zeros_(self.router.net[-1].bias)

        # --- FUSION HEAD (ANA KAFA) ---
        fusion_dim = self.cnn_out_dim + self.lstm_out_dim + self.tr_out_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

        self.current_epoch = 0

    def forward(self, x_cnn, x_lstm, x_tr, return_g_weights=True):
        # 1. Öznitelik Çıkarımı (Feature Extraction)
        x_c = x_cnn.transpose(1, 2)
        out_cnn = self.cnn_branch(x_c)

        lstm_out_seq, _ = self.lstm(x_lstm)
        out_lstm = lstm_out_seq[:, -1, :]

        x_t = self.tr_proj(x_tr)
        x_t = self.pos_enc(x_t)
        out_tr = self.transformer(x_t)
        out_tr = out_tr.mean(dim=1)

        # 2. AUXILIARY PREDICTIONS
        pred_aux_cnn = self.aux_cnn(out_cnn)
        pred_aux_lstm = self.aux_lstm(out_lstm)
        pred_aux_tr = self.aux_tr(out_tr)

        # 3. MOE ROUTER (Dinamik Ağırlıklar)
        # Gating Context: Son adımdaki volatilite, hacim ve trend özellikleri
        context = x_tr[:, -1, self.context_indices] # [batch, len(context_indices)]
        g_weights = self.router(context) # [batch, 3]

        # 4. GATED FUSION (Dalların Dinamik Ölçeklenmesi)
        out_cnn_scaled = out_cnn * g_weights[:, 0:1]
        out_lstm_scaled = out_lstm * g_weights[:, 1:2]
        out_tr_scaled = out_tr * g_weights[:, 2:3]

        # 5. BRANCH DROPOUT (Sabotaj)
        if self.training and self.current_epoch >= 3:
            if torch.rand(1).item() < 0.50:
                r = torch.rand(1).item()
                if r < 0.60:   out_cnn_scaled = torch.zeros_like(out_cnn_scaled)
                elif r < 0.80: out_lstm_scaled = torch.zeros_like(out_lstm_scaled)
                else:          out_tr_scaled = torch.zeros_like(out_tr_scaled)

        # 6. FUSION HEAD
        fused = torch.cat([out_cnn_scaled, out_lstm_scaled, out_tr_scaled], dim=1)
        pred_main = self.head(fused)

        if return_g_weights:
            return pred_main.squeeze(), pred_aux_cnn.squeeze(), pred_aux_lstm.squeeze(), pred_aux_tr.squeeze(), g_weights
        else:
            return pred_main.squeeze(), pred_aux_cnn.squeeze(), pred_aux_lstm.squeeze(), pred_aux_tr.squeeze()
