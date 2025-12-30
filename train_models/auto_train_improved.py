import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import gc
import json
import math
import matplotlib.pyplot as plt # Grafik için eklendi

# Senin kütüphanen (Aynen korundu)
from ai_engine_improved import MultiBranchCryptoDataset, MultiBranchModel

# =========================================================
# AYARLAR (SENİN BELİRLEDİĞİN DEĞERLER KORUNDU)
# =========================================================
TARGET_COIN = "BTC"
TIMEFRAME = "1h"
MONTH_PERIOD = 180

N_TRIALS = 25
FINAL_EPOCHS = 100

# Senin istediğin Window ayarları:
CNN_WINDOW = 12
LSTM_WINDOW = 120
TR_WINDOW = 120

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# OGM-GE MANTIĞI (Raporlama Özelliği Eklendi)
# =========================================================
def calc_ogm_loss(loss_main, loss_cnn, loss_lstm, loss_tr):
    """
    OGM-GE: Hangi kol başarısızsa ona 'Boost' ver.
    Dönüş: (Total Loss, Katsayılar Sözlüğü)
    """
    # Detach: Gradient takibini kes (sadece oran hesabı için)
    ratio_cnn = loss_main.detach() / (loss_cnn.detach() + 1e-6)
    ratio_lstm = loss_main.detach() / (loss_lstm.detach() + 1e-6)
    ratio_tr = loss_main.detach() / (loss_tr.detach() + 1e-6)

    # Katsayılar (Boost Factors)
    coeff_cnn = math.exp(1.0 - ratio_cnn)
    coeff_lstm = math.exp(1.0 - ratio_lstm)
    coeff_tr = math.exp(1.0 - ratio_tr)

    # Weighted Aux Losses
    weighted_loss_cnn = loss_cnn * coeff_cnn
    weighted_loss_lstm = loss_lstm * coeff_lstm
    weighted_loss_tr = loss_tr * coeff_tr

    # Ana formül
    total_loss = loss_main + 0.1 * (weighted_loss_cnn + weighted_loss_lstm + weighted_loss_tr)

    # İstatistikleri döndür (Grafik çizmek için lazım)
    stats = {
        'k_cnn': coeff_cnn,
        'k_lstm': coeff_lstm,
        'k_tr': coeff_tr
    }

    return total_loss, stats

# =========================================================
# RAPORLAMA VE GRAFİK FONKSİYONLARI (YENİ EKLENDİ)
# =========================================================
def generate_training_report(history):
    """Eğitim geçmişinden grafikler ve özet rapor oluşturur."""
    df_hist = pd.DataFrame(history)
    df_hist.to_csv("training_history.csv", index=False)
    print("\n💾 Detaylı veriler 'training_history.csv' dosyasına kaydedildi.")

    # Grafik Ayarları
    plt.figure(figsize=(18, 5))

    # 1. Grafik: Loss
    plt.subplot(1, 3, 1)
    plt.plot(df_hist['epoch'], df_hist['train_loss'], label='Train Loss', color='blue')
    plt.plot(df_hist['epoch'], df_hist['val_loss'], label='Val Loss', color='orange', linestyle='--')
    plt.title("Eğitim ve Doğrulama Kaybı")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Grafik: Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(df_hist['epoch'], df_hist['val_acc'], label='Val Accuracy', color='green')
    plt.title("Dürüst Başarı Oranı (Masked Accuracy)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.axhline(y=50, color='r', linestyle=':', label='Rastgele (%50)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Grafik: OGM-GE Aktivitesi
    plt.subplot(1, 3, 3)
    plt.plot(df_hist['epoch'], df_hist['avg_k_cnn'], label='CNN Boost', color='cyan')
    plt.plot(df_hist['epoch'], df_hist['avg_k_lstm'], label='LSTM Boost', color='magenta')
    plt.plot(df_hist['epoch'], df_hist['avg_k_tr'], label='Transformer Boost', color='yellow')
    plt.title("OGM-GE Modülasyonu (Zorlanma Düzeyi)")
    plt.xlabel("Epoch")
    plt.ylabel("Boost Katsayıları (k)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("TRAINING_REPORT_GRAPHICS.png")
    print("📊 Grafikler 'TRAINING_REPORT_GRAPHICS.png' olarak kaydedildi.")

# =========================================================
# OPTUNA OBJECTIVE (GÜNCELLENDİ)
# =========================================================
def objective(trial):
    embed_dim = trial.suggest_categorical("embed_dim", [64, 96, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024])
    dropout = trial.suggest_float("dropout", 0.15, 0.35)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)

    model = MultiBranchModel(embed_dim=embed_dim, dropout=dropout).to(DEVICE)
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.7)
    scaler = torch.amp.GradScaler("cuda")

    best_val = float("inf")

    try:
        for epoch in range(20): # Optuna araması için kısa tur
            model.train()
            model.current_epoch = epoch

            for batch in train_loader:
                x_cnn = batch["x_cnn"].to(DEVICE); x_lstm = batch["x_lstm"].to(DEVICE); x_tr = batch["x_tr"].to(DEVICE); y = batch["y"].to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    p_main, p_cnn, p_lstm, p_tr = model(x_cnn, x_lstm, x_tr)
                    y_s = y * 100.0
                    # Stats kısmını burada kullanmıyoruz ama fonksiyon değiştiği için almamız lazım
                    loss, _ = calc_ogm_loss(criterion(p_main, y_s), criterion(p_cnn, y_s), criterion(p_lstm, y_s), criterion(p_tr, y_s))

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()

            # Validation
            model.eval()
            val_loss_acc = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_cnn = batch["x_cnn"].to(DEVICE); x_lstm = batch["x_lstm"].to(DEVICE); x_tr = batch["x_tr"].to(DEVICE); y = batch["y"].to(DEVICE)
                    with torch.amp.autocast("cuda"):
                        p_main, _, _, _ = model(x_cnn, x_lstm, x_tr)
                        val_loss_acc += criterion(p_main, y * 100.0).item()

            avg_val_loss = val_loss_acc / len(val_loader)
            scheduler.step(avg_val_loss)

            trial.report(avg_val_loss, epoch)
            if trial.should_prune(): raise optuna.TrialPruned()

            if avg_val_loss < best_val: best_val = avg_val_loss

        return best_val

    except RuntimeError as e:
        if "out of memory" in str(e): return float("inf")
        raise e
    finally:
        del model; gc.collect(); torch.cuda.empty_cache()

# =========================================================
# FINAL TRAIN (VERİ TOPLAYAN SÜRÜM)
# =========================================================
def train_best_model(best_params):
    print("\n" + "=" * 60); print("🏆 FINAL MODEL TRAINING (VERİ TOPLAMA MODU)"); print(best_params); print("=" * 60)

    train_loader = DataLoader(train_ds, batch_size=best_params["batch_size"], shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=best_params["batch_size"], shuffle=False, drop_last=True, pin_memory=True)

    model = MultiBranchModel(embed_dim=best_params["embed_dim"], dropout=best_params["dropout"]).to(DEVICE)
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=best_params["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=7, factor=0.7)
    scaler = torch.amp.GradScaler("cuda")

    best_val = float("inf")
    history = [] # Tüm verileri burada biriktireceğiz

    for epoch in range(FINAL_EPOCHS):
        model.train()
        model.current_epoch = epoch

        # İstatistik Toplayıcılar
        train_loss_acc = 0
        ogm_stats_acc = {'k_cnn': 0, 'k_lstm': 0, 'k_tr': 0}
        batch_count = 0

        for batch in train_loader:
            x_cnn = batch["x_cnn"].to(DEVICE); x_lstm = batch["x_lstm"].to(DEVICE); x_tr = batch["x_tr"].to(DEVICE); y = batch["y"].to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                p_main, p_cnn, p_lstm, p_tr = model(x_cnn, x_lstm, x_tr)
                y_s = y * 100.0
                # Hem Loss'u hem İstatistikleri (Stats) alıyoruz
                loss, stats = calc_ogm_loss(criterion(p_main, y_s), criterion(p_cnn, y_s), criterion(p_lstm, y_s), criterion(p_tr, y_s))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            train_loss_acc += loss.item()
            # Batch istatistiklerini biriktir
            for k, v in stats.items(): ogm_stats_acc[k] += v
            batch_count += 1

        avg_train_loss = train_loss_acc / batch_count
        # Ortalama Boost katsayılarını hesapla
        avg_ogm = {k: v / batch_count for k, v in ogm_stats_acc.items()}

        # Validation
        model.eval()
        val_loss_acc = 0; correct = 0; total = 0
        with torch.no_grad():
            for batch in val_loader:
                x_cnn = batch["x_cnn"].to(DEVICE); x_lstm = batch["x_lstm"].to(DEVICE); x_tr = batch["x_tr"].to(DEVICE); y = batch["y"].to(DEVICE)
                with torch.amp.autocast("cuda"):
                    p_main, _, _, _ = model(x_cnn, x_lstm, x_tr)
                    val_loss_acc += criterion(p_main, y * 100.0).item()

                mask = torch.abs(y) > 0.11   #VERİ SETİ DEĞİŞİNCE BUNU DA DEĞİŞMEN GEREKİYOR
                correct += ((torch.sign(p_main) == torch.sign(y)) & mask).sum().item()
                total += mask.sum().item()

        avg_val_loss = val_loss_acc / len(val_loader)
        acc = (correct / max(total, 1)) * 100
        scheduler.step(avg_val_loss)

        # Geçmişe Kaydet
        history_record = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': acc,
            'avg_k_cnn': avg_ogm['k_cnn'],
            'avg_k_lstm': avg_ogm['k_lstm'],
            'avg_k_tr': avg_ogm['k_tr']
        }
        history.append(history_record)

        # Senin istediğin "Epoch 3 sonrası kayıt" mantığı
        if avg_val_loss < best_val and epoch > 3:
            best_val = avg_val_loss
            torch.save(model.state_dict(), "BEST_MODEL_FINAL.pth")
            print(f"Ep {epoch+1}: Loss {avg_val_loss:.5f} | Acc %{acc:.2f} | CNN-Boost: {avg_ogm['k_cnn']:.2f} | LSTM-Boost: {avg_ogm['k_lstm']:.2f} | TR-Boost: {avg_ogm['k_tr']:.2f} 💾")
        else:
            print(f"Ep {epoch+1}: Loss {avg_val_loss:.5f} | Acc %{acc:.2f} | CNN-Boost: {avg_ogm['k_cnn']:.2f} | LSTM-Boost: {avg_ogm['k_lstm']:.2f} | TR-Boost: {avg_ogm['k_tr']:.2f}")

    # EĞİTİM BİTTİ -> RAPORLARI ÜRET
    generate_training_report(history)
    print("\n✅ FINAL MODEL SAVED: BEST_MODEL_FINAL.pth")

if __name__ == "__main__":
    csv_path = f"{TARGET_COIN}_{MONTH_PERIOD}Ay_{TIMEFRAME}_AI_Ready.csv"
    df = pd.read_csv(csv_path)
    if [c for c in df.columns if "date" in c.lower()]: df.drop(columns=[c for c in df.columns if "date" in c.lower()], inplace=True)
    df = df.select_dtypes(include=[np.number]).fillna(0)

    train_end = int(0.70 * len(df)); val_end = int(0.85 * len(df))
    # Window boyutlarını yukarıdaki sabitlerden alıyor (12, 120, 120)
    train_ds = MultiBranchCryptoDataset(df.iloc[:train_end], cnn_window=CNN_WINDOW, lstm_window=LSTM_WINDOW, tr_window=TR_WINDOW)
    val_ds = MultiBranchCryptoDataset(df.iloc[train_end:val_end], mean=train_ds.mean, std=train_ds.std, cnn_window=CNN_WINDOW, lstm_window=LSTM_WINDOW, tr_window=TR_WINDOW)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=N_TRIALS)

    with open("best_params.json", "w") as f: json.dump(study.best_params, f, indent=4)
    train_best_model(study.best_params)