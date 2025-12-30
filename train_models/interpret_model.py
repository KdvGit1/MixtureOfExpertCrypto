import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import copy

# 🔥 DÜZELTME 1: Eski engine yerine 'improved' olanı çağırıyoruz
from ai_engine_improved import MultiBranchCryptoDataset, MultiBranchModel, CNN_FEATURES, LSTM_FEATURES, TR_FEATURES

# ========================================================
# AYARLAR (Yeni Sisteme Göre Güncellendi)
# ========================================================
TARGET_COIN = "BTC"
TIMEFRAME = "1h"
MONTH_PERIOD = 180
CSV_PATH = f"{TARGET_COIN}_{MONTH_PERIOD}Ay_{TIMEFRAME}_AI_Ready.csv"

# Model ve Parametre Dosyaları
MODEL_PATH = "BEST_MODEL_FINAL.pth"
PARAMS_PATH = "best_params.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pencereler (auto_train_improved.py ile birebir aynı olmalı)
CNN_WINDOW = 12
LSTM_WINDOW = 120
TR_WINDOW = 120

def load_model_and_params():
    """Otomatik eğitilen en iyi modeli ve ayarlarını yükler."""
    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError(f"❌ {PARAMS_PATH} bulunamadı! Önce 'auto_train_improved.py' çalıştırıp modeli eğitmelisin.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ {MODEL_PATH} bulunamadı! Model dosyası eksik.")

    print(f"📂 Parametreler okunuyor: {PARAMS_PATH}")
    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)

    embed_dim = int(params['embed_dim'])
    dropout = float(params['dropout'])

    print(f"🏆 İncelenen Model: {MODEL_PATH}")
    print(f"⚙️ Model Ayarları: Embed Dim={embed_dim}, Dropout={dropout}")

    return embed_dim, dropout

def evaluate_accuracy(model, loader, device):
    """
    Maskelenmiş (Dürüst) Accuracy Ölçer.
    Sadece 'y > 0.18' (önemli hareketler) dikkate alınır.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x_cnn = batch['x_cnn'].to(device)
            x_lstm = batch['x_lstm'].to(device)
            x_tr = batch['x_tr'].to(device)
            y = batch['y'].to(device)

            with torch.amp.autocast('cuda'):
                # 🔥 DÜZELTME 2: Yeni model 4 çıktı veriyor (Main, Aux1, Aux2, Aux3).
                # Biz sadece ilkiyle (Main Pred) ilgileniyoruz.
                pred, _, _, _ = model(x_cnn, x_lstm, x_tr)

                # Maskeleme mantığı (0.18 Sigma)
                mask = torch.abs(y) > 0.18

                if mask.sum() > 0:
                    correct += ((torch.sign(pred) == torch.sign(y)) & mask).sum().item()
                    total += mask.sum().item()

    if total == 0: return 0.0
    return (correct / total) * 100

def evaluate_branch_ablation(model, loader, device, mask_branch):
    """Bir kolu kör ederek (sıfırlayarak) performans ölçer."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x_cnn = batch['x_cnn'].to(device)
            x_lstm = batch['x_lstm'].to(device)
            x_tr = batch['x_tr'].to(device)
            y = batch['y'].to(device)

            # Seçilen kolu sustur
            if mask_branch == 'cnn': x_cnn = torch.zeros_like(x_cnn)
            elif mask_branch == 'lstm': x_lstm = torch.zeros_like(x_lstm)
            elif mask_branch == 'tr': x_tr = torch.zeros_like(x_tr)

            with torch.amp.autocast('cuda'):
                # 🔥 DÜZELTME 3: Burada da 4 çıktıyı karşılıyoruz
                pred, _, _, _ = model(x_cnn, x_lstm, x_tr)

                mask = torch.abs(y) > 0.18
                if mask.sum() > 0:
                    correct += ((torch.sign(pred) == torch.sign(y)) & mask).sum().item()
                    total += mask.sum().item()

    if total == 0: return 0.0
    return (correct / total) * 100

def run_analysis():
    # 1. VERİYİ YÜKLE
    print(f"📂 Veri Seti: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print("❌ CSV dosyası yok!"); return

    df = pd.read_csv(CSV_PATH)

    # Temizlik
    cols_to_drop = [c for c in df.columns if 'date' in c.lower() or 'unnamed' in c.lower()]
    if cols_to_drop: df.drop(columns=cols_to_drop, inplace=True)
    df = df.select_dtypes(include=[np.number]).fillna(0)

    # Split (Sadece Test verisini kullanacağız)
    total_len = len(df)
    train_end = int(0.70 * total_len)
    val_end = int(0.85 * total_len)

    # Normalizasyon için Train setinin istatistiklerini alıyoruz
    train_df = df.iloc[:train_end]
    test_df = df.iloc[val_end:]

    print("📊 İstatistikler Çıkarılıyor...")
    temp_train_ds = MultiBranchCryptoDataset(train_df, mean=None, std=None)
    train_mean, train_std = temp_train_ds.mean, temp_train_ds.std

    # 2. MODELİ YÜKLE
    embed_dim, dropout = load_model_and_params()
    model = MultiBranchModel(embed_dim=embed_dim, dropout=dropout).to(DEVICE)

    # weights_only=False hatasını önlemek için (Pickle uyarısı) ve Load işlemi
    # OGM-GE katmanları (aux_cnn vb.) artık model tanımında olduğu için hata vermeyecek.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3. BASELINE PERFORMANS (Maskelenmiş Accuracy)
    print("\n🔍 Temel Performans Ölçülüyor (Test Seti)...")
    test_ds = MultiBranchCryptoDataset(
        test_df,
        mean=train_mean,
        std=train_std,
        cnn_window=CNN_WINDOW,
        lstm_window=LSTM_WINDOW,
        tr_window=TR_WINDOW
    )
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    baseline_acc = evaluate_accuracy(model, test_loader, DEVICE)
    print(f"✅ Baseline Accuracy (Masked): %{baseline_acc:.2f}")
    print("-" * 50)

    # =====================================================
    # ANALİZ 1: ÖZELLİK ÖNEMİ (FEATURE IMPORTANCE)
    # =====================================================
    feature_importance = {}

    # Feature listelerini birleştirip unique yapalım (Sıralı kontrol)
    all_features = list(set(CNN_FEATURES + LSTM_FEATURES + TR_FEATURES))
    # Log_Ret genelde target olduğu için feature importance'da karıştırmak sonucu çok bozar ama görelim.

    print("🔬 Özellikler Test Ediliyor (Permutation Importance)...")

    for col in all_features:
        if col not in test_df.columns: continue

        # Sütunu karıştır
        temp_df = test_df.copy()
        temp_df[col] = np.random.permutation(temp_df[col].values)

        perm_ds = MultiBranchCryptoDataset(
            temp_df,
            mean=train_mean,
            std=train_std,
            cnn_window=CNN_WINDOW,
            lstm_window=LSTM_WINDOW,
            tr_window=TR_WINDOW
        )
        perm_loader = DataLoader(perm_ds, batch_size=1024, shuffle=False)

        perm_acc = evaluate_accuracy(model, perm_loader, DEVICE)
        drop = baseline_acc - perm_acc

        feature_importance[col] = drop
        print(f"   👉 {col:15s}: %{perm_acc:.2f} (Etki: {drop:+.2f})")

    # =====================================================
    # ANALİZ 2: BEYİN LOBU ANALİZİ (BRANCH ABLATION)
    # =====================================================
    print("-" * 50)
    print("🧠 Beyin Lobları Test Ediliyor...")

    cnn_acc = evaluate_branch_ablation(model, test_loader, DEVICE, 'cnn')
    print(f"   ❌ CNN Kapalı: %{cnn_acc:.2f} (Fark: {baseline_acc - cnn_acc:+.2f})")

    lstm_acc = evaluate_branch_ablation(model, test_loader, DEVICE, 'lstm')
    print(f"   ❌ LSTM Kapalı: %{lstm_acc:.2f} (Fark: {baseline_acc - lstm_acc:+.2f})")

    tr_acc = evaluate_branch_ablation(model, test_loader, DEVICE, 'tr')
    print(f"   ❌ Transformer Kapalı: %{tr_acc:.2f} (Fark: {baseline_acc - tr_acc:+.2f})")

    branch_importance = {
        'CNN (Refleks)': baseline_acc - cnn_acc,
        'LSTM (Trend)': baseline_acc - lstm_acc,
        'Transformer (Rejim)': baseline_acc - tr_acc
    }

    # =====================================================
    # GÖRSELLEŞTİRME VE KAYIT
    # =====================================================
    plt.figure(figsize=(18, 6))

    # Grafik 1
    plt.subplot(1, 2, 1)
    sorted_feats = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
    sns.barplot(x=list(sorted_feats.values()), y=list(sorted_feats.keys()), palette="viridis")
    plt.title("Feature Importance (Doğruluk Kaybı)")
    plt.xlabel("Accuracy Drop (%)")

    # Grafik 2
    plt.subplot(1, 2, 2)
    sns.barplot(x=list(branch_importance.keys()), y=list(branch_importance.values()), palette="magma")
    plt.title("Model Branch Importance")
    plt.ylabel("Accuracy Drop (%)")

    plt.tight_layout()
    plt.savefig("MODEL_INTERPRETATION.png")
    print(f"\n✅ Grafik kaydedildi: MODEL_INTERPRETATION.png")

    # JSON Raporu
    results = {
        "BASELINE_ACCURACY": baseline_acc,
        "FEATURE_IMPORTANCE": feature_importance,
        "BRANCH_IMPORTANCE": branch_importance
    }

    with open("ANALYSIS_RESULTS.txt", "w") as f:
        f.write("="*50 + "\n")
        f.write("       YENİ MODEL ANALİZ RAPORU\n")
        f.write("="*50 + "\n\n")
        f.write(json.dumps(results, indent=4))

    print("✅ Metin raporu kaydedildi: ANALYSIS_RESULTS.txt")

if __name__ == "__main__":
    run_analysis()