"""
================================================================================
🔬 BATCH MODEL ANALYZER - Tüm Modelleri Otomatik Test Et
================================================================================
Bu script kaggle_outputs klasöründeki tüm modelleri interpret_model mantığıyla
test eder ve sonuçları model ismiyle kaydeder.

Kullanım:
    python batch_model_analyzer.py
================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob
from datetime import datetime

# AI Engine import
from ai_engine_improved import MultiBranchCryptoDataset, MultiBranchModel, CNN_FEATURES, LSTM_FEATURES, TR_FEATURES
from data_fetcher import get_crypto_history, prepare_dual_dataframes

# ========================================================
# AYARLAR
# ========================================================
KAGGLE_OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "kaggle_outputs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "analysis_results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Timeframe ayarları
TIMEFRAME_CONFIGS = {
    "15m": {
        "months_back": 60,
        "cnn_window": 16,
        "lstm_window": 96,
        "tr_window": 96,
        "mask_threshold": 0.15
    },
    "1h": {
        "months_back": 180,
        "cnn_window": 12,
        "lstm_window": 120,
        "tr_window": 120,
        "mask_threshold": 0.11
    }
}

# ========================================================
# YARDIMCI FONKSİYONLAR
# ========================================================

def find_all_models(kaggle_dir):
    """Kaggle outputs klasöründeki tüm modelleri bul."""
    models = []
    model_files = glob.glob(os.path.join(kaggle_dir, "*_model.pth"))
    
    for model_path in model_files:
        filename = os.path.basename(model_path)
        # BTC_15m_model.pth -> BTC, 15m
        parts = filename.replace("_model.pth", "").split("_")
        if len(parts) >= 2:
            coin = parts[0]
            timeframe = parts[1]
            
            params_path = model_path.replace("_model.pth", "_params.json")
            stats_path = model_path.replace("_model.pth", "_stats.json")
            
            if os.path.exists(params_path):
                models.append({
                    "coin": coin,
                    "timeframe": timeframe,
                    "model_path": model_path,
                    "params_path": params_path,
                    "stats_path": stats_path
                })
    
    return models


def load_model(model_info):
    """Model ve parametreleri yükle."""
    with open(model_info["params_path"], 'r') as f:
        params = json.load(f)
    
    embed_dim = int(params['embed_dim'])
    dropout = float(params['dropout'])
    
    model = MultiBranchModel(embed_dim=embed_dim, dropout=dropout).to(DEVICE)
    model.load_state_dict(torch.load(model_info["model_path"], map_location=DEVICE))
    model.eval()
    
    return model, params


def get_or_fetch_data(coin, timeframe):
    """Veriyi çek veya cache'ten al."""
    tf_config = TIMEFRAME_CONFIGS[timeframe]
    
    # Veri dosyası var mı kontrol et
    csv_filename = f"{coin}_{tf_config['months_back']}Ay_{timeframe}_AI_Ready.csv"
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
    
    if os.path.exists(csv_path):
        print(f"   📂 Cache'ten yükleniyor: {csv_filename}")
        df = pd.read_csv(csv_path)
    else:
        print(f"   🌐 Veri çekiliyor: {coin}/{timeframe}...")
        symbol = f"{coin}/USDT"
        df_raw = get_crypto_history(symbol, timeframe, tf_config['months_back'])
        _, df = prepare_dual_dataframes(df_raw)
        df.to_csv(csv_path, index=False)
        print(f"   💾 Kaydedildi: {csv_filename}")
    
    return df


def evaluate_accuracy(model, loader, device, mask_threshold=0.18):
    """Maskelenmiş Accuracy ölç."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x_cnn = batch['x_cnn'].to(device)
            x_lstm = batch['x_lstm'].to(device)
            x_tr = batch['x_tr'].to(device)
            y = batch['y'].to(device)

            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                pred, _, _, _ = model(x_cnn, x_lstm, x_tr)
                mask = torch.abs(y) > mask_threshold

                if mask.sum() > 0:
                    correct += ((torch.sign(pred) == torch.sign(y)) & mask).sum().item()
                    total += mask.sum().item()

    if total == 0: return 0.0
    return (correct / total) * 100


def evaluate_branch_ablation(model, loader, device, mask_branch, mask_threshold=0.18):
    """Bir kolu kör ederek performans ölç."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x_cnn = batch['x_cnn'].to(device)
            x_lstm = batch['x_lstm'].to(device)
            x_tr = batch['x_tr'].to(device)
            y = batch['y'].to(device)

            if mask_branch == 'cnn': x_cnn = torch.zeros_like(x_cnn)
            elif mask_branch == 'lstm': x_lstm = torch.zeros_like(x_lstm)
            elif mask_branch == 'tr': x_tr = torch.zeros_like(x_tr)

            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                pred, _, _, _ = model(x_cnn, x_lstm, x_tr)
                mask = torch.abs(y) > mask_threshold
                
                if mask.sum() > 0:
                    correct += ((torch.sign(pred) == torch.sign(y)) & mask).sum().item()
                    total += mask.sum().item()

    if total == 0: return 0.0
    return (correct / total) * 100


def analyze_single_model(model_info):
    """Tek bir modeli analiz et."""
    coin = model_info["coin"]
    timeframe = model_info["timeframe"]
    model_name = f"{coin}_{timeframe}"
    
    print(f"\n{'='*60}")
    print(f"🔬 ANALİZ: {model_name}")
    print(f"{'='*60}")
    
    tf_config = TIMEFRAME_CONFIGS[timeframe]
    
    # 1. Veriyi al
    try:
        df = get_or_fetch_data(coin, timeframe)
    except Exception as e:
        print(f"   ❌ Veri çekilemedi: {e}")
        return None
    
    # Temizlik
    cols_to_drop = [c for c in df.columns if 'date' in c.lower() or 'unnamed' in c.lower()]
    if cols_to_drop: df.drop(columns=cols_to_drop, inplace=True)
    df = df.select_dtypes(include=[np.number]).fillna(0)
    
    # Split
    total_len = len(df)
    train_end = int(0.70 * total_len)
    val_end = int(0.85 * total_len)
    
    train_df = df.iloc[:train_end]
    test_df = df.iloc[val_end:]
    
    print(f"   📊 Test verisi: {len(test_df)} satır")
    
    # Normalizasyon istatistikleri
    temp_train_ds = MultiBranchCryptoDataset(train_df, mean=None, std=None)
    train_mean, train_std = temp_train_ds.mean, temp_train_ds.std
    
    # 2. Modeli yükle
    model, params = load_model(model_info)
    print(f"   ⚙️ Embed: {params['embed_dim']}, Dropout: {params['dropout']:.3f}")
    
    # 3. Test Dataset
    cnn_w = tf_config["cnn_window"]
    lstm_w = tf_config["lstm_window"]
    tr_w = tf_config["tr_window"]
    mask_threshold = tf_config["mask_threshold"]
    
    test_ds = MultiBranchCryptoDataset(
        test_df,
        mean=train_mean,
        std=train_std,
        cnn_window=cnn_w,
        lstm_window=lstm_w,
        tr_window=tr_w
    )
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
    
    # 4. Baseline Accuracy
    baseline_acc = evaluate_accuracy(model, test_loader, DEVICE, mask_threshold)
    print(f"   ✅ Baseline Accuracy: %{baseline_acc:.2f}")
    
    # 5. Feature Importance
    feature_importance = {}
    all_features = list(set(CNN_FEATURES + LSTM_FEATURES + TR_FEATURES))
    
    print("   🔬 Özellik testi...")
    for col in all_features:
        if col not in test_df.columns: continue
        
        temp_df = test_df.copy()
        temp_df[col] = np.random.permutation(temp_df[col].values)
        
        perm_ds = MultiBranchCryptoDataset(
            temp_df, mean=train_mean, std=train_std,
            cnn_window=cnn_w, lstm_window=lstm_w, tr_window=tr_w
        )
        perm_loader = DataLoader(perm_ds, batch_size=1024, shuffle=False)
        
        perm_acc = evaluate_accuracy(model, perm_loader, DEVICE, mask_threshold)
        feature_importance[col] = baseline_acc - perm_acc
    
    # 6. Branch Ablation
    print("   🧠 Dal analizi...")
    cnn_acc = evaluate_branch_ablation(model, test_loader, DEVICE, 'cnn', mask_threshold)
    lstm_acc = evaluate_branch_ablation(model, test_loader, DEVICE, 'lstm', mask_threshold)
    tr_acc = evaluate_branch_ablation(model, test_loader, DEVICE, 'tr', mask_threshold)
    
    branch_importance = {
        'CNN (Refleks)': baseline_acc - cnn_acc,
        'LSTM (Trend)': baseline_acc - lstm_acc,
        'Transformer (Rejim)': baseline_acc - tr_acc
    }
    
    # 7. Sonuçları kaydet
    results = {
        "model_name": model_name,
        "coin": coin,
        "timeframe": timeframe,
        "baseline_accuracy": baseline_acc,
        "feature_importance": feature_importance,
        "branch_importance": branch_importance,
        "params": params,
        "test_samples": len(test_df),
        "analyzed_at": datetime.now().isoformat()
    }
    
    return results


def save_results(results, results_dir):
    """Sonuçları kaydet."""
    model_name = results["model_name"]
    
    # JSON raporu
    json_path = os.path.join(results_dir, f"{model_name}_analysis.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Grafik
    plt.figure(figsize=(16, 6))
    
    # Feature Importance
    plt.subplot(1, 2, 1)
    sorted_feats = dict(sorted(results["feature_importance"].items(), key=lambda x: x[1], reverse=True))
    colors = ['green' if v > 0 else 'red' for v in sorted_feats.values()]
    sns.barplot(x=list(sorted_feats.values()), y=list(sorted_feats.keys()), palette=colors)
    plt.title(f"{model_name} - Feature Importance")
    plt.xlabel("Accuracy Drop (%)")
    
    # Branch Importance
    plt.subplot(1, 2, 2)
    branch_data = results["branch_importance"]
    colors = ['green' if v > 0 else 'red' for v in branch_data.values()]
    sns.barplot(x=list(branch_data.keys()), y=list(branch_data.values()), palette=colors)
    plt.title(f"{model_name} - Branch Importance")
    plt.ylabel("Accuracy Drop (%)")
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    png_path = os.path.join(results_dir, f"{model_name}_analysis.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    
    print(f"   💾 Kaydedildi: {model_name}_analysis.json/.png")


def generate_summary_report(all_results, results_dir):
    """Tüm sonuçların özet raporunu oluştur."""
    summary = []
    
    for r in all_results:
        summary.append({
            "Model": r["model_name"],
            "Coin": r["coin"],
            "Timeframe": r["timeframe"],
            "Accuracy (%)": round(r["baseline_accuracy"], 2),
            "CNN Impact": round(r["branch_importance"]["CNN (Refleks)"], 2),
            "LSTM Impact": round(r["branch_importance"]["LSTM (Trend)"], 2),
            "TR Impact": round(r["branch_importance"]["Transformer (Rejim)"], 2)
        })
    
    df_summary = pd.DataFrame(summary)
    df_summary = df_summary.sort_values("Accuracy (%)", ascending=False)
    
    # CSV kaydet
    csv_path = os.path.join(results_dir, "SUMMARY_ALL_MODELS.csv")
    df_summary.to_csv(csv_path, index=False)
    
    # Özet grafiği
    plt.figure(figsize=(14, 8))
    
    # Accuracy bar chart
    plt.subplot(2, 1, 1)
    colors = plt.cm.RdYlGn(df_summary["Accuracy (%)"].values / 100)
    plt.barh(df_summary["Model"], df_summary["Accuracy (%)"], color=colors)
    plt.xlabel("Accuracy (%)")
    plt.title("Model Performans Karşılaştırması")
    plt.axvline(x=50, color='red', linestyle='--', label='Random (%50)')
    plt.legend()
    
    # Branch importance heatmap
    plt.subplot(2, 1, 2)
    branch_data = df_summary[["Model", "CNN Impact", "LSTM Impact", "TR Impact"]].set_index("Model")
    sns.heatmap(branch_data, annot=True, cmap="RdYlGn", center=0, fmt=".1f")
    plt.title("Dal Önem Haritası (Accuracy Drop)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "SUMMARY_ALL_MODELS.png"), dpi=150)
    plt.close()
    
    print(f"\n✅ Özet rapor kaydedildi: SUMMARY_ALL_MODELS.csv/.png")
    print("\n" + "="*60)
    print("📊 EN İYİ 5 MODEL:")
    print("="*60)
    print(df_summary.head().to_string(index=False))


def main():
    print("\n" + "="*60)
    print("🔬 BATCH MODEL ANALYZER")
    print("="*60)
    
    # Klasörleri kontrol et
    if not os.path.exists(KAGGLE_OUTPUTS_DIR):
        print(f"❌ Kaggle outputs klasörü bulunamadı: {KAGGLE_OUTPUTS_DIR}")
        return
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Modelleri bul
    models = find_all_models(KAGGLE_OUTPUTS_DIR)
    print(f"📦 {len(models)} model bulundu")
    
    if not models:
        print("❌ Hiç model bulunamadı!")
        return
    
    # Sıralama (coin ve timeframe'e göre)
    models.sort(key=lambda x: (x["coin"], x["timeframe"]))
    
    # Tüm modelleri analiz et
    all_results = []
    
    for i, model_info in enumerate(models, 1):
        print(f"\n📈 İlerleme: {i}/{len(models)}")
        
        try:
            results = analyze_single_model(model_info)
            if results:
                save_results(results, RESULTS_DIR)
                all_results.append(results)
        except Exception as e:
            print(f"   ❌ HATA: {e}")
            continue
    
    # Özet rapor
    if all_results:
        generate_summary_report(all_results, RESULTS_DIR)
    
    print("\n🎉 Tüm analizler tamamlandı!")
    print(f"📁 Sonuçlar: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
