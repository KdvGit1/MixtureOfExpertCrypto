"""
ERROR ANALYSIS SCRIPT
=====================
Hatalı tahminlerin hangi feature değerlerinde yoğunlaştığını analiz eder.
Örneğin: Gece 4'te hata oranı %40 iken, sabah 9'da %25 gibi.

Kullanım: python error_analysis.py
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns

# Kendi modüllerimiz
from ai_engine_improved import MultiBranchCryptoDataset, MultiBranchModel

# =========================================================
# AYARLAR
# =========================================================
TARGET_COIN = "BTC"
TIMEFRAME = "15m"
MONTH_PERIOD = 60
MODEL_PATH = "finalized_models/3BranchApproach/6try/BEST_MODEL_FINAL.pth"  # Eğitilmiş model dosyası

# Window ayarları (auto_train ile aynı olmalı)
CNN_WINDOW = 12
LSTM_WINDOW = 120
TR_WINDOW = 120

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# SİNÜS/KOSINÜS'TEN SAAT/GÜN ÇIKARMA
# =========================================================
def sin_cos_to_hour(sin_val, cos_val):
    """Sin/Cos değerlerinden saati (0-23) hesaplar."""
    angle = np.arctan2(sin_val, cos_val)
    hour = (angle * 24 / (2 * np.pi)) % 24
    return int(round(hour))

def sin_cos_to_day(sin_val, cos_val):
    """Sin/Cos değerlerinden günü (0-6) hesaplar."""
    angle = np.arctan2(sin_val, cos_val)
    day = (angle * 7 / (2 * np.pi)) % 7
    return int(round(day))

def get_hour_label(hour):
    """Saat için okunabilir etiket."""
    ranges = [
        (0, 4, "00:00-04:00 (Gece)"),
        (4, 8, "04:00-08:00 (Sabah Erken)"),
        (8, 12, "08:00-12:00 (Sabah)"),
        (12, 16, "12:00-16:00 (Öğleden Sonra)"),
        (16, 20, "16:00-20:00 (Akşam)"),
        (20, 24, "20:00-24:00 (Gece Geç)")
    ]
    for start, end, label in ranges:
        if start <= hour < end:
            return label
    return "00:00-04:00 (Gece)"

def get_day_label(day):
    """Gün için okunabilir etiket."""
    days = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
    return days[day % 7]

# =========================================================
# MODEL TAHMİN FONKSİYONU
# =========================================================
def get_predictions(model, dataloader):
    """Model ile tahmin yap, gerçek değerleri ve tahminleri döndür."""
    model.eval()
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for batch in dataloader:
            x_cnn = batch["x_cnn"].to(DEVICE)
            x_lstm = batch["x_lstm"].to(DEVICE)
            x_tr = batch["x_tr"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            
            with torch.amp.autocast("cuda"):
                p_main, _, _, _ = model(x_cnn, x_lstm, x_tr)
            
            all_preds.extend(p_main.cpu().numpy())
            all_actuals.extend(y.cpu().numpy())
    
    return np.array(all_preds), np.array(all_actuals)

# =========================================================
# HATA ANALİZİ
# =========================================================
def analyze_errors_by_feature(df_original, predictions, actuals, feature_name, bin_func=None, label_func=None, n_bins=5):
    """
    Belirli bir feature için hata analizi yapar.
    
    Args:
        df_original: Normalize edilmemiş orijinal dataframe
        predictions: Model tahminleri
        actuals: Gerçek değerler
        feature_name: Analiz edilecek feature adı
        bin_func: Özel binning fonksiyonu (sin/cos için)
        label_func: Etiketleme fonksiyonu
        n_bins: Kaç gruba bölüneceği (bin_func yoksa)
    
    Returns:
        DataFrame: Her grup için hata istatistikleri
    """
    # Yön tahmini doğruluğunu hesapla
    pred_signs = np.sign(predictions)
    actual_signs = np.sign(actuals * 100)  # Scale düzeltmesi
    is_correct = pred_signs == actual_signs
    
    # Sadece anlamlı hareketleri al (mask)
    mask = np.abs(actuals) > 0.001  # %0.1'den büyük hareketler
    
    # Feature değerlerini al
    feature_values = df_original[feature_name].values[-len(predictions):]
    
    # Binning
    if bin_func is not None:
        bins = [bin_func(v) for v in feature_values]
    else:
        # Quantile-based binning
        bins = pd.qcut(feature_values, n_bins, labels=False, duplicates='drop')
    
    # Analiz DataFrame'i oluştur
    analysis_df = pd.DataFrame({
        'bin': bins,
        'is_correct': is_correct,
        'is_error': ~is_correct,
        'mask': mask,
        'feature_value': feature_values
    })
    
    # Sadece masked olanları al
    analysis_df = analysis_df[analysis_df['mask']]
    
    # Gruplara göre istatistikler
    results = analysis_df.groupby('bin').agg(
        total=('is_error', 'count'),
        errors=('is_error', 'sum'),
        avg_feature=('feature_value', 'mean')
    ).reset_index()
    
    results['error_rate'] = (results['errors'] / results['total'] * 100).round(2)
    
    # Etiketleme
    if label_func is not None:
        results['label'] = results['bin'].apply(label_func)
    else:
        results['label'] = results['avg_feature'].apply(lambda x: f"{x:.3f}")
    
    results['feature'] = feature_name
    
    return results

# =========================================================
# SAAT ANALİZİ (Özel fonksiyon)
# =========================================================
def analyze_hour_errors(df_original, predictions, actuals):
    """Sin/Cos'tan saat çıkarıp hata analizi yapar."""
    
    hour_sin = df_original['Hour_Sin'].values[-len(predictions):]
    hour_cos = df_original['Hour_Cos'].values[-len(predictions):]
    
    hours = [sin_cos_to_hour(s, c) for s, c in zip(hour_sin, hour_cos)]
    hour_bins = [h // 4 for h in hours]  # 4 saatlik dilimler
    
    pred_signs = np.sign(predictions)
    actual_signs = np.sign(actuals * 100)
    is_correct = pred_signs == actual_signs
    mask = np.abs(actuals) > 0.001
    
    analysis_df = pd.DataFrame({
        'hour': hours,
        'hour_bin': hour_bins,
        'is_error': ~is_correct,
        'mask': mask
    })
    
    analysis_df = analysis_df[analysis_df['mask']]
    
    results = analysis_df.groupby('hour_bin').agg(
        total=('is_error', 'count'),
        errors=('is_error', 'sum')
    ).reset_index()
    
    results['error_rate'] = (results['errors'] / results['total'] * 100).round(2)
    results['label'] = results['hour_bin'].apply(lambda x: get_hour_label(x * 4))
    results['feature'] = 'Hour (Saat)'
    
    return results

# =========================================================
# GÜN ANALİZİ (Özel fonksiyon)
# =========================================================
def analyze_day_errors(df_original, predictions, actuals):
    """Sin/Cos'tan gün çıkarıp hata analizi yapar."""
    
    day_sin = df_original['Day_Sin'].values[-len(predictions):]
    day_cos = df_original['Day_Cos'].values[-len(predictions):]
    
    days = [sin_cos_to_day(s, c) for s, c in zip(day_sin, day_cos)]
    
    pred_signs = np.sign(predictions)
    actual_signs = np.sign(actuals * 100)
    is_correct = pred_signs == actual_signs
    mask = np.abs(actuals) > 0.001
    
    analysis_df = pd.DataFrame({
        'day': days,
        'is_error': ~is_correct,
        'mask': mask
    })
    
    analysis_df = analysis_df[analysis_df['mask']]
    
    results = analysis_df.groupby('day').agg(
        total=('is_error', 'count'),
        errors=('is_error', 'sum')
    ).reset_index()
    
    results['error_rate'] = (results['errors'] / results['total'] * 100).round(2)
    results['label'] = results['day'].apply(get_day_label)
    results['feature'] = 'Day (Gün)'
    
    return results

# =========================================================
# GÖRSELLEŞTİRME
# =========================================================
def plot_error_analysis(all_results):
    """Tüm feature'lar için hata analizi grafiklerini çizer."""
    
    n_features = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = sns.color_palette("RdYlGn_r", 10)  # Kırmızı=Yüksek hata, Yeşil=Düşük hata
    
    for idx, (feature_name, df) in enumerate(all_results.items()):
        if idx >= 6:
            break
            
        ax = axes[idx]
        
        # Renkleri hata oranına göre belirle
        error_rates = df['error_rate'].values
        norm_rates = (error_rates - error_rates.min()) / (error_rates.max() - error_rates.min() + 1e-6)
        bar_colors = [colors[int(r * 9)] for r in norm_rates]
        
        bars = ax.bar(range(len(df)), df['error_rate'], color=bar_colors, edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['label'], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Hata Oranı (%)')
        ax.set_title(f'{feature_name} Bazında Hata Analizi', fontweight='bold')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Rastgele (%50)')
        
        # Değerleri bar üzerine yaz
        for bar, val in zip(bars, df['error_rate']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylim(0, max(error_rates) * 1.15)
        ax.grid(axis='y', alpha=0.3)
    
    # Kullanılmayan subplot'ları gizle
    for idx in range(len(all_results), 6):
        axes[idx].set_visible(False)
    
    plt.suptitle('🔍 FEATURE BAZINDA HATA ANALİZİ', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('ERROR_ANALYSIS_GRAPHICS.png', dpi=150, bbox_inches='tight')
    print("📊 Grafikler 'ERROR_ANALYSIS_GRAPHICS.png' olarak kaydedildi.")

# =========================================================
# KONSOL RAPORU
# =========================================================
def print_console_report(all_results):
    """Konsola detaylı rapor yazdırır."""
    
    print("\n" + "=" * 70)
    print("🔍 FEATURE BAZINDA HATA ANALİZİ RAPORU")
    print("=" * 70)
    
    for feature_name, df in all_results.items():
        print(f"\n{'='*50}")
        print(f"📌 {feature_name}")
        print(f"{'='*50}")
        print(f"{'Grup':<25} | {'Toplam':>8} | {'Hatalı':>8} | {'Hata %':>8}")
        print("-" * 55)
        
        # Hata oranına göre sırala
        df_sorted = df.sort_values('error_rate', ascending=False)
        
        for _, row in df_sorted.iterrows():
            label = row['label'][:24]  # Max 24 karakter
            indicator = "⚠️" if row['error_rate'] > 45 else "✅" if row['error_rate'] < 35 else "  "
            print(f"{indicator} {label:<22} | {row['total']:>8} | {row['errors']:>8} | {row['error_rate']:>7.1f}%")
        
        # En iyi ve en kötü
        best = df.loc[df['error_rate'].idxmin()]
        worst = df.loc[df['error_rate'].idxmax()]
        print(f"\n   ✅ En İyi: {best['label']} ({best['error_rate']:.1f}%)")
        print(f"   ⚠️ En Kötü: {worst['label']} ({worst['error_rate']:.1f}%)")

# =========================================================
# ANA SCRIPT
# =========================================================
def main():
    print("🚀 Error Analysis başlatılıyor...")
    
    # 1. VERİ YÜKLE
    csv_path = f"{TARGET_COIN}_{MONTH_PERIOD}Ay_{TIMEFRAME}_AI_Ready.csv"
    df = pd.read_csv(csv_path)
    
    # Tarih sütununu kaldır (varsa)
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        df.drop(columns=date_cols, inplace=True)
    
    df = df.select_dtypes(include=[np.number]).fillna(0)
    
    # Original dataframe'i de sakla (sin/cos decode için)
    df_original = df.copy()
    
    # 2. DATASET OLUŞTUR
    train_end = int(0.70 * len(df))
    val_end = int(0.85 * len(df))
    
    train_ds = MultiBranchCryptoDataset(df.iloc[:train_end], cnn_window=CNN_WINDOW, lstm_window=LSTM_WINDOW, tr_window=TR_WINDOW)
    val_ds = MultiBranchCryptoDataset(df.iloc[train_end:val_end], mean=train_ds.mean, std=train_ds.std, cnn_window=CNN_WINDOW, lstm_window=LSTM_WINDOW, tr_window=TR_WINDOW)
    
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, drop_last=False, pin_memory=True)
    
    # 3. MODEL YÜKLE
    print(f"📦 Model yükleniyor: {MODEL_PATH}")
    model = MultiBranchModel(embed_dim=96, dropout=0.3).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Model başarıyla yüklendi!")
    except FileNotFoundError:
        print(f"❌ HATA: {MODEL_PATH} bulunamadı!")
        print("   Önce modeli eğitmeniz gerekiyor: python auto_train_improved.py")
        return
    
    # 4. TAHMİN YAP
    print("🔮 Tahminler yapılıyor...")
    predictions, actuals = get_predictions(model, val_loader)
    print(f"   Toplam {len(predictions)} tahmin yapıldı.")
    
    # 5. Original dataframe'den validation kısmını al
    val_start = train_end + max(CNN_WINDOW, LSTM_WINDOW, TR_WINDOW)
    val_df_original = df_original.iloc[val_start:val_start+len(predictions)]
    
    # 6. HATA ANALİZLERİ
    print("📊 Hata analizleri hesaplanıyor...")
    
    all_results = {}
    
    # Saat analizi
    try:
        all_results['Hour (Saat)'] = analyze_hour_errors(val_df_original, predictions, actuals)
    except Exception as e:
        print(f"   ⚠️ Saat analizi hatası: {e}")
    
    # Gün analizi
    try:
        all_results['Day (Gün)'] = analyze_day_errors(val_df_original, predictions, actuals)
    except Exception as e:
        print(f"   ⚠️ Gün analizi hatası: {e}")
    
    # Diğer feature'lar
    features_to_analyze = ['RSI', 'Vol_Ratio', 'BB_PctB', 'ATR_Pct']
    
    for feature in features_to_analyze:
        if feature in val_df_original.columns:
            try:
                result = analyze_errors_by_feature(val_df_original, predictions, actuals, feature, n_bins=5)
                all_results[feature] = result
            except Exception as e:
                print(f"   ⚠️ {feature} analizi hatası: {e}")
    
    # 7. RAPOR VE GRAFİKLER
    print_console_report(all_results)
    plot_error_analysis(all_results)
    
    # 8. CSV KAYDET
    all_dfs = []
    for name, df in all_results.items():
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv('ERROR_ANALYSIS_REPORT.csv', index=False)
    print("\n💾 Detaylı veriler 'ERROR_ANALYSIS_REPORT.csv' olarak kaydedildi.")
    
    print("\n✅ Analiz tamamlandı!")

if __name__ == "__main__":
    main()
