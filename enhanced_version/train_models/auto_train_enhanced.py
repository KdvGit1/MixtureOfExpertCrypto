import sys
import argparse
import os
import gc
import json
import math
from pathlib import Path
from datetime import datetime

import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Setup sys path to import from enhanced and parent directories
ENHANCED_ROOT = Path(__file__).parent.parent
PARENT_ROOT = ENHANCED_ROOT.parent

sys.path.insert(0, str(PARENT_ROOT))
sys.path.insert(0, str(ENHANCED_ROOT))

# Enhanced mimariyi import et
from train_models.ai_engine_enhanced import (
    MultiBranchCryptoDataset, 
    MultiBranchModel,
    TimeSeriesContrastiveLoss,
    ContrastiveProjectionHead,
    augment_sequence
)

# TA-Lib / CCXT Safe imports
try:
    import ccxt
    from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes
    HAS_DATA_FETCHER = True
except ImportError:
    HAS_DATA_FETCHER = False

# Window ayarları
CNN_WINDOW = 12
LSTM_WINDOW = 120
TR_WINDOW = 120

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = ENHANCED_ROOT / "trained_models"

# =========================================================
# OGM-GE MANTIĞI
# =========================================================
def calc_ogm_loss(loss_main, loss_cnn, loss_lstm, loss_tr):
    """
    OGM-GE: Hangi kol başarısızsa ona 'Boost' ver.
    """
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

    stats = {
        'k_cnn': coeff_cnn,
        'k_lstm': coeff_lstm,
        'k_tr': coeff_tr
    }

    return total_loss, stats

# =========================================================
# RAPORLAMA VE GRAFİK FONKSİYONLARI (MoE EKLENDİ)
# =========================================================
def generate_training_report(coin, tf_name, history):
    """Eğitim geçmişinden grafikler ve özet rapor oluşturur (4-panel)."""
    df_hist = pd.DataFrame(history)
    
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    csv_path = OUTPUT_DIR / f"{coin}_{tf_name}_history.csv"
    df_hist.to_csv(csv_path, index=False)
    print(f"\n💾 Detaylı veriler '{csv_path}' dosyasına kaydedildi.")

    plt.figure(figsize=(24, 5))

    # 1. Grafik: Loss
    plt.subplot(1, 4, 1)
    plt.plot(df_hist['epoch'], df_hist['train_loss'], label='Train Loss', color='blue')
    plt.plot(df_hist['epoch'], df_hist['val_loss'], label='Val Loss', color='orange', linestyle='--')
    plt.title(f"{coin} {tf_name} - Eğitim ve Doğrulama Kaybı")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Grafik: Accuracy
    plt.subplot(1, 4, 2)
    plt.plot(df_hist['epoch'], df_hist['val_acc'], label='Val Accuracy', color='green')
    plt.title("Dürüst Başarı Oranı (Masked Accuracy)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.axhline(y=50, color='r', linestyle=':', label='Rastgele (%50)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Grafik: OGM-GE Aktivitesi
    plt.subplot(1, 4, 3)
    plt.plot(df_hist['epoch'], df_hist['avg_k_cnn'], label='CNN Boost', color='cyan')
    plt.plot(df_hist['epoch'], df_hist['avg_k_lstm'], label='LSTM Boost', color='magenta')
    plt.plot(df_hist['epoch'], df_hist['avg_k_tr'], label='Transformer Boost', color='yellow')
    plt.title("OGM-GE Modülasyonu (Zorlanma Düzeyi)")
    plt.xlabel("Epoch")
    plt.ylabel("Boost Katsayıları (k)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Grafik: MoE Router Ağırlıkları (YENİ)
    plt.subplot(1, 4, 4)
    if 'avg_g_cnn' in df_hist.columns:
        plt.plot(df_hist['epoch'], df_hist['avg_g_cnn'] * 100, label='CNN Ağırlığı', color='cyan')
        plt.plot(df_hist['epoch'], df_hist['avg_g_lstm'] * 100, label='LSTM Ağırlığı', color='magenta')
        plt.plot(df_hist['epoch'], df_hist['avg_g_tr'] * 100, label='Transformer Ağırlığı', color='yellow')
        plt.title("MoE Router Dinamik Ağırlık Dağılımı")
        plt.xlabel("Epoch")
        plt.ylabel("Yönlendirme Yüzdesi (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    graphics_path = OUTPUT_DIR / f"{coin}_{tf_name}_graphics.png"
    plt.savefig(graphics_path)
    plt.close()
    print(f"📊 Grafikler '{graphics_path}' olarak kaydedildi.")

# =========================================================
# OPTUNA HİPERPARAMETRE OPTİMİZASYONU
# =========================================================
def optimize_hyperparameters(coin, tf_name, train_ds, val_ds, n_trials=25):
    """Optuna araması gerçekleştirir."""
    print(f"\n🔍 Optuna Hiperparametre Optimizasyonu Başlıyor ({n_trials} Trial)...")

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
        is_cuda = (DEVICE.type == 'cuda')
        scaler = torch.amp.GradScaler(device_type="cuda", enabled=is_cuda)

        best_val = float("inf")

        try:
            for epoch in range(12):  # Optuna araması için kısa tur (12 epoch)
                model.train()
                model.current_epoch = epoch

                for batch in train_loader:
                    x_cnn = batch["x_cnn"].to(DEVICE)
                    x_lstm = batch["x_lstm"].to(DEVICE)
                    x_tr = batch["x_tr"].to(DEVICE)
                    y = batch["y"].to(DEVICE)
                    
                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type=("cuda" if is_cuda else "cpu"), enabled=is_cuda):
                        p_main, p_cnn, p_lstm, p_tr, _ = model(x_cnn, x_lstm, x_tr)
                        y_s = y * 100.0
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
                        x_cnn = batch["x_cnn"].to(DEVICE)
                        x_lstm = batch["x_lstm"].to(DEVICE)
                        x_tr = batch["x_tr"].to(DEVICE)
                        y = batch["y"].to(DEVICE)
                        with torch.amp.autocast(device_type=("cuda" if is_cuda else "cpu"), enabled=is_cuda):
                            p_main, _, _, _, _ = model(x_cnn, x_lstm, x_tr)
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

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=4))
    study.optimize(objective, n_trials=n_trials)
    
    print(f"✅ Optuna Tamamlandı. En iyi Trial Değeri: {study.best_value:.5f}")
    return study.best_params

# =========================================================
# FINAL TRAIN (2-STAGE: PRE-TRAIN & FINE-TUNE)
# =========================================================
def train_best_model(coin, tf_name, train_ds, val_ds, best_params, pretrain_epochs=15, final_epochs=100):
    print("\n" + "=" * 60)
    print(f"🏆 AŞAMA 1: {coin} - KONTRASTİF ÖN EĞİTİM ({pretrain_epochs} Epoch)")
    print("=" * 60)

    train_loader = DataLoader(train_ds, batch_size=best_params["batch_size"], shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=best_params["batch_size"], shuffle=False, drop_last=True, pin_memory=True)

    model = MultiBranchModel(embed_dim=best_params["embed_dim"], dropout=best_params["dropout"]).to(DEVICE)
    
    # Projeksiyon Kafaları
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

    # STAGE 1 LOOP
    for epoch in range(pretrain_epochs):
        model.train()
        proj_cnn.train()
        proj_lstm.train()
        proj_tr.train()
        
        pretrain_loss_acc = 0.0
        batch_count = 0
        
        for batch in train_loader:
            x_cnn = batch["x_cnn"].to(DEVICE)
            x_lstm = batch["x_lstm"].to(DEVICE)
            x_tr = batch["x_tr"].to(DEVICE)
            
            # Artırılmış görünümler
            x_cnn_1 = augment_sequence(x_cnn)
            x_lstm_1 = augment_sequence(x_lstm)
            x_tr_1 = augment_sequence(x_tr)
            
            x_cnn_2 = augment_sequence(x_cnn)
            x_lstm_2 = augment_sequence(x_lstm)
            x_tr_2 = augment_sequence(x_tr)
            
            pretrain_optimizer.zero_grad()
            with torch.amp.autocast(device_type=("cuda" if is_cuda else "cpu"), enabled=is_cuda):
                # 1. Görünüm omurgalar
                out_cnn_1 = model.cnn_branch(x_cnn_1.transpose(1, 2))
                lstm_out_seq_1, _ = model.lstm(x_lstm_1)
                out_lstm_1 = lstm_out_seq_1[:, -1, :]
                out_tr_1 = model.transformer(model.pos_enc(model.tr_proj(x_tr_1))).mean(dim=1)
                
                # 2. Görünüm omurgalar
                out_cnn_2 = model.cnn_branch(x_cnn_2.transpose(1, 2))
                lstm_out_seq_2, _ = model.lstm(x_lstm_2)
                out_lstm_2 = lstm_out_seq_2[:, -1, :]
                out_tr_2 = model.transformer(model.pos_enc(model.tr_proj(x_tr_2))).mean(dim=1)
                
                # Kafalardan geçiş
                z_cnn_1 = proj_cnn(out_cnn_1)
                z_cnn_2 = proj_cnn(out_cnn_2)
                z_lstm_1 = proj_lstm(out_lstm_1)
                z_lstm_2 = proj_lstm(out_lstm_2)
                z_tr_1 = proj_tr(out_tr_1)
                z_tr_2 = proj_tr(out_tr_2)
                
                # Loss
                loss_cnn = contrastive_criterion(z_cnn_1, z_cnn_2)
                loss_lstm = contrastive_criterion(z_lstm_1, z_lstm_2)
                loss_tr = contrastive_criterion(z_tr_1, z_tr_2)
                
                total_loss = loss_cnn + loss_lstm + loss_tr
                
            scaler.scale(total_loss).backward()
            scaler.step(pretrain_optimizer)
            scaler.update()
            
            pretrain_loss_acc += total_loss.item()
            batch_count += 1
            
        avg_pretrain_loss = pretrain_loss_acc / batch_count
        print(f"Pre-train Ep {epoch+1}/{pretrain_epochs} | InfoNCE Loss: {avg_pretrain_loss:.5f}")

    print("\n" + "=" * 60)
    print(f"🏆 AŞAMA 2: {coin} - DENETİMLİ MoE İNCE AYAR ({final_epochs} Epoch)")
    print("=" * 60)

    optimizer = optim.AdamW(model.parameters(), lr=best_params["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=7, factor=0.7)
    criterion = nn.HuberLoss(delta=0.5)

    best_val = float("inf")
    history = []

    for epoch in range(final_epochs):
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
            with torch.amp.autocast(device_type=("cuda" if is_cuda else "cpu"), enabled=is_cuda):
                p_main, p_cnn, p_lstm, p_tr, g_weights = model(x_cnn, x_lstm, x_tr)
                y_s = y * 100.0
                loss, stats = calc_ogm_loss(criterion(p_main, y_s), criterion(p_cnn, y_s), criterion(p_lstm, y_s), criterion(p_tr, y_s))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            train_loss_acc += loss.item()
            for k, v in stats.items(): ogm_stats_acc[k] += v
            moe_weights_acc += g_weights.mean(dim=0)
            batch_count += 1

        avg_train_loss = train_loss_acc / batch_count
        avg_ogm = {k: v / batch_count for k, v in ogm_stats_acc.items()}
        avg_moe = moe_weights_acc / batch_count

        # Validation
        model.eval()
        val_loss_acc = 0; correct = 0; total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_cnn = batch["x_cnn"].to(DEVICE)
                x_lstm = batch["x_lstm"].to(DEVICE)
                x_tr = batch["x_tr"].to(DEVICE)
                y = batch["y"].to(DEVICE)
                with torch.amp.autocast(device_type=("cuda" if is_cuda else "cpu"), enabled=is_cuda):
                    p_main, _, _, _, _ = model(x_cnn, x_lstm, x_tr)
                    val_loss_acc += criterion(p_main, y * 100.0).item()

                mask = torch.abs(y) > 0.11
                correct += ((torch.sign(p_main) == torch.sign(y)) & mask).sum().item()
                total += mask.sum().item()

        avg_val_loss = val_loss_acc / len(val_loader)
        acc = (correct / max(total, 1)) * 100
        scheduler.step(avg_val_loss)

        history_record = {
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
        }
        history.append(history_record)

        # Model Dosyalarını Kaydet
        model_path = OUTPUT_DIR / f"{coin}_{tf_name}_model.pth"
        params_path = OUTPUT_DIR / f"{coin}_{tf_name}_params.json"
        stats_path = OUTPUT_DIR / f"{coin}_{tf_name}_stats.json"

        if avg_val_loss < best_val and epoch > 3:
            best_val = avg_val_loss
            
            # Kaydet
            torch.save(model.state_dict(), model_path)
            
            params_with_info = {
                "coin": coin,
                "timeframe": tf_name,
                "embed_dim": best_params["embed_dim"],
                "dropout": best_params["dropout"],
                "learning_rate": best_params["learning_rate"],
                "batch_size": best_params["batch_size"]
            }
            with open(params_path, "w") as f:
                json.dump(params_with_info, f, indent=4)
                
            stats = {
                "mean": train_ds.mean.to_dict(),
                "std": train_ds.std.to_dict()
            }
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=4)

            print(f"Ep {epoch+1}: Loss {avg_val_loss:.5f} | Acc %{acc:.2f} | MoE-Weights: [C: {avg_moe[0]:.2f}, L: {avg_moe[1]:.2f}, T: {avg_moe[2]:.2f}] 💾")
        elif epoch % 5 == 0:
            print(f"Ep {epoch+1}: Loss {avg_val_loss:.5f} | Acc %{acc:.2f} | MoE-Weights: [C: {avg_moe[0]:.2f}, L: {avg_moe[1]:.2f}, T: {avg_moe[2]:.2f}]")

    # Rapor ve grafik üret
    generate_training_report(coin, tf_name, history)
    print(f"✅ MODEL EĞİTİMİ TAMAMLANDI: {coin}_{tf_name}")

# =========================================================
# MAİN ÇALIŞTIRICI
# =========================================================
def run_training_pipeline(coins, timeframes, months_back, trials, pretrain, epochs):
    """Tekli veya Çoklu Coin/Timeframe eğitimlerini koordine eder."""
    print("\n" + "="*60)
    print("🚀 ENHANCED MULTI-COIN MOE TRAINING PIPELINE")
    print(f"   Coins: {coins}")
    print(f"   Timeframes: {timeframes}")
    print("="*60)

    for coin in coins:
        for tf in timeframes:
            print(f"\n{'-'*60}")
            print(f"⚙️ Eğitim Başlıyor: {coin} - {tf} (Veri: {months_back} Ay)")
            print(f"{'-'*60}")

            # 1. Veriyi çek / yükle
            csv_path = PARENT_ROOT / f"train_models/{coin}_{months_back}Ay_{tf}_AI_Ready.csv"
            if not csv_path.exists():
                csv_path = PARENT_ROOT / f"{coin}_{months_back}Ay_{tf}_AI_Ready.csv"
            if not csv_path.exists():
                csv_path = ENHANCED_ROOT / f"{coin}_{months_back}Ay_{tf}_AI_Ready.csv"

            df = None
            if csv_path.exists():
                print(f"📂 Yerel veri bulundu: {csv_path}")
                df = pd.read_csv(csv_path)
            elif HAS_DATA_FETCHER:
                print(f"🔌 Yerel veri bulunamadı. Binance Vadeli İşlemler'den veri çekiliyor...")
                try:
                    df_raw = get_crypto_history(f"{coin}/USDT", tf, months_back)
                    _, df = prepare_dual_dataframes(df_raw)
                    # Yerel olarak kaydet (bir sonraki eğitim için)
                    df.to_csv(ENHANCED_ROOT / f"{coin}_{months_back}Ay_{tf}_AI_Ready.csv")
                except Exception as e:
                    print(f"❌ Veri çekilemedi: {e}")
            
            if df is None:
                print(f"⚠️ {coin}/{tf} için veri elde edilemedi, dummy veri üretiliyor...")
                df = pd.DataFrame(np.random.randn(2000, 14), columns=[
                    'Log_Ret', 'RSI', 'Dist_SMA_50', 'Dist_EMA_200',
                    'BB_PctB', 'BB_Width', 'MACD_Norm', 'ATR_Pct',
                    'Vol_Ratio', 'Vol_Spike',
                    'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
                ])

            # Veriyi temizle ve sayısal hale getir
            if [c for c in df.columns if "date" in c.lower()]: 
                df.drop(columns=[c for c in df.columns if "date" in c.lower()], inplace=True)
            df = df.select_dtypes(include=[np.number]).fillna(0)

            # Dataset oluşumu
            train_end = int(0.70 * len(df))
            val_end = int(0.85 * len(df))
            
            train_ds = MultiBranchCryptoDataset(df.iloc[:train_end], cnn_window=CNN_WINDOW, lstm_window=LSTM_WINDOW, tr_window=TR_WINDOW)
            val_ds = MultiBranchCryptoDataset(df.iloc[train_end:val_end], mean=train_ds.mean, std=train_ds.std, cnn_window=CNN_WINDOW, lstm_window=LSTM_WINDOW, tr_window=TR_WINDOW)

            print(f"📊 Veri Sınırları: Train: {len(train_ds)}, Val: {len(val_ds)}")

            # 2. Optuna Hiperparametre Arama
            best_params = optimize_hyperparameters(coin, tf, train_ds, val_ds, n_trials=trials)
            
            # Parametreleri kaydet
            OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
            with open(OUTPUT_DIR / f"{coin}_{tf}_best_params.json", "w") as f:
                json.dump(best_params, f, indent=4)

            # 3. Final 2-Aşamalı Eğitim
            train_best_model(
                coin=coin, 
                tf_name=tf, 
                train_ds=train_ds, 
                val_ds=val_ds, 
                best_params=best_params,
                pretrain_epochs=pretrain,
                final_epochs=epochs
            )

            # Temizlik
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced multi-coin / multi-timeframe MoE training script.")
    parser.add_argument("--coins", nargs="+", default=["BTC"], help="Coins to train (e.g. BTC LTC ADA)")
    parser.add_argument("--timeframes", nargs="+", default=["1h"], help="Timeframes to train (e.g. 15m 1h)")
    parser.add_argument("--months", type=int, default=180, help="Months of data back to fetch (e.g. 60 or 180)")
    parser.add_argument("--trials", type=int, default=25, help="Number of Optuna trials")
    parser.add_argument("--pretrain", type=int, default=15, help="Stage-1 pretraining epochs")
    parser.add_argument("--epochs", type=int, default=100, help="Stage-2 supervised epochs")

    args = parser.parse_args()
    
    run_training_pipeline(
        coins=[c.upper() for c in args.coins],
        timeframes=args.timeframes,
        months_back=args.months,
        trials=args.trials,
        pretrain=args.pretrain,
        epochs=args.epochs
    )
