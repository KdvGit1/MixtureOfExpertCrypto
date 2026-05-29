import ccxt
import pandas as pd
import json
import os
import numpy as np
import torch
from datetime import datetime
from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes
from train_models.ai_engine_improved import MultiBranchModel, CNN_FEATURES, LSTM_FEATURES, TR_FEATURES

JSON_FILENAME = "live_market_data.json"

MODEL_MAP = {
    '15m': 'train_models/finalized_models/3BranchApproach/6try/BEST_MODEL_FINAL.pth',
    '1h': 'train_models/finalized_models/3BranchApproach/7try_1h/BEST_MODEL_FINAL.pth'
}

MODEL_PARAMS = {
    '15m': {'embed_dim': 96, 'dropout': 0.31},
    '1h': {'embed_dim': 128, 'dropout': 0.32}
}

# Try loading from json if available
for tf in MODEL_MAP.keys():
    params_file = os.path.join(os.path.dirname(__file__), 'train_models', 'CryptoMoeApp', f'best_params_{tf}.json')
    if os.path.exists(params_file):
        try:
            with open(params_file) as f:
                params = json.load(f)
                MODEL_PARAMS[tf] = {
                    'embed_dim': params.get('embed_dim', 128),
                    'dropout': params.get('dropout', 0.15)
                }
        except Exception:
            pass

_MODEL_CACHE = {}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ai_model(timeframe):
    """
    İstenilen timeframe için doğru modeli yükler ve hafızada tutar.
    """
    global _MODEL_CACHE

    # 1. Cache Kontrolü
    if timeframe in _MODEL_CACHE:
        return _MODEL_CACHE[timeframe]

    # 2. Dosya Yolunu Bul
    if timeframe not in MODEL_MAP:
        print(f"❌ '{timeframe}' için tanımlı bir model yok! (MODEL_MAP'i kontrol et)")
        return None

    model_path = MODEL_MAP[timeframe]

    if not os.path.exists(model_path):
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        return None

    print(f"🧠 '{timeframe}' için Yapay Zeka Modeli Yükleniyor: {model_path} ...")

    # 3. Modeli Oluştur
    try:
        params = MODEL_PARAMS.get(timeframe, {'embed_dim': 128, 'dropout': 0.15})
        model = MultiBranchModel(
            embed_dim=params['embed_dim'],
            dropout=params['dropout']
        ).to(_DEVICE)

        # 4. Ağırlıkları Yükle
        state_dict = torch.load(model_path, map_location=_DEVICE, weights_only=True)
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)

        model.eval()  # Sınav modu

        # 5. Cache'e At
        _MODEL_CACHE[timeframe] = model
        return model

    except Exception as e:
        print(f"❌ Model yükleme hatası ({timeframe}): {e}")
        return None

def get_available_exhanges():
    return ccxt.exchanges

def get_all_pairs(exchange_name="binance"):
    exchange_name = exchange_name.lower()
    exchange = getattr(ccxt, exchange_name)()
    exchange.load_markets()
    pair_list = [ symbol for symbol in exchange.symbols
                  if symbol.endswith(("/USDT"))
                  ]
    print(pair_list)
    return pair_list

def calculate_needed_months(timeframe_str, candle_count=500):
    """
    İstenilen mum sayısı için kaç ay geriye gidilmesi gerektiğini hesaplar.
    Güvenlik payı olarak %10 fazlasını hesaplar.
    """
    # 1. Zaman dilimini dakikaya çevir
    tf_minutes = 0
    if timeframe_str == '1h':
        tf_minutes = 60
    elif timeframe_str == '15m':
        tf_minutes = 15
    elif timeframe_str == '5m':
        tf_minutes = 5
    else:
        # Bilinmeyen bir time frame ise varsayılan 1 ay döndür
        return 60

    # 2. Toplam gereken dakika (500 mum * periyot)
    total_minutes = candle_count * tf_minutes

    # 3. Bir aydaki dakika sayısı (30 gün * 24 saat * 60 dk)
    minutes_in_month = 30 * 24 * 60

    # 4. Oranla ve %10 güvenlik payı ekle (Veri eksik gelmesin)
    months_needed = (total_minutes / minutes_in_month) * 1.1

    return months_needed


def prepare_model_input(df_ai, cnn_window=12, lstm_window=120, tr_window=120):
    """Model girdilerini hazırlar."""
    cnn_cols = [df_ai.columns.get_loc(c) for c in CNN_FEATURES if c in df_ai.columns]
    lstm_cols = [df_ai.columns.get_loc(c) for c in LSTM_FEATURES if c in df_ai.columns]
    tr_cols = [df_ai.columns.get_loc(c) for c in TR_FEATURES if c in df_ai.columns]
    
    # Normalize et
    mean = df_ai.mean()
    std = df_ai.std()
    if 'Log_Ret' in mean:
        mean['Log_Ret'] = 0.0
    std[std == 0] = 1.0
    df_normalized = (df_ai - mean) / std
    
    data = df_normalized.values
    max_window = max(cnn_window, lstm_window, tr_window)
    
    if len(data) < max_window:
        return None, None, None
        
    t = len(data)
    x_cnn = data[t - cnn_window:t, cnn_cols]
    x_lstm = data[t - lstm_window:t, lstm_cols]
    x_tr = data[t - tr_window:t, tr_cols]
    
    x_cnn = torch.tensor(x_cnn, dtype=torch.float32).unsqueeze(0).to(_DEVICE)
    x_lstm = torch.tensor(x_lstm, dtype=torch.float32).unsqueeze(0).to(_DEVICE)
    x_tr = torch.tensor(x_tr, dtype=torch.float32).unsqueeze(0).to(_DEVICE)
    
    return x_cnn, x_lstm, x_tr


def scan_market(timeframe, exchange_name="binance"):
    # 1. Kaç ay (float) gerektiğin hesapla
    months_to_fetch = calculate_needed_months(timeframe, candle_count=500)

    print(f"🛠️ {timeframe} için son 500 mum yaklaşık {months_to_fetch:.4f} ay ediyor.")

    ai_model = load_ai_model(timeframe)

    if ai_model is None:
        print("⚠️ Model YÜKLENEMEDİ! Tahminler 0.0 olacak.")

    all_pairs = get_all_pairs(exchange_name)
    market_data_storage = {}

    for pair in all_pairs:
        try:
            df = get_crypto_history(
                symbol=pair,
                timeframe=timeframe,
                months_back=months_to_fetch,
                exchange_name=exchange_name
            )

            if len(df) < 120:
                print(f"{pair} yetersiz veriye sahip. Atlanıyor.")
                continue

            if len(df) > 500:
                raw_df = df.tail(500)
            else:
                raw_df = df

            print(f"{pair} -> {len(raw_df)} mum alındı. İşleme hazır.")

            df_display, df_ai = prepare_dual_dataframes(raw_df)

            ai_prediction_value = 0.0

            if ai_model is not None:
                x_cnn, x_lstm, x_tr = prepare_model_input(df_ai)
                if x_cnn is not None:
                    # Eğer veride hala NaN veya Sonsuz varsa bu coini atla
                    if np.isnan(df_ai.values).any() or np.isinf(df_ai.values).any():
                        print(f"⚠️ {pair}: Veri bozuk (NaN/Inf tespit edildi), atlanıyor.")
                        continue
                    
                    with torch.no_grad():
                        pred_main, _, _, _ = ai_model(x_cnn, x_lstm, x_tr)
                    
                    # REVERSE SCALING
                    ai_prediction_value = pred_main.item() / 100.0
                    print(f"   🤖 AI Tahmini ({timeframe}): %{ai_prediction_value * 100:.4f}")

            export_df = df_display.copy()
            export_df.reset_index(inplace=True)
            export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Veriyi Sözlüğe Ekle
            market_data_storage[pair] = {
                "last_indicators": export_df.tail(5).to_dict(orient='records'),
                "ai_prediction": ai_prediction_value,
                "updated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            print(f"❌ {pair} hatası: {e}")
            continue

    if market_data_storage:
        print(f"\n💾 Veriler '{JSON_FILENAME}' dosyasına yazılıyor...")
        with open(JSON_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(market_data_storage, f, indent=4, ensure_ascii=False)

        print("🏁 İşlem Başarıyla Tamamlandı.")
    else:
        print("⚠️ Kaydedilecek veri bulunamadı.")


def scan_single_coin(coin_name, timeframe, exchange_name="binance"):
    coin_name = coin_name.upper()
    if "/" not in coin_name:
        pair = f"{coin_name}/USDT"
    else:
        pair = coin_name

    exchange_name = exchange_name.lower()
    ai_model = load_ai_model(timeframe)
    months_to_fetch = calculate_needed_months(timeframe, candle_count=500)

    try:
        df = get_crypto_history(
            symbol=pair,
            timeframe=timeframe,
            months_back=months_to_fetch,
            exchange_name=exchange_name
        )

        if len(df) < 120:
            return {pair: {"error": "Yetersiz veri", "candles": len(df)}}

        if len(df) > 500:
            raw_df = df.tail(500)
        else:
            raw_df = df

        print(f"✅ {pair} -> {len(raw_df)} mum alındı. Analiz yapılıyor...")

        df_display, df_ai = prepare_dual_dataframes(raw_df)
        ai_prediction_value = 0.0

        if ai_model is not None:
            x_cnn, x_lstm, x_tr = prepare_model_input(df_ai)
            if x_cnn is not None:
                if np.isnan(df_ai.values).any() or np.isinf(df_ai.values).any():
                    return {pair: {"error": "Veri bozuk (NaN/Inf tespit edildi)"}}

                with torch.no_grad():
                    pred_main, _, _, _ = ai_model(x_cnn, x_lstm, x_tr)
                
                ai_prediction_value = pred_main.item() / 100.0

        export_df = df_display.copy()
        export_df.reset_index(inplace=True)
        export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        result_data = {
            pair: {
                "last_indicators": export_df.tail(5).to_dict(orient='records'),
                "ai_prediction": ai_prediction_value,
                "updated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }

        return result_data

    except Exception as e:
        print(f"❌ Hata ({pair}): {e}")
        return {pair: {"error": str(e)}}


if __name__ == "__main__":
    scan_market("1h","binance")