import ccxt
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta

_EXCHANGE_CACHE={}

def get_exchange_instance(exchange_name="binance"):
    """
    Borsa nesnesini oluşturur veya varsa hafızadan getirir (Singleton).
    """
    exchange_name = exchange_name.lower()

    # 1. Eğer hafızada varsa DİREKT ONU DÖNDÜR
    if exchange_name in _EXCHANGE_CACHE:
        return _EXCHANGE_CACHE[exchange_name]

    # 2. Yoksa YENİ OLUŞTUR
    print(f"🔌 {exchange_name.upper()} bağlantısı ilk kez kuruluyor... (Piyasalar yükleniyor)")

    if exchange_name == "binance":
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'futures'}
        })
    elif exchange_name == "bitget":
        exchange = ccxt.bitget({
            'enableRateLimit': True,
            'options': {'defaultType': 'futures'}
        })
    else:
        raise ValueError(f"{exchange_name} is not supported yet. Please appeal to developers.")

    # Piyasaları yükle (Bu işlem ağırdır, artık sadece 1 kere yapılacak)
    exchange.load_markets()

    # 3. Hafızaya kaydet
    _EXCHANGE_CACHE[exchange_name] = exchange
    return exchange

# ==========================================
# 1. VERİ ÇEKME KATMANI
# ==========================================
def get_crypto_history(symbol, timeframe, months_back,exchange_name="binance"):
    """Borsadan ham mum verilerini çeker."""
    exchange = get_exchange_instance(exchange_name)

    now = datetime.now()
    start_date = now - timedelta(days=30 * months_back)
    since = int(start_date.timestamp() * 1000)

    print(f"🚀 BAŞLIYOR: {symbol} - {timeframe}")
    all_candles = []

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not candles:
                break

            all_candles += candles
            last_candle_time = candles[-1][0]
            since = last_candle_time + 1

            # İlerleme göstergesi
            if len(all_candles) % 5000 == 0:
                print(f"📦 Çekilen: {len(all_candles)} mum...")

            if last_candle_time >= exchange.milliseconds():
                print("✅ Veri çekimi tamamlandı.")
                break

        except Exception as e:
            print(f"❌ Hata: {e}")
            break

    df = pd.DataFrame(all_candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df.drop(columns=['Timestamp'], inplace=True)
    return df

# ==========================================
# 2. İNDİKATÖR HESAPLAMA KATMANI
# ==========================================
def add_smart_indicators(df):
    """
    Hem AI için oranları hem de İnsanlar için gerçek değerleri hesaplar.
    """
    df = df.copy()

    # --- HAM İNDİKATÖRLER (İnsanlar ve Grafik İçin) ---
    # Bunları JSON'a koyacağız ki kullanıcı "SMA kaç?" diye bakabilsin.

    # Hareketli Ortalamalar
    df['SMA_50_Val'] = talib.SMA(df['Close'], timeperiod=50)  # Örn: 94500.5
    df['EMA_200_Val'] = talib.EMA(df['Close'], timeperiod=200)  # Örn: 92100.0

    # Bollinger Bands (Ham Değerler)
    upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20)
    df['BB_Upper_Val'] = upper
    df['BB_Middle_Val'] = middle
    df['BB_Lower_Val'] = lower

    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['ATR_Val'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # MACD (Ham Değerler)
    macd, macdsignal, macdhist = talib.MACD(df['Close'])
    df['MACD_Val'] = macd
    df['MACD_Signal_Val'] = macdsignal
    df['MACD_Hist_Val'] = macdhist

    # --- AI İÇİN DÖNÜŞÜMLER (Feature Engineering) ---
    # Bu sütunlar modele girecek, kullanıcıya göstermeye gerek yok (Kafa karıştırır)

    # Fiyatın ortalamalara uzaklığı (Oran)
    df['Dist_SMA_50'] = (df['Close'] - df['SMA_50_Val']) / df['SMA_50_Val']
    df['Dist_EMA_200'] = (df['Close'] - df['EMA_200_Val']) / df['EMA_200_Val']

    # Bollinger %B ve Genişlik
    df['BB_PctB'] = (df['Close'] - lower) / (upper - lower)
    df['BB_Width'] = (upper - lower) / middle

    # MACD Normalize
    df['MACD_Norm'] = df['MACD_Val'] / df['Close']

    # ATR Yüzdesi
    df['ATR_Pct'] = df['ATR_Val'] / df['Close']

    # Hacim Analizi
    df['Vol_SMA_20'] = talib.SMA(df['Volume'], timeperiod=20)
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA_20']
    df['Vol_Spike'] = (df['Vol_Ratio'] > 2.0).astype(int)

    # Zaman Döngüleri
    df['Hour_Sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['Day_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Hedef (Log Return)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

    return df

# ==========================================
# 3. AYRIŞTIRMA VE KAYDETME KATMANI (YENİ)
# ==========================================
def prepare_dual_dataframes(df):
    """
    Veriyi 3 parçaya ayırır:
    1. Display (Kullanıcı için ham indikatörler)
    2. AI (Model için normalize veriler)
    """
    df_calculated = add_smart_indicators(df)
    df_calculated = df_calculated.replace([np.inf, -np.inf], np.nan)
    df_clean = df_calculated.dropna()
    print(f"🧹 Temizlik: İlk {len(df_calculated) - len(df_clean)} satır (NaN) silindi.")

    # A) DISPLAY DATA (Kullanıcıya Gösterilecekler)
    # Fiyatlar + Ham İndikatör Değerleri
    display_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', # Temel
        'RSI',                                    # Popüler
        'SMA_50_Val', 'EMA_200_Val',              # Ortalamalar
        'BB_Upper_Val', 'BB_Lower_Val',           # Bollinger Sınırları
        'MACD_Val', 'MACD_Signal_Val',            # Trend Gücü
        'ATR_Val'                                 # Volatilite (Dolar bazında)
    ]
    df_display = df_clean[display_cols].copy()

    # B) AI DATA (Modele Girecekler)
    ai_cols = [
        'Log_Ret',
        'RSI',
        'Dist_SMA_50',
        'Dist_EMA_200',
        'BB_PctB',
        'BB_Width',
        'MACD_Norm',
        'ATR_Pct',
        'Vol_Ratio',
        'Vol_Spike',
        'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
    ]
    df_ai = df_clean[ai_cols].copy()
    df_ai['RSI'] = df_ai['RSI'] / 100.0

    return df_display, df_ai

def workflow_runner(coin_name,desired_month, desired_timeframes):
    """Tüm süreci yöneten ana fonksiyon."""

    for tf in desired_timeframes:
        # 1. Veriyi Çek
        df_raw = get_crypto_history(f"{coin_name.upper()}/USDT", tf, desired_month)

        # 2. Hesapla ve İkiye Böl
        df_orig, df_ai = prepare_dual_dataframes(df_raw)

        # 3. Kontrol Et (Satır sayıları eşit mi?)
        if len(df_orig) == len(df_ai):
            print(f"✅ Eşleşme Başarılı: İki tabloda da {len(df_orig)} satır var.")
        else:
            print("❌ HATA: Satır sayıları tutmuyor!")

        # 4. Kaydet
        file_orig = f"{coin_name}_{desired_month}Ay_{tf}_ORIGINAL.csv"
        file_ai = f"{coin_name}_{desired_month}Ay_{tf}_AI_Ready.csv"

        df_orig.to_csv(file_orig)
        df_ai.to_csv(file_ai)

        print(f"💾 Kaydedildi:\n  -> {file_orig}\n  -> {file_ai}")
        print("-" * 40)

if __name__ == "__main__":
    workflow_runner("BTC",180, ('1h',))