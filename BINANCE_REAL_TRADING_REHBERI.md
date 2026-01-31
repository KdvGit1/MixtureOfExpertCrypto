# 🤖 Binance Real Trading Kurulum Rehberi

Bu rehber, Telegram Auto-Trading Bot'u gerçek Binance hesabınızla nasıl kullanacağınızı anlatır.

---

## ⚠️ ÖNEMLİ UYARILAR

> **DİKKAT**: Bu bot gerçek para ile işlem yapabilir. Kaybetmeyi göze alamayacağınız parayı asla riske atmayın!

- Önce **TESTNET** modunda test edin
- İlk gerçek işlemlerde **küçük miktarlar** kullanın
- **Stop-loss** ayarlarını mutlaka aktif tutun
- API anahtarlarınızı **asla paylaşmayın**

---

## 📋 Adım 1: Binance API Anahtarı Oluşturma

### 1.1 Binance'e Giriş
1. [binance.com](https://www.binance.com) adresine gidin
2. Hesabınıza giriş yapın

### 1.2 API Yönetimine Git
1. Sağ üstte profil simgesine tıklayın
2. **API Management** (API Yönetimi) seçin
3. **Create API** (API Oluştur) butonuna tıklayın

### 1.3 API Ayarları
1. API'ye bir isim verin: `TradingBot`
2. **System Generated** (Sistem Oluşturmuş) seçin
3. 2FA doğrulamasını tamamlayın

### 1.4 İzinleri Ayarlayın

| İzin | Spot Trading | Futures Trading | Önerilen |
|------|-------------|-----------------|----------|
| ✅ Enable Reading | Gerekli | Gerekli | ✅ |
| ✅ Enable Spot & Margin Trading | Gerekli | - | ✅ (Spot için) |
| ✅ Enable Futures | - | Gerekli | ✅ (Futures için) |
| ❌ Enable Withdrawals | - | - | ❌ KAPALI TUTUN! |
| 🔒 IP Restrictions | Önerilen | Önerilen | Raspberry Pi IP'si |

> **UYARI**: Çekim (Withdrawal) iznini ASLA açmayın! Bot'un buna ihtiyacı yok.

### 1.5 Anahtarları Kaydedin
- **API Key**: `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
- **Secret Key**: `yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy`

> Secret Key sadece bir kez gösterilir, mutlaka kopyalayın!

---

## 📝 Adım 2: .env Dosyasını Yapılandırma

Proje klasöründeki `.env` dosyasını düzenleyin:

```env
# ============================================
# 🔑 BINANCE API AYARLARI (TESTNET)
# ============================================
# Demo mod için testnet anahtarları
BINANCE_API_KEY=your_testnet_api_key
BINANCE_API_SECRET=your_testnet_api_secret

# ============================================
# 🔑 BINANCE API AYARLARI (GERÇEK HESAP)
# ============================================
# /real confirm komutu ile aktif olur
# ⚠️ DİKKAT: Gerçek para ile işlem yapar!
REAL_API_KEY=your_real_api_key
REAL_API_SECRET=your_real_api_secret

# ============================================
# ⚠️ KRİTİK AYARLAR
# ============================================
# Bot varsayılan olarak DEMO modunda başlar
TESTNET=true
DRY_RUN=true

# ============================================
# 📊 TRADİNG AYARLARI
# ============================================
TRADING_MODE=spot
LEVERAGE=5
AUTO_TRADE_COINS=BTC,ETH,SOL
DEFAULT_TIMEFRAME=15m
MAX_POSITIONS=3
POSITION_PCT=0.10

# ============================================
# 🛡️ RİSK YÖNETİMİ
# ============================================
PREDICTION_THRESHOLD=0.005
STOP_LOSS_PCT=-0.03
MIN_PROFIT_TO_EXIT=0.003

# ============================================
# 🔒 GÜVENLİK LİMİTLERİ
# ============================================
DAILY_LOSS_LIMIT_PCT=-5.0
MAX_DAILY_TRADES=20
MIN_BALANCE_USDT=50.0

# ============================================
# 📱 TELEGRAM AYARLARI
# ============================================
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

---

## 🚀 Adım 3: Test Aşamaları

### Aşama 1: Testnet + Dry Run (Tamamen Güvenli)
```env
TESTNET=true
DRY_RUN=true
```
- Trade simülasyonu
- Gerçek veri, sahte işlem
- Risk: **SIFIR**

### Aşama 2: Testnet + Real Orders (Test Ortamı)
```env
TESTNET=true
DRY_RUN=false
```
- Binance Testnet'te gerçek siparişler
- Sahte para ile gerçek mekanik
- Risk: **SIFIR**

### Aşama 3: Mainnet + Dry Run (Canlı Takip)
```env
TESTNET=false
DRY_RUN=true
```
- Gerçek piyasa verisi
- İşlem simülasyonu
- Risk: **SIFIR** (işlem yapılmaz)
- Birkaç gün çalıştırıp `/history` ile sonuçları inceleyin

### Aşama 4: Mainnet + Real Trading (GERÇEK PARA!)
```env
TESTNET=false
DRY_RUN=false
```
- ⚠️ **GERÇEK PARA İLE İŞLEM!**
- Küçük miktarlarla başlayın
- Sürekli takip edin

---

## 📊 Adım 4: Önerilen Başlangıç Ayarları

| Ayar | Yeni Başlayanlar | Orta Seviye | Deneyimli |
|------|------------------|-------------|-----------|
| `POSITION_PCT` | 0.05 (5%) | 0.10 (10%) | 0.15-0.20 |
| `MAX_POSITIONS` | 2 | 3 | 5 |
| `STOP_LOSS_PCT` | -0.02 (-2%) | -0.03 (-3%) | -0.05 (-5%) |
| `DAILY_LOSS_LIMIT_PCT` | -3% | -5% | -10% |
| `MAX_DAILY_TRADES` | 10 | 20 | 50 |
| `PREDICTION_THRESHOLD` | 0.008 | 0.005 | 0.003 |

**Örnek**: $500 ile başlıyorsanız:
- `POSITION_PCT=0.10` → Her trade $50
- `MAX_POSITIONS=2` → Maksimum $100 açık pozisyon
- `STOP_LOSS_PCT=-0.02` → Trade başına maksimum $1 kayıp

---

## 🔧 Adım 5: Bot'u Başlatma

```bash
# Klasöre git
cd /path/to/MixtureOfExpertCrypto

# Bot'u başlat
python telegram_trading_bot.py
```

### Telegram Komutları

| Komut | Açıklama |
|-------|----------|
| `/start` | Bot'u aktif et |
| `/stop` | Bot'u duraklat |
| `/status` | Durum ve açık pozisyonlar |
| `/demo` | 🎮 Demo moduna geç (testnet + dry-run) |
| `/real confirm` | 💰 Gerçek trading moduna geç |
| `/history` | Trade geçmişi ve istatistikler |
| `/history export` | CSV'ye aktar |
| `/safety` | Güvenlik durumu ve limitler |
| `/safety reset` | Güvenlik kilidini sıfırla |
| `/spot` | Spot moduna geç |
| `/futures 5` | Futures moduna geç (5x) |
| `/coins` | Aktif coinleri göster |
| `/help` | Tüm komutlar |

---

## 🛡️ Güvenlik İpuçları

1. **API Anahtarı Güvenliği**
   - Withdrawal iznini AÇMAYIN
   - IP kısıtlaması kullanın
   - Anahtarları paylaşmayın

2. **Risk Yönetimi**
   - Küçük başlayın, yavaş büyütün
   - Stop-loss'u aktif tutun
   - Günlük kayıp limitini ayarlayın

3. **İzleme**
   - Telegram bildirimlerini takip edin
   - `/history` ile performansı kontrol edin
   - Log dosyalarını inceleyin

4. **Acil Durum**
   - `/stop` komutu bot'u durdurur
   - Binance'den API'yi devre dışı bırakabilirsiniz

---

## 📈 Trade Geçmişi

Bot her trade'i otomatik olarak kaydeder:
- **Konum**: `trade_history/trades.json`
- **Telegram**: `/history` komutu
- **CSV Export**: `/history export`

### Kaydedilen Bilgiler:
- Trade ID, Coin, Giriş/Çıkış fiyatları
- Miktar, Zaman damgaları
- P&L (% ve USDT)
- AI tahmini ve güven skoru
- Trading modu, leverage
- Dry run durumu

---

## ❓ Sık Sorulan Sorular

**S: Testnet'te işlem yapamıyorum?**
C: Binance Testnet için ayrı API key gerekir: https://testnet.binance.vision/

**S: "Insufficient balance" hatası alıyorum?**
C: Testnet hesabınıza test USDT yüklemeniz gerekiyor.

**S: Bot durdu ama pozisyonlar açık?**
C: Binance web/app'ten manuel kapatabilirsiniz.

**S: Günlük kayıp limiti aşıldı mesajı?**
C: Bot güvenlik için durdu. Ertesi gün otomatik sıfırlanır veya `/start` ile tekrar başlatın.

---

## 📞 Destek

- Log dosyaları: `bot_logs/` klasörü
- Trade geçmişi: `trade_history/` klasörü
- Hata durumunda log'ları inceleyin

---

**İyi tradeler! 🚀📈**
