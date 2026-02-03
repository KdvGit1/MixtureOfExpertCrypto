# 🤖 Multi-User Bot Kurulum Rehberi (Raspberry Pi)

Bu rehber, tek Raspberry Pi üzerinde birden fazla kullanıcının bağımsız bot instance'ları çalıştırmasını sağlar.

---

## 📋 Ön Gereksinimler

- Raspberry Pi 4 (8GB RAM önerilir)
- Raspberry Pi OS (64-bit)
- Python 3.9+
- İnternet bağlantısı

---

## 🚀 Adım 1: Projeyi Raspberry Pi'a Kopyalama

```bash
# SSH ile bağlan
ssh pi@RASPBERRY_IP

# Projeyi kopyala (USB veya SCP ile)
scp -r /path/to/MixtureOfExpertCrypto pi@RASPBERRY_IP:/home/pi/

# Veya Git ile çek
cd /home/pi
git clone https://github.com/USERNAME/MixtureOfExpertCrypto.git
```

---

## 🔧 Adım 2: Python Ortamını Kurma

```bash
cd /home/pi/MixtureOfExpertCrypto

# Virtual environment oluştur
python3 -m venv ~/venv

# Aktive et
source ~/venv/bin/activate

# Gereksinimleri kur
pip install --upgrade pip
pip install -r requirements.txt

# PyTorch CPU-only (Raspberry Pi için)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 👤 Adım 3: Kullanıcı Ekleme

Her arkadaşınız için ayrı kullanıcı oluşturun:

```bash
# İnteraktif kurulum
python setup_user.py
```

Bu script şunları soracak:
- Kullanıcı adı (örn: kaan, ali, mehmet)
- Telegram Bot Token (@BotFather'dan alınır)
- Telegram Chat ID (@userinfobot'tan alınır)
- Binance API anahtarları (testnet ve/veya mainnet)
- Trade edilecek coinler
- Risk ayarları

### 📁 Kullanıcılar Nereye Kaydedilir?

Her kullanıcı `users/` klasörüne kaydedilir:

```
users/
├── kaan/
│   └── config.json    ← Kaan'ın tüm ayarları burada
├── ali/
│   └── config.json
└── mehmet/
    └── config.json
```

### ✅ Kayıtlı Kullanıcıları Kontrol Etme

```bash
# Tüm kayıtlı kullanıcıları listele
python run_user.py --list

# Çıktı:
# 📋 Kayıtlı Kullanıcılar:
#    • kaan
#    • ali
#    • mehmet
```

### Her Arkadaş İçin Telegram Botu Oluşturma:

1. Telegram'da @BotFather'a git
2. `/newbot` yaz
3. Bot adı ve username ver
4. Token'ı kopyala (bu token'ı setup_user.py'ye gireceksin)

### Chat ID Bulma:

1. Telegram'da @userinfobot'a git
2. `/start` yaz
3. "Id" yazan sayıyı kopyala

### 🔄 Tekrar: Her arkadaş için setup_user.py'yi çalıştır

```bash
python setup_user.py   # Kaan için
python setup_user.py   # Ali için
python setup_user.py   # Mehmet için
# ... her arkadaş için tekrar
```

## ✅ Adım 4: Botu Test Etme

```bash
# Aktive et (eğer değilse)
source ~/venv/bin/activate

# Kaan'ın botunu test et
python run_user.py kaan

# Başka terminalde Ali'nin botunu test et
python run_user.py ali
```

Her bot başladığında Telegram'dan `/start` mesajı gelebilir.

---

## ⚙️ Adım 5: Systemd Service Kurulumu

Botların otomatik başlaması için systemd servisleri kurun:

```bash
# Service dosyasını kopyala
sudo cp services/bot@.service /etc/systemd/system/

# Systemd'yi yenile
sudo systemctl daemon-reload

# Her kullanıcı için servisi aktive et
sudo systemctl enable bot@kaan
sudo systemctl enable bot@ali
sudo systemctl enable bot@mehmet
# ... diğer kullanıcılar için tekrarla

# Servisleri başlat
sudo systemctl start bot@kaan
sudo systemctl start bot@ali
sudo systemctl start bot@mehmet
```

### Service Komutları:

```bash
# Durum kontrol
sudo systemctl status bot@kaan

# Durdur
sudo systemctl stop bot@kaan

# Yeniden başlat
sudo systemctl restart bot@kaan

# Logları göster
sudo journalctl -u bot@kaan -f

# Tüm bot loglarını göster
sudo journalctl -u 'bot@*' -f
```

---

## 📁 Kullanıcı Dizin Yapısı

```
MixtureOfExpertCrypto/
├── users/
│   ├── kaan/
│   │   ├── config.json          # Kaan'ın ayarları
│   │   ├── trade_history/       # Kaan'ın trade geçmişi
│   │   │   └── trades.json
│   │   └── bot_logs/            # Kaan'ın log dosyaları
│   │       └── bot_kaan_20260203.log
│   ├── ali/
│   │   ├── config.json
│   │   ├── trade_history/
│   │   └── bot_logs/
│   └── mehmet/
│       └── ...
├── run_user.py                  # Kullanıcı başlatma scripti
├── setup_user.py                # Kullanıcı ekleme scripti
└── telegram_trading_bot.py      # Ana bot
```

---

## 🛡️ Güvenlik Önerileri

1. **API Anahtarları**: Sadece gerekli izinleri verin (Spot/Futures Trading)
2. **IP Kısıtlama**: Binance'te API anahtarlarını Raspberry Pi IP'sine kısıtlayın
3. **Testnet**: Önce testnet ile test edin
4. **Dry-Run**: Gerçek paraya geçmeden önce simülasyon modunda test edin

---

## 🔧 Sorun Giderme

### Bot Başlamıyor
```bash
# Logları kontrol et
sudo journalctl -u bot@KULLANICI -n 50

# Manuel çalıştır ve hatayı gör
source ~/venv/bin/activate
python run_user.py KULLANICI
```

### RAM Yetersiz
```bash
# RAM kullanımını kontrol et
free -h

# Her bot için model cache'i azalt (config.json'da)
"max_loaded_models": 3
```

### Telegram Bağlantı Hatası
- Bot token'ının doğru olduğundan emin ol
- Chat ID'nin doğru olduğundan emin ol
- İnternet bağlantısını kontrol et

---

## 📊 Kaynak Kullanımı

| Kullanıcı Sayısı | Tahmini RAM | CPU Kullanımı |
|------------------|-------------|---------------|
| 1                | 500MB       | ~10%          |
| 3                | 1.5GB       | ~25%          |
| 5                | 2.5GB       | ~40%          |
| 6+               | 3GB+        | ~50%          |

> ⚠️ 8GB RAM ile maksimum 6-8 kullanıcı önerilir.

---

## 📞 Yardım

Sorularınız için: @KaanTG (veya grup admin'i)
