# MarketMoE

MarketMoE; kripto ve dünya hisse piyasaları için yerel çalışan, ücretsiz kamu verisi
kullanan bir araştırma ve karar-destek platformudur. Veriyi ortak bir sözleşmeye çevirir,
Mixture-of-Experts modellerini eğitir, maliyetli backtest yapar, piyasa evrenlerini tarar
ve tamamen manuel bir paper portfolio tutar.

Gerçek emir göndermez, broker hesabına bağlanmaz, private exchange endpointi çağırmaz ve
ücretli API anahtarı istemez. Çıktılar yatırım tavsiyesi değildir.

## Öne çıkanlar

- Kripto: CCXT üzerinden public spot OHLCV; varsayılan Binance, kimlik doğrulama yok.
- Global hisseler: yfinance best-effort; günlük Stooq ve CSV/Parquet fallback.
- BIST, ABD, Avrupa, Asya ve majör kripto için sürümlü YAML evrenleri.
- Tek canonical bar ve feature sözleşmesi; TA-Lib runtime bağımlılığı yok.
- CNN + GRU + Transformer uzmanları ve açıklanabilir softmax router.
- Getiri, yön, volatilite ve belirsizlik için multi-task model.
- Train-only normalizasyon, feature hash, purge/embargo ve kilitli test katı.
- Komisyon, spread, slippage, funding/borrow, FX ve corporate-action destekli Backtest V2.
- Local model registry, model card, calibration ve eğitim geçmişi içeren bundle.
- FastAPI + Jinja dashboard; varsayılan `127.0.0.1`, wildcard CORS yok.
- Manuel paper trade CRUD, PnL/değerleme ve CSV/JSON export.

## Gereksinimler

- Python 3.11 veya üstü; doğrulanan geliştirme ortamı Python 3.12'dir.
- CPU ile temel kullanım mümkündür. CUDA isteğe bağlıdır.
- Veri indirmek için internet; daha sonra cache üzerinden offline araştırma yapılabilir.

## Kurulum

Windows PowerShell:

```powershell
.\scripts\setup.ps1
```

Linux/macOS:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

Elle kurulum:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
market-moe doctor
```

Tam tekrarlanabilir ortam için `requirements.lock` içindeki sürümler kullanılabilir.

## Hızlı başlangıç

12 GB CUDA'lı Windows geliştirme bilgisayarında ücretsiz veriyi indirmek, gerekli modelleri
eğitmek ve kilitli test backtestlerini çalıştırmak için proje kökündeki dosyaya çift tıklayın:

```bat
RUN_FULL_CUDA_PIPELINE.bat
```

Akış kesintiden devam eder ve hiçbir gerçek emir göndermez. Ayrıntılar:
[Windows CUDA tam eğitim rehberi](docs/windows-cuda-pipeline.md).

```bash
# Ortam ve güvenlik politikası
market-moe doctor

# Evrenler
market-moe universes

# Ücretsiz günlük hisse verisi
market-moe fetch us_large_cap AAPL --timeframe 1d --days 2500

# Public kripto verisi
market-moe fetch crypto_major BTC/USDT --timeframe 1h --days 730

# Gerçek/yerel veriden candidate model
market-moe train us_large_cap AAPL --timeframe 1d --epochs 20

# Test ve model card incelendikten sonra açıkça production'a terfi
market-moe models list
market-moe models promote equity_moe VERSION production

# Cache + production model ile tarama
market-moe scan us_large_cap --timeframe 1d

# Dashboard
market-moe serve
```

Dashboard: [http://127.0.0.1:8080](http://127.0.0.1:8080). Yerel ağ bind'i ancak
`market-moe serve --host 0.0.0.0 --allow-network` ile bilinçli biçimde açılabilir.

## Backtest

Girdi canonical bar sütunlarını ve varsayılan olarak `signal` sütununu içerir. `t`
kapanışındaki sinyal en erken `t+1` açılışında simüle edilir.

```bash
market-moe backtest data/imports/example.parquet --output artifacts/backtests/example
```

Çıktı `metrics.json`, `equity.parquet`, `trades.parquet` ve standalone `report.html`
dosyalarını üretir. Maliyetler varsayılan olarak sıfır değildir; sıfır maliyetli sonuç
`idealized_no_cost` olarak işaretlenir ve production değerlendirmesi sayılmaz.

## Manuel paper portfolio

UI veya `/api/portfolio/trades` ile pozisyon kullanıcı tarafından manuel açılır/kapanır.

```bash
market-moe portfolio list
market-moe portfolio export artifacts/reports/paper-trades.csv
```

Bu defter broker bakiyesi değildir; gerçek para veya otomatik stop emri temsil etmez.

## Veri ve lisans notları

- CCXT adaptörü yalnız public OHLCV kullanır; borsa erişilebilirliği ve geçmiş kapsamı değişir.
- yfinance resmi bir borsa feed'i değildir ve kişisel araştırma kullanım koşulları gözetilmelidir.
- Stooq yalnız günlük fallback'tir; adjustment/provenance ekranda ve catalog'da saklanır.
- Universe listeleri güncel bileşenlerin geçmişte de aynı olduğu anlamına gelmez; survivorship
  bias uyarısı raporlarda açıktır.
- Kritik çalışma için yerel, lisanslı ve doğrulanmış CSV/Parquet verisi tercih edilebilir.

## Kalite kapıları

```bash
ruff check src tests scripts
mypy src/market_moe
pytest --cov=market_moe --cov-fail-under=70
market-moe train-smoke --asset-class crypto --epochs 1
market-moe train-smoke --asset-class equity --epochs 1
```

## Proje yapısı

```text
src/market_moe/       Aktif ürün paketi
configs/              Evren, model, veri ve strateji ayarları
tests/                Birim, entegrasyon, leakage ve güvenlik testleri
scripts/              Kurulum, CUDA pipeline ve data-health araçları
docs/                 Kullanıcı, doğrulama ve sınırlama belgeleri
data/                  Yerel cache/import (Git dışında)
artifacts/             Model/backtest/rapor çıktıları (Git dışında)
```

## Dokümantasyon

- [Kullanıcı rehberi](docs/user-guide.md)
- [Model ve doğrulama protokolü](docs/model-validation.md)
- [Bilinen sınırlamalar](docs/known-limitations.md)
- [Release checklist](docs/release-checklist.md)
- [Windows 12 GB CUDA tam eğitim akışı](docs/windows-cuda-pipeline.md)
- [Bağlayıcı uygulama planı](plan.md)
