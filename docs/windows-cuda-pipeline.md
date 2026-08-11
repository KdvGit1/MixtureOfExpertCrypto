# Windows 12 GB CUDA tam eğitim akışı

`RUN_FULL_CUDA_PIPELINE.bat`, yeni geliştirme bilgisayarında aşağıdaki araştırma zincirini
tek komutla çalıştırır:

1. Python sanal ortamını ve proje bağımlılıklarını kurar/günceller.
2. NVIDIA sürücüsünü ve CUDA destekli PyTorch'u kontrol eder; gerekirse resmi PyTorch
   CUDA wheel deposundan kurar.
3. Ücretsiz public endpointlerden kripto ve dünya hissesi verilerini indirir, doğrular ve
   Parquet cache'e yazar.
4. Aynı varlık sınıfındaki enstrümanları ortak, kronolojik havuzlarda eğitir.
5. Validation katında early stopping ve olasılık kalibrasyonu yapar.
6. Model seçimi sırasında görülmeyen kilitli test katında tahmin üretir.
7. Komisyon, spread, slippage ve uygun olduğunda funding/borrow maliyetlerini içeren
   `t+1` backtestleri çalıştırır.
8. Candidate model bundle'larını, enstrüman bazlı raporları, birleşik özeti ve logu kaydeder.

Akış gerçek emir göndermez, broker/private exchange API'sine bağlanmaz, ücretli API anahtarı
istemez ve modeli otomatik olarak production'a yükseltmez.

## Geliştirme bilgisayarı gereksinimleri

- Windows 10/11 x64
- Python 3.11 veya 3.12 (`py` launcher önerilir)
- Güncel NVIDIA sürücüsü ve `nvidia-smi` komutu
- En az 12 GB ekran kartı belleği (config güvenlik eşiği: 10 GB)
- En az 20 GB boş disk alanı
- İlk kurulum ve veri indirme için internet

BAT dosyası Python veya NVIDIA sistem sürücüsünü sessizce kurmaz; bunlar yönetici yetkisi ve
cihaza özel sürücü seçimi gerektirir. Proje bağımlılıklarını ve CUDA PyTorch paketini ise
yerel `.venv` içine otomatik kurar.

## Başlatma

Dosya Gezgini'nde proje kökündeki `RUN_FULL_CUDA_PIPELINE.bat` dosyasına çift tıklayın.
Terminalden çalıştırmak isterseniz:

```bat
RUN_FULL_CUDA_PIPELINE.bat
```

Bilgisayarın uykuya geçmesi eğitim süresince engellenir; ekranın kapanması normaldir. Windows
Update veya elektrik kesintisi süreci durdurursa aynı BAT dosyasını yeniden çalıştırın. İndirilen
veriler cache'den okunur, tamamlanmış işler atlanır ve yarım eğitim mevcut epoch checkpoint'inden
devam eder.

## Varsayılan işler

| İş | Evren | Periyot | Yön | Durum |
|---|---|---:|---|---|
| `crypto_15m` | BTC, ETH, BNB, SOL, XRP | 15 dakika | Long/short | Etkin |
| `crypto_1h` | BTC, ETH, BNB, SOL, XRP | 1 saat | Long/short | Etkin |
| `crypto_1d` | BTC, ETH, BNB, SOL, XRP | 1 gün | Long/short | Etkin |
| `global_equity_1d` | BIST30, ABD, Avrupa, Asya | 1 gün | Long-only | Etkin |
| `global_equity_1h` | Aynı hisse evrenleri | 1 saat | Long-only | Devre dışı |

Global hisse intraday işi varsayılan olarak kapalıdır; ücretsiz kaynaklar bütün ülkelerde yeterli,
uzun ve eşit kapsamlı saatlik geçmiş sağlamaz. Yerel/lisanslı veri eklendiğinde
`configs/pipelines/windows_cuda_12gb.yaml` içinden bilinçli biçimde etkinleştirilebilir.

## Seçenekler

Yalnız bir işi çalıştırma:

```bat
RUN_FULL_CUDA_PIPELINE.bat --only crypto_1h
```

Birden çok işi seçme:

```bat
RUN_FULL_CUDA_PIPELINE.bat --only crypto_1h --only global_equity_1d
```

Sadece mevcut cache ile çalışma:

```bat
RUN_FULL_CUDA_PIPELINE.bat --offline
```

Tamamlanmış işler için yeni model sürümleri oluşturup yeniden eğitim:

```bat
RUN_FULL_CUDA_PIPELINE.bat --force
```

CPU üzerinde yalnız geliştirme/smoke denemesi için Python entrypoint doğrudan kullanılabilir:

```bat
.venv\Scripts\python.exe scripts\windows_full_pipeline.py --allow-cpu --only crypto_1d
```

Tam üretim akışında `--allow-cpu` önerilmez.

## CUDA wheel ayarı

Varsayılan resmi wheel kanalı `https://download.pytorch.org/whl/cu128` adresidir. NVIDIA
sürücünüz başka bir resmi PyTorch CUDA kanalını gerektiriyorsa BAT'ı aynı terminalde şu şekilde
başlatabilirsiniz:

```bat
set MARKET_MOE_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
RUN_FULL_CUDA_PIPELINE.bat
```

Kurulum zaten yapılmışsa bakım amaçlı atlama değişkenleri kullanılabilir:

```bat
set MARKET_MOE_SKIP_SETUP=1
set MARKET_MOE_SKIP_CUDA_INSTALL=1
RUN_FULL_CUDA_PIPELINE.bat --only crypto_1h
```

## Çıktılar

- Ana durum: `artifacts/pipeline_runs/windows_cuda_full/state.json`
- Okunabilir özet: `artifacts/pipeline_runs/windows_cuda_full/summary.html`
- Canlı/tam log: `artifacts/pipeline_runs/windows_cuda_full/pipeline.log`
- Epoch checkpointleri: `artifacts/pipeline_runs/windows_cuda_full/checkpoints/`
- Candidate modeller: `artifacts/models/<model_id>/<version>/`
- Backtestler: `artifacts/backtests/<model_id>/<version>/`
- Birleşik backtest özeti: ilgili sürüm dizinindeki `summary.json`

Her enstrüman klasöründe `predictions.parquet`, `equity.parquet`, `trades.parquet`,
`metrics.json` ve `report.html` bulunur. Bundle içinde model ağırlığı, manifest, feature schema,
train-only normalizasyon, calibration, test metrikleri, model card ve eğitim geçmişi vardır.

## Hata davranışı

- Kriptoda Binance public veri erişimi başarısızsa public Bitget kaynağı denenir.
- Veri isteği üç kez artan bekleme ile denenir; yeterli cache varsa ağ hatasında cache kullanılır.
- Birkaç sembol başarısız olabilir; iş yalnız config'deki minimum enstrüman sayısı sağlanırsa sürer.
- CUDA bellek taşmasında batch `256 -> 128 -> 64 -> 32` küçültülür ve checkpoint'ten devam edilir.
- Bir işin hatası kaydedilir; kalan bağımsız işler çalışmaya devam eder.
- Süreç sonunda sıfır olmayan çıkış kodu, en az bir işin tamamlanamadığını ifade eder.

Modeli production'a almak ayrı ve bilinçli bir doğrulama adımıdır. Model card, kilitli test
metrikleri ve maliyetli backtestler incelenmeden promotion yapılmamalıdır.
