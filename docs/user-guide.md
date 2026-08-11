# MarketMoE Kullanıcı Rehberi

## 1. Güvenli başlangıç

Kurulumdan sonra önce `market-moe doctor` çalıştırın. `healthy: true`; Python,
bağımlılıklar, universe dosyaları, localhost bind, CORS ve anahtarsız veri politikasının
geçtiğini gösterir. Doctor ağ çağrısı yapmaz.

MarketMoE'nin çalışma sırası şöyledir:

1. Universe ve enstrüman seçilir.
2. Ücretsiz public veya yerel veri alınır, canonical forma çevrilir ve kalite kontrol edilir.
3. Ortak feature pipeline çalışır.
4. Model train/validation üzerinde seçilir; test katı en sona kadar kilitlidir.
5. Candidate bundle incelenir ve kullanıcı isterse production'a terfi ettirir.
6. Scanner yalnız production bundle ve yerel cache ile sıralama yapar.
7. Backtest sinyali bir sonraki mumda, zorunlu maliyet modeliyle simüle eder.
8. Kullanıcı kararını isterse manuel paper portfolio'ya kaydeder.

## 2. Universe ve veri

`market-moe universes` yapılandırılmış evrenleri listeler. Universe dosyaları
`configs/universes` altındadır ve yeni bir YAML dosyası eklenerek genişletilebilir.
Kimlik yalnız ticker değildir: `equity:XNAS:AAPL` ile aynı ticker'ın başka listing'i
birbirinden ayrılır.

Örnek veri çağrıları:

```bash
market-moe fetch bist30 THYAO --timeframe 1d --days 1500
market-moe fetch europe_large_cap SAP --timeframe 1d --days 1500
market-moe fetch asia_large_cap 7203 --timeframe 1d --days 1500
market-moe fetch crypto_major ETH/USDT --timeframe 15m --days 180
```

Her indirme Parquet cache'e atomik yazılır ve DuckDB catalog'a satır sayısı, dönem,
provider, geçerlilik ve uyarılar kaydedilir. İntraday hisse geçmişi ücretsiz sağlayıcının
sınırları nedeniyle best-effort'tur; hisse için güvenilir kabul tabanı günlük bardır.

Yerel dosya fallback'i canonical veya en az tarih, open, high, low, close ve volume
sütunlarını içeren CSV/Parquet olabilir. Dosyalar yalnız izin verilen `data/imports`
kökünden okunur; path traversal reddedilir.

## 3. Model eğitimi ve yaşam döngüsü

```bash
market-moe train bist30 THYAO --timeframe 1d --window 32 --epochs 20
market-moe models list --status candidate
```

Bundle içinde model ağırlığı, manifest, feature schema/hash, train-only normalizasyon,
metrics, calibration, eğitim geçmişi ve model card birlikte bulunur. Schema veya
normalizasyon uyuşmazsa inference başlamaz.

Candidate model otomatik biçimde production olmaz. Model card, test metrikleri,
baseline ve walk-forward sonuçları incelendikten sonra:

```bash
market-moe models promote equity_moe 20260811T120000Z validated
market-moe models promote equity_moe 20260811T120000Z production
```

Yeni production sürümü, eski production sürümünü `validated` durumuna indirir. Dosyalar
silinmez; model yeniden üretilebilir kalır.

## 4. Scanner

```bash
market-moe scan bist30 --timeframe 1d
```

Scanner beklenen maliyet sonrası edge, calibrated yön olasılığı, volatilite, belirsizlik,
veri tazeliği ve expert ağırlık uyumunu birleştirir. Sonuç bir emir değildir. Her satırda
prediction ve ondan ayrı strategy signal sözleşmesi bulunur. Eksik cache, production model
veya schema uyumsuzluğu `skipped` içinde sebebiyle gösterilir.

## 5. Backtest V2

Backtestte temel kurallar:

- Sinyal `t` kapanışından sonra bilinir ve `t+1` açılışında uygulanır.
- Komisyon, yarım spread ve slippage açıkça raporlanır.
- Aynı mumda stop ve target görülürse alt timeframe yokken stop önce kabul edilir.
- USD dışı pozisyon için tarihsel FX serisi zorunludur.
- Adjusted bar ile explicit dividend/split birlikte verilmez; double counting reddedilir.
- Gross ve net getiri, benchmark, drawdown, turnover, exposure ve bootstrap aralığı ayrıdır.

HTML rapor tarayıcıda tek başına açılabilir. Backtest geçmiş performans simülasyonudur;
gelecekteki performansın garantisi değildir.

## 6. Manuel paper portfolio

Dashboard'da Paper Portfolio sayfası veya REST API kullanılabilir. Açılış ve kapanış
tarihi timezone içermelidir. Kullanıcı miktar, fiyat, FX, ücret, not ve isteğe bağlı
prediction snapshot'ını kendisi girer. Eksik değerleme fiyatı/FX olduğunda sistem sayı
uydurmaz ve değerlemeyi reddeder.

## 7. Dashboard ve API

`market-moe serve` sonrası temel uçlar:

- `GET /api/health`
- `GET /api/data/universes`
- `GET /api/data/catalog`
- `GET /api/models`
- `GET /api/scanner?universe_id=bist30&timeframe=1d`
- `POST /api/backtests`
- `GET|POST|DELETE /api/portfolio/trades`

API varsayılan olarak yalnız localhost'tadır. Uzak ağa açmak authentication sağlamaz;
bu nedenle ancak güvenilen, firewall ile sınırlı bir ortamda açık bayrakla yapılmalıdır.

## 8. Sorun giderme

- Boş veri: sembol/provider eşlemesini, timeframe kapsamını ve piyasa tatilini kontrol edin.
- Stale veri: provider yerine cache tarihi gösterilir; scanner stale sonucu nötrleştirir.
- Schema mismatch: aynı bundle'ın eğitiminde kullanılan feature pipeline sürümünü kullanın.
- CUDA hatası: `--device cpu` seçin.
- Model yok: önce eğitim yapın, metrikleri inceleyin ve production'a açıkça terfi ettirin.
