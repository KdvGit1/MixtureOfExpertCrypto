# MarketMoE Nihai Ürün Uygulama Planı

> Durum: Aktif ve bağlayıcı uygulama planı  
> Plan sürümü: 1.0  
> Oluşturulma tarihi: 2026-08-11  
> Ürün tipi: Local-first, çoklu piyasa araştırma, analiz ve karar destek platformu  
> Uygulama sahibi: Proje sahibi ve Codex  

---

## 1. Bu belgenin amacı

Bu dosya, mevcut `MixtureOfExpertCrypto` araştırma kodunu sadeleştirerek kripto ve dünya hisse piyasalarını destekleyen, kurulabilir, test edilebilir ve son kullanıcıya teslim edilebilir bir ürüne dönüştürmek için izlenecek bağlayıcı plandır.

Kodlama bu plana göre yürütülecektir. Her uygulama aşaması tamamlandıkça ilgili kontrol kutuları işaretlenecek, doğrulama sonuçları bu dosyaya veya bağlantılı raporlara eklenecek ve plandan önemli bir sapma yapılması gerekiyorsa önce “Karar Günlüğü” bölümünde gerekçesi kaydedilecektir.

Bu planın amacı daha fazla deneysel dosya üretmek değil; tek bir veri hattı, tek bir model sözleşmesi, tek bir backtest motoru ve tek bir kullanıcı uygulaması bulunan nihai ürün oluşturmaktır.

---

## 2. Değiştirilemez ürün kısıtları

Aşağıdaki maddeler kullanıcı tarafından belirlenmiştir ve planın geri kalanından daha yüksek önceliğe sahiptir:

1. Ücretli piyasa verisi API’leri kullanılmayacaktır.
2. Uygulama hiçbir borsaya veya brokera gerçek alım-satım emri göndermeyecektir.
3. API anahtarı gerektiren ücretli veri sağlayıcıları zorunlu bağımlılık olmayacaktır.
4. Ürün kripto piyasaları ile sınırlı kalmayacak, yapılandırılmış global hisse evrenlerini de destekleyecektir.
5. Uygulama local-first çalışacaktır; kullanıcı verileri ve model artefaktları varsayılan olarak yerel diskte tutulacaktır.
6. Backtest, paper portfolio ve karar desteği bulunacak; gerçek işlem yürütme bulunmayacaktır.
7. Eski kod davranışları ölçülmeden ve karşılaştırma testleri geçmeden çalışan dosyalar kaldırılmayacaktır.
8. Test seti model veya strateji parametrelerini optimize etmek için kullanılmayacaktır.
9. Kârlılık iddiası yalnızca yön doğruluğuna veya tek bir başarılı backtest’e dayandırılmayacaktır.
10. Ücretsiz kaynakların lisans, gecikme, kapsam ve güvenilirlik sınırları kullanıcı arayüzünde açıkça belirtilecektir.

---

## 3. Nihai ürün tanımı

Ürünün çalışma adı `MarketMoE` olacaktır. Depo adı daha sonra değiştirilebilir; ilk refaktör sırasında depo yeniden adlandırılmayacaktır.

MarketMoE aşağıdaki işleri yapan bir piyasa araştırma ve karar destek platformudur:

- Kripto ve global hisse sembolleri için ücretsiz kaynaklardan OHLCV verisi toplar.
- Veriyi yerel cache ve Parquet veri gölünde saklar.
- Piyasa takvimi, timezone ve şirket aksiyonlarını hesaba katar.
- Ortak ve piyasa-özel teknik özellikler üretir.
- Kripto ve hisse senetleri için ayrı fakat aynı çekirdeği paylaşan MoE modelleri eğitir.
- Model tahminlerini açık birimlerle ve model sürümüyle sunar.
- Piyasaları tarar ve fırsatları risk düzeltilmiş skorlarla sıralar.
- Maliyetleri hesaba katan güvenli backtest ve walk-forward değerlendirmesi yapar.
- Kullanıcının manuel olarak işlem ekleyebildiği paper portfolio sağlar.
- Sonuçları web dashboard, rapor ve dışa aktarılabilir JSON/CSV/Parquet dosyalarıyla gösterir.
- Veri/model sağlığını ve tahminlerin güncelliğini takip eder.

### 3.1 Nihai kullanıcı akışları

Son ürün aşağıdaki akışları eksiksiz destekleyecektir:

1. Kullanıcı uygulamayı tek komutla kurar ve başlatır.
2. Dashboard ilk açılışta örnek evrenleri gösterir.
3. Kullanıcı `BIST 30`, `ABD büyük şirketler`, `Avrupa büyük şirketler`, `Asya büyük şirketler` veya kripto evreni seçer.
4. Uygulama ücretsiz kaynaklardan veriyi indirir ve local cache’e kaydeder.
5. Kullanıcı veri sağlığını, son güncelleme zamanını ve eksik veri uyarılarını görür.
6. Kullanıcı bir sembolün fiyat grafiğini, özelliklerini, model tahminini, expert ağırlıklarını ve belirsizliğini inceler.
7. Kullanıcı tüm evreni taratır ve sembolleri risk düzeltilmiş model skoruna göre sıralar.
8. Kullanıcı seçilen model ve strateji için out-of-sample backtest çalıştırır.
9. Kullanıcı farklı maliyet ve risk ayarlarını validation döneminde karşılaştırır.
10. Kullanıcı test dönemini yalnızca final değerlendirmede açar.
11. Kullanıcı manuel paper işlem ekler, kapatır ve performansını izler.
12. Kullanıcı hiçbir aşamada gerçek borsa emri oluşturamaz veya gönderemez.

### 3.2 Kapsam dışı özellikler

Aşağıdakiler nihai ürün kapsamı dışındadır:

- Otomatik broker veya borsa emirleri
- API anahtarıyla gerçek hesap bakiyesi okuma
- Otomatik portföy rebalancing
- Yüksek frekanslı işlem
- Tick/order-book tabanlı HFT stratejileri
- Gerçek zaman garantili profesyonel piyasa verisi
- Her dünyadaki menkul kıymeti otomatik keşfetme garantisi
- Vergi hesaplama veya yatırım danışmanlığı
- Çok kiracılı SaaS altyapısı
- Ücretli veri sağlayıcıya sessiz fallback
- Ücretsiz veri kaynaklarının yeniden dağıtılması

---

## 4. Ücretsiz veri stratejisi

### 4.1 Kaynak önceliği

#### Kripto

Birincil kaynak: CCXT üzerinden borsaların public OHLCV uçları.

- Varsayılan borsa: Binance public market data
- İkinci adaptör: Bitget public market data
- Kimlik doğrulama: Yok
- Private endpoint kullanımı: Yasak
- Emir endpointleri: Uygulamada bulunmayacak
- Kaynak dokümantasyonu: <https://github.com/ccxt/ccxt/wiki/manual>

CCXT public piyasa verisi hesap veya API anahtarı olmadan kullanılabilir. Uygulama her borsanın rate limit değerine uymalı ve henüz kapanmamış son mumu eğitim/backtest verisine dahil etmemelidir.

#### Global hisseler ve ETF’ler

Birincil adaptör: `yfinance`.

- API anahtarı: Yok
- Kullanım amacı: Kişisel araştırma ve local analiz
- Birincil zaman dilimi: `1d`
- Best-effort zaman dilimi: `1h`
- Şirket aksiyonları: split ve temettü verisi mümkün olduğunda alınacak
- Kaynak uyarısı: `yfinance`, Yahoo Finance ile bağlı veya Yahoo tarafından garanti edilen resmî bir SDK değildir ve veri kullanımı kişisel kullanım koşullarına tabidir.
- Kaynak: <https://pypi.org/project/yfinance/>

İkincil/fallback adaptör: Stooq günlük tarihsel veri.

- API anahtarı: Yok
- Öncelik: Günlük veri ve yfinance hata durumlarında sembol bazlı fallback
- Kullanım: Yalnızca izin verilen local araştırma ve cache senaryoları
- Intraday garantisi: Yok
- Veri yeniden dağıtımı: Yapılmayacak

Üçüncü adaptör: Kullanıcının kendi CSV veya Parquet dosyasını içe aktarması.

Bu adaptör ücretsiz kaynaklar kesildiğinde ürünün tamamen kullanılamaz hale gelmesini engeller. İçe aktarılan verinin şeması doğrulanır, kaynağı `local_import` olarak etiketlenir ve kullanıcıya veri sorumluluğu açıkça gösterilir.

#### Piyasa takvimleri

`exchange_calendars` kullanılacaktır.

- 50’den fazla piyasa takvimi için local Python tanımları sunar.
- NYSE, Nasdaq ile aynı takvim ailesi, Xetra, LSE ve diğer desteklenen MIC kodları kullanılabilir.
- BIST takvimi mevcut sürümde doğrulanacak; eksik veya hatalıysa proje içinde testli özel takvim eklenecektir.
- Kaynak: <https://github.com/gerrymanoim/exchange_calendars>

### 4.2 Ücretsiz veriyle ilgili ürün sınırları

Ücretsiz veri kullanımı nedeniyle aşağıdaki sınırlamalar açık kabul edilir:

- Global hisselerde gerçek zaman garantisi verilmeyecektir.
- Intraday geçmiş derinliği sembole ve sağlayıcıya göre değişebilir.
- Bazı borsalarda volume veya adjusted price alanları eksik olabilir.
- Delist edilmiş sembollerin tam point-in-time evreni ücretsiz kaynaklarda bulunmayabilir.
- Survivorship bias tamamen ortadan kaldırılamıyorsa backtest raporunda yüksek görünürlüklü uyarı gösterilecektir.
- Sağlayıcı rate limit veya format değiştirdiğinde adaptör güncellemesi gerekebilir.
- Ürün ticari dağıtıma açılmadan önce Yahoo Finance ve diğer veri kaynaklarının lisans koşulları yeniden incelenecektir.

### 4.3 Sağlayıcı seçim ve fallback sırası

Hisse verisi için çözümleme sırası:

1. Local cache güncelse cache kullan.
2. `yfinance` ile istenen aralığı çek.
3. Günlük veri isteğiyse ve yfinance başarısızsa Stooq adaptörünü dene.
4. Local import kayıtlıysa onu kullan.
5. Veri yoksa sessizce sahte veri üretme; kullanıcıya açık hata ve kapsam raporu göster.

Kripto verisi için çözümleme sırası:

1. Local cache güncelse cache kullan.
2. Yapılandırılmış primary CCXT exchange’i kullan.
3. Aynı sembol secondary exchange’de mevcutsa ve kullanıcı izin vermişse fallback yap.
4. Farklı borsalardan gelen verileri kaynak etiketi olmadan birleştirme.
5. Hiç veri yoksa açık hata üret.

---

## 5. Ürün mimarisi

### 5.1 Katmanlar

Sistem aşağıdaki yönlü bağımlılık kuralına uyacaktır:

```text
apps -> services -> domain
               -> data
               -> features
               -> models
               -> backtest
               -> portfolio
```

Alt katmanlar uygulama/UI katmanını import edemez. Model kodu FastAPI, Telegram veya HTML katmanını bilmez. Veri sağlayıcıları model sınıflarını bilmez. Backtest, canlı olmayan simüle execution arayüzü üzerinden stratejiyi çalıştırır.

### 5.2 Hedef klasör yapısı

```text
MixtureOfExpertCrypto/
├── src/
│   └── market_moe/
│       ├── __init__.py
│       ├── cli.py
│       ├── settings.py
│       ├── domain/
│       │   ├── instruments.py
│       │   ├── bars.py
│       │   ├── predictions.py
│       │   ├── signals.py
│       │   ├── portfolio.py
│       │   └── errors.py
│       ├── data/
│       │   ├── protocols.py
│       │   ├── catalog.py
│       │   ├── cache.py
│       │   ├── quality.py
│       │   ├── calendars.py
│       │   ├── corporate_actions.py
│       │   └── providers/
│       │       ├── ccxt_provider.py
│       │       ├── yfinance_provider.py
│       │       ├── stooq_provider.py
│       │       └── local_provider.py
│       ├── features/
│       │   ├── schema.py
│       │   ├── pipeline.py
│       │   ├── indicators.py
│       │   ├── common.py
│       │   ├── crypto.py
│       │   ├── equity.py
│       │   └── normalization.py
│       ├── models/
│       │   ├── experts.py
│       │   ├── router.py
│       │   ├── moe.py
│       │   ├── losses.py
│       │   ├── bundle.py
│       │   ├── registry.py
│       │   └── inference.py
│       ├── training/
│       │   ├── dataset.py
│       │   ├── splits.py
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   ├── calibration.py
│       │   └── experiments.py
│       ├── backtest/
│       │   ├── engine.py
│       │   ├── execution.py
│       │   ├── costs.py
│       │   ├── accounting.py
│       │   ├── metrics.py
│       │   ├── walk_forward.py
│       │   └── reports.py
│       ├── scanner/
│       │   ├── service.py
│       │   ├── ranking.py
│       │   └── filters.py
│       ├── portfolio/
│       │   ├── paper.py
│       │   ├── valuation.py
│       │   └── persistence.py
│       ├── services/
│       │   ├── data_service.py
│       │   ├── model_service.py
│       │   ├── analysis_service.py
│       │   ├── backtest_service.py
│       │   └── health_service.py
│       └── apps/
│           └── web/
│               ├── app.py
│               ├── routes/
│               ├── schemas/
│               ├── templates/
│               └── static/
├── configs/
│   ├── app.yaml
│   ├── data.yaml
│   ├── features/
│   ├── models/
│   ├── strategies/
│   └── universes/
│       ├── crypto_major.yaml
│       ├── bist30.yaml
│       ├── us_large_cap.yaml
│       ├── europe_large_cap.yaml
│       └── asia_large_cap.yaml
├── data/
│   ├── raw/
│   ├── normalized/
│   ├── features/
│   └── catalog.duckdb
├── artifacts/
│   ├── models/
│   ├── experiments/
│   ├── backtests/
│   └── reports/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── golden/
│   ├── property/
│   └── fixtures/
├── scripts/
│   ├── setup.ps1
│   ├── setup.sh
│   └── windows_full_pipeline.py
├── docs/
│   ├── architecture.md
│   ├── data-sources.md
│   ├── model-card.md
│   ├── validation.md
│   └── user-guide.md
├── pyproject.toml
├── requirements.lock
├── README.md
└── plan.md
```

### 5.3 Depolama yaklaşımı

- OHLCV ve feature tabloları Parquet olarak tutulacaktır.
- DuckDB yalnızca local katalog, sorgu ve özet metrikler için kullanılacaktır.
- Model dosyaları ve büyük veri dosyaları Git’e eklenmeyecektir.
- Her veri dosyasının yanında veya katalog kaydında provider, çekim zamanı, sembol, timeframe, timezone ve adjustment bilgisi tutulacaktır.
- Atomic write kullanılacaktır: önce geçici dosyaya yaz, doğrula, sonra hedef dosyayla değiştir.
- Cache anahtarı provider + instrument_id + timeframe + date range + adjustment mode içerecektir.

---

## 6. Domain sözleşmeleri

### 6.1 Instrument kimliği

Ticker tek başına kimlik olarak kullanılmayacaktır.

Örnekler:

```text
crypto:BINANCE:BTCUSDT
equity:XIST:THYAO
equity:XNAS:AAPL
equity:XETR:SAP
equity:XTKS:7203
```

`Instrument` alanları:

```text
instrument_id: str
symbol: str
display_name: str
asset_class: crypto | equity | etf | index | fx
exchange_mic: str
currency: str
timezone: str
calendar: str
country: str | null
sector: str | null
provider_symbols: dict[str, str]
active: bool
tradable: bool
metadata_as_of: datetime
```

`tradable` burada gerçek emir anlamına gelmez. Paper portfolio/backtest içinde işlem simülasyonu yapılabilir olup olmadığını ifade eder.

### 6.2 Bar sözleşmesi

Tüm sağlayıcılar veriyi aşağıdaki canonical şemaya dönüştürecektir:

```text
instrument_id
timeframe
open_time_utc
close_time_utc
open
high
low
close
volume
vwap | null
currency
session_type
is_adjusted
provider
provider_symbol
ingested_at_utc
quality_flags
```

Kurallar:

- UTC kolonları timezone-aware olacaktır.
- OHLC değerleri pozitif olmalıdır.
- `low <= open/close <= high` doğrulanmalıdır.
- Timestamp tekrarları reddedilmeli veya deterministik biçimde çözülmelidir.
- Henüz kapanmamış mum `quality_flags` ile işaretlenmeli ve eğitim/backtest dışında tutulmalıdır.
- Hisse günlük verisinde adjustment modu dosya ve model manifestinde açıkça yazmalıdır.

### 6.3 Prediction sözleşmesi

Model çıktısı yalnızca tek bir belirsiz float olamaz.

```text
prediction_id
instrument_id
asset_class
timeframe
as_of_utc
horizon
target_type
expected_log_return
expected_return_pct
probability_up
predicted_volatility
uncertainty
confidence
expert_weights
raw_model_output
model_id
model_version
feature_schema_hash
normalization_id
data_freshness_seconds
warnings
```

Kurallar:

- Kullanıcıya gösterilen getiri de-normalize edilmiş gerçek birimde olacaktır.
- Z-score yalnızca `raw_model_output` veya diagnostic alanında bulunabilir.
- `confidence`, yön oylaması gibi geçici formüllerden değil kalibre edilmiş olasılık ve belirsizlikten türetilmelidir.
- Tahminin üretildiği veri timestamp’i kaydedilmelidir.
- Model ve feature schema uyumsuzsa inference reddedilmelidir.

### 6.4 Signal sözleşmesi

Model tahmini ile strateji sinyali ayrılacaktır.

```text
signal: strong_long | long | neutral | reduce | short | strong_short
score: [-1, 1]
reason_codes: list[str]
expected_edge_after_cost
risk_level
model_prediction_id
strategy_version
```

Hisse ürününün varsayılan kullanıcı modu long/neutral olacaktır. Short sinyalleri araştırma/backtest görünümü olarak açılabilir; paper portfolio varsayılanında kapalıdır.

---

## 7. Piyasa evrenleri

### 7.1 Destek modeli

Ücretsiz kaynaklarla tüm dünya sembollerini güvenli ve otomatik keşfetmek garanti edilemez. Bu nedenle final ürün, test edilmiş ve versiyonlanmış evren dosyalarıyla gelecektir. Kullanıcı YAML üzerinden sembol ekleyebilir.

### 7.2 Varsayılan evrenler

#### Kripto major

- Yüksek likiditeli USDT çiftleri
- İlk sürümde mevcut 20 coin listesi gözden geçirilerek korunur
- Delist veya düşük likidite kontrolü veri çekiminde yapılır

#### BIST 30

- Yahoo sembolleri `.IS` uzantısıyla tutulur
- MIC: `XIST`
- Para birimi: TRY
- Üyelik listesi tarih damgalı YAML olarak saklanır
- Üyelik değişiklikleri otomatik varsayılmaz

#### ABD büyük şirketler

- S&P 100 benzeri, yüksek likiditeli, elle doğrulanmış sembol listesi
- NYSE/Nasdaq MIC eşlemesi sembol metadata’sında tutulur
- ETF’ler ayrı kategoride işaretlenir

#### Avrupa büyük şirketler

- Xetra, LSE ve Euronext’ten test edilmiş sınırlı sembol seti
- EUR ve GBP dönüşümleri ayrı tutulur
- Aynı şirketin farklı listing’leri tek kimlik sanılmaz

#### Asya büyük şirketler

- Tokyo ve desteklenen diğer büyük borsalardan test edilmiş sembol seti
- Lunch break ve timezone davranışı calendar ile doğrulanır

### 7.3 Evrene eklenme kalite kapısı

Bir sembol varsayılan evrene ancak şu koşullarla eklenir:

- En az tanımlı minimum tarihsel günlük veri bulunuyor.
- Son veri tarihi beklenen seansa uygun.
- OHLC doğrulaması geçiyor.
- Currency ve exchange metadata’sı mevcut.
- Kurumsal aksiyon/adjustment davranışı test edilmiş.
- Sembol için provider suffix mapping doğrulanmış.
- Sembol test fixture’ına eklenmiş.

---

## 8. Feature engineering tasarımı

### 8.1 Ortak özellikler

Tüm asset class’larda kullanılabilecek, ölçekten bağımsız özellikler:

- 1, 3, 5, 10, 20 bar log return
- rolling volatility
- downside volatility
- true range ve ATR yüzdesi
- RSI
- MACD / fiyat
- Bollinger bandwidth ve percent-B
- hareketli ortalamaya normalize uzaklık
- volume ratio
- volume z-score
- candle body/wick oranları
- rolling high/low uzaklığı
- trend slope
- realized skew/kurtosis için yeterli pencere varsa diagnostic özellikler
- veri kalite/missingness maskeleri

### 8.2 Kripto özellikleri

- 24 saat ve haftanın günü cyclic özellikleri
- exchange/provider kimliği
- mümkünse public funding rate
- mümkünse public open interest
- spot/perpetual market türü
- weekend işareti
- volatility regime
- volume/liquidity regime

Funding ve open interest her exchange’de güvenilir değilse zorunlu model girdisi olmayacak; missing mask ile opsiyonel feature olacaktır.

### 8.3 Hisse özellikleri

- session progress
- minutes since open / minutes to close
- opening window / closing window
- overnight gap
- previous close return
- index-relative return
- sector-relative return, metadata mevcutsa
- market breadth, evren yeterliyse
- rolling dollar volume
- illiquidity proxy
- earnings proximity yalnızca ücretsiz ve güvenilir veri varsa
- ex-dividend/split proximity, corporate action verisi varsa
- base currency’ye göre FX return

`Hour_Sin / 24` hisse modelinde kullanılmayacaktır. Hisse zaman özellikleri yerel borsa seansına göre hesaplanacaktır.

### 8.4 Feature şema versiyonlama

- Her feature listesi sıralı ve isimlendirilmiş manifest olarak kaydedilir.
- Manifest SHA-256 hash’i model bundle’a yazılır.
- Feature kodundaki anlam değişikliği schema version artırır.
- Model yalnızca eğitildiği schema hash’iyle inference yapar.
- Train, validation, test ve inference aynı `FeaturePipeline` sınıfını kullanır.
- Notebook içinde kopyalanmış feature kodu bulunmaz.

### 8.5 TA-Lib bağımlılığının kaldırılması

Final ürün native TA-Lib kurulumuna zorunlu olmayacaktır.

- Gerekli indikatörler pandas/numpy tabanlı proje içi fonksiyonlarla uygulanır.
- Mevcut TA-Lib sonuçlarıyla toleranslı golden test yapılır.
- Sapmalar dokümante edilir.
- TA-Lib opsiyonel doğrulama bağımlılığı olabilir, runtime zorunluluğu olamaz.

---

## 9. Model tasarımı

### 9.1 Model ailesi

İki ana model ailesi bulunacaktır:

1. `crypto_moe`
2. `equity_moe`

Mimari kodları ortak olacak, ağırlıklar ve domain özellikleri ayrı olacaktır. İlk final sürümde tek bir ortak dünya modeli kullanılmayacaktır.

### 9.2 Timeframe’ler

Kripto:

- `15m`
- `1h`
- `1d`

Hisse:

- Birincil: `1d`
- İkincil/best-effort: `1h`

Global hisselerde `15m`, ücretsiz veri geçmişi ve mikro yapı sorunları nedeniyle final sürümün zorunlu kabul kriteri değildir.

### 9.3 MoE uzmanları

Model üç ana uzmana sahip olacaktır:

1. Yerel hareket uzmanı
   - Temporal CNN
   - kısa pencere
   - candle ve ani momentum özellikleri

2. Trend/hafıza uzmanı
   - LSTM veya GRU
   - orta pencere
   - trend, volume ve volatilite akışı

3. Rejim/bağlam uzmanı
   - Transformer encoder
   - uzun pencere
   - rejim, seans ve piyasa bağlamı

Router, uzman ağırlıklarını son piyasa bağlamından üretir. Router çıktıları softmax ile toplam 1 olacak şekilde sınırlandırılır ve inference sonucunda kullanıcıya gösterilir.

### 9.4 Çıkış görevleri

Model multi-task olacaktır:

- Regresyon: ileri dönem beklenen log return
- Sınıflandırma: maliyet eşiğini aşan yukarı/aşağı hareket olasılığı
- Volatilite: ileri dönem gerçekleşen volatilite tahmini

Toplam loss açık katsayılı olacaktır. Katsayılar config ve model manifestinde saklanacaktır.

### 9.5 Eğitim mekanizmaları

Varsayılan stabil sürüm:

- Auxiliary heads: Açık
- Scheduled branch dropout: Warm-up sonrasında açık
- Router load-balancing regularization: Açık
- Gradient clipping: Açık
- Early stopping: Validation metriğiyle açık
- Automatic mixed precision: CUDA varsa açık
- Deterministic seed ve manifest: Açık
- Contrastive learning: Deneysel feature flag
- OGM-GE: Deneysel feature flag, varsayılan kapalı

OGM-GE, branch dropout ve auxiliary loss kombinasyonu ablation ile faydasını kanıtlamadan üretim modelinin varsayılanı olmayacaktır.

### 9.6 Model bundle

Her model tek klasörde aşağıdakilerle saklanacaktır:

```text
model.pt
manifest.json
feature_schema.json
normalization.json
metrics.json
model_card.md
calibration.json
training_history.parquet
```

Manifest en az şunları içerir:

- model_id ve version
- asset class
- timeframe ve horizon
- desteklenen feature schema hash
- train/validation/test tarih aralıkları
- kullanılan semboller
- data provider ve adjustment mode
- random seed
- git commit
- Python ve bağımlılık sürümleri
- hyperparameters
- loss config
- test sonuçları
- bilinen sınırlamalar

### 9.7 Model registry

- Registry local dosya sistemi ve DuckDB indeksinden oluşur.
- `candidate`, `validated`, `production`, `retired` durumları bulunur.
- Yalnızca acceptance gate geçen model `production` olabilir.
- UI model sürümünü seçebilir fakat varsayılan yalnızca production modeldir.
- Model dosyaları Git dışında tutulur.

---

## 10. Eğitim ve doğrulama protokolü

### 10.1 Veri bölme

Chronological bölme zorunludur:

- Train: En eski dönem
- Validation: Train sonrasındaki dönem
- Test: En yeni ve kilitli dönem

Global/cross-asset modelde tüm semboller tarih bazında ortak cutoff ile bölünür. Aynı tarihte bir sembol train, başka sembol test tarafında olmayacak şekilde split manifesti tutulur.

### 10.2 Purge ve embargo

- Split sınırında maksimum feature window kadar purge uygulanır.
- Target horizon kadar embargo uygulanır.
- Overlapping label örneklerinin train ve validation/test tarafına sızması engellenir.
- Purge/embargo uzunlukları manifestte kaydedilir.

### 10.3 Walk-forward

En az üç walk-forward fold desteklenir:

```text
Fold 1: Train A -> Validate B
Fold 2: Train A+B -> Validate C
Fold 3: Train A+B+C -> Test D
```

Fold sayısı veri uzunluğuna göre config edilebilir. Tek bir dönem başarısı yeterli sayılmaz.

### 10.4 Hyperparameter seçimi

- Optimizasyon yalnızca train/validation üzerinde yapılır.
- Test fold sonuçları hyperparameter seçim koduna verilmez.
- İlk sürümde küçük ve kontrollü arama alanı kullanılır.
- Optuna açık kaynak olarak kullanılabilir; ücretli servis kullanılmaz.
- Çok sayıda deneme sonucu oluşan multiple-testing riski raporlanır.
- En iyi deneme değil, fold’lar arasında kararlı parametre tercih edilir.

### 10.5 Baseline’lar

Her model aşağıdakilerle karşılaştırılır:

- Sıfır getiri tahmini
- Son getiri yönünü tekrar eden naive momentum
- Basit moving-average trend
- Buy-and-hold
- Aynı işlem sıklığında random signal
- Volatility-scaled momentum

Model baseline’dan yalnızca accuracy ile değil, maliyet sonrası edge ve risk metrikleriyle ayrışmalıdır.

### 10.6 Model metrikleri

- MAE ve Huber loss
- Sign accuracy
- Balanced accuracy
- ROC AUC ve PR AUC
- Brier score
- Calibration error
- IC/Pearson ve rank IC/Spearman
- Tahmin quantile’larına göre gerçekleşen return
- Expert ablation
- Feature permutation importance
- Piyasa, sembol ve rejim bazında performans

---

## 11. Backtest V2 tasarımı

### 11.1 Look-ahead önleme

- `t` mumunun close bilgisiyle üretilen sinyal en erken `t+1` mumunun open fiyatında işlenir.
- Corporate action bilgisi yalnızca bilindiği tarihten itibaren kullanılabilir.
- Günlük mumun high/low bilgisi aynı gün açılış emrine karar vermek için kullanılamaz.
- Model normalizasyonu yalnızca train verisinden fit edilir.
- Backtest sırasında feature pipeline gelecek satırlara erişemez.

### 11.2 Execution simülasyonu

Gerçek emir gönderilmeyecektir. Execution yalnızca simülasyondur.

Desteklenen simüle emir türleri:

- next-open market
- next-bar VWAP yaklaşımı, veri varsa
- limit order için konservatif fill modeli
- stop-loss
- take-profit
- time exit
- signal reversal exit

### 11.3 Aynı mum SL/TP belirsizliği

Aynı mum içinde hem stop hem hedef görülürse:

1. Alt timeframe veri varsa sıra alt timeframe’den belirlenir.
2. Alt timeframe yoksa konservatif olarak stop önce gerçekleşmiş kabul edilir.
3. Sonuç `intrabar_ambiguous` sayacıyla raporlanır.

### 11.4 Maliyet modeli

Tüm backtestlerde maliyet modeli zorunludur:

- commission
- bid/ask spread tahmini
- slippage
- crypto funding, veri varsa
- equity FX conversion etkisi
- equity short borrow maliyeti, short araştırması açıksa
- turnover maliyeti

Maliyet `0` yapılabilir fakat rapor bu sonucu “idealized/no-cost” olarak işaretler ve production değerlendirmesine sokmaz.

### 11.5 Portföy muhasebesi

- Configurable base currency: Varsayılan USD, kullanıcı TRY seçebilir.
- Nakit ve pozisyonlar ayrı tutulur.
- FX dönüşümleri tarihsel kurla yapılır; veri yoksa pozisyon simülasyonu reddedilir veya açık uyarı üretilir.
- Split pozisyon adetlerine uygulanır.
- Temettü seçilen adjustment/accounting moduna göre iki kez sayılmayacak şekilde işlenir.
- Equity ve crypto sonuçları ayrı ve birleşik raporlanabilir.

### 11.6 Risk kuralları

Backtest ve paper portfolio için:

- maksimum açık pozisyon
- sembol başına maksimum ağırlık
- asset class başına maksimum ağırlık
- piyasa/ülke başına maksimum ağırlık
- günlük kayıp limiti
- portföy drawdown limiti
- volatilite tabanlı position sizing
- stale data kilidi
- eksik fiyat kilidi
- aşırı gap kontrolü

Varsayılan risk değerleri yatırım önerisi olarak sunulmayacak; kullanıcıya simülasyon ayarı olduğu açıkça gösterilecektir.

### 11.7 Backtest metrikleri

- Net ve gross return
- CAGR, süre yeterliyse
- Max drawdown ve drawdown süresi
- Sharpe, Sortino, Calmar
- Profit factor
- Win rate
- Ortalama kazanç/kayıp
- Expectancy
- Turnover
- Exposure
- Trade count
- Ortalama holding süresi
- Fee/spread/slippage toplamı
- Benchmark excess return
- Regime breakdown
- Instrument contribution
- Bootstrap confidence intervals

---

## 12. Scanner ve karar destek

### 12.1 Tarama akışı

1. Evreni yükle.
2. Verinin güncel ve kaliteli olduğunu doğrula.
3. Feature pipeline çalıştır.
4. Uyumlu production modeli registry’den yükle.
5. Prediction sözleşmesini üret.
6. Strateji katmanında maliyet sonrası edge hesapla.
7. Risk ve veri kalitesi filtrelerini uygula.
8. Sonuçları sıralayıp açıklama kodlarıyla kaydet.

### 12.2 Ranking skoru

Ranking yalnızca expected return olmayacaktır. Örnek bileşenler:

- calibrated direction probability
- expected return after cost
- predicted volatility cezası
- uncertainty cezası
- data freshness
- liquidity score
- expert agreement
- regime confidence

Skor formülü config ve strateji versiyonuyla saklanacaktır.

### 12.3 Kullanıcıya gösterilecek açıklamalar

Her tahmin için:

- tahmin ufku
- beklenen getiri
- yön olasılığı
- tahmin belirsizliği
- expert ağırlıkları
- piyasa rejimi
- veri son güncelleme zamanı
- risk uyarıları
- model version
- backtest kapsamı
- ücretli/gerçek zaman veri kullanılmadığı uyarısı

---

## 13. Paper portfolio

Paper portfolio tamamen manueldir.

Desteklenen işlemler:

- Kullanıcı manuel pozisyon açar.
- Giriş tarihi, fiyatı, miktarı ve notu girer.
- Uygulama güncel ücretsiz veriyle değerleme yapar.
- Kullanıcı pozisyonu manuel kapatır.
- Sanal nakit ve gerçekleşen/gerçekleşmemiş PnL hesaplanır.
- Model tahminiyle manuel karar eşleşmesi raporlanır.
- CSV/JSON dışa aktarım sağlanır.

Olmayacak işlemler:

- Broker bağlantısı
- Gerçek bakiye okuma
- Otomatik emir
- Otomatik stop emri
- Gerçek para durumunu temsil ettiği iddiası

---

## 14. Web uygulaması

### 14.1 Teknoloji

- Backend: FastAPI
- Template: Jinja2
- Frontend: Mevcut HTML/CSS/vanilla JS yaklaşımı sadeleştirilerek devam eder
- Build zorunluluğu: Yok
- API: JSON REST
- Varsayılan bind: `127.0.0.1`
- Public network erişimi: Varsayılan kapalı

### 14.2 Sayfalar

1. Ana görünüm
   - sistem sağlığı
   - son veri güncellemesi
   - aktif evrenler
   - en güçlü ve en zayıf tarama sonuçları

2. Market scanner
   - asset class, ülke, borsa, timeframe filtreleri
   - model skoru
   - risk ve veri kalite uyarıları

3. Enstrüman detayı
   - OHLCV grafik
   - indikatörler
   - tahmin ve expert ağırlıkları
   - geçmiş tahmin doğruluğu
   - benchmark karşılaştırması

4. Model laboratuvarı
   - model registry
   - model card
   - fold metrikleri
   - calibration
   - ablation ve feature importance

5. Backtest
   - config seçimi
   - validation/test ayrımı
   - maliyet ayarları
   - equity curve ve drawdown
   - trade tablosu

6. Paper portfolio
   - manuel pozisyon ekleme/kapatma
   - değerleme
   - PnL ve attribution

7. Veri yöneticisi
   - indirilen semboller
   - kapsam ve eksik veri
   - cache yenileme
   - local CSV/Parquet import

8. Sistem ve dokümantasyon
   - ücretsiz veri uyarıları
   - model sınırlamaları
   - sürüm bilgisi
   - health checks

### 14.3 API endpoint grupları

```text
/api/health
/api/instruments
/api/universes
/api/data/status
/api/data/refresh
/api/predictions
/api/scanner
/api/models
/api/backtests
/api/paper-portfolio
/api/reports
```

Endpointler thin controller olacak; iş mantığı service katmanında bulunacaktır.

---

## 15. Güvenlik ve gizlilik

- Gerçek exchange/broker API key alanları nihai config şemasından kaldırılacaktır.
- Dış kullanıcı config dosyalarındaki secret alanları okunmayacak veya taşınmayacaktır.
- `.env` yalnızca opsiyonel uygulama ayarları için kullanılabilir; secrets Git’e giremez.
- `users/*/config.json`, local data, models ve logs Gitignore’a alınacaktır.
- FastAPI varsayılan olarak `127.0.0.1` üzerinde çalışacaktır.
- Wildcard CORS kaldırılacaktır.
- Arbitrary model path yükleme endpointi kaldırılacak veya yalnızca registry kimliği kabul edecektir.
- Local import dosyaları path traversal’a karşı doğrulanacaktır.
- PyTorch modelleri `weights_only=True` ile ve yalnızca trusted local registry’den yüklenecektir.
- Uygulama hiçbir private exchange endpointini çağırmayacaktır.
- Loglar token, path dışı kullanıcı bilgisi veya kişisel veri içermeyecektir.

---

## 16. Bağımlılık ve kurulum

### 16.1 Temel runtime bağımlılıkları

Planlanan ana paketler:

- fastapi
- uvicorn
- pydantic
- pydantic-settings
- pandas
- numpy
- pyarrow
- duckdb
- yfinance
- pandas-datareader veya proje içi Stooq adaptörü
- ccxt
- exchange-calendars
- torch
- scikit-learn
- scipy
- xgboost
- optuna
- matplotlib
- plotly
- jinja2
- httpx
- tenacity
- pyyaml

### 16.2 Geliştirme bağımlılıkları

- pytest
- pytest-asyncio
- pytest-cov
- hypothesis
- ruff
- mypy

### 16.3 Kurulum hedefleri

- `pyproject.toml` tek bağımlılık kaynağı olacaktır.
- `requirements.lock` tekrar üretilebilir kurulum için tutulacaktır.
- Windows için `scripts/setup.ps1` oluşturulacaktır.
- Linux/macOS için `scripts/setup.sh` oluşturulacaktır.
- Kurulum native TA-Lib gerektirmeyecektir.
- CUDA opsiyonel olacaktır; CPU modunda tüm temel özellikler çalışacaktır.

### 16.4 CLI hedefleri

```text
market-moe doctor
market-moe data refresh --universe bist30 --timeframe 1d
market-moe data validate --universe bist30
market-moe train --asset-class equity --timeframe 1d
market-moe evaluate --model equity-1d-v1
market-moe backtest --config configs/strategies/equity_long.yaml
market-moe scan --universe us_large_cap --timeframe 1d
market-moe serve
```

`doctor` komutu Python, bağımlılıklar, veri klasörleri, model registry ve ücretsiz sağlayıcı erişimini kontrol edecektir.

---

## 17. Test stratejisi

### 17.1 Unit testler

- Instrument ID oluşturma ve parse
- Timezone dönüşümleri
- Bar schema doğrulaması
- OHLC invariants
- Calendar session hesapları
- İndikatör fonksiyonları
- Normalizasyon yalnızca train fit
- Window slicing
- Prediction de-normalizasyonu
- Cost hesapları
- Portfolio accounting
- Metric hesapları
- Registry uyumluluk kontrolleri

### 17.2 Integration testler

- CCXT public veri çekimi küçük örnek
- yfinance küçük örnek
- Stooq fallback
- Cache write/read roundtrip
- Feature pipeline end-to-end
- Model bundle save/load
- Inference service
- Backtest small fixture
- FastAPI endpoint smoke test
- Paper portfolio persistence

Network integration testleri varsayılan CI’da mock kullanmalı; gerçek network testleri ayrı işaretlenmelidir.

### 17.3 Golden testler

- Mevcut feature pipeline’dan seçilmiş BTC/ETH çıktıları
- Mevcut modelden sabit örnek inference
- Mevcut ve yeni feature sonuçlarının tolerans karşılaştırması
- Aynı küçük veri üzerinde deterministik backtest
- Dashboard API response şemaları

### 17.4 Property testler

- Fiyat ölçeği değişince oran özelliklerinin değişmemesi
- Hiçbir işlem yokken equity’nin sabit kalması
- Pozisyon kapatılınca cash + realized PnL korunumu
- Maliyet arttıkça net PnL’nin artmaması
- Gelecek bar değiştiğinde geçmiş feature’ın değişmemesi
- Expert weights toplamının 1 olması

### 17.5 Test kalite kapıları

- Yeni core modüllerde minimum %80 line coverage hedefi
- Domain, cost ve accounting modüllerinde minimum %90 coverage hedefi
- Tüm syntax ve import smoke testleri geçmeli
- Ruff hatası olmamalı
- Kritik mypy hatası olmamalı
- Test fixture’larında gerçek secret olmamalı

---

## 18. Aktif modül haritası

### 18.1 Veri ve feature kodu

| Sorumluluk | Aktif modül |
|---|---|
| Public market verisi | `data/providers/*` ve `services/data_service.py` |
| Canonical bar/cache | `data/quality.py`, `data/cache.py`, `data/catalog.py` |
| Ortak feature üretimi | `features/pipeline.py` ve `features/indicators.py` |

### 18.2 Model ve eğitim

| Sorumluluk | Aktif modül |
|---|---|
| Expert/router/MoE | `models/experts.py`, `models/router.py`, `models/moe.py` |
| Eğitim/checkpoint | `training/trainer.py` ve `training/pipeline.py` |
| Bundle/registry | `models/bundle.py` ve `models/registry.py` |
| Calibration | `models/calibration.py` |

### 18.3 Backtest

| Sorumluluk | Aktif modül |
|---|---|
| Event engine | `backtest/engine.py` |
| Maliyet/risk/muhasebe | `backtest/costs.py`, `risk.py`, `accounting.py` |
| Metrik ve rapor | `backtest/metrics.py` ve `backtest/reports.py` |
| Walk-forward | `backtest/walk_forward.py` |

### 18.4 Uygulamalar

| Aktif giriş | Hedef |
|---|---|
| `market-moe serve` | Web scanner routes + services |
| `market-moe portfolio` | Yalnız manuel paper portfolio |
| `RUN_FULL_CUDA_PIPELINE.bat` | Ücretsiz veri, eğitim ve backtest otomasyonu |

### 18.5 Artefaktlar

- Model, veri ve backtest çıktıları `data/` ile `artifacts/` altında ve Git dışında tutulacaktır.
- Önceki ürünün runtime ve artefaktları kullanıcı onayıyla çalışma ağacından kaldırılmıştır.
- Büyük dosyalar yeni commitlerden önce `.gitignore` kapsamına alınacaktır.

---

## 19. Uygulama aşamaları

Her aşama bir öncekinin kabul kriteri tamamlanmadan başlatılmayacaktır. Paralel yapılabilecek küçük işler olsa bile ana branch’e birleşme sırası korunacaktır.

### Aşama 0 — Canonical davranışın dondurulması

Amaç: Aktif ürün sözleşmelerini deterministik testlerle korumak.

- [x] Git çalışma ağacı ve dosya envanteri çıkarılacak.
- [x] Canonical bar, feature, prediction ve bundle sözleşmeleri tanımlanacak.
- [x] Crypto ve equity için deterministik test fixture’ları seçilecek.
- [x] Feature leakage ve normalization kontrolleri yazılacak.
- [x] Model inference ve backtest entegrasyon testleri yazılacak.
- [x] Bilinen ölçek/eşik tutarsızlıkları yeni domain sözleşmeleriyle giderilecek.

Kabul kriteri:

- Canonical davranışı doğrulayan crypto/equity fixture seti var.
- Model dosyaları ve feature şemaları eşleştirilmiş.
- Bilinen hatalar “korunacak davranış” olarak yanlışlıkla onaylanmamış.

### Aşama 1 — Proje temeli ve güvenlik

Amaç: Tekrar üretilebilir ve güvenli geliştirme ortamı.

- [x] `pyproject.toml` oluşturulacak.
- [x] Eksik bağımlılıklar eklenecek.
- [x] Lock dosyası üretilecek.
- [x] `src/market_moe` paketi kurulacak.
- [x] Pytest, Ruff ve mypy yapılandırılacak.
- [x] Windows/Linux setup scriptleri yazılacak.
- [x] README başlangıç kurulumu yazılacak.
- [x] `.gitignore` data, artifacts, models, user config ve secrets için düzeltilecek.
- [x] Public CORS ve `0.0.0.0` varsayımları kaldırılacak.
- [x] Gerçek trading secret alanları yeni configten çıkarılacak.
- [x] `market-moe doctor` ilk sürümü yazılacak.

Kabul kriteri:

- Temiz makinede dokümante komutlarla kurulum yapılabiliyor.
- `pytest`, `ruff`, import smoke test geçiyor.
- Uygulama hiçbir exchange/broker secret istemiyor.

### Aşama 2 — Canonical domain ve veri katmanı

Amaç: Kripto ve hisseleri aynı canonical veri sözleşmesine getirmek.

- [x] Instrument, Bar ve DataQuality modelleri yazılacak.
- [x] Provider protocol tanımlanacak.
- [x] CCXT public provider yazılacak.
- [x] yfinance provider yazılacak.
- [x] Stooq daily fallback yazılacak.
- [x] Local CSV/Parquet provider yazılacak.
- [x] Parquet cache ve DuckDB catalog yazılacak.
- [x] Rate limit, retry ve timeout politikası eklenecek.
- [x] Exchange calendar servisi eklenecek.
- [x] Corporate action şeması eklenecek.
- [x] Data quality raporu eklenecek.
- [x] Varsayılan universe YAML dosyaları oluşturulacak.

Kabul kriteri:

- BTC/USDT, AAPL, THYAO.IS ve en az bir Avrupa/Asya sembolü canonical Bar şemasına çekilebiliyor.
- Provider kesilince cache çalışıyor.
- Günlük hisse verisi calendar ve timezone doğrulamasından geçiyor.
- Hiçbir ücretli API veya key gerekmiyor.

### Aşama 3 — Ortak feature pipeline

Amaç: Eğitim, backtest ve inference arasında tek feature uygulaması.

- [x] TA-Lib bağımsız indikatör modülü yazılacak.
- [x] Ortak feature seti yazılacak.
- [x] Kripto domain feature’ları yazılacak.
- [x] Equity session/corporate action feature’ları yazılacak.
- [x] Feature schema manifest ve hash üretimi yazılacak.
- [x] Train-only normalization yazılacak.
- [x] Missing feature mask desteği eklenecek.
- [x] Golden karşılaştırmalar yapılacak.
- [x] Leakage property testleri eklenecek.

Kabul kriteri:

- Aynı input, train/inference/backtest yollarında aynı feature matrisini üretiyor.
- Gelecek veri değişikliği geçmiş feature satırını değiştirmiyor.
- Feature schema uyuşmazlığı açık hata veriyor.

### Aşama 4 — Canonical MoE ve model registry

Amaç: Base/enhanced ayrımını bitiren tek model sistemi.

- [x] Expert modülleri ayrıştırılacak.
- [x] Router tek implementation olacak.
- [x] Multi-task heads yazılacak.
- [x] Auxiliary losses yazılacak.
- [x] Scheduled branch dropout yazılacak.
- [x] ModelBundle save/load yazılacak.
- [x] Local model registry yazılacak.
- [x] Yalnız self-describing ModelBundle yükleme yolu desteklenecek.
- [x] Prediction contract ve de-normalizasyon uygulanacak.
- [x] Calibration modülü yazılacak.

Kabul kriteri:

- Model yükleme yalnızca bundle + schema kontrolüyle çalışıyor.
- Prediction sonucu gerçek birimde ve açık horizon ile dönüyor.
- Expert weights toplamı 1 ve UI/API’ye aktarılabiliyor.
- Aynı checkpoint aynı fixture’da deterministik tolerans içinde sonuç veriyor.

### Aşama 5 — Eğitim ve güvenilir değerlendirme

Amaç: Kripto ve equity model ailelerini leakage olmadan eğitmek.

- [x] Purged chronological split yazılacak.
- [x] Embargo desteği yazılacak.
- [x] Walk-forward runner yazılacak.
- [x] Baseline modeller yazılacak.
- [x] Crypto trainer config yazılacak.
- [x] Equity trainer config yazılacak.
- [x] Metrics ve calibration raporları yazılacak.
- [x] Ablation runner yazılacak.
- [x] Model card otomatik üretilecek.
- [x] Test fold kilidi kod ve config düzeyinde uygulanacak.

Kabul kriteri:

- Train/validation/test tarihleri model manifestinde var.
- Test verisi hyperparameter aramasına giremiyor.
- En az bir crypto ve bir equity model tüm pipeline’dan geçiyor.
- Model baseline karşılaştırmalı rapor üretiyor.

### Aşama 6 — Backtest V2

Amaç: Gerçekçi, ortak ve denetlenebilir simülasyon motoru.

- [x] Event/bar tabanlı engine yazılacak.
- [x] Next-bar execution zorunlu olacak.
- [x] Commission/spread/slippage modeli yazılacak.
- [x] Intrabar ambiguity politikası uygulanacak.
- [x] Multi-currency accounting yazılacak.
- [x] Corporate action accounting yazılacak.
- [x] Risk limitleri yazılacak.
- [x] Benchmark ve baseline karşılaştırması eklenecek.
- [x] Bootstrap confidence interval eklenecek.
- [x] JSON/Parquet/HTML rapor üretilecek.
- [x] Deterministik maliyet ve next-bar regression testleri hazırlanacak.

Kabul kriteri:

- Cost arttığında net sonuç kötüleşiyor veya aynı kalıyor.
- Look-ahead fixture testleri geçiyor.
- Aynı strategy kodu scanner/paper/backtest tarafından kullanılabiliyor.
- Rapor gross ve net sonucu ayrı gösteriyor.

### Aşama 7 — Scanner, API ve dashboard

Amaç: Nihai kullanıcı deneyimini oluşturmak.

- [x] Service katmanı yazılacak.
- [x] FastAPI app factory yazılacak.
- [x] Health/data/models/scanner/backtest endpointleri yazılacak.
- [x] Ana dashboard yapılacak.
- [x] Market scanner yapılacak.
- [x] Enstrüman detay sayfası yapılacak.
- [x] Model laboratuvarı yapılacak.
- [x] Backtest UI yapılacak.
- [x] Veri yöneticisi yapılacak.
- [x] Ücretsiz veri ve yatırım tavsiyesi uyarıları eklenecek.
- [x] Responsive temel görünüm doğrulanacak.

Kabul kriteri:

- Kullanıcı terminal dışına çıkmadan veri yenileyebiliyor, tarama ve backtest çalıştırabiliyor.
- UI her tahminde model/data zamanını gösteriyor.
- API ağır işlemleri bloke etmeden job status döndürüyor.
- Uygulama varsayılan olarak yalnızca localhost’a açılıyor.

### Aşama 8 — Paper portfolio ve raporlama

Amaç: Emir otomasyonu olmadan kararların takip edilebilmesi.

- [x] Manual trade CRUD yazılacak.
- [x] Sanal nakit ve pozisyon muhasebesi yazılacak.
- [x] Multi-currency değerleme yazılacak.
- [x] Prediction snapshot trade’e bağlanacak.
- [x] PnL/attribution raporu yazılacak.
- [x] CSV/JSON export yazılacak.
- [x] UI sayfası tamamlanacak.

Kabul kriteri:

- Paper işlem yalnızca kullanıcı eylemiyle açılıp kapanıyor.
- Gerçek emir/broker kodu bulunmuyor.
- Muhasebe invariants testleri geçiyor.

### Aşama 9 — Çalışma ağacı sadeleştirme

Amaç: Tek ürün, tek çekirdek ve anlaşılır depo.

- [x] Tüm aktif giriş noktaları yeni pakete taşınacak.
- [x] Notebook ve kopya runtime dosyaları aktif çalışma ağacından çıkarılacak.
- [x] Gerçek işlem bot kodları kaldırılacak.
- [x] Secret içeren user setup akışı kaldırılacak; eski anahtarlar için rotation uyarısı verilecek.
- [x] Tek canonical `src/market_moe` paketi kullanılacak.
- [x] Büyük artefaktlar `data/` ve `artifacts/` altında Git dışında üretilecek.
- [x] Eski dosya kaldırma listesi kullanıcıya sunulacak ve açık onayla uygulanacak.
- [x] Import ve dokümantasyon referansları güncellenecek.

Kabul kriteri:

- Ürünün tek runtime kaynağı `src/market_moe` paketidir.
- Aynı model veya feature implementasyonunun kopyası yok.
- Git çalışma ağacı büyük üretilmiş artefaktlarla büyümüyor.

### Aşama 10 — Nihai QA ve teslim

Amaç: Son ürünün kurulabilir, belgeli ve doğrulanmış olarak teslimi.

- [x] Temiz Windows kurulum testi yapılacak.
- [x] CPU-only end-to-end testi yapılacak.
- [x] En az bir crypto ve dört bölgeden equity sembolü test edilecek.
- [x] Tüm default universe data health raporu üretilecek.
- [x] Model eğitim smoke testi yapılacak.
- [x] Backtest ve paper portfolio E2E testi yapılacak.
- [x] API/UI smoke test yapılacak.
- [x] README tamamlanacak.
- [x] Kullanıcı rehberi tamamlanacak.
- [x] Model card ve validation dokümanı tamamlanacak.
- [x] Bilinen sınırlamalar listesi tamamlanacak.
- [x] Release checklist ve sürüm etiketi hazırlanacak.

Kabul kriteri:

- Bölüm 20’deki nihai ürün kabul kriterlerinin tamamı geçiyor.
- Açık kritik veya yüksek öncelikli hata yok.
- Ücretli API veya gerçek emir entegrasyonu yok.

---

## 20. Nihai ürün kabul kriterleri

Ürün ancak aşağıdaki kriterlerin tamamı sağlandığında teslim edilmiş kabul edilir.

### Kurulum ve çalıştırma

- [x] README’deki temiz kurulum adımları çalışıyor.
- [x] `market-moe doctor` tüm kritik kontrolleri yapıyor.
- [x] `market-moe serve` dashboard’u açıyor.
- [x] CPU-only sistemde temel kullanım mümkün.
- [x] Kullanıcıdan ücretli API key istenmiyor.

### Veri

- [x] CCXT public crypto verisi çalışıyor.
- [x] yfinance global equity adaptörü çalışıyor.
- [x] Günlük equity fallback veya local import çalışıyor.
- [x] BIST, ABD, Avrupa ve Asya’dan örnek semboller geçiyor.
- [x] Data quality ve freshness UI’da gösteriliyor.
- [x] Corporate action ve adjustment modu açıkça gösteriliyor.

### Model

- [x] Crypto ve equity model aileleri ayrı production bundle üretebiliyor.
- [x] Feature schema kontrolü zorunlu.
- [x] Tahmin birimleri açık ve de-normalize.
- [x] Calibration ve uncertainty raporu var.
- [x] Baseline karşılaştırması var.
- [x] Test fold hyperparameter seçiminde kullanılmıyor.

### Backtest

- [x] Look-ahead testleri geçiyor.
- [x] Maliyet modeli zorunlu.
- [x] Gross/net sonuçlar ayrı.
- [x] Benchmark ve drawdown raporları var.
- [x] Walk-forward sonuçları var.
- [x] Survivorship/data kapsam uyarıları görünür.

### Uygulama

- [x] Scanner çalışıyor.
- [x] Enstrüman detayı çalışıyor.
- [x] Model laboratuvarı çalışıyor.
- [x] Backtest UI çalışıyor.
- [x] Paper portfolio manuel çalışıyor.
- [x] Export çalışıyor.
- [x] Health sayfası çalışıyor.

### Güvenlik ve kapsam

- [x] Gerçek emir gönderen kod aktif üründe yok.
- [x] Broker bağlantısı yok.
- [x] Private exchange endpointi yok.
- [x] Wildcard CORS yok.
- [x] Varsayılan bind localhost.
- [x] Secret dosyaları Git dışında.
- [x] Ücretsiz veri lisans/sınırlama uyarıları var.

### Kalite

- [x] Test suite geçiyor.
- [x] Ruff geçiyor.
- [x] Kritik type hatası yok.
- [x] Kritik modül coverage hedefleri geçiyor.
- [x] Dokümantasyon güncel.
- [x] Bilinen kritik bug yok.

---

## 21. Risk kaydı

| Risk | Etki | Olasılık | Önlem |
|---|---|---:|---|
| yfinance format/rate değişikliği | Equity veri kesintisi | Yüksek | Cache, retry, Stooq/local fallback, adapter izolasyonu |
| Ücretsiz intraday veri yetersizliği | Equity 1h kapsamı daralır | Yüksek | 1d’yi zorunlu, 1h’yi best-effort yap |
| Survivorship bias | Backtest şişer | Yüksek | Uyarı, point-in-time evren imkânı, delisted import desteği |
| Corporate action hatası | Return ve muhasebe bozulur | Orta | Raw/adjusted ayrımı, split/dividend testleri |
| Model leakage | Sahte performans | Orta | Purge, embargo, locked test, property tests |
| Backtest/live parity ihtiyacı | Gerçek emir yok; karar simülasyonu sapar | Düşük/Orta | Scanner ve backtest aynı signal/risk kodunu kullanır |
| Büyük Git artefaktları | Depo yavaşlığı | Yüksek | Data/artifacts Gitignore ve yerel çıktı politikası |
| CPU eğitim süresi | Kullanıcı deneyimi kötü | Orta | Küçük preset, batch ayarı, önceden doğrulanmış model bundle |
| Global timezone hataları | Yanlış bar/seans | Orta | UTC canonical, exchange_calendars, fixture tests |
| Çoklu para birimi hatası | Yanlış portfolio PnL | Orta | FX zorunluluğu, accounting tests |
| Model karmaşıklığı baseline’ı geçmez | Fazladan bakım | Orta | Ablation, basit baseline gate, gerekirse sade model |

---

## 22. Kodlama sırasında uyulacak kurallar

1. Her aşamaya başlamadan önce bu dosyadaki ilgili görevler ve kabul kriterleri okunur.
2. Kod değişiklikleri küçük, test edilebilir adımlarla yapılır.
3. Mevcut kullanıcı değişiklikleri korunur; ilgisiz dosyalar değiştirilmez.
4. Aynı iş mantığı ikinci kez kopyalanmaz; ortak modüle taşınır.
5. Notebooklar kaynak kodun ana implementasyonu olamaz.
6. Model çıktısı birimsiz float olarak katmanlar arasında taşınmaz.
7. Train normalizasyonu inference sırasında yeniden fit edilmez.
8. Test setine bakılarak eşik veya hyperparameter değiştirilmez.
9. Veri bulunamadığında sahte/sentetik market data ile sessiz devam edilmez.
10. Network ve ücretsiz sağlayıcı hataları kullanıcıya anlaşılır biçimde gösterilir.
11. Her model/rapor kaynak veri, tarih aralığı ve sürüm bilgisi taşır.
12. Gerçek emir gönderme özelliği eklenmez.
13. Ücretli veri API’si eklenmez.
14. Destructive cleanup, checksum ve kullanıcı onayı olmadan yapılmaz.
15. Her aşama sonunda test, git diff ve dokümantasyon kontrolü yapılır.

---

## 23. Karar günlüğü

### ADR-001 — Gerçek işlem otomasyonu kaldırıldı

Karar: Nihai ürün broker veya exchange private API kullanmayacak.  
Gerekçe: Kullanıcı otomatik işlem istemiyor; güvenlik ve ürün karmaşıklığı önemli ölçüde azalıyor.  
Sonuç: Trading bot dosyaları kaldırıldı; paper portfolio ve simüle execution korunuyor.

### ADR-002 — Ücretli API kullanılmayacak

Karar: Veri katmanı CCXT public, yfinance, Stooq ve local import ile sınırlı olacak.  
Gerekçe: Kullanıcı açıkça ücretli API istemiyor.  
Sonuç: Global gerçek zaman garantisi verilemez; `1d` equity ana timeframe olur.

### ADR-003 — Tek dünya modeli yerine domain modelleri

Karar: Crypto ve equity aynı mimari kodunu paylaşır fakat ayrı ağırlıklarla eğitilir.  
Gerekçe: 24/7 kripto ile seanslı/corporate-action içeren hisselerin veri üretim süreçleri farklıdır.  
Sonuç: İleride ortak trunk denenebilir, ilk final sürümde zorunlu değildir.

### ADR-004 — Backtest ve scanner aynı strategy katmanını kullanır

Karar: Sinyal/risk iş mantığı tek implementation olacaktır.  
Gerekçe: Araştırma ve kullanıcıya gösterilen kararlar arasında tutarlılık sağlamak.  
Sonuç: Uygulamaya özel threshold kopyaları kaldırılır.

### ADR-005 — Equity için günlük timeframe zorunlu, 1h best-effort

Karar: Global equity final acceptance `1d` üzerinde kurulacak; `1h` ücretsiz veri mevcutsa desteklenecek.  
Gerekçe: Ücretsiz intraday verinin kapsam ve tarihsel derinliği garanti edilemez.  
Sonuç: Ürün ücretsiz kalırken global kapsama daha dürüst biçimde ulaşır.

### ADR-006 — TA-Lib runtime zorunluluğu kaldırılacak

Karar: Gerekli indikatörler proje içinde pandas/numpy ile uygulanacak.  
Gerekçe: Windows ve temiz kurulum deneyimini iyileştirmek.  
Sonuç: Golden tolerans testleriyle davranış farkları kontrol edilir.

---

## 24. Plan değişiklik günlüğü

- 2026-08-11: Kullanıcı onayıyla önceki runtime, model, veri ve sonuç dosyaları aktif
  çalışma ağacından kaldırıldı. Git geçmişi geri dönüş amacıyla korundu.
- 2026-08-12: Yahoo adjusted OHLC kayan-nokta sapmaları için kontrollü provider sanitization
  eklendi; sabit minimum-enstrüman eşiği kaldırılıp tam-universe politikası getirildi.
- 2026-08-12: Evren 20 kripto ve 120 global hisse olmak üzere 140 enstrümana genişletildi;
  Yahoo sembollerinin 120/120'si, Binance paritelerinin 20/20'si canlı doğrulandı.
- 2026-08-11: BIST resmi 24.11.2025 işlem kodu değişikliğine göre KOZAL kimliği
  TRALT olarak güncellendi.

| Sürüm | Tarih | Değişiklik |
|---|---|---|
| 1.0 | 2026-08-11 | İlk bağlayıcı nihai ürün planı oluşturuldu; ücretsiz veri ve emir otomasyonu olmaması koşulları işlendi. |

---

## 25. İlk uygulama teslimatı (tamamlandı)

1. Canonical `src/market_moe` ürün paketi
2. Deterministik unit/integration test seti
3. Tekrar üretilebilir `pyproject.toml`, lock ve kurulum temeli
4. Ücretsiz veri, eğitim, backtest ve yerel dashboard akışı

---

## 26. Windows CUDA tek-komut eğitim teslimatı (tamamlandı)

- [x] 12 GB VRAM için güvenli başlangıç batch/model ayarları tanımlandı.
- [x] Kripto 15m, 1h, 1d ve global equity 1d zorunlu iş matrisi oluşturuldu.
- [x] Ücretsiz public veri indirme, doğrulama, retry, fallback ve cache tekrar kullanımı bağlandı.
- [x] Çoklu enstrüman pencereleri birbirine karışmadan pooled eğitim uygulandı.
- [x] Global kronolojik train/validation/locked-test sınırları ve purge uygulandı.
- [x] Normalizasyon yalnız train havuzunda fit edildi.
- [x] Early stopping ve isotonic calibration yalnız validation katında uygulandı.
- [x] Epoch checkpoint'i ve aynı komutla kesintiden devam davranışı eklendi.
- [x] CUDA OOM halinde 256 -> 128 -> 64 -> 32 batch fallback'i eklendi.
- [x] Kilitli test tahminleriyle t+1, maliyetli ve enstrüman bazlı backtest üretildi.
- [x] Model bundle'ları candidate olarak kaydedildi; otomatik production promotion engellendi.
- [x] State JSON, UTF-8 log, HTML özet ve birleşik backtest özeti üretildi.
- [x] Windows uyku engelleme ve sıfır olmayan hata çıkış kodu uygulandı.
- [x] `RUN_FULL_CUDA_PIPELINE.bat` tek-tık başlatıcısı oluşturuldu.
- [x] BAT gerçek `cmd.exe` ile check modunda doğrulandı.
- [x] PowerShell kurulum dosyası parser ile doğrulandı.
- [x] Pooled eğitim/backtest ve checkpoint resume entegrasyon testleri eklendi.
- [x] Ruff, mypy ve tüm test/coverage kalite kapıları geçti.

Karar: Ücretsiz kaynakların global hisselerde uzun ve eşit intraday kapsam vermemesi nedeniyle
`global_equity_1h` tanımlı fakat varsayılan olarak devre dışıdır. Bu, ADR-005 ile uyumludur.
