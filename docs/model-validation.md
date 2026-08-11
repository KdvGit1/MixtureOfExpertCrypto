# Model ve Doğrulama Protokolü

## Veri ayrımı

Tüm örnekler ortak UTC tarih ekseninde kronolojik ayrılır. Train en eski, validation
sonraki, test en yeni dönemdir. Target horizon kadar purge ve embargo uygulanır. Test
katı `test_locked` olarak manifeste yazılır ve early stopping/hyperparameter seçim koduna
verilmez. Model seçildikten sonra yalnız bir kez nihai değerlendirme için açılır.

## Feature ve normalizasyon

Feature'lar geçmişe bakan pandas/numpy fonksiyonlarıyla tek pipeline'da üretilir. Golden
ve truncated-frame testleri, gelecekte satır eklenmesinin geçmiş feature değerini
değiştirmediğini doğrular. Ortalama/std yalnız train satırlarında fit edilir. Feature sırası,
versiyonu ve SHA-256 hash'i bundle yüklemede zorunlu kapıdır.

## Metrikler

Model raporunda MAE, sign accuracy, balanced accuracy, Brier score, Pearson IC ve rank IC
bulunur. Yön olasılığı validation katında isotonic calibration ile fit edilir. Testte hem
kalibrasyon öncesi/sonrası Brier hem performans raporlanabilir. Strateji kabulü yalnız
accuracy'ye değil maliyet sonrası edge, risk, turnover ve walk-forward kararlılığına dayanır.

## Baseline'lar

Sıfır getiri, naive momentum, moving-average trend, aynı frekansta random signal,
volatility-scaled momentum ve buy-and-hold karşılaştırmaları mevcuttur. Production terfisi
otomatik değildir; model card ve backtest raporu kullanıcı tarafından değerlendirilir.

## Model bundle kabulü

Bir bundle şu dosyaların tümünü içermelidir: `model.pt`, `manifest.json`,
`feature_schema.json`, `normalization.json`, `metrics.json`, `model_card.md`,
`calibration.json`, `training_history.parquet`. Eksik dosya, checksum farkı veya alan sırası
uyuşmazlığı inference'ı durdurur.

## Smoke ve regression kanıtı

- `market-moe train-smoke --asset-class crypto --epochs 1`
- `market-moe train-smoke --asset-class equity --epochs 1`
- Pooled crypto/equity eğitim, epoch-resume ve candidate bundle entegrasyon testleri.
- Canonical feature leakage, router toplamı, bundle round-trip ve kilitli split testleri.
- Backtest next-bar, maliyet, FX ve corporate-action testleri.

Synthetic smoke bundle yalnız yazılım yolunu doğrular; araştırma kararında kullanılamaz ve
manifestte bu sınırlama açıkça bulunur.
