# Changelog

## 0.2.0 - 2026-08-12

- Uzun adjusted Yahoo serilerindeki kayan-nokta OHLC zarf hataları güvenli biçimde onarılıyor.
- Sabit minimum-enstrüman eşiği kaldırıldı; tüm seçili universe varsayılan olarak zorunlu.
- Evren 120 global hisse ve 20 likit kripto paritesine genişletildi.
- Hisse geçmiş isteği 25 yıla, kripto 15m/1h istekleri 2/5 yıla çıkarıldı.
- Yeni listelenen enstrümanlar uygun global split katlarına katkı verebiliyor.
- Config/universe imzası değiştiğinde tamamlanmış eski iş yerine yeni sürüm eğitiliyor.

## 0.1.0 - 2026-08-11

- Kripto ve global equity için canonical ücretsiz veri katmanı.
- Sürümlü feature pipeline ve train-only normalization.
- Canonical three-expert multi-task MoE, local bundle ve registry.
- Purged chronological split, calibration, baseline ve walk-forward araçları.
- Next-bar, maliyet/FX/corporate-action aware Backtest V2.
- Cache tabanlı scanner ve manuel paper portfolio.
- Localhost FastAPI/Jinja dashboard ve CLI.
- Önceki runtime ve artefaktların çalışma ağacından kaldırılması; güvenlik testleri ve CI.
