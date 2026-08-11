# v0.1.0 Release Checklist

- [x] Ürün kapsamı: araştırma, scanner, backtest, manuel paper portfolio.
- [x] Ücretli API, broker ve gerçek emir aktif pakette yok.
- [x] Varsayılan bind localhost; wildcard CORS yok.
- [x] Windows ve POSIX setup scriptleri var.
- [x] Exact bağımlılık snapshot'ı `requirements.lock` içinde.
- [x] Önceki runtime ve artefaktlar aktif çalışma ağacından kaldırıldı.
- [x] Crypto ve equity universe/provider sözleşmeleri var.
- [x] Feature leakage ve train-only normalizasyon testleri geçiyor.
- [x] MoE, bundle, registry, calibration ve test-fold kilidi var.
- [x] Backtest next-bar, maliyet, FX ve corporate-action testleri geçiyor.
- [x] API/dashboard ve manual portfolio smoke testleri geçiyor.
- [x] Ruff, mypy, pytest ve coverage kapıları geçiyor.
- [x] Kullanıcı rehberi, model doğrulama ve sınırlamalar tamamlandı.
- [ ] Git tag oluşturma proje sahibinin release kararına bırakıldı.

Sürüm adayı: `0.1.0`. Release artefaktları local-firsttir; model/veri dosyaları Git'e dahil
edilmez.
