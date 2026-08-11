# Bilinen Sınırlamalar

- Ücretsiz feed'ler resmi, garantili veya gerçek zamanlı borsa feed'i değildir; gecikme,
  rate limit, revizyon ve boşluk olabilir.
- yfinance kişisel araştırma için best-effort bir istemcidir. Ticari/yeniden dağıtım
  lisansını MarketMoE sağlamaz.
- Stooq fallback yalnız günlük veridir. Provider adjustment anlamları farklı olabilir.
- Global intraday hisse geçmişi kabul kriteri değildir; ücretsiz kaynaklarda dönem kısadır.
- Varsayılan universe dosyaları güncel araştırma sepetidir. Tarihsel constituent veri tabanı
  değildir ve survivorship bias üretir.
- BIST ve bazı Asya/Avrupa sembollerinde provider ticker eşlemesi değişebilir.
- BIST sembol değişiklikleri universe kimliğine işlenir; örneğin resmi 24 Kasım 2025
  değişikliğiyle eski KOZAL kimliği TRALT olarak tutulur.
- USDT/USDC için USD paritesi varsayımı uyarıyla yapılır; depeg riski modellenmez.
- Spread/slippage varsayımları simülasyon parametresidir, yatırım tavsiyesi değildir.
- İntrabar alt-timeframe yoksa stop önce kabul edilen konservatif politika kullanılır.
- Corporate action ancak provider-adjusted veya explicit modlarından biriyle sayılır.
- Model tahmini nedensellik, kârlılık veya gelecekte performans garantisi değildir.
- İlk candidate eğitim komutu tek enstrüman içindir; production öncesi cross-asset ve
  walk-forward doğrulama gerektirir.
- Paper portfolio kullanıcı girişlerine dayanır; broker bakiyesi değildir.
- Uygulamada authentication yoktur çünkü varsayılan yalnız localhost'tur. Uzak bind ancak
  kullanıcının açık sorumluluğuyla açılır.
