import json

trades = json.load(open(r'c:\Users\KDV\Desktop\trade_history\trades.json'))

# 6 Subat islemleri
feb6 = [t for t in trades if t.get('entry_time','').startswith('2026-02-06')]
total_feb6 = sum(t.get('pnl_usdt',0) for t in feb6)

print(f"=== 6 Subat 2026 ===")
print(f"Islem sayisi: {len(feb6)}")
print(f"Toplam PnL: ${total_feb6:.2f}")
print("---")
for t in feb6:
    print(f"{t['trade_id'][:8]}: {t['pnl_usdt']:+.2f}$ | {t['side']} | {t['reason'][:30]}")

# 5 Subat gece (18:00+) islemleri
print(f"\n=== 5 Subat Gece (18:00+) ===")
feb5_night = [t for t in trades if t.get('entry_time','').startswith('2026-02-05T') and int(t.get('entry_time','')[11:13]) >= 18]
total_feb5_night = sum(t.get('pnl_usdt',0) for t in feb5_night)
print(f"Islem sayisi: {len(feb5_night)}")
print(f"Toplam PnL: ${total_feb5_night:.2f}")

# Fear filter sonrasi (1402. satirdan sonra - 5 Subat 18:45+)
print(f"\n=== Fear Filter Sonrasi (Tum) ===")
fear_filter_trades = feb5_night + feb6
total_ff = sum(t.get('pnl_usdt',0) for t in fear_filter_trades)
print(f"Islem sayisi: {len(fear_filter_trades)}")
print(f"Toplam PnL: ${total_ff:.2f}")
