"""SL Cooldown PnL Analysis v2 — Tarih bazli, dogru cooldown simulasyonu."""
import json, glob, os
from datetime import datetime, timedelta

results_dir = os.path.join(os.path.dirname(__file__), 'backtest_results')
trade_files = glob.glob(os.path.join(results_dir, 'backtest_trades_*.json'))

print("=" * 80)
print("SL COOLDOWN PnL ANALİZİ v2 — Tarih Bazlı Tam Simülasyon")
print("=" * 80)

# ─────────────────────────────────────────────────────────
# 1) Genel: SL sonrasi trade'lerin PnL profili
# ─────────────────────────────────────────────────────────
print("\n=== 1. SL SONRASI TRADE PnL PROFİLİ ===")
for f in sorted(trade_files):
    fname = os.path.basename(f).replace('backtest_trades_', '').replace('.json', '')
    with open(f) as fh:
        trades = json.load(fh)
    
    sl_pnls = [t['pnl_pct'] for t in trades if t.get('is_stop_loss')]
    win_pnls = [t['pnl_pct'] for t in trades if t.get('is_winning')]
    total_sl_pnl = sum(sl_pnls)
    total_win_pnl = sum(win_pnls)
    
    print(f"  {fname}:")
    print(f"    SL: n={len(sl_pnls)}, ort={sum(sl_pnls)/len(sl_pnls):+.1f}%, toplam={total_sl_pnl:+.0f}%")
    print(f"    Win: n={len(win_pnls)}, ort={sum(win_pnls)/len(win_pnls):+.1f}%, toplam={total_win_pnl:+.0f}%")


# ─────────────────────────────────────────────────────────
# 2) Tam Compound Bakiye Simulasyonu — Farkli cooldown degerleri
#    Her SL sonrasi, cooldown_until zamani hesapla
#    O zamana kadar olan TUM trade'leri atla
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("=== 2. COMPOUND BAKİYE SİMÜLASYONU (Her SL sonrasi cooldown) ===")
print("=" * 80)

cooldown_values = [0, 2, 4, 8, 12, 16, 24, 32, 48]  # candle count (15dk each)

for f in sorted(trade_files):
    fname = os.path.basename(f).replace('backtest_trades_', '').replace('.json', '')
    with open(f) as fh:
        trades = json.load(fh)
    
    print(f"\n  {fname} ({len(trades)} trades):")
    print(f"    {'Cooldown':>12} | {'Final $':>8} | {'Trade':>6} | {'Atlan':>6} | {'Fark':>10}")
    print(f"    {'-'*55}")
    
    baseline = None
    for cd in cooldown_values:
        balance = 100.0
        cooldown_until = None
        trades_taken = 0
        trades_skipped = 0
        consecutive_sl = 0
        
        for t in trades:
            entry_time = datetime.fromisoformat(t['entry_time'])
            pnl = t.get('pnl_pct', 0)
            
            # Cooldown active? Check by timestamp
            if cooldown_until is not None:
                if entry_time < cooldown_until:
                    trades_skipped += 1
                    continue
                else:
                    cooldown_until = None
                    consecutive_sl = 0  # Reset streak after cooldown
            
            # Take the trade
            balance *= (1 + pnl / 100)
            trades_taken += 1
            
            # Track consecutive SLs
            if t.get('is_stop_loss'):
                consecutive_sl += 1
            else:
                consecutive_sl = 0
            
            # Trigger cooldown after SL
            if t.get('is_stop_loss') and cd > 0:
                exit_time = datetime.fromisoformat(t['exit_time'])
                cooldown_until = exit_time + timedelta(minutes=cd * 15)
        
        if cd == 0:
            baseline = balance
        
        diff = balance - baseline if baseline else 0
        marker = " <-- baseline" if cd == 0 else (" ✅ BEST" if diff > 0 else "")
        print(f"    {cd*15:>8} dk  | ${balance:>7.1f} | {trades_taken:>5} | {trades_skipped:>5} | {diff:>+9.1f}{marker}")


# ─────────────────────────────────────────────────────────
# 3) Tam Compound Simulasyon — Sadece 2+ ARDISIK SL sonrasi cooldown
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("=== 3. COMPOUND SİMÜLASYON — 2+ ARDIŞIK SL SONRASI COOLDOWN ===")
print("=" * 80)

for f in sorted(trade_files):
    fname = os.path.basename(f).replace('backtest_trades_', '').replace('.json', '')
    with open(f) as fh:
        trades = json.load(fh)
    
    print(f"\n  {fname}:")
    print(f"    {'Cooldown':>12} | {'Final $':>8} | {'Trade':>6} | {'Atlan':>6} | {'Fark':>10}")
    print(f"    {'-'*55}")
    
    baseline = None
    for cd in cooldown_values:
        balance = 100.0
        cooldown_until = None
        trades_taken = 0
        trades_skipped = 0
        consecutive_sl = 0
        
        for t in trades:
            entry_time = datetime.fromisoformat(t['entry_time'])
            pnl = t.get('pnl_pct', 0)
            
            # Cooldown active?
            if cooldown_until is not None:
                if entry_time < cooldown_until:
                    trades_skipped += 1
                    continue
                else:
                    cooldown_until = None
                    consecutive_sl = 0
            
            # Take the trade
            balance *= (1 + pnl / 100)
            trades_taken += 1
            
            # Track consecutive SLs
            if t.get('is_stop_loss'):
                consecutive_sl += 1
            else:
                consecutive_sl = 0
            
            # Trigger cooldown ONLY after 2+ consecutive SLs
            if t.get('is_stop_loss') and consecutive_sl >= 2 and cd > 0:
                exit_time = datetime.fromisoformat(t['exit_time'])
                cooldown_until = exit_time + timedelta(minutes=cd * 15)
        
        if cd == 0:
            baseline = balance
        
        diff = balance - baseline if baseline else 0
        marker = " <-- baseline" if cd == 0 else (" ✅" if diff > 0 else "")
        print(f"    {cd*15:>8} dk  | ${balance:>7.1f} | {trades_taken:>5} | {trades_skipped:>5} | {diff:>+9.1f}{marker}")


# ─────────────────────────────────────────────────────────
# 4) AGGREGATE: Tum dosyalar toplam
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("=== 4. AGGREGATE — TÜM DOSYALARIN TOPLAMI ===")
print("=" * 80)

for label, min_streak in [("Her SL sonrasi", 1), ("2+ ardisik SL sonrasi", 2)]:
    print(f"\n  [{label}]")
    print(f"    {'Cooldown':>12} | {'Toplam Final$':>14} | {'Ort. Final$':>12} | {'Fark':>10}")
    print(f"    {'-'*60}")
    
    baseline_total = None
    for cd in cooldown_values:
        total_balance = 0
        
        for f in sorted(trade_files):
            with open(f) as fh:
                trades = json.load(fh)
            
            balance = 100.0
            cooldown_until = None
            consecutive_sl = 0
            
            for t in trades:
                entry_time = datetime.fromisoformat(t['entry_time'])
                pnl = t.get('pnl_pct', 0)
                
                if cooldown_until is not None:
                    if entry_time < cooldown_until:
                        continue
                    else:
                        cooldown_until = None
                        consecutive_sl = 0
                
                balance *= (1 + pnl / 100)
                
                if t.get('is_stop_loss'):
                    consecutive_sl += 1
                else:
                    consecutive_sl = 0
                
                if t.get('is_stop_loss') and consecutive_sl >= min_streak and cd > 0:
                    exit_time = datetime.fromisoformat(t['exit_time'])
                    cooldown_until = exit_time + timedelta(minutes=cd * 15)
            
            total_balance += balance
        
        avg_balance = total_balance / len(trade_files)
        if cd == 0:
            baseline_total = total_balance
        
        diff = total_balance - baseline_total if baseline_total else 0
        marker = " <-- baseline" if cd == 0 else (" ✅ BEST" if diff == max(
            (sum(100 * 1  for _ in trade_files) if ccc == 0 else total_balance) - baseline_total
            for ccc in [cd]
        ) else (" ✅" if diff > 0 else ""))
        print(f"    {cd*15:>8} dk  | ${total_balance:>13.1f} | ${avg_balance:>11.1f} | {diff:>+9.1f}")

print("\n" + "=" * 80)
print("Bitti!")
