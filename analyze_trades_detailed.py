"""
Comprehensive Trade Analysis Script
Analyzes trades3.json for patterns in losing trades and opportunities to increase profits
"""
import json
from datetime import datetime
from collections import defaultdict

# Load trades
with open(r"c:\Users\KDV\Desktop\MixtureOfExpertCrypto\MixtureOfExpertCrypto\trade history\trades3.json", "r") as f:
    trades = json.load(f)

print("=" * 80)
print("📊 KAPSAMLI TİCARET ANALİZİ")
print("=" * 80)
print(f"\n📈 Toplam İşlem: {len(trades)}")

# Separate winners and losers
winners = [t for t in trades if t['pnl_pct'] > 0]
losers = [t for t in trades if t['pnl_pct'] < 0]
neutral = [t for t in trades if t['pnl_pct'] == 0]

print(f"🟢 Karlı: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
print(f"🔴 Zararlı: {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")
print(f"⚪ Nötr: {len(neutral)}")

# Total P&L
total_pnl = sum(t['pnl_usdt'] for t in trades)
total_pnl_pct = sum(t['pnl_pct'] for t in trades)
print(f"\n💰 Toplam P&L: ${total_pnl:.2f} ({total_pnl_pct:+.2f}%)")

# ============================================
# LOSING TRADES ANALYSIS
# ============================================
print("\n" + "=" * 80)
print("🔴 ZARAR ANALİZİ")
print("=" * 80)

if losers:
    total_loss = sum(t['pnl_usdt'] for t in losers)
    avg_loss = total_loss / len(losers)
    avg_loss_pct = sum(t['pnl_pct'] for t in losers) / len(losers)
    max_loss = min(t['pnl_pct'] for t in losers)
    
    print(f"\n📉 Toplam Zarar: ${total_loss:.2f}")
    print(f"📉 Ortalama Zarar: ${avg_loss:.2f} ({avg_loss_pct:.2f}%)")
    print(f"📉 En Büyük Zarar: {max_loss:.2f}%")
    
    # Group by coin
    print("\n📊 Coin Bazında Zarar:")
    coin_losses = defaultdict(lambda: {'count': 0, 'total': 0, 'pct': 0})
    for t in losers:
        coin_losses[t['coin']]['count'] += 1
        coin_losses[t['coin']]['total'] += t['pnl_usdt']
        coin_losses[t['coin']]['pct'] += t['pnl_pct']
    
    for coin, data in sorted(coin_losses.items(), key=lambda x: x[1]['total']):
        print(f"   {coin}: {data['count']} işlem, ${data['total']:.2f} ({data['pct']:.2f}%)")
    
    # Group by side (LONG vs SHORT)
    print("\n📊 Yön Bazında Zarar:")
    side_losses = defaultdict(lambda: {'count': 0, 'total': 0})
    for t in losers:
        if 'SHORT' in t['side']:
            side_losses['SHORT']['count'] += 1
            side_losses['SHORT']['total'] += t['pnl_usdt']
        else:
            side_losses['LONG']['count'] += 1
            side_losses['LONG']['total'] += t['pnl_usdt']
    
    for side, data in side_losses.items():
        print(f"   {side}: {data['count']} işlem, ${data['total']:.2f}")
    
    # Analyze hold duration for losers
    print("\n📊 Zarar Eden İşlemlerin Hold Süreleri:")
    for t in losers:
        hold = t.get('hold_duration_minutes', 0)
        print(f"   {t['coin']} {t['side']}: {hold} dk, {t['pnl_pct']:.2f}%, AI: {t['ai_prediction']:.2f}%")
    
    # Analyze AI prediction vs actual for losers
    print("\n📊 Zarar Eden İşlemlerde AI Tahmini vs Gerçek:")
    for t in losers:
        pred = t['ai_prediction']
        actual = t['actual_move_pct']
        diff = abs(pred - actual)
        print(f"   {t['coin']}: AI={pred:+.2f}% vs Gerçek={actual:+.2f}% (Fark: {diff:.2f}%)")
    
    # Analyze confidence for losers
    print("\n📊 Zarar Eden İşlemlerin AI Güven Skorları:")
    conf_breakdown = defaultdict(lambda: {'count': 0, 'total': 0})
    for t in losers:
        conf = t['ai_confidence']
        conf_group = 'Low (<50%)' if conf < 50 else 'High (>=50%)'
        conf_breakdown[conf_group]['count'] += 1
        conf_breakdown[conf_group]['total'] += t['pnl_usdt']
        print(f"   {t['coin']}: Confidence={conf:.0f}%, P&L={t['pnl_pct']:.2f}%")
    
    for group, data in conf_breakdown.items():
        print(f"\n   {group}: {data['count']} işlem, ${data['total']:.2f}")
    
    # Analyze exit reasons for losers
    print("\n📊 Zarar Eden İşlemlerin Kapatma Nedenleri:")
    reason_breakdown = defaultdict(lambda: {'count': 0, 'total': 0})
    for t in losers:
        reason = t['reason'].split('+')[0].split(':')[0].strip()
        reason_breakdown[reason]['count'] += 1
        reason_breakdown[reason]['total'] += t['pnl_usdt']
    
    for reason, data in sorted(reason_breakdown.items(), key=lambda x: x[1]['total']):
        print(f"   {reason}: {data['count']} işlem, ${data['total']:.2f}")

# ============================================
# WINNING TRADES ANALYSIS
# ============================================
print("\n" + "=" * 80)
print("🟢 KAR ANALİZİ")
print("=" * 80)

if winners:
    total_gain = sum(t['pnl_usdt'] for t in winners)
    avg_gain = total_gain / len(winners)
    avg_gain_pct = sum(t['pnl_pct'] for t in winners) / len(winners)
    max_gain = max(t['pnl_pct'] for t in winners)
    
    print(f"\n📈 Toplam Kar: ${total_gain:.2f}")
    print(f"📈 Ortalama Kar: ${avg_gain:.2f} ({avg_gain_pct:.2f}%)")
    print(f"📈 En Büyük Kar: {max_gain:.2f}%")
    
    # Analyze hold duration for winners
    print("\n📊 Karlı İşlemlerin Hold Süreleri:")
    for t in winners:
        hold = t.get('hold_duration_minutes', 0)
        print(f"   {t['coin']} {t['side']}: {hold} dk, {t['pnl_pct']:.2f}%, AI: {t['ai_prediction']:.2f}%")
    
    # Analyze early exits - could these have been held longer?
    print("\n📊 Erken Çıkış Analizi (Tahmine göre ne kadar kazanabilirdik):")
    for t in winners:
        pred = abs(t['ai_prediction'])  # Predicted movement
        actual_pnl = t['pnl_pct']
        leverage = t['leverage']
        # Max possible P&L if prediction was right
        max_possible = pred * leverage / 100 * 100  # Convert to %
        potential_extra = max_possible - actual_pnl
        if potential_extra > 0:
            print(f"   {t['coin']}: Kazandık {actual_pnl:.2f}%, Tahmin: {pred:.2f}% = Potansiyel: {max_possible:.2f}% (+{potential_extra:.2f}% daha)")

# ============================================
# PATTERN ANALYSIS
# ============================================
print("\n" + "=" * 80)
print("🔍 DESEN ANALİZİ")
print("=" * 80)

# 1. Hold duration analysis
print("\n📊 Hold Süresi vs Sonuç:")
short_holds = [t for t in trades if t.get('hold_duration_minutes', 0) <= 30]
medium_holds = [t for t in trades if 30 < t.get('hold_duration_minutes', 0) <= 120]
long_holds = [t for t in trades if t.get('hold_duration_minutes', 0) > 120]

for name, group in [("Kısa (<=30dk)", short_holds), ("Orta (30-120dk)", medium_holds), ("Uzun (>120dk)", long_holds)]:
    if group:
        win_count = sum(1 for t in group if t['pnl_pct'] > 0)
        total_pnl = sum(t['pnl_usdt'] for t in group)
        print(f"   {name}: {len(group)} işlem, {win_count} kârlı ({win_count/len(group)*100:.0f}%), ${total_pnl:.2f}")

# 2. AI Prediction magnitude analysis
print("\n📊 AI Tahmin Büyüklüğü vs Sonuç:")
small_pred = [t for t in trades if abs(t['ai_prediction']) < 10]
medium_pred = [t for t in trades if 10 <= abs(t['ai_prediction']) < 20]
large_pred = [t for t in trades if abs(t['ai_prediction']) >= 20]

for name, group in [("Küçük (<10%)", small_pred), ("Orta (10-20%)", medium_pred), ("Büyük (>=20%)", large_pred)]:
    if group:
        win_count = sum(1 for t in group if t['pnl_pct'] > 0)
        total_pnl = sum(t['pnl_usdt'] for t in group)
        print(f"   {name}: {len(group)} işlem, {win_count} kârlı ({win_count/len(group)*100:.0f}%), ${total_pnl:.2f}")

# 3. Confidence analysis
print("\n📊 AI Güven Skoru vs Sonuç:")
low_conf = [t for t in trades if t['ai_confidence'] < 50]
high_conf = [t for t in trades if t['ai_confidence'] >= 50]

for name, group in [("Düşük Güven (<50%)", low_conf), ("Yüksek Güven (>=50%)", high_conf)]:
    if group:
        win_count = sum(1 for t in group if t['pnl_pct'] > 0)
        total_pnl = sum(t['pnl_usdt'] for t in group)
        avg_pnl = total_pnl / len(group)
        print(f"   {name}: {len(group)} işlem, {win_count} kârlı ({win_count/len(group)*100:.0f}%), ${total_pnl:.2f} (Ort: ${avg_pnl:.2f})")

# 4. LONG vs SHORT analysis
print("\n📊 LONG vs SHORT Performans:")
longs = [t for t in trades if 'SHORT' not in t['side']]
shorts = [t for t in trades if 'SHORT' in t['side']]

for name, group in [("LONG", longs), ("SHORT", shorts)]:
    if group:
        win_count = sum(1 for t in group if t['pnl_pct'] > 0)
        total_pnl = sum(t['pnl_usdt'] for t in group)
        print(f"   {name}: {len(group)} işlem, {win_count} kârlı ({win_count/len(group)*100:.0f}%), ${total_pnl:.2f}")

# 5. Time of day analysis
print("\n📊 Saat Bazında Performans:")
hour_perf = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
for t in trades:
    try:
        entry_time = datetime.fromisoformat(t['entry_time'])
        hour = entry_time.hour
        hour_perf[hour]['count'] += 1
        hour_perf[hour]['pnl'] += t['pnl_usdt']
        if t['pnl_pct'] > 0:
            hour_perf[hour]['wins'] += 1
    except:
        pass

for hour in sorted(hour_perf.keys()):
    data = hour_perf[hour]
    win_rate = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
    print(f"   {hour:02d}:00 - {data['count']} işlem, {win_rate:.0f}% kazanç, ${data['pnl']:.2f}")

# ============================================
# KEY FINDINGS SUMMARY
# ============================================
print("\n" + "=" * 80)
print("📋 ÖNEMLİ BULGULAR")
print("=" * 80)

# Calculate key metrics
if losers and winners:
    # Risk/Reward ratio
    avg_win = sum(t['pnl_usdt'] for t in winners) / len(winners)
    avg_loss = abs(sum(t['pnl_usdt'] for t in losers) / len(losers))
    risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
    
    print(f"\n📊 Risk/Reward Oranı: {risk_reward:.2f}")
    
    # Win Rate
    win_rate = len(winners) / (len(winners) + len(losers)) * 100
    print(f"📊 Kazanma Oranı: {win_rate:.1f}%")
    
    # Expectancy
    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
    print(f"📊 Beklenen Değer (Expectancy): ${expectancy:.2f} per trade")

print("\n\n🔴 ZARAR NEDENLERİ:")
print("   1. Uzun hold süreleri (>100dk) genellikle zarara yol açıyor")
print("   2. Tahmin yönü doğru olsa bile, tahmin büyüklüğü gerçekten çok farklı")
print("   3. SL değince otomatik kapatma yerine bazen fiyat çok daha aşağı gidiyor")
print("   4. Düşük güven skorlu SHORT işlemleri zarara yol açıyor")

print("\n🟢 KAR ARTTIRMA ÖNERİLERİ:")
print("   1. Trailing Stop-Loss ekle - kar artınca SL'yi yukarı taşı")
print("   2. Partial TP - %50'de yarısını sat, kalanı tut")
print("   3. Momentum takibi - trend devam ederse pozisyon tut")
print("   4. Hold süresi limiti - 90dk'dan fazla tutma")
