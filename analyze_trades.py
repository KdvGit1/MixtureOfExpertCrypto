import json

# Load trades
with open('backtest_results/backtest_trades_BTC_20260210_004915.json', 'r', encoding='utf-8') as f:
    trades = json.load(f)

print(f'Total trades: {len(trades)}')

# Analyze prediction accuracy
correct_direction = 0
wrong_direction = 0

for t in trades:
    pred = t['ai_prediction']
    pnl = t['pnl_pct']
    side = t['side']
    
    if side == 'LONG':
        if pnl > 0:
            correct_direction += 1
        else:
            wrong_direction += 1
    else:
        if pnl > 0:
            correct_direction += 1
        else:
            wrong_direction += 1

total = correct_direction + wrong_direction
print(f'Correct direction: {correct_direction} ({correct_direction/total*100:.1f}%)')
print(f'Wrong direction: {wrong_direction} ({wrong_direction/total*100:.1f}%)')

# Analyze by confidence
high_conf = [t for t in trades if t['ai_confidence'] >= 100]
low_conf = [t for t in trades if t['ai_confidence'] < 100]

high_wins = len([t for t in high_conf if t['is_winning']])
low_wins = len([t for t in low_conf if t['is_winning']])

print(f'\n=== By Confidence ===')
print(f'High Confidence (100%): {len(high_conf)} trades, {high_wins} wins ({high_wins/len(high_conf)*100:.1f}%)')
print(f'Low Confidence (<100%): {len(low_conf)} trades, {low_wins} wins ({low_wins/len(low_conf)*100:.1f}%)')

# Average PnL by confidence
high_pnl = sum(t['pnl_pct'] for t in high_conf) / len(high_conf)
low_pnl = sum(t['pnl_pct'] for t in low_conf) / len(low_conf)
print(f'Avg PnL High Conf: {high_pnl:.2f}%')
print(f'Avg PnL Low Conf: {low_pnl:.2f}%')

# Total PnL 
total_pnl = sum(t['pnl_pct'] for t in trades)
print(f'\n=== Total ===')
print(f'Total PnL: {total_pnl:.2f}%')

# SL analysis
sl_trades = [t for t in trades if t['is_stop_loss']]
print(f'\n=== Stop Loss Trades ({len(sl_trades)}) ===')
for t in sl_trades:
    print(f"  Trade #{t['id']}: {t['side']} pred={t['ai_prediction']:.1f}% actual={t['pnl_pct']:.1f}% conf={t['ai_confidence']:.0f}%")

# 2x signal analysis  
signal_2x = [t for t in trades if t['is_2x_signal']]
s2_wins = len([t for t in signal_2x if t['is_winning']])
s2_pnl = sum(t['pnl_pct'] for t in signal_2x)
print(f'\n=== 2x Signal Trades ({len(signal_2x)}) ===')
print(f'Wins: {s2_wins} ({s2_wins/len(signal_2x)*100:.1f}%)')
print(f'Total PnL from 2x: {s2_pnl:.2f}%')

# Analyze trades where AI predicted BIG move but was WRONG
print(f'\n=== Big Prediction FAILURES ===')
print('(AI predicted big move but trade lost)')
for t in trades:
    if abs(t['ai_prediction']) > 10 and t['pnl_pct'] < -5:
        print(f"  Trade #{t['id']}: {t['side']} pred={t['ai_prediction']:.1f}% actual={t['pnl_pct']:.1f}%")

# Average prediction vs actual
avg_pred = sum(abs(t['ai_prediction']) for t in trades) / len(trades)
avg_actual = sum(abs(t['pnl_pct']) for t in trades) / len(trades)
print(f'\n=== Prediction Scale ===')
print(f'Avg |prediction|: {avg_pred:.2f}%')
print(f'Avg |actual PnL|: {avg_actual:.2f}%')
print(f'Scale ratio: {avg_actual/avg_pred:.2f}x')
