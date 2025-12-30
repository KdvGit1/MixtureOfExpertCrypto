import pandas as pd
import numpy as np

# Load the correct file
filename = 'BTC_180Ay_1h_AI_Ready.csv'
df = pd.read_csv(filename)

# Check statistics of Log_Ret
log_ret = df['Log_Ret']
abs_ret = log_ret.abs()

# Standard Deviation (Sigma)
std_dev = log_ret.std()

# Percentiles of Absolute Returns (Noise Analysis)
p25 = abs_ret.quantile(0.25)
p50 = abs_ret.quantile(0.50) # Median
p75 = abs_ret.quantile(0.75)

print(f"--- 1 Saatlik (1h) Veri Analizi ---")
print(f"Standart Sapma (Sigma): {std_dev:.6f}")
print(f"Medyan Mutlak Hareket: {p50:.6f}")
print(f"Gürültü Tabanı (25. Yüzdelik): {p25:.6f}")
print(f"Güçlü Hareket (75. Yüzdelik): {p75:.6f}")

# Calculate normalized threshold candidates
# If we want to filter out the bottom 25% (noise), we use p25 as the raw threshold.
# Normalized Threshold = Raw Threshold / Std Dev

candidates = [p25, p50]
print("\n--- Aday Eşik Değerleri (Sigma Cinsinden) ---")
for raw_t in candidates:
    norm_t = raw_t / std_dev
    print(f"Ham Eşik: {raw_t:.6f} -> Normalize (Kod için): {norm_t:.4f}")