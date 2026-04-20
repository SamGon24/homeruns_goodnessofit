import pandas as pd
import numpy as np
from scipy import stats

# ── 1. Load & Sample ─────────────────────────────────────────────
df = pd.read_csv('sample_200.csv')
sample = df['HR'].dropna().astype(int).reset_index(drop=True) 

# ── 2. Descriptive Stats ─────────────────────────────────────────
n    = len(sample)
mean = sample.mean()
std  = sample.std(ddof=1)

print("=" * 50)
print("DESCRIPTIVE STATISTICS")
print("=" * 50)
print(f"n      = {n}")
print(f"mean   = {mean:.4f}")
print(f"std    = {std:.4f}")
print(f"min    = {sample.min()}")
print(f"max    = {sample.max()}")

# ── 3. Define Intervals ──────────────────────────────────────────
# Adjust these bin edges to suit your data / professor's preference
bins = [
    (float('-inf'), 0.5,   "0"),
    (0.5,           5.5,   "1-5"),
    (5.5,           10.5,  "6-10"),
    (10.5,          18.5,  "11-18"),
    (18.5,          27.5,  "19-27"),
    (27.5,          float('inf'), "28+"),
]

# ── 4. Chi-Square Table ──────────────────────────────────────────
print("\n" + "=" * 50)
print("CHI-SQUARE GOODNESS-OF-FIT TABLE")
print("=" * 50)
print(f"{'Interval':<10} {'fi':>5} {'pi':>8} {'npi':>8} {'fi-npi':>8} {'(fi-npi)^2':>12} {'chi2_i':>8}")
print("-" * 65)

chi2_stat = 0.0
rows = []

for lo, hi, label in bins:
    # Observed frequency
    if lo == float('-inf'):
        fi = int((sample <= hi).sum())
        p  = stats.norm.cdf(hi, loc=mean, scale=std)
    elif hi == float('inf'):
        fi = int((sample > lo).sum())
        p  = 1 - stats.norm.cdf(lo, loc=mean, scale=std)
    else:
        fi = int(((sample > lo) & (sample <= hi)).sum())
        p  = stats.norm.cdf(hi, loc=mean, scale=std) - stats.norm.cdf(lo, loc=mean, scale=std)

    npi    = n * p
    diff   = fi - npi
    chi2_i = (diff ** 2) / npi
    chi2_stat += chi2_i
    rows.append((label, fi, p, npi, diff, diff**2, chi2_i))
    print(f"{label:<10} {fi:>5} {p:>8.4f} {npi:>8.4f} {diff:>8.4f} {diff**2:>12.4f} {chi2_i:>8.4f}")

print("-" * 65)
print(f"{'TOTAL':<10} {sum(r[1] for r in rows):>5} {'':>8} {'':>8} {'':>8} {'':>12} {chi2_stat:>8.4f}")

# ── 5. Test Result ───────────────────────────────────────────────
k        = len(bins)
r        = 2          # parameters estimated from sample (mean + std)
df_chi   = k - 1 - r
alpha    = 0.05
critical = stats.chi2.ppf(1 - alpha, df=df_chi)

print("\n" + "=" * 50)
print("TEST RESULT")
print("=" * 50)
print(f"chi2 statistic     = {chi2_stat:.4f}")
print(f"df (k-1-r)         = {k} - 1 - {r} = {df_chi}")
print(f"critical value     = chi2_({alpha},{df_chi}) = {critical:.4f}")
print(f"reject H0?         = {chi2_stat > critical}  ({chi2_stat:.4f} > {critical:.4f})")
print()
if chi2_stat > critical:
    print("CONCLUSION: Reject H0. HR totals do NOT follow a Normal distribution.")
else:
    print("CONCLUSION: Fail to reject H0. No evidence against Normal distribution.")