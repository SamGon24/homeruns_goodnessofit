import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── 1. Load & Sample ─────────────────────────────────────────────
df = pd.read_csv('sample_200.csv')
sample = df['WAR'].dropna().reset_index(drop=True) 

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
bins = [
    (float('-inf'), 0.0,  "< 0"),
    (0.0,           1.0,  "0-1"),
    (1.0,           2.0,  "1-2"),
    (2.0,           3.5,  "2-3.5"),
    (3.5,  float('inf'),  "3.5+"),
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
    print("CONCLUSION: Reject H0. WAR totals do NOT follow a Normal distribution.")
else:
    print("CONCLUSION: Fail to reject H0. No evidence against Normal distribution.")
    
# ── 6. Plots ──────────────────────────────────────────────────────
labels   = [r[0] for r in rows]
observed = [r[1] for r in rows]
expected = [r[3] for r in rows]
chi2_per = [r[6] for r in rows]
 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('WAR Goodness-of-Fit Test (Normal Distribution)', fontsize=14, fontweight='bold')
 
# -- Plot 1: Histogram with Normal curve overlay --
ax1 = axes[0]
ax1.hist(sample, bins=30, density=True, color='steelblue', edgecolor='white', alpha=0.7, label='Observed')
x = np.linspace(sample.min() - 1, sample.max() + 1, 300)
ax1.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2, label=f'N({mean:.2f}, {std:.2f})')
ax1.set_title('Sample Distribution vs Normal Curve')
ax1.set_xlabel('WAR')
ax1.set_ylabel('Density')
ax1.legend()
 
# -- Plot 2: Observed vs Expected bar chart --
ax2 = axes[1]
x_pos = np.arange(len(labels))
width = 0.35
ax2.bar(x_pos - width/2, observed, width, label='Observed', color='steelblue', edgecolor='white')
ax2.bar(x_pos + width/2, expected, width, label='Expected', color='tomato', edgecolor='white', alpha=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels)
ax2.set_title('Observed vs Expected Frequencies')
ax2.set_xlabel('WAR Interval')
ax2.set_ylabel('Frequency')
ax2.legend()
 
# -- Plot 3: Chi-square critical region --
ax3 = axes[2]
x_chi = np.linspace(0, max(chi2_stat * 1.1, critical * 1.5), 300)
ax3.plot(x_chi, stats.chi2.pdf(x_chi, df=df_chi), 'k-', linewidth=2)
ax3.fill_between(x_chi, stats.chi2.pdf(x_chi, df=df_chi),
                 where=(x_chi >= critical), color='tomato', alpha=0.4, label=f'Reject region (α={alpha})')
ax3.axvline(critical, color='tomato', linestyle='--', linewidth=1.5, label=f'Critical = {critical:.4f}')
ax3.axvline(chi2_stat, color='steelblue', linestyle='--', linewidth=1.5, label=f'χ² = {chi2_stat:.4f}')
ax3.set_title(f'χ² Distribution (df={df_chi})')
ax3.set_xlabel('χ²')
ax3.set_ylabel('Density')
ax3.legend(fontsize=8)
 
plt.tight_layout()
plt.savefig('war_goodness_of_fit.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to war_goodness_of_fit.png")
 