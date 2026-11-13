"""Extra course-method checks used on the slides (run after analysis.py).

Adds three results that are not in results.txt:
  - chi-square test of independence (slide 13): is "rural > urban" linked to "food > non-food"?
  - Wilcoxon signed-rank test on F - O (limitations slide; non-parametric check of Finding 1)
  - Pearson correlation of food with non-food inflation, with p-values (slide 12)

Usage: python extra_tests.py   (reads data/inflation_series_derived.csv)
"""
import pandas as pd
from scipy import stats

d = pd.read_csv("data/inflation_series_derived.csv", parse_dates=["month"])

# 2x2 table: months where average food inflation beats average non-food inflation, vs rural > urban
food_beats_nonfood = d["rel_food"] > 0          # rel_food = mean(F_rural, F_urban) - mean(NF_rural, NF_urban)
rural_higher = d["gap_H"] > 0                    # gap_H = H_rural - H_urban
table = pd.crosstab(food_beats_nonfood, rural_higher)
chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
print("Chi-square test of independence")
print(table)
print(f"chi2 = {chi2:.1f}, df = {dof}, p = {p:.2e}")
print("expected counts:\n", expected.round(1))

print("\nWilcoxon signed-rank test, H0: median(F - O) = 0")
for s in ["rural", "urban"]:
    res = stats.wilcoxon(d[f"F_{s}"] - d[f"H_{s}"])
    print(f"  {s}: W = {res.statistic:.0f}, p = {res.pvalue:.2f}")

print("\nCorrelation of food with non-food inflation")
for s in ["rural", "urban"]:
    r, p = stats.pearsonr(d[f"F_{s}"], d[f"NF_{s}"])
    print(f"  {s}: r = {r:.2f}, p = {p:.2f}")