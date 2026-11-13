"""Charts and equation images for the deck. Run after analysis.py:  python charts.py"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "charts")
os.makedirs(OUT, exist_ok=True)
o = pd.read_csv(os.path.join(HERE, "data", "inflation_series_derived.csv"), index_col=0, parse_dates=True)
R = json.load(open(os.path.join(HERE, "results.json")))

BG, INK, INK2, GRID = "#FAF8F4", "#16233B", "#5A6475", "#E2DCD0"
RURAL, URBAN, FOOD = "#EB6834", "#2A78D6", "#008300"
C_W, C_F, C_N = "#008300", "#EDA100", "#6250D6"
DPI = 200
if any("Inter" == f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Inter"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.axisbelow": True,
                     "xtick.major.size": 0, "ytick.major.size": 0, "axes.linewidth": 1.2})


def pt(css_px):          # slide px -> matplotlib points (image is rendered at 2x)
    return css_px * 2 * 72 / DPI


def fig(w_css, h_css, **kw):
    return plt.subplots(figsize=(w_css * 2 / DPI, h_css * 2 / DPI), dpi=DPI, **kw)


def pct(ax, ticks):
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}%".replace("-", "−") for t in ticks], fontsize=pt(24))


def years(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=pt(24), pad=10)
    ax.grid(axis="x", visible=False)


# ------------------------------------------------------------------ C1 food vs headline, all-India
f, ax = fig(1664, 600)
ax.axhspan(2, 6, color="#16233B", alpha=0.055, lw=0)
ax.axhline(0, color=INK2, lw=1.2)
ax.plot(o.index, o["F_combined"], color=FOOD, lw=pt(3.2), solid_capstyle="round")
ax.plot(o.index, o["H_combined"], color=INK, lw=pt(3.6), solid_capstyle="round")
pct(ax, [-5, 0, 5, 10, 15])
years(ax)
ax.set_xlim(o.index[0], pd.Timestamp("2027-12-01"))
ax.set_ylim(-6.5, 16.5)
last = o.index[-1]
ax.text(pd.Timestamp("2026-02-01"), o["H_combined"].iloc[-1] + 0.6, "Overall (CPI)", color=INK, fontsize=pt(26), fontweight="semibold", va="center")
ax.text(pd.Timestamp("2026-02-01"), o["F_combined"].iloc[-1] - 0.3, "Food (CFPI)", color=FOOD, fontsize=pt(26), fontweight="semibold", va="center")
ax.text(pd.Timestamp("2026-02-01"), 5.3, "RBI band, 2–6%", color=INK2, fontsize=pt(24), va="center")
pk, tr = o["F_combined"].idxmax(), o["F_combined"].idxmin()
ax.annotate(f"Dec 2019: food {o.loc[pk, 'F_combined']:.1f}%", (pk, o.loc[pk, "F_combined"]), xytext=(-18, 4), textcoords="offset points",
            ha="right", va="center", fontsize=pt(24), color=INK)
ax.annotate(f"Oct 2025: food −{abs(o.loc[tr, 'F_combined']):.1f}%, overall {o.loc[tr, 'H_combined']:.2f}%", (tr, o.loc[tr, "F_combined"]),
            xytext=(-18, 0), textcoords="offset points", ha="right", va="center", fontsize=pt(24), color=INK)
ax.scatter([pk, tr], [o.loc[pk, "F_combined"], o.loc[tr, "F_combined"]], s=110, color=FOOD, edgecolor=BG, linewidth=2.5, zorder=5)
f.subplots_adjust(left=0.05, right=0.995, top=0.97, bottom=0.11)
f.savefig(os.path.join(OUT, "c1_food_vs_overall.png"))
plt.close(f)

# ------------------------------------------------------------------ C2 pass-through scatter, rural | urban
f, axes = fig(1000, 600, ncols=2, sharey=True, sharex=True)
for ax, s, col, name in [(axes[0], "rural", RURAL, "Rural"), (axes[1], "urban", URBAN, "Urban")]:
    x, y = o[f"F_{s}"], o[f"H_{s}"]
    ax.scatter(x, y, s=70, color=col, alpha=0.55, edgecolor=BG, linewidth=1.2)
    r = R["passthrough"][s]
    xs = np.array([-6, 17])
    ax.plot(xs, r["alpha"] + r["beta"] * xs, color=INK, lw=pt(2.6))
    ax.set_xlim(-6.5, 17.5)
    ax.set_ylim(-1, 9.6)
    ax.set_xticks([-5, 0, 5, 10, 15])
    ax.set_xticklabels(["−5%", "0%", "5%", "10%", "15%"], fontsize=pt(24))
    ax.text(-5.6, 9.1, name, fontsize=pt(30), fontweight="semibold", color=col, va="center")
    ax.text(17.0, -0.35, f"slope {r['beta']:.2f}  ·  R² {r['r2']:.2f}", fontsize=pt(24), color=INK, va="center", ha="right")
    ax.tick_params(axis="x", pad=10)
pct(axes[0], [0, 2, 4, 6, 8])
axes[0].set_ylabel("Overall inflation", fontsize=pt(24), labelpad=12)
f.supxlabel("Food inflation (CFPI, y-o-y)", fontsize=pt(24), color=INK2, y=0.012)
f.subplots_adjust(left=0.115, right=0.99, top=0.985, bottom=0.165, wspace=0.07)
f.savefig(os.path.join(OUT, "c2_passthrough.png"))
plt.close(f)

# ------------------------------------------------------------------ C3 annual decomposition of the rural-urban gap
a = o[["dec_weight", "dec_foodprice", "dec_nonfood", "gap_H"]].groupby(o.index.year).mean()
f, ax = fig(1000, 600)
xs = np.arange(len(a))
pos, neg = np.zeros(len(a)), np.zeros(len(a))
for c, col, lab in [("dec_weight", C_W, "Weight effect"), ("dec_foodprice", C_F, "Food-price effect"), ("dec_nonfood", C_N, "Non-food effect")]:
    v = a[c].values
    base = np.where(v >= 0, pos, neg)
    ax.bar(xs, v, bottom=base, width=0.62, color=col, edgecolor=BG, linewidth=2.5, label=lab)
    pos += np.where(v >= 0, v, 0)
    neg += np.where(v < 0, v, 0)
ax.axhline(0, color=INK2, lw=1.2)
ax.plot(xs, a["gap_H"].values, color=INK, lw=pt(1.6), marker="o", ms=13, mec=BG, mew=2.5, label="Rural − urban gap", zorder=6)
ax.set_xticks(xs)
ax.set_xticklabels([f"’{str(y)[2:]}" for y in a.index], fontsize=pt(24))
ax.tick_params(axis="x", pad=10)
ax.grid(axis="x", visible=False)
ax.set_yticks([-2, -1, 0, 1])
ax.set_yticklabels(["−2", "−1", "0", "+1"], fontsize=pt(24))
ax.set_ylim(-2.3, 2.2)
ax.set_ylabel("percentage points", fontsize=pt(24), labelpad=10)
for yr in [2016, 2019]:
    i = list(a.index).index(yr)
    v = a.loc[yr, "gap_H"]
    ax.annotate(f"{v:+.2f}".replace("-", "−"), (i, v), xytext=(21, 0), textcoords="offset points", ha="left", va="center",
                fontsize=pt(24), fontweight="semibold", color=INK)
h, l = ax.get_legend_handles_labels()
ordr = [l.index(k) for k in ["Weight effect", "Food-price effect", "Non-food effect", "Rural − urban gap"]]
leg = ax.legend([h[i] for i in ordr], [l[i] for i in ordr], loc="upper center", ncol=2, frameon=False, fontsize=pt(24), handlelength=1.2, columnspacing=1.6, handletextpad=0.6,
                bbox_to_anchor=(0.5, 1.02))
f.subplots_adjust(left=0.115, right=0.995, top=0.985, bottom=0.1)
f.savefig(os.path.join(OUT, "c3_gap_decomposition.png"))
plt.close(f)

# ------------------------------------------------------------------ C4 rural vs urban overall inflation and the gap
f, (a1, a2) = fig(1000, 620, nrows=2, sharex=True, gridspec_kw={"height_ratios": [1.55, 1], "hspace": 0.16})
a1.plot(o.index, o["H_rural"], color=RURAL, lw=pt(3))
a1.plot(o.index, o["H_urban"], color=URBAN, lw=pt(3))
pct(a1, [0, 2, 4, 6, 8])
a1.set_ylim(-0.8, 9.4)
a1.text(pd.Timestamp("2016-01-01"), 7.7, "Rural", color=RURAL, fontsize=pt(26), fontweight="semibold", va="center")
a1.text(pd.Timestamp("2019-07-01"), 1.6, "Urban", color=URBAN, fontsize=pt(26), fontweight="semibold", va="center")
g = o["gap_H"]
a2.bar(o.index, g, width=24, color=np.where(g >= 0, RURAL, URBAN), linewidth=0)
a2.axhline(0, color=INK2, lw=1.2)
a2.set_yticks([-2, 0, 2])
a2.set_yticklabels(["−2", "0", "+2"], fontsize=pt(24))
a2.set_ylim(-3.1, 2.6)
a2.text(pd.Timestamp("2014-02-01"), -2.35, "Gap: rural − urban (pp)", color=INK2, fontsize=pt(24), va="center")
years(a2)
a1.grid(axis="x", visible=False)
a2.set_xlim(o.index[0] - pd.Timedelta(days=20), o.index[-1] + pd.Timedelta(days=20))
f.subplots_adjust(left=0.085, right=0.995, top=0.985, bottom=0.1)
f.savefig(os.path.join(OUT, "c4_rural_urban_gap.png"))
plt.close(f)

# ------------------------------------------------------------------ equations (mathtext -> transparent PNG)
plt.rcParams["mathtext.fontset"] = "stix"
EQ = {
    "e1_identity": r"$\pi^{H}_{s,t}\;=\;\tilde{w}_{s,t}\,\pi^{F}_{s,t}\;+\;(1-\tilde{w}_{s,t})\,\pi^{NF}_{s,t}$",
    "e1b_weight": r"$\tilde{w}_{s,t}\;=\;w_{s}\,\dfrac{I^{F}_{s,t-12}}{I^{H}_{s,t-12}}$",
    "e2_slope": r"$\beta_{s}\;=\;\dfrac{\mathrm{Cov}(\pi^{H}_{s},\pi^{F}_{s})}{\mathrm{Var}(\pi^{F}_{s})}\;=\;w_{s}+(1-w_{s})\,\dfrac{\mathrm{Cov}(\pi^{NF}_{s},\pi^{F}_{s})}{\mathrm{Var}(\pi^{F}_{s})}$",
    "e3_variance": r"$\mathrm{Var}(\pi^{H}_{s})\;=\;w_{s}^{2}\,\sigma^{2}_{F}+(1-w_{s})^{2}\,\sigma^{2}_{NF}+2\,w_{s}(1-w_{s})\,\sigma_{F,NF}$",
    "e4_gap": r"$\pi^{H}_{R}-\pi^{H}_{U}\;=\;(w_{R}-w_{U})(\bar{\pi}^{F}-\bar{\pi}^{NF})\;+\;\bar{w}\,(\pi^{F}_{R}-\pi^{F}_{U})\;+\;(1-\bar{w})(\pi^{NF}_{R}-\pi^{NF}_{U})$",
    "e4a_weight": r"$(w_{R}-w_{U})(\bar{\pi}^{F}-\bar{\pi}^{NF})$",
    "e4b_food": r"$\bar{w}\,(\pi^{F}_{R}-\pi^{F}_{U})$",
    "e4c_nonfood": r"$(1-\bar{w})(\pi^{NF}_{R}-\pi^{NF}_{U})$",
    "e5a_reg": r"$\pi^{H}_{s,t}\;=\;\alpha_{s}+\beta_{s}\,\pi^{F}_{s,t}+\varepsilon_{s,t}$",
    "e5b_gapreg": r"$g_{t}\;=\;c+\gamma\,(\bar{\pi}^{F}_{t}-\bar{\pi}^{NF}_{t})+u_{t}$",
    "e5c_ar": r"$g_{t}\;=\;c+\rho\,g_{t-1}+e_{t}\,,\quad \mathrm{half\text{-}life}=\dfrac{\ln 0.5}{\ln \rho}$",
    "e5d_tests": r"$H_{0}\!:\;\mu_{d}=0\,,\qquad H_{0}\!:\;\sigma^{2}_{1}=\sigma^{2}_{2}$",
}
sizes = {}
for name, tex in EQ.items():
    f = plt.figure(figsize=(0.1, 0.1), dpi=300)
    f.text(0, 0, tex, fontsize=30, color=INK)
    p = os.path.join(OUT, name + ".png")
    f.savefig(p, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(f)
    from PIL import Image
    sizes[name] = Image.open(p).size
json.dump(sizes, open(os.path.join(OUT, "equation_sizes.json"), "w"), indent=1)
print(sizes)
