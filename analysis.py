"""
Food inflation vs overall inflation in rural and urban India
ECON F213 (Mathematical and Statistical Methods) presentation -- empirical work

Data  : MoSPI All-India CPI (General) and Consumer Food Price Index (CFPI),
        Rural / Urban / Combined, base 2012=100, Jan 2013 - Dec 2025
        (downloaded from the MoSPI eSankhyiki API, api.mospi.gov.in/api/cpi/getCPIIndex)
Run   : python analysis.py        (needs pandas, numpy, scipy, statsmodels)
Output: results.json and results.txt in the same folder
"""
import json
import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint, kpss

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
W = {"rural": 0.4725, "urban": 0.2962, "combined": 0.3906}   # CFPI weight in CPI (2012 base)
HAC_LAGS = 12                                                  # y-o-y rates overlap 12 months

df = pd.read_csv(os.path.join(HERE, "data", "cpi_india_2012base.csv"))
df["month"] = pd.to_datetime(df["month"])
df = df.set_index("month")

# ---------------------------------------------------------------- 1. build series
out = pd.DataFrame(index=df.index)
for s, w in W.items():
    H, F = df[f"cpi_{s}"], df[f"cfpi_{s}"]
    NF = (H - w * F) / (1 - w)                    # implied non-food index (Laspeyres identity)
    out[f"H_{s}"] = 100 * (H / H.shift(12) - 1)   # headline inflation, % y-o-y
    out[f"F_{s}"] = 100 * (F / F.shift(12) - 1)   # food inflation
    out[f"NF_{s}"] = 100 * (NF / NF.shift(12) - 1)
    out[f"weff_{s}"] = w * F.shift(12) / H.shift(12)   # effective food weight in y-o-y inflation
    out[f"Fcontrib_{s}"] = out[f"weff_{s}"] * out[f"F_{s}"]
out = out.dropna()
out["gap_H"] = out["H_rural"] - out["H_urban"]
out["gap_F"] = out["F_rural"] - out["F_urban"]
out["wedge_rural"] = out["F_rural"] - out["H_rural"]
out["wedge_urban"] = out["F_urban"] - out["H_urban"]
N = len(out)
R = {"sample": [str(out.index[0].date()), str(out.index[-1].date())], "N": N, "weights": W}

# identity check: headline = weff*food + (1-weff)*nonfood
chk = (out["weff_rural"] * out["F_rural"] + (1 - out["weff_rural"]) * out["NF_rural"] - out["H_rural"]).abs().max()
R["identity_max_abs_error_pp"] = float(chk)


def hac_mean(x):
    """mean with Newey-West (HAC) standard error -> (mean, se, t, p)"""
    m = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return dict(mean=float(m.params[0]), se=float(m.bse[0]), t=float(m.tvalues[0]), p=float(m.pvalues[0]))


# ---------------------------------------------------------------- 2. descriptive statistics
desc = {}
for c in ["H_rural", "H_urban", "H_combined", "F_rural", "F_urban", "F_combined", "NF_rural", "NF_urban", "NF_combined"]:
    x = out[c]
    desc[c] = dict(mean=x.mean(), median=x.median(), sd=x.std(ddof=1), min=x.min(), max=x.max(),
                   cv=x.std(ddof=1) / x.mean(), ar1=x.autocorr(1),
                   min_month=str(x.idxmin().date())[:7], max_month=str(x.idxmax().date())[:7])
R["descriptive"] = desc
R["corr"] = {
    "H_rural~H_urban": out["H_rural"].corr(out["H_urban"]),
    "F_rural~F_urban": out["F_rural"].corr(out["F_urban"]),
    "H_rural~F_rural": out["H_rural"].corr(out["F_rural"]),
    "H_urban~F_urban": out["H_urban"].corr(out["F_urban"]),
    "F_rural~NF_rural": out["F_rural"].corr(out["NF_rural"]),
    "F_urban~NF_urban": out["F_urban"].corr(out["NF_urban"]),
}

# ---------------------------------------------------------------- 3. tests on means (paired)
tests = {}
pairs = {
    "food_minus_headline_rural": out["wedge_rural"],
    "food_minus_headline_urban": out["wedge_urban"],
    "rural_minus_urban_headline": out["gap_H"],
    "rural_minus_urban_food": out["gap_F"],
}
for k, x in pairs.items():
    t_naive = st.ttest_1samp(x, 0.0)
    tests[k] = dict(naive_t=float(t_naive.statistic), naive_p=float(t_naive.pvalue), sd=float(x.std(ddof=1)),
                    share_positive=float((x > 0).mean()), mean_abs=float(x.abs().mean()), **hac_mean(x))
R["mean_tests"] = tests

# ---------------------------------------------------------------- 4. tests on variances
var = {}
for k, (a, b) in {"food_vs_headline_rural": ("F_rural", "H_rural"), "food_vs_headline_urban": ("F_urban", "H_urban"),
                  "food_vs_nonfood_rural": ("F_rural", "NF_rural"), "food_vs_nonfood_urban": ("F_urban", "NF_urban"),
                  "headline_rural_vs_urban": ("H_rural", "H_urban"), "food_urban_vs_rural": ("F_urban", "F_rural")}.items():
    Fst = out[a].var(ddof=1) / out[b].var(ddof=1)
    p_f = 2 * min(st.f.cdf(Fst, N - 1, N - 1), 1 - st.f.cdf(Fst, N - 1, N - 1))
    lev = st.levene(out[a], out[b], center="median")
    var[k] = dict(var_ratio=float(Fst), f_p=float(p_f), levene_stat=float(lev.statistic), levene_p=float(lev.pvalue))
R["variance_tests"] = var

# moving-block bootstrap (block = 12 months) for SD ratios: robust to serial correlation and pairing
rng = np.random.default_rng(213)
B, L = 5000, 12
starts = np.arange(N - L + 1)
vals = out[["H_rural", "H_urban", "F_rural", "F_urban"]].values
ratios = {"sd_Hrural_over_Hurban": [], "sd_Frural_over_Furban": [], "sd_Frural_over_Hrural": [], "sd_Furban_over_Hurban": []}
for _ in range(B):
    idx = np.concatenate([np.arange(s0, s0 + L) for s0 in rng.choice(starts, size=N // L)])
    sd = vals[idx].std(axis=0, ddof=1)
    ratios["sd_Hrural_over_Hurban"].append(sd[0] / sd[1])
    ratios["sd_Frural_over_Furban"].append(sd[2] / sd[3])
    ratios["sd_Frural_over_Hrural"].append(sd[2] / sd[0])
    ratios["sd_Furban_over_Hurban"].append(sd[3] / sd[1])
sd0 = out[["H_rural", "H_urban", "F_rural", "F_urban"]].std(ddof=1)
point = {"sd_Hrural_over_Hurban": sd0["H_rural"] / sd0["H_urban"], "sd_Frural_over_Furban": sd0["F_rural"] / sd0["F_urban"],
         "sd_Frural_over_Hrural": sd0["F_rural"] / sd0["H_rural"], "sd_Furban_over_Hurban": sd0["F_urban"] / sd0["H_urban"]}
R["sd_ratio_block_bootstrap"] = {k: dict(point=float(point[k]), ci95_lo=float(np.percentile(v, 2.5)), ci95_hi=float(np.percentile(v, 97.5)),
                                         share_above_1=float(np.mean(np.array(v) > 1))) for k, v in ratios.items()}

# ---------------------------------------------------------------- 5. pass-through regressions  H = a + b*F
reg = {}
for s in ["rural", "urban", "combined"]:
    X = sm.add_constant(out[[f"F_{s}"]])
    m = sm.OLS(out[f"H_{s}"], X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    m0 = sm.OLS(out[f"H_{s}"], X).fit()
    b, se = m.params[f"F_{s}"], m.bse[f"F_{s}"]
    weff = out[f"weff_{s}"].mean()
    z = (b - weff) / se
    # omitted-variable identity: b = weff + (1-weff)*cov(NF,F)/var(F)   (approx., fixed weights)
    slope_nf = np.cov(out[f"NF_{s}"], out[f"F_{s}"])[0, 1] / out[f"F_{s}"].var(ddof=1)
    reg[s] = dict(alpha=float(m.params["const"]), alpha_se=float(m.bse["const"]), beta=float(b), beta_se_hac=float(se),
                  beta_se_ols=float(m0.bse[f"F_{s}"]), r2=float(m.rsquared), dw=float(sm.stats.durbin_watson(m.resid)),
                  mean_eff_weight=float(weff), z_beta_eq_weight=float(z), p_beta_eq_weight=float(2 * (1 - st.norm.cdf(abs(z)))),
                  slope_nonfood_on_food=float(slope_nf), implied_beta=float(weff + (1 - weff) * slope_nf))
# pooled with urban interaction: test beta_rural = beta_urban
pool = pd.concat([
    pd.DataFrame({"H": out["H_rural"].values, "F": out["F_rural"].values, "U": 0.0}),
    pd.DataFrame({"H": out["H_urban"].values, "F": out["F_urban"].values, "U": 1.0})], ignore_index=True)
pool["FU"] = pool["F"] * pool["U"]
mp = sm.OLS(pool["H"], sm.add_constant(pool[["F", "U", "FU"]])).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
reg["pooled_interaction"] = dict(beta_diff_urban_minus_rural=float(mp.params["FU"]), se=float(mp.bse["FU"]),
                                 t=float(mp.tvalues["FU"]), p=float(mp.pvalues["FU"]))
R["passthrough"] = reg

# ---------------------------------------------------------------- 6. decomposing the rural-urban gap
a, b = out["weff_rural"], out["weff_urban"]
Fbar, NFbar = (out["F_rural"] + out["F_urban"]) / 2, (out["NF_rural"] + out["NF_urban"]) / 2
wbar = (a + b) / 2
out["dec_weight"] = (a - b) * (Fbar - NFbar)                       # same prices, different baskets
out["dec_foodprice"] = wbar * (out["F_rural"] - out["F_urban"])    # different food inflation
out["dec_nonfood"] = (1 - wbar) * (out["NF_rural"] - out["NF_urban"])
resid = (out["gap_H"] - out[["dec_weight", "dec_foodprice", "dec_nonfood"]].sum(axis=1)).abs().max()
vg = out["gap_H"].var(ddof=1)
dec = {"max_abs_residual_pp": float(resid), "mean_gap": float(out["gap_H"].mean())}
for c in ["dec_weight", "dec_foodprice", "dec_nonfood"]:
    dec[c] = dict(mean=float(out[c].mean()), mean_abs=float(out[c].abs().mean()),
                  var_share=float(np.cov(out[c], out["gap_H"])[0, 1] / vg))
dec["food_total_var_share"] = dec["dec_weight"]["var_share"] + dec["dec_foodprice"]["var_share"]
R["gap_decomposition"] = dec

# gap regression: gap = c + g*(Fbar - NFbar); theory (weight channel alone) says g = w_R - w_U
out["rel_food"] = Fbar - NFbar
mg = sm.OLS(out["gap_H"], sm.add_constant(out[["rel_food"]])).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
g, gse = mg.params["rel_food"], mg.bse["rel_food"]
wd = float((a - b).mean())
R["gap_regression"] = dict(const=float(mg.params["const"]), const_se=float(mg.bse["const"]), gamma=float(g), gamma_se=float(gse),
                           t_gamma_eq_0=float(g / gse), p_gamma_eq_0=float(mg.pvalues["rel_food"]), r2=float(mg.rsquared),
                           weight_diff=wd, z_gamma_eq_wdiff=float((g - wd) / gse),
                           p_gamma_eq_wdiff=float(2 * (1 - st.norm.cdf(abs((g - wd) / gse)))),
                           corr_gap_relfood=float(out["gap_H"].corr(out["rel_food"])))
# two-regressor version: add the rural-urban food inflation differential
mg2 = sm.OLS(out["gap_H"], sm.add_constant(out[["rel_food", "gap_F"]])).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
R["gap_regression_2"] = dict(params={k: float(v) for k, v in mg2.params.items()}, se={k: float(v) for k, v in mg2.bse.items()},
                             p={k: float(v) for k, v in mg2.pvalues.items()}, r2=float(mg2.rsquared))

# ---------------------------------------------------------------- 7. is the gap temporary?  (convergence)
conv = {}
for k in ["gap_H", "gap_F", "wedge_rural", "wedge_urban"]:
    x = out[k]
    adf = adfuller(x, autolag="AIC", regression="c")
    ar = sm.OLS(x.iloc[1:].values, sm.add_constant(x.shift(1).iloc[1:].values)).fit()
    rho = float(ar.params[1])
    adf_b = adfuller(x, autolag="BIC", regression="c")
    kp = kpss(x, regression="c", nlags="auto")
    conv[k] = dict(adf_aic_stat=float(adf[0]), adf_aic_p=float(adf[1]), adf_aic_lags=int(adf[2]), adf_crit5=float(adf[4]["5%"]),
                   adf_bic_stat=float(adf_b[0]), adf_bic_p=float(adf_b[1]), adf_bic_lags=int(adf_b[2]),
                   kpss_stat=float(kp[0]), kpss_p_lower_bound=float(kp[1]), kpss_crit5=float(kp[3]["5%"]),
                   ar1_rho=rho, ar1_se=float(ar.bse[1]), half_life_months=float(np.log(0.5) / np.log(rho)))
for k in ["H_rural", "H_urban", "F_rural", "F_urban"]:
    adf = adfuller(out[k], autolag="AIC", regression="c")
    conv["level_" + k] = dict(adf_stat=float(adf[0]), adf_p=float(adf[1]), adf_lags=int(adf[2]))
eg = coint(out["H_urban"], out["H_rural"], trend="c", autolag="AIC")
conv["engle_granger_urban_on_rural"] = dict(stat=float(eg[0]), p=float(eg[1]), crit5=float(eg[2][1]))
lr = sm.OLS(out["H_urban"], sm.add_constant(out[["H_rural"]])).fit()
ect = lr.resid
d = pd.DataFrame({"dU": out["H_urban"].diff(), "dR": out["H_rural"].diff(), "ect1": ect.shift(1)}).dropna()
ecm_u = sm.OLS(d["dU"], sm.add_constant(d[["dR", "ect1"]])).fit()
ecm_r = sm.OLS(d["dR"], sm.add_constant(d[["dU", "ect1"]])).fit()
conv["long_run_slope_urban_on_rural"] = float(lr.params["H_rural"])
conv["ecm_urban"] = dict(ect=float(ecm_u.params["ect1"]), se=float(ecm_u.bse["ect1"]), p=float(ecm_u.pvalues["ect1"]), r2=float(ecm_u.rsquared))
conv["ecm_rural"] = dict(ect=float(ecm_r.params["ect1"]), se=float(ecm_r.bse["ect1"]), p=float(ecm_r.pvalues["ect1"]))
R["convergence"] = conv

# lagged co-movement of non-food with food inflation (a crude look at second-round effects)
R["crosscorr_food_lead_nonfood"] = {s_: {f"lag{k}": float(out[f"F_{s_}"].shift(k).corr(out[f"NF_{s_}"])) for k in [0, 3, 6, 12, 18, 24]}
                                    for s_ in ["rural", "urban", "combined"]}

# ---------------------------------------------------------------- 8. sub-periods
sub = {}
for name, (lo, hi) in {"2014-2019": ("2014-01", "2019-12"), "2020-2022": ("2020-01", "2022-12"), "2023-2025": ("2023-01", "2025-12")}.items():
    x = out.loc[lo:hi]
    sub[name] = {c: dict(mean=float(x[c].mean()), sd=float(x[c].std(ddof=1))) for c in
                 ["H_rural", "H_urban", "F_rural", "F_urban", "gap_H", "gap_F"]}
    sub[name]["n"] = len(x)
R["subperiods"] = sub

# episodes of the rural-urban gap
R["gap_extremes"] = dict(max=float(out["gap_H"].max()), max_month=str(out["gap_H"].idxmax().date())[:7],
                         min=float(out["gap_H"].min()), min_month=str(out["gap_H"].idxmin().date())[:7],
                         share_abs_gt_1pp=float((out["gap_H"].abs() > 1).mean()))

out.round(4).to_csv(os.path.join(HERE, "data", "inflation_series_derived.csv"))
with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(R, f, indent=1, default=float)


def show(d, ind=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print(" " * ind + str(k) + ":")
            show(v, ind + 2)
        elif isinstance(v, float):
            print(" " * ind + f"{k}: {v:.4f}")
        else:
            print(" " * ind + f"{k}: {v}")


show(R)
