"""Exploratory 2024 multi-metro public-data association.

The input uses values displayed on Data USA metro profiles. Retail employment
is resident industry employment (ACS/PUMS), not destination retail jobs or
establishments. Total employment is recorded as the rounded value displayed
on each page. The estimates are descriptive and not causal.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "multicity_public_2024.csv"
OUT = ROOT / "results"


def robust_ols(x, y, weights=None):
    """OLS/WLS coefficient and HC1 covariance; t reference has n-k df."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = x.shape
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)
    xtwx = x.T @ (w[:, None] * x)
    xtwy = x.T @ (w * y)
    bread = np.linalg.inv(xtwx)
    beta = bread @ xtwy
    resid = y - x @ beta
    meat = sum((w[i] * resid[i]) ** 2 * np.outer(x[i], x[i]) for i in range(n))
    vcov = bread @ meat @ bread * (n / (n - k))
    se = np.sqrt(np.diag(vcov))
    crit = t_dist.ppf(0.975, n - k)
    ci = np.column_stack((beta - crit * se, beta + crit * se))
    stat = beta / se
    p = 2 * t_dist.sf(np.abs(stat), n - k)
    return beta, se, ci, stat, p


def main():
    df = pd.read_csv(DATA)
    df["retail_employment_share_pct"] = 100 * df["retail_employment"] / df["total_employment_rounded"]
    x = np.column_stack((np.ones(len(df)), df["wfh_pct"].to_numpy()))
    y = df["retail_employment_share_pct"].to_numpy()
    beta, se, ci, stat, p = robust_ols(x, y)
    beta_w, se_w, ci_w, stat_w, p_w = robust_ols(x, y, df["total_employment_rounded"])

    # A size-control sensitivity is intentionally reported as exploratory only.
    x_pop = np.column_stack((np.ones(len(df)), df["wfh_pct"], np.log(df["total_employment_rounded"])))
    beta_pop, se_pop, ci_pop, stat_pop, p_pop = robust_ols(x_pop, y)

    df.to_csv(OUT / "multicity_public_2024.csv", index=False, float_format="%.6f")
    results = {
        "n_metros": int(len(df)),
        "outcome": "resident retail-employment share (percent)",
        "exposure": "ACS 2024 Worked At Home share (percent)",
        "total_employment_note": "rounded values displayed by Data USA; see input and protocol",
        "ols_hc1": {
            "intercept": float(beta[0]), "wfh_slope_pp_per_pp": float(beta[1]),
            "se": float(se[1]), "ci95": [float(ci[1, 0]), float(ci[1, 1])],
            "p_value": float(p[1]), "ten_pp_contrast": float(10 * beta[1]),
            "ten_pp_ci95": [float(10 * ci[1, 0]), float(10 * ci[1, 1])],
        },
        "employment_weighted_hc1": {
            "wfh_slope_pp_per_pp": float(beta_w[1]), "se": float(se_w[1]),
            "ci95": [float(ci_w[1, 0]), float(ci_w[1, 1])],
            "p_value": float(p_w[1]), "ten_pp_contrast": float(10 * beta_w[1]),
        },
        "log_employment_control_hc1": {
            "wfh_slope_pp_per_pp": float(beta_pop[1]), "se": float(se_pop[1]),
            "ci95": [float(ci_pop[1, 0]), float(ci_pop[1, 1])],
            "p_value": float(p_pop[1]), "ten_pp_contrast": float(10 * beta_pop[1]),
        },
    }
    (OUT / "multicity_public_2024.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
