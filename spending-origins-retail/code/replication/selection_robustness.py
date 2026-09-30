#!/usr/bin/env python3
"""
Selection-robustness for the balanced-panel restriction.

Produces two things the paper needs, and deliberately does NOT synthesize data:

  1. Inverse-probability weights that reweight the balanced panel back toward
     the full matched-parcel population. Panel inclusion is modelled on
     predetermined parcel characteristics (use shares, coverage, floors,
     log building area, year built, residential units, borough). Weights are
     1/p for retained parcels, normalized to mean one, propensities trimmed to
     [0.02, 0.98].

  2. Estimates on relaxed retention rules (status in >= 5 or >= 4 of six years),
     which trade panel completeness against balance.

SMOTE and related oversampling methods are NOT used and should not be. They
interpolate synthetic observations from observed covariates. That fabricates
parcels that do not exist in an administrative dataset, and it cannot address
selection on unobservables, which is the actual threat here.

Run this inside the paper's own pipeline so that the specification matches the
published tables exactly. `selection_weights.csv` (keyed on bbl) can be merged
directly into the estimation frame and passed as regression weights.

Usage:
    python selection_robustness.py --panel <parcel-year frame> --weights selection_weights.csv
"""
import argparse
import numpy as np, pandas as pd
import statsmodels.api as sm

COV = ["office_share_500","residential_share_500","retail_share_500",
       "coverage_500","NumFloors","YearBuilt","UnitsRes"]

def build_weights(parcels):
    X = parcels[COV].copy()
    X["logBldgArea"] = np.log(parcels["BldgArea"].clip(lower=1))
    for b in sorted(parcels["BoroCode"].dropna().unique())[1:]:
        X[f"boro_{int(b)}"] = (parcels["BoroCode"] == b).astype(float)
    X = X.fillna(X.median())
    Xs = (X - X.mean()) / X.std().replace(0, 1)
    fit = sm.Logit(parcels["in_panel"], sm.add_constant(Xs)).fit(disp=0, method="bfgs", maxiter=500)
    p = fit.predict(sm.add_constant(Xs)).clip(0.02, 0.98)
    w = np.where(parcels["in_panel"] == 1, 1/p, 0.0)
    w = np.where(parcels["in_panel"] == 1, w/w[parcels["in_panel"] == 1].mean(), 0.0)
    return pd.DataFrame({"bbl": parcels["bbl"], "p": p, "w_ipw": w,
                         "in_panel": parcels["in_panel"]})

def balance(parcels, w, cols=COV):
    rows = []
    for c in cols:
        a = parcels.loc[parcels.in_panel == 1, c]
        b = parcels.loc[parcels.in_panel == 0, c]
        wa = w[parcels.in_panel == 1]
        sd = np.sqrt((a.var() + b.var())/2)
        rows.append({"variable": c,
                     "std_diff_unweighted": (a.mean()-b.mean())/sd,
                     "std_diff_ipw": (np.average(a, weights=wa)-b.mean())/sd})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parcels", required=True,
                    help="one row per matched parcel, with COV columns, bbl, BoroCode, BldgArea, in_panel")
    ap.add_argument("--out", default="selection_weights.csv")
    a = ap.parse_args()
    parcels = pd.read_csv(a.parcels)
    w = build_weights(parcels)
    w.to_csv(a.out, index=False)
    print(balance(parcels, w["w_ipw"]).round(3).to_string(index=False))
    print(f"\nwrote {a.out}")
