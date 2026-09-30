#!/usr/bin/env python3
"""
Build the sample-selection table: 45,317 matched parcels -> 13,826 balanced panel.

This CANNOT be built from the shared replication extract, which carries parcel
exposures but no panel-membership flag. Run it where the LL157 registry and the
panel-construction step live.

Required inputs
---------------
  exposures : storefront_exposures.csv.gz  (the 45,317 matched parcels; has bbl)
  panel     : any file listing the bbl values that survive into the balanced
              analysis panel -- the estimation frame, or an intermediate with
              one row per retained parcel.

Usage
-----
  python build_selection_table.py --exposures data/storefront_exposures.csv.gz \
                                  --panel data/storefront_panel.csv.gz \
                                  --bbl-col bbl
Output
------
  results/selection_table.csv  and a LaTeX fragment on stdout.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

VARS = {
    "exposure":             "Residential-for-office contrast (500 m)",
    "office_share_500":     "Office floor-area share (500 m)",
    "residential_share_500":"Residential floor-area share (500 m)",
    "retail_share_500":     "Retail floor-area share (500 m)",
    "coverage_500":         "Built coverage (500 m)",
    "BldgArea":             "Building area (sq ft)",
    "NumFloors":            "Floors",
    "YearBuilt":            "Year built",
    "UnitsRes":             "Residential units",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposures", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--bbl-col", default="bbl")
    ap.add_argument("--out", default="results/selection_table.csv")
    a = ap.parse_args()

    e = pd.read_csv(a.exposures)
    e["exposure"] = e["residential_share_500"] - e["office_share_500"]
    kept = set(pd.read_csv(a.panel, usecols=[a.bbl_col])[a.bbl_col].astype(str))
    e["in_panel"] = e[a.bbl_col].astype(str).isin(kept)

    print(f"matched parcels : {len(e):,}")
    print(f"in balanced panel: {e.in_panel.sum():,}")
    print(f"dropped          : {(~e.in_panel).sum():,}")

    rows = []
    for col, label in VARS.items():
        if col not in e.columns:
            continue
        a_, b_ = e.loc[e.in_panel, col].dropna(), e.loc[~e.in_panel, col].dropna()
        # Welch t on the difference in means
        se = np.sqrt(a_.var(ddof=1)/len(a_) + b_.var(ddof=1)/len(b_))
        rows.append({"Variable": label,
                     "Panel mean": a_.mean(), "Panel SD": a_.std(),
                     "Dropped mean": b_.mean(), "Dropped SD": b_.std(),
                     "Difference": a_.mean()-b_.mean(),
                     "t": (a_.mean()-b_.mean())/se if se > 0 else np.nan,
                     "Std. diff": (a_.mean()-b_.mean()) /
                                  np.sqrt((a_.var(ddof=1)+b_.var(ddof=1))/2)})

    # composition by borough and broad building class
    for key, name in [("BoroCode", "Borough"), ("BldgClass", "Building class")]:
        if key not in e.columns:
            continue
        g = e.groupby(key)["in_panel"].agg(["size", "mean"])
        print(f"\nRetention rate by {name.lower()}:")
        print((g.assign(retained=lambda t: (100*t["mean"]).round(1))
                 [["size", "retained"]]).head(30).to_string())

    t = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False)
    print("\n" + t.round(3).to_string(index=False))
    print(f"\nwrote {a.out}")
    print("\nStandardized differences above 0.10 in absolute value are the ones "
          "to discuss in the text.")

if __name__ == "__main__":
    main()
