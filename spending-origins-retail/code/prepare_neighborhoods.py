"""Predetermined, leave-one-tax-lot-out land-use measures from PLUTO 16v2."""
from pathlib import Path
import gc
import json
import zipfile
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"
COLS = ["BBL", "BoroCode", "BldgClass", "BldgArea", "ResArea", "OfficeArea", "RetailArea", "LotArea", "XCoord", "YCoord", "YearBuilt", "NumFloors", "UnitsRes", "AreaSource", "CD", "Version"]
parts = []
with zipfile.ZipFile(ROOT / "raw/nyc_pluto_16v2.zip") as z:
    for name in z.namelist():
        if name.lower().endswith(".csv"):
            with z.open(name) as f:
                d = pd.read_csv(f, usecols=COLS, low_memory=False, encoding="latin1")
            parts.append(d)
            print(name, len(d), flush=True)
p = pd.concat(parts, ignore_index=True)
del parts
for c in set(COLS) - {"BldgClass", "Version"}:
    p[c] = pd.to_numeric(p[c], errors="coerce")
assert not p.BBL.duplicated().any(), "Duplicate 2016 tax lot"
p["BBL"] = p.BBL.astype("int64")
summary = {"pluto_records": len(p), "versions": p.Version.value_counts().to_dict()}
valid = p.XCoord.between(850000, 1100000) & p.YCoord.between(100000, 300000)
summary["invalid_coordinates"] = int((~valid).sum())
p = p.loc[valid].reset_index(drop=True)
p.to_csv(OUT / "pluto_2016_relevant_fields.csv.gz", index=False)
sales = pd.read_csv(OUT / "cre_sales_raw.csv.gz", parse_dates=["date"])
focal = p[p.BBL.isin(sales.bbl.unique())].copy()
focal_idx = focal.index.to_numpy()
print("Focal matched tax lots:", len(focal), flush=True)
xy = p[["XCoord", "YCoord"]].to_numpy(dtype=float)
tree = cKDTree(xy)
areas = p[["BldgArea", "ResArea", "OfficeArea", "RetailArea"]].fillna(0).clip(lower=0).to_numpy(dtype=float)
leisure = p.BldgClass.str.startswith(("J", "P"), na=False).to_numpy(dtype=int)
food = p.BldgClass.eq("K5").to_numpy(dtype=int)
values = np.column_stack([areas, leisure, food, np.ones(len(p))])
radii = [250, 500, 1000]
# New York Long Island State Plane coordinates use US survey feet.
feet_per_meter = 3937 / 1200
results = np.zeros((len(focal), len(radii), values.shape[1]))
for start in range(0, len(focal), 256):
    batch = focal_idx[start:start + 256]
    neighbors = tree.query_ball_point(xy[batch], radii[-1] * feet_per_meter, workers=1)
    for offset, (idx, nbs) in enumerate(zip(batch, neighbors)):
        nbs = np.asarray(nbs, dtype=int)
        nbs = nbs[nbs != idx]
        distance = np.sqrt(((xy[nbs] - xy[idx]) ** 2).sum(axis=1)) / feet_per_meter
        for ri, radius in enumerate(radii):
            results[start + offset, ri] = values[nbs[distance <= radius]].sum(axis=0)
    if start % 5120 == 0:
        print("Neighborhoods", start, "/", len(focal), flush=True)
focal = focal.rename(columns={"BBL": "bbl"}).reset_index(drop=True)
for ri, radius in enumerate(radii):
    for ci, name in enumerate(["building", "residential", "office", "retail", "culture_count", "food_count", "lot_count"]):
        focal[f"{name}_{radius}"] = results[:, ri, ci]
    for name in ["residential", "office", "retail"]:
        focal[f"{name}_share_{radius}"] = focal[f"{name}_{radius}"] / focal[f"building_{radius}"].replace(0, np.nan)
    focal[f"coverage_{radius}"] = focal[[f"{x}_{radius}" for x in ["residential", "office", "retail"]]].sum(axis=1) / focal[f"building_{radius}"].replace(0, np.nan)
focal.to_csv(OUT / "neighborhood_exposures.csv.gz", index=False)
summary["matched_focal_lots"] = len(focal)
summary["coverage_500_quantiles"] = focal.coverage_500.quantile([0, .01, .5, .99, 1]).to_dict()
summary["shares_above_one_500"] = {x: int((focal[f"{x}_share_500"] > 1).sum()) for x in ["residential", "office", "retail"]}
(OUT / "neighborhood_construction.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
