"""Transcribe official DOF annual sales, retaining a documented CRE universe."""
from pathlib import Path
import json
import re
import gc
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)
frames, counts = [], []
cache = OUT / "sales_cache"
cache.mkdir(exist_ok=True)
for path in sorted((ROOT / "raw/sales").glob("*.xls*")):
    cached = cache / (path.name + ".csv.gz")
    countpath = cache / (path.name + ".json")
    if cached.exists() and countpath.exists():
        frames.append(pd.read_csv(cached, parse_dates=["date"]))
        counts.append(json.loads(countpath.read_text()))
        continue
    raw = pd.read_excel(path, header=None)
    h = next(i for i in range(15) if str(raw.iloc[i, 0]).strip() == "BOROUGH")
    df = raw.iloc[h + 1:].copy()
    df.columns = [re.sub(r"\s+", " ", str(x)).strip().lower().replace(" ", "_") for x in raw.iloc[h]]
    keep = ["borough", "neighborhood", "block", "lot", "zip_code", "residential_units", "commercial_units", "total_units", "land_square_feet", "gross_square_feet", "year_built", "building_class_at_time_of_sale", "sale_price", "sale_date"]
    df = df[keep].rename(columns={"building_class_at_time_of_sale": "class", "gross_square_feet": "area", "land_square_feet": "land", "sale_price": "price", "sale_date": "date"})
    for col in ["borough", "block", "lot", "zip_code", "residential_units", "commercial_units", "total_units", "land", "area", "year_built", "price"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip(), errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["class"] = df["class"].astype(str).str.strip().str.upper()
    df["neighborhood"] = df["neighborhood"].astype(str).str.strip()
    df = df[df.borough.between(1, 5) & df.date.notna()]
    n_all = len(df)
    df["year"] = df.date.dt.year
    source_year = int(path.name[:4])
    n_wrong_year = int((df.year != source_year).sum())
    df = df[df.year == source_year]
    df["sector"] = df["class"].str[0].map({"K": "Retail", "O": "Office", "C": "Multifamily", "D": "Multifamily", "E": "Industrial", "F": "Industrial"})
    df = df[df.sector.notna() & ~df["class"].isin(["C6", "C8", "CC", "D0", "D4", "DC"])].copy()
    df["bbl"] = (df.borough * 10**9 + df.block * 10**4 + df.lot).astype("Int64")
    df["source_file"] = path.name
    counts.append({"file": path.name, "dated_sales": n_all, "year_mismatch": n_wrong_year, "cre_records_before_filters": len(df)})
    df.to_csv(cached, index=False)
    countpath.write_text(json.dumps(counts[-1]))
    frames.append(df)
    print(path.name, n_all, len(df), flush=True)
    del raw
    gc.collect()
all_cre = pd.concat(frames, ignore_index=True)
all_cre.to_csv(OUT / "cre_sales_raw.csv.gz", index=False)
pd.DataFrame(counts).to_csv(OUT / "sales_source_counts.csv", index=False)
print(all_cre.groupby(["year", "sector"]).size().unstack().to_string())
