"""Observational event-study and sensitivity estimates; no causal label is imposed."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import linalg, stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

def fit_cluster(y, x, absorb, cluster):
    """OLS with one absorbed FE and CR1 cluster covariance, t(G-1) inference."""
    names = list(x.columns)
    a, alevels = pd.factorize(absorb)
    g, glevels = pd.factorize(cluster)
    z = np.column_stack([np.asarray(y, float), x.to_numpy(dtype=float)])
    sums = np.zeros((len(alevels), z.shape[1]))
    np.add.at(sums, a, z)
    z -= (sums / np.bincount(a)[:, None])[a]
    yw, xw = z[:, 0], z[:, 1:]
    _, r, pivot = linalg.qr(xw, mode="economic", pivoting=True)
    rank = int((np.abs(np.diag(r)) > np.max(np.abs(np.diag(r))) * 1e-10).sum())
    cols = np.sort(pivot[:rank])
    xr = xw[:, cols]
    beta = linalg.lstsq(xr, yw)[0]
    resid = yw - xr @ beta
    bread = linalg.inv(xr.T @ xr)
    score = np.zeros((len(glevels), rank))
    np.add.at(score, g, xr * resid[:, None])
    n, k, ng = len(y), rank + len(alevels), len(glevels)
    cov = bread @ (score.T @ score) @ bread * ng / (ng - 1) * (n - 1) / (n - k)
    selected = [names[i] for i in cols]
    return {"beta": pd.Series(beta, index=selected), "cov": pd.DataFrame(cov, index=selected, columns=selected), "n": n, "k": k, "groups": ng, "absorbed_groups": len(alevels), "resid": resid, "r2_within": 1 - (resid @ resid)/(yw @ yw), "dropped": [names[i] for i in range(len(names)) if i not in cols]}

def infer(fit, weights):
    w = pd.Series(0., index=fit["beta"].index)
    for key, val in weights.items():
        if key not in w.index:
            raise ValueError("Unidentified requested coefficient: " + key)
        w[key] = val
    b = float(w @ fit["beta"])
    se = float(np.sqrt(max(0, w @ fit["cov"] @ w)))
    crit = stats.t.ppf(.975, fit["groups"] - 1)
    return {"estimate": b, "se": se, "lower": b-crit*se, "upper": b+crit*se, "p": float(2*stats.t.sf(abs(b/se), fit["groups"]-1)) if se else 0.0}

def design(d, radius=500, event=False, diversity=False, zip_year=False, repeat=False, trends=False):
    x = pd.DataFrame(index=d.index)
    for label, col in [("R", "residential"), ("O", "office"), ("K", "retail")]:
        x[label] = d[f"{col}_share_{radius}"]
    x["density"] = np.log(d[f"building_{radius}"] / (np.pi * radius**2))
    x["log_area"] = np.log(d.area)
    x["age100"] = ((d.year - d.year_built).clip(0, 250).fillna(75))/100
    x["age_missing"] = d.year_built.isna().astype(float)
    if diversity:
        sh = x[["R", "O", "K"]].to_numpy()
        # Evenness among the three reported focal uses; undefined all-zero areas excluded upstream.
        sh = sh / sh.sum(axis=1, keepdims=True)
        x["mix"] = 1.5 * (1 - (sh**2).sum(axis=1))
        interacted = ["mix", "density"]
    else:
        interacted = ["R", "O", "K", "density"]
    if event:
        for year in sorted(d.year.unique()):
            if year != 2019:
                for c in interacted:
                    x[f"{c}_{year}"] = x[c] * (d.year == year)
    else:
        for c in interacted:
            x[f"{c}_post"] = x[c] * (d.year >= 2021)
    if trends:
        for c in interacted:
            x[f"{c}_trend"] = x[c] * (d.year - 2019)
    by = d.borough.astype(str) + "_" + d.year.astype(str)
    if zip_year:
        by = d.zip_code.fillna(0).astype(int).astype(str) + "_" + d.year.astype(str)
    x = pd.concat([x, pd.get_dummies(by, prefix="time", dtype=float), pd.get_dummies(d["class"], prefix="class", dtype=float)], axis=1)
    return x

def clean():
    d = pd.read_csv(ROOT / "data/cre_sales_raw.csv.gz", parse_dates=["date"])
    flow = []
    def record(label, frame):
        flow.append({"step": label, "all": len(frame), **frame.sector.value_counts().to_dict()})
    record("CRE class universe", d)
    d = d.drop_duplicates(["bbl", "date", "price", "class", "area"])
    record("Remove exact transaction duplicates", d)
    # Do not infer a price allocation for ambiguous multiple records on the same parcel/date.
    ambiguous = d.groupby(["bbl", "date"])["price"].transform("nunique") > 1
    d = d[~ambiguous].drop_duplicates(["bbl", "date", "price"])
    record("Remove ambiguous same-parcel/date records", d)
    d = d[d.price.ge(100000) & d.area.between(250, 5000000)]
    record("Price >= $100k; reported area 250–5m sq ft", d)
    d["ppsf"] = d.price / d.area
    d = d[d.ppsf.between(20, 10000)]
    record("Price per sq ft $20–$10,000", d)
    d["package_flag"] = d.groupby(["borough", "date", "price"])["bbl"].transform("nunique").gt(1) & d.price.ge(1000000)
    e = pd.read_csv(ROOT / "data/neighborhood_exposures.csv.gz")
    d = d.merge(e, on="bbl", how="inner", validate="many_to_one")
    record("Matched to 2016 PLUTO coordinates", d)
    valid = d.building_500.gt(0)
    for c in ["residential", "office", "retail"]:
        valid &= d[f"{c}_share_500"].between(0, 1)
    d = d[valid & d.coverage_500.gt(0)]
    record("Valid 500m exposure ratios", d)
    # Hold the parcel's neighborhood identifier fixed across its observed transactions.
    d = d.sort_values(["bbl", "date"])
    d["neigh"] = d.borough.astype(int).astype(str) + "_" + d.groupby("bbl").neighborhood.transform("first")
    d["cd"] = d.CD.astype(str)
    d.loc[~d.year_built.between(1800, d.year), "year_built"] = np.nan
    d["log_ppsf"] = np.log(d.ppsf)
    d.to_csv(ROOT / "data/estimation_sample_with_package_flags.csv.gz", index=False)
    record("Exclude potential same-price/date multi-parcel sales", d[~d.package_flag])
    pd.DataFrame(flow).fillna(0).to_csv(OUT / "sample_flow.csv", index=False)
    return d

def main():
    d = clean()
    records, event_records, diagnostics = [], [], []
    for sector in ["Retail", "Office", "Multifamily", "Industrial"]:
        ds = d[d.sector == sector].copy()
        for label, rad, keep_packages, repeat, zip_year, diversity, cluster_cd, sameclass in [
            ("Primary", 500, False, False, False, False, False, False),
            ("Retain flagged packages", 500, True, False, False, False, False, False),
            ("250m radius", 250, False, False, False, False, False, False),
            ("1000m radius", 1000, False, False, False, False, False, False),
            ("Repeated parcels", 500, False, True, False, False, False, False),
            ("ZIP by year controls", 500, False, False, True, False, False, False),
            ("Community district clusters", 500, False, False, False, False, True, False),
            ("Stable broad building class", 500, False, False, False, False, False, True),
            ("Three-use evenness", 500, False, False, False, True, False, False),
        ]:
            s = ds[ds.year != 2020].copy()
            if not keep_packages:
                s = s[~s.package_flag]
            if sameclass:
                s = s[s["class"].str[0] == s.BldgClass.str[0]]
            if repeat:
                c = s.groupby("bbl").year.agg(["min", "max"])
                eligible = c[(c["min"] <= 2019) & (c["max"] >= 2021)].index
                s = s[s.bbl.isin(eligible)]
            valid = s[f"building_{rad}"].gt(0) & s[f"coverage_{rad}"].gt(0)
            for col in ["residential", "office", "retail"]:
                valid &= s[f"{col}_share_{rad}"].between(0, 1)
            s = s[valid].reset_index(drop=True)
            if len(s) < 100 or s.neigh.nunique() < 10:
                diagnostics.append({"sector": sector, "spec": label, "unavailable": "Insufficient repeated-parcel sample", "n":len(s)})
                continue
            fit = fit_cluster(s.log_ppsf, design(s, rad, diversity=diversity, zip_year=zip_year), s.bbl if repeat else s.neigh, s.cd if cluster_cd else s.neigh)
            weights = {"mix_post":.1} if diversity else {"R_post":.1, "O_post":-.1}
            rec = {"sector":sector, "spec":label, "n":fit["n"], "clusters":fit["groups"], "parcels":s.bbl.nunique(), "r2_within":fit["r2_within"], **infer(fit,weights)}
            records.append(rec)
            if label == "Primary":
                for c in ["R", "O", "K"]:
                    records.append({"sector":sector, "spec":"Primary coefficient " + c, "n":fit["n"], "clusters":fit["groups"], "parcels":s.bbl.nunique(), **infer(fit,{c+"_post":.1})})
            print(sector, label, len(s), round(rec["estimate"],4), round(rec["p"],4), flush=True)
        s = ds[~ds.package_flag].reset_index(drop=True)
        fit = fit_cluster(s.log_ppsf, design(s,event=True), s.neigh, s.neigh)
        for year in sorted(s.year.unique()):
            if year == 2019:
                rec = {"estimate":0., "se":0., "lower":0., "upper":0., "p":None}
            else:
                rec = infer(fit,{f"R_{year}":.1, f"O_{year}":-.1})
            event_records.append({"sector":sector, "year":int(year), "n":fit["n"], "clusters":fit["groups"], **rec})
        q = np.zeros((2,len(fit["beta"])))
        for j,yr in enumerate([2017,2018]):
            q[j,fit["beta"].index.get_loc(f"R_{yr}")] = .1
            q[j,fit["beta"].index.get_loc(f"O_{yr}")] = -.1
        qb = q @ fit["beta"].to_numpy(); qc = q @ fit["cov"].to_numpy() @ q.T
        fstat = float(qb @ linalg.solve(qc,qb)/2)
        diagnostics.append({"sector":sector,"spec":"Annual interactions", "n":fit["n"], "clusters":fit["groups"],"pretrend_f":fstat,"pretrend_p":float(stats.f.sf(fstat,2,fit["groups"]-1)),"dropped_columns":fit["dropped"]})
    pd.DataFrame(records).to_csv(OUT / "estimates.csv",index=False)
    pd.DataFrame(event_records).to_csv(OUT / "event_study.csv",index=False)
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
    d[~d.package_flag].groupby(["sector","year"]).agg(transactions=("price","size"),median_ppsf=("ppsf","median"),median_price=("price","median"),median_area=("area","median"),res_share=("residential_share_500","mean"),office_share=("office_share_500","mean")).reset_index().to_csv(OUT/"annual_description.csv",index=False)
    print("All specifications saved.", flush=True)

if __name__ == "__main__":
    main()
