"""Close the remaining gap in the measurement check: match the MAIN sample's request and agreement dates (the April 2021 relief
extract, as cached in out/inq_timing.parquet by round2_a.py) to the month-end loads, loan by loan. Aggregate output only.
Usage (secure machine):  python match_main_sample_to_loads.py <path to out/inq_timing.parquet>
Requires the same environment variables as contemporaneous_records.py, and its cached frame in AING_SECURE_TMP."""
import os, sys, json, numpy as np, pandas as pd
TMP = os.environ.get("AING_SECURE_TMP", "<SECURE_TMP>")
OUTJ = os.environ.get("AING_OUT3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "validation", "main_sample_match.json"))
D = pd.read_parquet(sys.argv[1]); L = pd.read_parquet(os.path.join(TMP, "frame_contemp.parquet"))
D.index = D.index.astype(str); L.index = L.index.astype(str)
D["r"] = (D.inq.dt.normalize() - pd.Timestamp("2020-04-07")).dt.days; W = D[D.r.between(-14, 14)].copy()
W = W.join(L[["inq", "agree", "inq_apr", "agr_apr"]].add_suffix("_loads"))
out = {}
for g in (0, 1):
    x = W[W.Gov == g]; inl = x.inq_loads.notna()
    same_inq = (x.inq.dt.normalize() == x.inq_loads.dt.normalize())
    later_in_loads = (x.inq_loads.dt.normalize() > x.inq.dt.normalize())
    fb = x.agree.notna() & (x.agree <= "2020-09-30")
    same_agr = (x.agree.dt.normalize() == x.agree_loads.dt.normalize())
    res = {"n_window": int(len(x)), "in_loads": 100 * float(inl.mean()),
           "same_first_inquiry": 100 * float(same_inq[inl].mean()), "loads_date_later": 100 * float(later_in_loads[inl].mean()),
           "same_agreement_if_fb": 100 * float(same_agr[inl & fb].mean()) if (inl & fb).any() else None,
           "on_apr_load_if_inq_by_28apr": 100 * float(x.loc[inl & (x.inq <= "2020-04-28"), "inq_apr_loads"].notna().mean())}
    for side in (0, 1):
        z = x[(x.r >= 0) == bool(side)]; zl = z.inq_loads.notna()
        res[f"side{side}"] = {"n": int(len(z)), "in_loads": 100 * float(zl.mean()), "completion_in_loads": 100 * float((z.agree.notna() & (z.agree <= "2020-09-30"))[zl].mean()) if zl.any() else None,
                              "completion_not_in_loads": 100 * float((z.agree.notna() & (z.agree <= "2020-09-30"))[~zl].mean()) if (~zl).any() else None}
    out["gov" if g else "conv"] = res
os.makedirs(os.path.dirname(OUTJ), exist_ok=True); json.dump(out, open(OUTJ, "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
