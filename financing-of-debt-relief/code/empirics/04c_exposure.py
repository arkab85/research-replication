"""Issuer-level funding pressure. The model's budget is K = A - M, where M is the cash that existing
obligations absorb; for a Ginnie Mae issuer the largest such obligation is the stock of principal and
interest advanced on delinquent loans. The issuer-month summary built from the disclosures gives, for
every issuer's whole portfolio, the outstanding advances (scheduled payment times months delinquent,
summed over loans) relative to unpaid balance. This script relates issuers' exercise to their own
advance burden within issuer and month, separately for nonbanks (capped facilities) and depositories
(deposit funding), which serve as the placebo for the value channel.

Outcomes at the issuer-month level (aggregated from the decision cells): funded share, relative
response, level response, and the two benchmarks. Loan-level: a logit with cell fixed effects and the
interaction of the advance burden with the ranking index (a fixed ranking predicts zero). A second
exposure, the issuer's 2019 state mix times the February-April 2020 rise in state unemployment,
appears as a shift-share cross-section in the appendix table."""
import pandas as pd, numpy as np, json
from felogit import fe_logit
from config import WORK, OUT, ISSUER_MONTHLY, STATE_UR, PERIODS
PER = [p for p, _, _ in PERIODS]

def wls_fe(df, y, x, w, fe_cols, cluster, iters=200):
    """Weighted least squares with several fixed effects (iterative weighted demeaning), cluster-robust SE."""
    d = df[list(dict.fromkeys([y] + x + [w] + fe_cols + [cluster]))].dropna().copy()
    W = d[w].values.astype(float); cols = [y] + x
    Z = d[cols].values.astype(float)
    codes = {f: pd.factorize(d[f].values)[0] for f in fe_cols}
    for _ in range(iters):
        Zold = Z.copy()
        for f in fe_cols:
            g = codes[f]; K = g.max() + 1
            den = np.bincount(g, W, K)
            for j in range(Z.shape[1]):
                Z[:, j] = Z[:, j] - (np.bincount(g, W * Z[:, j], K) / den)[g]
        if np.max(np.abs(Z - Zold)) < 1e-10: break
    yv = Z[:, 0]; X = Z[:, 1:]
    sw = np.sqrt(W); A = (X * W[:, None]).T @ X; b = np.linalg.solve(A, (X * W[:, None]).T @ yv)
    e = yv - X @ b; s = X * (W * e)[:, None]
    S = pd.DataFrame(s).groupby(d[cluster].values).sum().values; G = S.shape[0]
    V = np.linalg.inv(A) @ (S.T @ S * G / (G - 1)) @ np.linalg.inv(A)
    return b, np.sqrt(np.diag(V)), int(len(d)), int(G)

# ---------- issuer-month panel ----------
im = pd.read_csv(ISSUER_MONTHLY); im["ym"] = pd.to_datetime(im.as_of_date).dt.strftime("%Y%m").astype(int)
im["adv"] = im.tot_port_advance / im.port_upb * 100          # outstanding advances, % of portfolio UPB
im["fbsh"] = im.fb_cnt / im.total_cnt * 100                   # share of portfolio loans in forbearance
im["dlq3"] = im.dlq3_cnt / im.total_cnt * 100
im = im[["issuer_id", "ym", "adv", "fbsh", "dlq3", "port_upb", "total_cnt"]]
rows = []
for it in ["nonbank", "depository"]:
    cs = pd.read_csv(OUT / f"cells_{it}.csv")
    g = cs.groupby(["issuer", "ym", "period"]).agg(n=("n", "sum"), nx=("nx", "sum"), cov=("cov", "sum"), covp=("covp", "sum"), covr=("covr", "sum"), var=("var", "sum")).reset_index()
    g["itype"] = it; rows.append(g)
P = pd.concat(rows).rename(columns={"issuer": "issuer_id"})
P["m"] = P.nx / P.n; P["level"] = P["cov"] / P["var"]; P["level_pred"] = P.covp / P["var"]; P["level_rand"] = P.covr / P["var"]
P["rel"] = P.level / P.m; P["rel_pred"] = P.level_pred / P.m; P["rel_rand"] = P.level_rand / P.m
P = P.merge(im, on=["issuer_id", "ym"], how="left")
P["lag_adv"] = P.groupby("issuer_id").adv.shift(1)
P.to_csv(OUT / "issuer_month.csv", index=False)
R = {"adv_summary": {}}
for it in ["nonbank", "depository"]:
    h = P[(P.itype == it) & P.ym.between(201901, 202209)]
    R["adv_summary"][it] = {"pre": float(np.average(h[h.period == "pre"].adv, weights=h[h.period == "pre"].n)),
                            "freeze": float(np.average(h[h.period == "freeze"].adv, weights=h[h.period == "freeze"].n)),
                            "recovery1": float(np.average(h[h.period == "recovery1"].adv, weights=h[h.period == "recovery1"].n)),
                            "ratehike": float(np.average(h[h.period == "ratehike"].adv, weights=h[h.period == "ratehike"].n)),
                            "sd_within": float((h.adv - h.groupby("issuer_id").adv.transform("mean")).std())}
im2 = pd.read_csv(ISSUER_MONTHLY); im2["ym"] = pd.to_datetime(im2.as_of_date).dt.strftime("%Y%m").astype(int)
im2["adv_pmt"] = im2.tot_port_advance / im2.tot_port_pmt        # advances outstanding in months of scheduled principal and interest
P = P.merge(im2[["issuer_id", "ym", "adv_pmt"]], on=["issuer_id", "ym"], how="left")
P["ladv"] = np.log(P.adv.clip(lower=0.01))                     # main measure: log of advances outstanding, % of UPB
# monthly exercise rate on the eligible stock, by issuer-month
st = pd.read_parquet(WORK / "stock.parquet", columns=["issuer_id", "ym", "ex", "itype"])
st = st[st.itype.isin(["nonbank", "depository"]) & st.ym.between(201901, 202209)]
stm = st.groupby(["issuer_id", "ym"]).ex.agg(mx="mean", ns="size").reset_index()
P = P.merge(stm, on=["issuer_id", "ym"], how="left")
R["panel"] = {}
MEASURES = [("ladv", "Log advances outstanding (\\% of UPB)"), ("adv", "Advances outstanding (\\% of UPB)"), ("adv_pmt", "Advances outstanding (months of P\\&I)"), ("fbsh", "Share of portfolio in forbearance (\\%)")]
for it in ["nonbank", "depository"]:
    h = P[(P.itype == it) & P.ym.between(201901, 202209) & (P.n >= 20)].copy()
    out = {}
    for xv, _ in MEASURES:
        hh = h.dropna(subset=[xv]); row = {"within_sd": float((hh[xv] - hh.groupby("issuer_id")[xv].transform("mean")).std())}
        for yv, wv, sub in [("m", "n", hh), ("level", "n", hh), ("rel", "n", hh[hh.m > 0]), ("rel_pred", "n", hh[hh.m > 0]), ("rel_rand", "n", hh[hh.m > 0]), ("mx", "ns", hh[hh.ns >= 50])]:
            b, se, N, G = wls_fe(sub, yv, [xv], wv, ["issuer_id", "ym"], "issuer_id")
            row[yv] = {"b": float(b[0]), "se": float(se[0]), "n": N, "issuers": G}
        b, se, N, G = wls_fe(hh, "m", [xv, "dlq3"], "n", ["issuer_id", "ym"], "issuer_id")
        row["m_ctrl"] = {"b": float(b[0]), "se": float(se[0]), "b_dlq": float(b[1]), "se_dlq": float(se[1])}
        out[xv] = row
        print(it, xv, {k: (round(v["b"], 3), round(v["se"], 3)) for k, v in row.items() if isinstance(v, dict)}, flush=True)
    R["panel"][it] = out
# difference between types for the main measure (pooled regression with type interactions)
h = P[P.ym.between(201901, 202209) & (P.n >= 20)].copy(); h["nb"] = (h.itype == "nonbank").astype(float)
h["ladv_nb"] = h.ladv * h.nb; h["ym_t"] = h.ym.astype(str) + "_" + h.itype
b, se, N, G = wls_fe(h, "m", ["ladv", "ladv_nb"], "n", ["issuer_id", "ym_t"], "issuer_id")
R["panel"]["diff_ladv_m"] = {"b_dep": float(b[0]), "se_dep": float(se[0]), "b_diff": float(b[1]), "se_diff": float(se[1]), "n": N, "issuers": G}
print("difference", R["panel"]["diff_ladv_m"], flush=True)
# ---------- loan-level: advance burden x index in the cell-FE logit ----------
d = pd.read_parquet(WORK / "scored.parquet", columns=["issuer_id", "itype", "ym", "period", "x", "forbear", "r_main"])
d = d[d.itype.isin(["nonbank", "depository"]) & d.ym.between(201901, 202209)].copy()
d["r"] = (d.r_main - d.r_main.mean()) / d.r_main.std()
d = d.merge(im[["issuer_id", "ym", "adv"]], on=["issuer_id", "ym"], how="left").dropna(subset=["adv"])
d["cell"] = d.issuer_id.astype(str) + "_" + d.ym.astype(str) + "_" + d.forbear.astype(int).astype(str)
R["logit"] = {}
for it in ["nonbank", "depository"]:
    g = d[d.itype == it]; la = np.log(g.adv.clip(lower=0.01)); ad = la - la.groupby(g.issuer_id).transform("mean")   # within-issuer variation in the log advance burden
    Z = np.column_stack([g.r.values, (g.r * ad).values])
    b, se, N, C, *_ = fe_logit(g.x.values, Z, g.cell.values, g.issuer_id.values)
    # with index-by-month interactions, so that the advance term is identified from cross-issuer differences within a month
    D = pd.get_dummies(g.ym, drop_first=True).astype(float).values * g.r.values[:, None]
    Z2 = np.column_stack([g.r.values, D, (g.r * ad).values])
    b2, se2, *_ = fe_logit(g.x.values, Z2, g.cell.values, g.issuer_id.values)
    R["logit"][it] = {"beta": float(b[0]), "se": float(se[0]), "beta_adv": float(b[1]), "se_adv": float(se[1]),
                      "beta_adv_within_month": float(b2[-1]), "se_adv_within_month": float(se2[-1]), "n": int(N), "cells": int(C), "adv_sd": float(ad.std())}
    print(it, "logit", R["logit"][it], flush=True)
# ---------- shift-share cross-section: 2019 state mix x state unemployment rise, Feb-Apr 2020 ----------
sc = pd.read_parquet(WORK / "scored.parquet", columns=["issuer_id", "itype", "ym", "state"])
ur = pd.read_csv(STATE_UR).pivot(index="state", columns="ym", values="ur"); du = ur[202004] - ur[202002]
mix = sc[sc.ym.between(201901, 201912) & sc.itype.isin(["nonbank", "depository"])].groupby(["issuer_id", "state"]).size().unstack(fill_value=0)
mix = mix.div(mix.sum(axis=1), axis=0); common = [s for s in mix.columns if s in du.index]
exp_lab = (mix[common] * du[common]).sum(axis=1).rename("exp_lab")
fb5 = im[im.ym == 202005].set_index("issuer_id").fbsh.rename("exp_fb")
I = P[P.period.isin(["pre", "freeze"])].groupby(["issuer_id", "itype", "period"]).agg(n=("n", "sum"), nx=("nx", "sum"), cov=("cov", "sum"), var=("var", "sum")).reset_index()
I["m"] = I.nx / I.n; I["rel"] = I["cov"] / I["var"] / I.m
Wd = I.pivot_table(index=["issuer_id", "itype"], columns="period", values=["m", "rel", "n"]).reset_index()
Wd.columns = ["_".join([c for c in col if c]) for col in Wd.columns]
Wd = Wd.merge(exp_lab, left_on="issuer_id", right_index=True, how="left").merge(fb5, left_on="issuer_id", right_index=True, how="left")
Wd["dm"] = 100 * (Wd.m_freeze - Wd.m_pre); Wd["drel"] = Wd.rel_freeze - Wd.rel_pre
R["shiftshare"] = {}
for it in ["nonbank", "depository"]:
    h = Wd[(Wd.itype == it) & (Wd.n_pre >= 100) & (Wd.n_freeze >= 100)].dropna(subset=["exp_lab", "exp_fb", "dm"])
    out = {"issuers": int(len(h)), "exp_lab_sd": float(h.exp_lab.std()), "exp_fb_sd": float(h.exp_fb.std())}
    for xv in ["exp_lab", "exp_fb"]:
        for yv in ["dm", "drel"]:
            hh = h.dropna(subset=[yv]); w = hh.n_freeze.values; X = np.column_stack([np.ones(len(hh)), hh[xv].values]); y = hh[yv].values
            b = np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ y); e = y - X @ b
            V = np.linalg.inv((X * w[:, None]).T @ X) @ ((X * (w * e)[:, None]).T @ (X * (w * e)[:, None])) @ np.linalg.inv((X * w[:, None]).T @ X) * len(hh) / (len(hh) - 2)
            out[f"{yv}_on_{xv}"] = {"b": float(b[1]), "se": float(np.sqrt(V[1, 1]))}
    R["shiftshare"][it] = out; print(it, "shiftshare", {k: v for k, v in out.items()}, flush=True)
json.dump(R, open(OUT / "exposure.json", "w"), indent=1)
