"""Tests of Propositions 1, 3, 5 and the dynamic extension on Ginnie Mae buyout decisions, 2019m1-2022m9.
Decision cells are issuer x vesting month x forbearance status: options in and out of forbearance
tie up cash for different lengths of time (Internet Appendix IA.2), and the index is blind to
forbearance, so the ranking is tested among comparable options. Output: results.json, series.csv,
cells_*.csv, boot_*.csv, loo_*.csv."""
import pandas as pd, numpy as np, json, time, sys
from felogit import fe_logit
from rationing import cell_stats, agg, prepare, issuer_pre_share
from config import WORK, OUT, PERIODS
PER = [p for p, _, _ in PERIODS]
B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(20260926)
d = pd.read_parquet(WORK / "scored.parquet")
d = d[d.ym.between(201901, 202209) & d.itype.isin(["depository", "nonbank"])].copy()
mu, sd = d.r_main.mean(), d.r_main.std(); d["r"] = (d.r_main - mu) / sd
d = prepare(d, by_forbearance=True)
R = {"N_decisions": int(len(d)), "N_issuers": int(d.issuer_id.nunique()), "N_by_type": d.itype.value_counts().to_dict(),
     "N_cells": int(d.cell.nunique())}
series = []; periods = {}
for it in ["nonbank", "depository"]:
    g = d[d.itype == it]; pre = g[g.period == "pre"]
    b, se, n, C, *_ = fe_logit(pre.x.values, pre.r.values[:, None], pre.cell.values, pre.issuer_id.values)
    R[f"{it}_beta_pre"] = float(b[0]); R[f"{it}_beta_pre_se"] = float(se[0])
    m0, m0d = issuer_pre_share(g)
    cs = cell_stats(g, b[0], m0, m0d); cs.to_csv(OUT / f"cells_{it}.csv", index=False)
    for ym, cc in cs.groupby("ym"):
        a = agg(cc); a.update(itype=it, ym=int(ym)); series.append(a)
    for per in PER:
        periods[f"{it}_{per}"] = agg(cs[cs.period == per])
    fz = cs[cs.period == "freeze"]
    loo = []
    for i in fz.issuer.unique():
        a = agg(fz[fz.issuer != i]); loo.append(dict(issuer=int(i), rel=a["rel"], rel_pred=a["rel_pred"], rel_random=a["rel_random"]))
    loo = pd.DataFrame(loo); loo.to_csv(OUT / f"loo_{it}.csv", index=False)
    R[f"{it}_loo"] = dict(rel_min=float(loo.rel.min()), rel_max=float(loo.rel.max()),
        gap_fixed_min=float((loo.rel - loo.rel_pred).min()), gap_fixed_max=float((loo.rel - loo.rel_pred).max()),
        gap_random_min=float((loo.rel - loo.rel_random).min()), gap_random_max=float((loo.rel - loo.rel_random).max()))
    contrib = fz.groupby("issuer")["cov"].sum().sort_values(ascending=False)
    R[f"{it}_top2_cov_share"] = float(contrib.head(2).sum() / contrib.sum())
    top2 = list(contrib.index[:2]); R[f"{it}_top2_option_share"] = float(fz[fz.issuer.isin(top2)].n.sum() / fz.n.sum())
    # the two largest contributors and the rest, each with the pre-period weight re-estimated on its own decisions
    for lab, gg in [("top2", g[g.issuer_id.isin(top2)]), ("rest", g[~g.issuer_id.isin(top2)])]:
        pg = gg[gg.period == "pre"]
        bg, seg, *_ = fe_logit(pg.x.values, pg.r.values[:, None], pg.cell.values, pg.issuer_id.values)
        m0g, m0gd = issuer_pre_share(gg); csg = cell_stats(gg, bg[0], m0g, m0gd)
        R[f"{it}_{lab}"] = {"beta_pre": float(bg[0]), "se_pre": float(seg[0]), "issuers": int(gg.issuer_id.nunique()),
                            **{f"{per}_{k}": v for per in ["pre", "freeze", "release", "recovery1"] for k, v in agg(csg[csg.period == per]).items()}}
    R[f"{it}_excl_top2"] = R[f"{it}_rest"]
    # bootstrap over issuers, re-estimating the pre-period weight in every draw
    iss = g.issuer_id.unique(); boot = []; gi = {i: gg for i, gg in g.groupby("issuer_id")}; t = time.time()
    for bb in range(B):
        draw = rng.choice(iss, len(iss), replace=True); parts = []
        for k, i in enumerate(draw):
            gg = gi[i].copy(); gg["issuer_id"] = k; gg["cell"] = str(k) + "_" + gg.ym.astype(str) + "_" + gg.forbear.astype(int).astype(str); parts.append(gg)
        gb = pd.concat(parts); preb = gb[gb.period == "pre"]
        try: bb_, *_ = fe_logit(preb.x.values, preb.r.values[:, None], preb.cell.values, preb.issuer_id.values)
        except Exception: continue
        m0b, m0bd = issuer_pre_share(gb); csb = cell_stats(gb, bb_[0], m0b, m0bd)
        apre = agg(csb[csb.period == "pre"]); row = {"beta": bb_[0], "pre_rel": apre["rel"], "pre_rel_random": apre["rel_random"]}
        for per in PER[1:]:
            a = agg(csb[csb.period == per])
            row.update({f"{per}_rel": a["rel"], f"{per}_rel_pred": a["rel_pred"], f"{per}_rel_random": a["rel_random"],
                        f"{per}_gap_fixed": a["rel"] - a["rel_pred"], f"{per}_gap_random": a["rel"] - a["rel_random"],
                        f"{per}_level": a["level"], f"{per}_level_pred": a["level_pred"],
                        f"{per}_did_random": (a["rel"] - apre["rel"]) - (a["rel_random"] - apre["rel_random"])})
        boot.append(row)
    bt = pd.DataFrame(boot); bt.to_csv(OUT / f"boot_{it}.csv", index=False)
    R[f"{it}_boot_draws"] = int(len(bt)); R[f"{it}_boot_seconds"] = round(time.time() - t, 1); R[f"{it}_boot_issuers"] = int(len(iss))
    for col in bt.columns: R[f"{it}_boot_se_{col}"] = float(bt[col].std())
    print(it, "done", round(time.time() - t, 1), flush=True)
pd.DataFrame(series).to_csv(OUT / "series.csv", index=False)
R["periods"] = periods

# latent weight by period (FE logit, cells, cluster by issuer), with forbearance-by-period controls
for it in ["nonbank", "depository"]:
    g = d[(d.itype == it) & d.period.isin(PER)]
    Z = [g.r] + [g.r * (g.period == p) for p in PER[1:]]
    b, se, n, C, *_ = fe_logit(g.x.values, np.column_stack(Z), g.cell.values, g.issuer_id.values)
    R[f"{it}_felogit"] = {"beta_pre": float(b[0]), "se_pre": float(se[0]), "n": int(n), "cells": int(C),
                          **{f"d_{p}": float(b[i + 1]) for i, p in enumerate(PER[1:])}, **{f"se_{p}": float(se[i + 1]) for i, p in enumerate(PER[1:])}}

# the queue: among options not exercised within the window and still in their pool, later exercise (months 4-12) on the index
d["late"] = ((d.x12 == 1) & (d.x == 0)).astype(float)
for it in ["nonbank", "depository"]:
    for p in ["pre", "freeze", "release", "recovery1"]:
        g = d[(d.itype == it) & (d.period == p) & (d.x == 0) & (d.gone3 == 0)]
        b, se, n, C, *_ = fe_logit(g.late.values, g.r.values[:, None], g.cell.values, g.issuer_id.values)
        R[f"queue_{it}_{p}"] = dict(n=int(len(g)), rate=float(g.late.mean()), ever=float(((g.x_ever == 1)).mean()), beta=float(b[0]), se=float(se[0]))
        ga = d[(d.itype == it) & (d.period == p) & (d.x == 0)]
        ba, sea, *_ = fe_logit(ga.late.values, ga.r.values[:, None], ga.cell.values, ga.issuer_id.values)
        R[f"queue_all_{it}_{p}"] = dict(n=int(len(ga)), rate=float(ga.late.mean()), ever=float(((ga.x_ever == 1)).mean()), beta=float(ba[0]), se=float(sea[0]))
# month by month in 2020 (cells are effectively issuer-month in March and April, before the forbearance flag exists)
for it in ["nonbank", "depository"]:
    cs = pd.read_csv(OUT / f"cells_{it}.csv")
    R[f"monthly_{it}"] = {str(int(ym)): agg(cc) for ym, cc in cs[cs.ym.between(202001, 202112)].groupby("ym")}
# calendar months of the delayed exercises of freeze-vested nonbank options
g = d[(d.itype == "nonbank") & (d.period == "freeze") & (d.late == 1)]
cal = (g.ym // 100) * 12 + (g.ym % 100 - 1) + g.t_ex.astype(int)
calym = (cal // 12) * 100 + cal % 12 + 1
R["queue_calendar"] = {str(int(k)): int(v) for k, v in calym.value_counts().sort_index().items()}
R["queue_share_by_2021m6"] = float((calym <= 202106).mean())

# forbearance contrast, May-June 2020: LPM with issuer-month FE (cells not split by forbearance)
f = d[d.ym.between(202005, 202006)].copy(); f["cell2"] = f.issuer_id.astype(str) + "_" + f.ym.astype(str)
for it in ["nonbank", "depository"]:
    g = f[f.itype == it].copy()
    g["xd"] = g.x - g.groupby("cell2").x.transform("mean"); g["fd"] = g.forbear - g.groupby("cell2").forbear.transform("mean")
    beta = (g.xd * g.fd).sum() / (g.fd ** 2).sum(); e = g.xd - beta * g.fd
    s = (g.fd * e).groupby(g.issuer_id).sum(); G = len(s); se = np.sqrt((s ** 2).sum() * G / (G - 1)) / (g.fd ** 2).sum()
    g["dec"] = pd.qcut(g.r, 10, labels=False, duplicates="drop")
    D = pd.get_dummies(g.dec, prefix="d", drop_first=True).astype(float)
    Xm = np.column_stack([g.forbear.values, D.values]); Xm = Xm - pd.DataFrame(Xm).groupby(g.cell2.values).transform("mean").values
    yv = g.xd.values; coef = np.linalg.lstsq(Xm, yv, rcond=None)[0]; e2 = yv - Xm @ coef
    A = Xm.T @ Xm; Sg = pd.DataFrame(Xm * e2[:, None]).groupby(g.issuer_id.values).sum().values
    V = np.linalg.inv(A) @ (Sg.T @ Sg * G / (G - 1)) @ np.linalg.inv(A)
    R[f"{it}_forbear_lpm_ctrl"] = dict(coef=float(coef[0]), se=float(np.sqrt(V[0, 0])))
    byjune = ((g.x_ever == 1) & ((g.ym % 100) + g.t_ever.fillna(99) <= 6)).astype(float)   # repurchased by June 30, 2020, before the redelivery restriction applied
    R[f"{it}_forbear_lpm"] = dict(coef=float(beta), se=float(se), n=int(len(g)), issuers=int(G),
        rate_forborne=float(g.x[g.forbear == 1].mean()), rate_nonforborne=float(g.x[g.forbear == 0].mean()),
        share_forborne=float(g.forbear.mean()), ever_forborne=float(g.x_ever[g.forbear == 1].mean()),
        ever_nonforborne=float(g.x_ever[g.forbear == 0].mean()), x12_forborne=float(g.x12[g.forbear == 1].mean()),
        byjune_forborne=float(byjune[g.forbear == 1].mean()), byjune_nonforborne=float(byjune[g.forbear == 0].mean()))
# outcomes over the following twelve months while in the pool (loan-in-pool outcomes), freeze cohort
g = d[d.period == "freeze"]
R["freeze_outcomes"] = g.groupby("itype")[["x", "x12", "x_ever", "foreclosed12", "cured12"]].mean().to_dict()
R["period_outcomes"] = d.groupby(["period", "itype"])[["x", "x12", "x_ever"]].mean().reset_index().to_dict("records")
json.dump(R, open(OUT / "results.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if not k.endswith("seconds")}, indent=1, default=float)[:6000])
