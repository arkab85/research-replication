"""Robustness of the freeze test for nonbank issuers to the ranking variable, the decision cell, the
exercise window, and the sample. Each row re-estimates the pre-period weight and recomputes the
fixed-ranking and random-rationing benchmarks, with 100 bootstrap draws over issuers."""
import pandas as pd, numpy as np, sys
from felogit import fe_logit
from rationing import cell_stats, agg, prepare, issuer_pre_share
from config import WORK, OUT
B = int(sys.argv[1]) if len(sys.argv) > 1 else 100
full = pd.read_parquet(WORK / "scored.parquet")
ref = full[full.ym.between(201901, 202209) & full.itype.isin(["depository", "nonbank"])]
MS = {c: (ref[c].mean(), ref[c].std()) for c in ["r_main", "r_t10", "r_gb", "r_dep", "r_pool", "mspread"]}   # same scaling as Table 3
base = full[full.ym.between(201903, 202006) & full.itype.isin(["depository", "nonbank", "techfirst"])].copy(); del full, ref

def run(dd, col, label, byf=True, y="x"):
    dd = dd.copy(); dd["r"] = (dd[col] - MS[col][0]) / MS[col][1]; dd["x"] = dd[y]
    dd = prepare(dd, by_forbearance=byf); pre = dd[dd.period == "pre"]
    b, se, *_ = fe_logit(pre.x.values, pre.r.values[:, None], pre.cell.values, pre.issuer_id.values)
    b2, se2, *_ = fe_logit(dd.x.values, np.column_stack([dd.r, dd.r * (dd.period == "freeze")]), dd.cell.values, dd.issuer_id.values)
    m0, m0d = issuer_pre_share(dd); cs = cell_stats(dd, b[0], m0, m0d)
    out = {"label": label, "beta_pre": float(b[0]), "d_freeze": float(b2[1]), "se_d_freeze": float(se2[1])}
    for per in ["pre", "freeze"]:
        a = agg(cs[cs.period == per]); out.update({f"{per}_m": a["m"], f"{per}_rel": a["rel"], f"{per}_rel_pred": a["rel_pred"], f"{per}_rel_random": a["rel_random"]})
    rng = np.random.default_rng(11); gi = {i: gg for i, gg in dd.groupby("issuer_id")}; iss = list(gi); gf = []; gr = []
    for _ in range(B):
        draw = rng.choice(iss, len(iss), replace=True); parts = []
        for k, i in enumerate(draw):
            gg = gi[i].copy(); gg["issuer_id"] = k; gg["cell"] = str(k) + "_" + gg.cell.str.split("_", n=1).str[1]; parts.append(gg)
        gb = pd.concat(parts); pb = gb[gb.period == "pre"]
        try: bb, *_ = fe_logit(pb.x.values, pb.r.values[:, None], pb.cell.values, pb.issuer_id.values)
        except Exception: continue
        mb, mbd = issuer_pre_share(gb); cb = cell_stats(gb, bb[0], mb, mbd); a = agg(cb[cb.period == "freeze"])
        gf.append(a["rel"] - a["rel_pred"]); gr.append(a["rel"] - a["rel_random"])
    out["se_gap_fixed"] = float(np.std(gf, ddof=1)); out["se_gap_random"] = float(np.std(gr, ddof=1))
    print(label, {k: round(v, 3) for k, v in out.items() if k != "label"}, flush=True)
    return out

nb = base[base.itype == "nonbank"]
rows = [run(nb, "r_main", "Baseline"),
        run(nb, "r_main", "Issuer-month cells", byf=False),
        run(nb, "r_t10", "Treasury-spread index, piecewise"),
        run(nb, "r_gb", "Boosted ranking index"),
        run(nb, "r_dep", "Index from depository decisions"),
        run(nb, "r_pool", "Depository exercise classifier"),
        run(nb, "mspread", "Mortgage spread only"),
        run(nb, "r_main", "Exercise within 12 months", y="x12")]
big = nb.groupby("issuer_id").size().idxmax()
rows.append(run(nb[nb.issuer_id != big], "r_main", "Excluding the largest nonbank"))
rows.append(run(base[base.itype.isin(["nonbank", "techfirst"])], "r_main", "Nonbanks and technology-based lenders"))
rows.append(run(nb[nb.forbear == 0], "r_main", "Loans not in forbearance"))
pd.DataFrame(rows).to_csv(OUT / "robust.csv", index=False)
