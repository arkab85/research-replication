"""Issuer-level view of the freeze and the recovery for the largest nonbanks (ranked ex ante by the
number of options that vested in the pre-shock period), each with the pre-shock weight re-estimated on
its own decisions; and exercise of the pre-shock and freeze cohorts at fixed horizons."""
import pandas as pd, numpy as np, json
from felogit import fe_logit
from rationing import cell_stats, agg, prepare, issuer_pre_share
from config import WORK, OUT
sc = pd.read_parquet(WORK / "scored.parquet")
sc = sc[sc.ym.between(201901, 202209) & sc.itype.isin(["depository", "nonbank"])].copy()
mu, sd = sc.r_main.mean(), sc.r_main.std()
d = sc[sc.itype == "nonbank"].copy(); d["r"] = (d.r_main - mu) / sd; d = prepare(d, True)
pre_n = d[d.period == "pre"].groupby("issuer_id").size().sort_values(ascending=False)
pre_m = d[d.period == "pre"].groupby("issuer_id").x.mean()
top = [i for i in pre_n.index if pre_m[i] >= 0.05][:5]
rows = []
for k, i in enumerate(top):
    g = d[d.issuer_id == i]; pg = g[g.period == "pre"]
    b, *_ = fe_logit(pg.x.values, pg.r.values[:, None], pg.cell.values)
    m0, m0d = issuer_pre_share(g); cs = cell_stats(g, b[0], m0, m0d)
    row = {"label": "ABCDE"[k], "issuer_id": int(i), "pre_n": int(pre_n[i]), "beta_pre": float(b[0])}
    for p in ["pre", "freeze", "release", "recovery1"]:
        a = agg(cs[cs.period == p]); row.update({f"{p}_n": int(cs[cs.period == p].n.sum()), f"{p}_m": a["m"], f"{p}_rel": a["rel"], f"{p}_pred": a["rel_pred"], f"{p}_rand": a["rel_random"]})
    rows.append(row)
rest = d[~d.issuer_id.isin(top)]; pg = rest[rest.period == "pre"]
b, *_ = fe_logit(pg.x.values, pg.r.values[:, None], pg.cell.values, pg.issuer_id.values)
m0, m0d = issuer_pre_share(rest); cs = cell_stats(rest, b[0], m0, m0d)
row = {"label": "Others", "issuer_id": 0, "pre_n": int(pg.shape[0]), "beta_pre": float(b[0]), "issuers": int(rest.issuer_id.nunique())}
for p in ["pre", "freeze", "release", "recovery1"]:
    a = agg(cs[cs.period == p]); row.update({f"{p}_n": int(cs[cs.period == p].n.sum()), f"{p}_m": a["m"], f"{p}_rel": a["rel"], f"{p}_pred": a["rel_pred"], f"{p}_rand": a["rel_random"]})
rows.append(row)
R = {"issuers": rows, "top_share_pre": float(pre_n[top].sum() / pre_n.sum()),
     "top_share_freeze": float(d[(d.period == "freeze") & d.issuer_id.isin(top)].shape[0] / d[d.period == "freeze"].shape[0])}
# fixed-horizon exercise, pre-shock and freeze cohorts, nonbanks and depositories
for it in ["nonbank", "depository"]:
    for p in ["pre", "freeze"]:
        g = sc[(sc.itype == it) & (sc.period == p)]
        for h in [3, 12, 24]:
            R[f"h{h}_{it}_{p}"] = float(((g.x_ever == 1) & (g.t_ever <= h)).mean())
json.dump(R, open(OUT / "issuers.json", "w"), indent=1)
print(pd.DataFrame(rows).round(3).to_string()); print({k: round(v, 3) for k, v in R.items() if k.startswith(("h", "top"))})
