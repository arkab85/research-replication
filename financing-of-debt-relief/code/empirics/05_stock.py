"""Fixed-ranking test on the monthly eligible stock: every loan at least three payments delinquent and
still in its pool is an option the issuer may exercise that month. Cells are issuer x month x
forbearance status; the pre-period weight is estimated on the stock in March 2019-February 2020."""
import pandas as pd, numpy as np, json
from felogit import fe_logit
from rationing import cell_stats, agg, issuer_pre_share
from config import WORK, OUT, PERIODS
PER = [p for p, _, _ in PERIODS]
import pyarrow.parquet as pq, pyarrow.compute as pc
t = pq.read_table(WORK / "stock.parquet", columns=["issuer_id", "ym", "ex", "fb", "itype", "r_main"])
t = t.filter(pc.and_(pc.is_in(t["itype"], value_set=__import__("pyarrow").array(["depository", "nonbank"])),
                     pc.and_(pc.greater_equal(t["ym"], 201901), pc.less_equal(t["ym"], 202209))))
s = t.to_pandas(categories=["itype"]); del t
sc = pd.read_parquet(WORK / "scored.parquet", columns=["r_main", "ym", "itype"])
sc = sc[sc.ym.between(201901, 202209) & sc.itype.isin(["depository", "nonbank"])]
mu, sd = sc.r_main.mean(), sc.r_main.std(); del sc
s["r"] = ((s.r_main - mu) / sd).astype(np.float64); s["x"] = s.ex.astype(np.float64); s["forbear"] = s.fb.astype(np.int8)
s = s.drop(columns=["ex", "fb", "r_main"])
s["cell"] = s.issuer_id.astype(np.int64) * 10_000_000 + s.ym.astype(np.int64) * 10 + s.forbear.astype(np.int64)
per = np.full(len(s), "early", dtype=object)
for p, a, b in PERIODS: per[s.ym.between(a, b).values] = p
s["period"] = pd.Categorical(per); del per
R = {"N_loan_months": int(len(s)), "N_repurchases": int(s.x.sum())}
rows = []
for it in ["nonbank", "depository"]:
    g = s[s.itype == it]; pre = g[g.period == "pre"]
    b, se, n, C, *_ = fe_logit(pre.x.values, pre.r.values[:, None], pre.cell.values, pre.issuer_id.values)
    R[f"{it}_beta_pre"] = float(b[0]); R[f"{it}_beta_pre_se"] = float(se[0])
    m0, m0d = issuer_pre_share(g); cs = cell_stats(g, b[0], m0, m0d)
    for per in PER:
        c = cs[cs.period == per]; a = agg(c); a.update(itype=it, period=per, n=int(c.n.sum())); rows.append(a)
    gg = g[g.period.isin(PER)]
    Z = [gg.r.values] + [(gg.r * (gg.period == p)).values for p in PER[1:]]
    b2, se2, *_ = fe_logit(gg.x.values, np.column_stack(Z), gg.cell.values, gg.issuer_id.values)
    R[f"{it}_felogit"] = {**{f"d_{p}": float(b2[i + 1]) for i, p in enumerate(PER[1:])}, **{f"se_{p}": float(se2[i + 1]) for i, p in enumerate(PER[1:])}}
    print(it, "done", flush=True)
R["periods"] = rows
json.dump(R, open(OUT / "stock_results.json", "w"), indent=1, default=float)
print(pd.DataFrame(rows).round(3).to_string()); print(json.dumps({k: v for k, v in R.items() if k != "periods"}, indent=1))
