"""Rambachan-Roth on the within-pool design, full window and the flat window."""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
from config import OUT
warnings.filterwarnings("ignore")
from honestdid import constructOriginalCS, createSensitivityResults_relativeMagnitudes

CTRL = ["coupon", "fico", "cltv", "age"]

d = pd.read_parquet(os.path.join(OUT, "mixed_pools.parquet"))
d = d.rename(columns={"interest_rate": "coupon", "credit_score": "fico",
                      "ltv_current": "cltv", "loan_age": "age"})
for c in CTRL: d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=CTRL + ["buyout"]).copy()
d["nonbank"] = (d.ext == "nonbank").astype(float)
d["pm"] = d.pool_id.astype(str) + "_" + d.ym.astype(str)

def val(x):
    v = x
    for _ in range(4):
        if hasattr(v, "iloc"): v = v.iloc[0]
        elif isinstance(v, (list, tuple, np.ndarray)): v = np.asarray(v).ravel()[0]
        else: break
    return float(v)

def run(dat, label, base=202002):
    dat = dat.copy()
    k = dat.groupby("pm").nonbank.transform(lambda s: s.nunique()); dat = dat[k > 1]
    months = sorted(dat.ym.unique())
    cols = []
    for m_ in months:
        if m_ != base:
            dat[f"d{m_}"] = dat.nonbank * (dat.ym == m_); cols.append(f"d{m_}")
    es = pf.feols(f"buyout ~ {' + '.join(cols)} + {' + '.join(CTRL)} | issuer_id + pm",
                  data=dat, vcov={"CRV1": "issuer_id"})
    idx = [list(es._coefnames).index(c) for c in cols]
    beta = np.asarray(es.coef())[idx]
    V = np.asarray(es._vcov)[np.ix_(idx, idx)]
    ms = [int(c[1:]) for c in cols]
    post_pos = [i for i, x in enumerate(ms) if x >= 202003]
    nP, nQ = len(ms) - len(post_pos), len(post_pos)
    ev = pd.DataFrame({"m": ms, "coef": beta, "se": np.sqrt(np.diag(V))})
    print(f"\n### {label}   N={int(es._N):,}  pre={nP} post={nQ}")
    print(ev.round(3).to_string(index=False))
    pre = ev[ev.m <= 202001]
    print(f"   largest |pre| = {pre.coef.abs().max():.3f}")
    out = {"label": label, "n": int(es._N),
           "max_abs_pre": round(float(pre.coef.abs().max()), 4)}
    # June 2020 target
    if 202006 in ms[nP:]:
        l = np.zeros(nQ); l[ms[nP:].index(202006)] = 1.0
        cs0 = constructOriginalCS(beta, V, nP, nQ, l_vec=l)
        pt = float(l @ beta[nP:])
        print(f"   June 2020 point {pt:+.3f}   parallel trends "
              f"[{val(cs0['lb']):+.3f}, {val(cs0['ub']):+.3f}]")
        out["point"] = round(pt, 3)
        out["pt"] = [round(val(cs0["lb"]), 3), round(val(cs0["ub"]), 3)]
        bd = None
        for M in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            try:
                r = createSensitivityResults_relativeMagnitudes(
                    beta, V, nP, nQ, Mbarvec=[M], l_vec=l, gridPoints=200)
                lb, ub = val(r["lb"]), val(r["ub"])
                inc = lb <= 0 <= ub
                print(f"     Mbar={M:<4} [{lb:+.3f}, {ub:+.3f}]"
                      f"{'  <- includes 0' if inc else ''}", flush=True)
                out[f"M{M}"] = [round(lb, 3), round(ub, 3), bool(inc)]
                if inc and bd is None: bd = M
            except Exception as e:
                print(f"     Mbar={M:<4} failed {type(e).__name__}")
        out["breakdown"] = bd if bd else ">4"
        print(f"   BREAKDOWN VALUE: {out['breakdown']}")
    return out

res = {}
res["full"] = run(d, "FULL WINDOW  Jan 2019 - Sep 2020")
res["flat"] = run(d[d.ym >= 201907], "FLAT WINDOW  Jul 2019 - Sep 2020")
json.dump(res, open(os.path.join(OUT, "results_honest_pool.json"), "w"),
          indent=1, default=str)
print("\nwrote results_honest_pool.json")
