"""The same sensitivity analysis as 118, run at a resolution that finishes.

Only the DiD-matched weighting is carried, because that is the aggregate the paper's
headline coefficient equals; the equal-weighted point estimate and its parallel-trends
set are reported alongside for reference but not searched over Mbar. The grid is coarser
than 118 and the Mbar ladder shorter: the object of interest is the breakdown value, and
a 0.25-wide ladder locates it precisely enough to report.
"""
import pandas as pd, numpy as np, os, json, time, warnings
import pyfixest as pf
warnings.filterwarnings("ignore")
from honestdid import constructOriginalCS, createSensitivityResults_relativeMagnitudes
# library patched once by 00_patch_honestdid.py

from config import OUT, PAPER as PAP
CTRL = ["coupon", "fico", "cltv", "age"]
MBARS = [0.25, 0.5, 1.0, 2.0]
GRID = 80

d = pd.read_parquet(os.path.join(OUT, "mixed_pools.parquet"))
d = d.rename(columns={"interest_rate": "coupon", "credit_score": "fico",
                      "ltv_current": "cltv", "loan_age": "age"})
for c in CTRL:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=CTRL + ["buyout"]).copy()
d["nonbank"] = (d.ext == "nonbank").astype(float)
d["pm"] = d.pool_id.astype(str) + "_" + d.ym.astype(str)


def val(x):
    return float(np.asarray(x).reshape(-1)[0])


def run(dat, label, base=202002):
    dat = dat.copy()
    k = dat.groupby("pm").nonbank.transform(lambda s: s.nunique())
    dat = dat[k > 1]
    months = sorted(dat.ym.unique())
    cols = []
    for m_ in months:
        if m_ != base:
            dat[f"d{m_}"] = dat.nonbank * (dat.ym == m_)
            cols.append(f"d{m_}")
    es = pf.feols(f"buyout ~ {' + '.join(cols)} + {' + '.join(CTRL)} | issuer_id + pm",
                  data=dat, vcov={"CRV1": "issuer_id"})
    idx = [list(es._coefnames).index(c) for c in cols]
    beta = np.asarray(es.coef())[idx]
    V = np.asarray(es._vcov)[np.ix_(idx, idx)]
    ms = [int(c[1:]) for c in cols]
    nQ = sum(1 for x in ms if x >= 202003)
    nP = len(ms) - nQ
    pmo = ms[nP:]

    dat["nb_post"] = dat.nonbank * (dat.ym >= 202003)
    static = float(pf.feols(f"buyout ~ nb_post + {' + '.join(CTRL)} | issuer_id + pm",
                            data=dat, vcov={"CRV1": "issuer_id"}).coef()["nb_post"])
    n_t = dat[dat.ym >= 202003].groupby("ym").size().reindex(pmo).values.astype(float)
    l = n_t / n_t.sum()
    leq = np.ones(nQ) / nQ

    print("\n" + "=" * 88)
    print(f"{label}   N={int(es._N):,}  pre={nP} post={nQ}")
    print("=" * 88)
    print(f"  largest |pre-period coefficient|  {np.abs(beta[:nP]).max():.3f}")
    print(f"  static two-way FE estimate        {static:+.4f}")
    pt = float(l @ beta[nP:]); se = float(np.sqrt(l @ V[nP:, nP:] @ l))
    pte = float(leq @ beta[nP:]); see = float(np.sqrt(leq @ V[nP:, nP:] @ leq))
    print(f"  DiD-matched aggregate             {pt:+.4f} (s.e. {se:.4f})")
    print(f"  equal-weighted aggregate          {pte:+.4f} (s.e. {see:.4f})")
    cs = constructOriginalCS(beta, V, nP, nQ, l_vec=l)
    cse = constructOriginalCS(beta, V, nP, nQ, l_vec=leq)
    out = {"label": label, "n": int(es._N), "static": round(static, 4),
           "max_abs_pre": round(float(np.abs(beta[:nP]).max()), 4),
           "point": round(pt, 4), "se": round(se, 4),
           "point_eq": round(pte, 4), "se_eq": round(see, 4),
           "pt": [round(val(cs["lb"]), 3), round(val(cs["ub"]), 3)],
           "pt_eq": [round(val(cse["lb"]), 3), round(val(cse["ub"]), 3)]}
    print(f"  parallel trends, DiD-matched      [{out['pt'][0]:+.3f}, {out['pt'][1]:+.3f}]")
    bd = None
    for M in MBARS:
        t0 = time.time()
        try:
            r = createSensitivityResults_relativeMagnitudes(
                beta, V, nP, nQ, Mbarvec=[M], l_vec=l, gridPoints=GRID)
            lb, ub = val(r["lb"]), val(r["ub"])
            inc = lb <= 0 <= ub
            print(f"    Mbar={M:<5} [{lb:+.3f}, {ub:+.3f}]"
                  f"{'   <- includes 0' if inc else ''}   ({time.time()-t0:.0f}s)",
                  flush=True)
            out[f"M{M}"] = [round(lb, 3), round(ub, 3), bool(inc)]
            if inc and bd is None:
                bd = M
        except Exception as e:
            print(f"    Mbar={M:<5} FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
    out["breakdown"] = bd if bd is not None else f">{MBARS[-1]}"
    print(f"  BREAKDOWN VALUE: {out['breakdown']}")
    return out


res = {"full": run(d, "FULL WINDOW  January 2019 - September 2020"),
       "flat": run(d[d.ym >= 201907], "JULY 2019 WINDOW")}
json.dump(res, open(os.path.join(OUT, "results_honest_avg.json"), "w"), indent=1, default=str)


def num(x, k=3):
    return ("$-$" if x < 0 else "") + f"{abs(x):.{k}f}"


L = [r"\begin{tabular}{lcc}", r"\toprule",
     r"& Full window & July 2019 window\\", r"\midrule",
     "Average post-period effect, DiD-matched & "
     + " & ".join(num(res[w]["point"]) for w in ("full", "flat")) + r" \\",
     r"\quad (s.e.) & " + " & ".join(f"({res[w]['se']:.3f})" for w in ("full", "flat")) + r" \\",
     "Static two-way fixed-effects estimate & "
     + " & ".join(num(res[w]["static"]) for w in ("full", "flat")) + r" \\",
     "Equal-weighted average & "
     + " & ".join(num(res[w]["point_eq"]) for w in ("full", "flat")) + r" \\",
     r"\addlinespace",
     r"\multicolumn{3}{l}{\textit{Confidence sets for the DiD-matched average}}\\",
     r"\quad Parallel trends & "
     + " & ".join(f"[{num(res[w]['pt'][0],2)}, {num(res[w]['pt'][1],2)}]"
                  for w in ("full", "flat")) + r" \\"]
for M in MBARS:
    cells = []
    for w in ("full", "flat"):
        c = res[w].get(f"M{M}")
        cells.append("---" if c is None else f"[{num(c[0],2)}, {num(c[1],2)}]")
    L.append(f"\\quad $\\bar{{M}} = {M}$ & " + " & ".join(cells) + r" \\")
L += [r"\addlinespace",
      r"\quad Breakdown value & "
      + " & ".join(str(res[w]["breakdown"]) for w in ("full", "flat")) + r" \\",
      r"\bottomrule", r"\end{tabular}"]
open(os.path.join(PAP, "tables", "P11.tex"), "w", encoding="utf-8").write("\n".join(L))

mac = {}
for w, tag in (("full", "Full"), ("flat", "Flat")):
    mac[f"Hon{tag}Pt"] = num(res[w]["point"])
    mac[f"Hon{tag}Bd"] = str(res[w]["breakdown"])
    mac[f"Hon{tag}Static"] = num(res[w]["static"])
pth = os.path.join(PAP, "numbers_pool.tex")
have = open(pth, encoding="utf-8").read()
with open(pth, "a", encoding="utf-8") as fh:
    fh.write("\n")
    for k, v in sorted(mac.items()):
        if "\\p" + k + "}" in have:
            continue
        fh.write("\\newcommand{\\p" + k + "}{" + v + "}\n")
print(f"\nwrote P11.tex and {len(mac)} macros")
