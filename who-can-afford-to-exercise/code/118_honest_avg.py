"""Rambachan-Roth sensitivity for the estimand the paper actually reports.

The published sets were for June 2020, a single event-study coefficient. The headline is
the average post-period effect, and a set for one month neither validates nor overturns
it. This runs the sensitivity on that average, under two weightings:

  equal        l_t = 1/Q, the simple average post-period effect
  DiD-matched  l_t proportional to the month's share of post-period observations, which
               is what the static two-way fixed-effects coefficient aggregates to

and checks that the DiD-matched aggregate reproduces the static estimate. Both the full
window and the July 2019 window are reported, as the reviewer asked.
"""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
warnings.filterwarnings("ignore")
from honestdid import constructOriginalCS, createSensitivityResults_relativeMagnitudes
# The library is patched once by 00_patch_honestdid.py; see that script for why.

from config import OUT, PAPER as PAP
CTRL = ["coupon", "fico", "cltv", "age"]

d = pd.read_parquet(os.path.join(OUT, "mixed_pools.parquet"))
d = d.rename(columns={"interest_rate": "coupon", "credit_score": "fico",
                      "ltv_current": "cltv", "loan_age": "age"})
for c in CTRL:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=CTRL + ["buyout"]).copy()
d["nonbank"] = (d.ext == "nonbank").astype(float)
d["pm"] = d.pool_id.astype(str) + "_" + d.ym.astype(str)


def val(x):
    v = x
    for _ in range(4):
        if hasattr(v, "iloc"):
            v = v.iloc[0]
        elif isinstance(v, (list, tuple, np.ndarray)):
            v = np.asarray(v).ravel()[0]
        else:
            break
    return float(v)


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
    post = [i for i, x in enumerate(ms) if x >= 202003]
    nP, nQ = len(ms) - len(post), len(post)
    post_months = ms[nP:]

    # the static estimate this aggregate should reproduce
    dat["nb_post"] = dat.nonbank * (dat.ym >= 202003)
    st = pf.feols(f"buyout ~ nb_post + {' + '.join(CTRL)} | issuer_id + pm",
                  data=dat, vcov={"CRV1": "issuer_id"})
    static = float(st.coef()["nb_post"])

    n_t = dat[dat.ym >= 202003].groupby("ym").size().reindex(post_months).values.astype(float)
    W = {"equal": np.ones(nQ) / nQ, "DiD-matched": n_t / n_t.sum()}

    print("\n" + "=" * 92)
    print(f"{label}    N={int(es._N):,}   pre={nP}  post={nQ}")
    print("=" * 92)
    pre_max = np.abs(beta[:nP]).max()
    print(f"  largest pre-period coefficient in absolute value : {pre_max:.3f}")
    print(f"  static two-way FE estimate                       : {static:+.4f}")

    out = {"label": label, "n": int(es._N), "static": round(static, 4),
           "max_abs_pre": round(float(pre_max), 4)}
    for wname, l in W.items():
        pt = float(l @ beta[nP:])
        se = float(np.sqrt(l @ V[nP:, nP:] @ l))
        cs0 = constructOriginalCS(beta, V, nP, nQ, l_vec=l)
        print(f"\n  --- {wname} weights: point {pt:+.4f} (s.e. {se:.4f})"
              + (f", static is {static:+.4f}" if wname.startswith("DiD") else ""))
        print(f"      parallel trends  [{val(cs0['lb']):+.3f}, {val(cs0['ub']):+.3f}]")
        rec = {"point": round(pt, 4), "se": round(se, 4),
               "pt": [round(val(cs0["lb"]), 3), round(val(cs0["ub"]), 3)]}
        bd = None
        for M in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            try:
                r = createSensitivityResults_relativeMagnitudes(
                    beta, V, nP, nQ, Mbarvec=[M], l_vec=l, gridPoints=300)
                lb, ub = val(r["lb"]), val(r["ub"])
                inc = lb <= 0 <= ub
                print(f"      Mbar={M:<5} [{lb:+.3f}, {ub:+.3f}]"
                      f"{'   <- includes 0' if inc else ''}", flush=True)
                rec[f"M{M}"] = [round(lb, 3), round(ub, 3), bool(inc)]
                if inc and bd is None:
                    bd = M
            except Exception as e:
                print(f"      Mbar={M:<5} failed {type(e).__name__}")
        rec["breakdown"] = bd if bd else ">3"
        print(f"      BREAKDOWN VALUE: {rec['breakdown']}")
        out[wname] = rec
    return out


res = {"full": run(d, "FULL WINDOW  January 2019 - September 2020"),
       "flat": run(d[d.ym >= 201907], "JULY 2019 WINDOW")}
json.dump(res, open(os.path.join(OUT, "results_honest_avg.json"), "w"),
          indent=1, default=str)

# ------------------------------------------------------------------ table
L = [r"\begin{tabular}{lcccc}", r"\toprule",
     r"& \multicolumn{2}{c}{Full window} & \multicolumn{2}{c}{July 2019 window}\\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
     r"& Equal & DiD-matched & Equal & DiD-matched\\", r"\midrule"]


def num(x, k=3):
    """LaTeX number with a text minus sign, never inside math mode."""
    return ("$-$" if x < 0 else "") + f"{abs(x):.{k}f}"


row = ["Average post-period effect"]
for w in ("full", "flat"):
    for k in ("equal", "DiD-matched"):
        row.append(num(res[w][k]["point"]))
L.append(" & ".join(row) + r" \\")
row = [r"\quad (s.e.)"]
for w in ("full", "flat"):
    for k in ("equal", "DiD-matched"):
        row.append(f"({res[w][k]['se']:.3f})")
L.append(" & ".join(row) + r" \\")
L.append(r"\addlinespace")
L.append(r"\multicolumn{5}{l}{\textit{Relative-magnitudes confidence sets}}\\")
for M in [0.25, 0.5, 1.0, 2.0]:
    row = [f"\\quad $\\bar M = {M}$"]
    for w in ("full", "flat"):
        for k in ("equal", "DiD-matched"):
            c = res[w][k].get(f"M{M}")
            row.append("---" if c is None else
                       f"[{num(c[0], 2)}, {num(c[1], 2)}]")
    L.append(" & ".join(row) + r" \\")
L.append(r"\addlinespace")
row = ["Breakdown value"]
for w in ("full", "flat"):
    for k in ("equal", "DiD-matched"):
        row.append(str(res[w][k]["breakdown"]))
L.append(" & ".join(row) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(PAP, "tables", "P11.tex"), "w", encoding="utf-8").write("\n".join(L))

mac = {}
for w, tag in (("full", "Full"), ("flat", "Flat")):
    for k, kt in (("equal", "Eq"), ("DiD-matched", "Did")):
        r = res[w][k]
        mac[f"Hon{tag}{kt}Pt"] = num(r["point"])
        mac[f"Hon{tag}{kt}Bd"] = str(r["breakdown"])
pth = os.path.join(PAP, "numbers_pool.tex")
have = open(pth, encoding="utf-8").read()
with open(pth, "a", encoding="utf-8") as fh:
    fh.write("\n")
    for k, v in sorted(mac.items()):
        if "\\p" + k + "}" in have:
            continue
        fh.write("\\newcommand{\\p" + k + "}{" + v + "}\n")
print(f"\nwrote P11.tex, results_honest_avg.json and {len(mac)} macros")
