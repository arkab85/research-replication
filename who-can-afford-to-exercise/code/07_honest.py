"""Rambachan-Roth sensitivity with the FULL clustered covariance matrix of the
event-study coefficients (the draft used a diagonal approximation and flagged it)."""
import numpy as np, pandas as pd, os, json, warnings
from config import OUT
warnings.filterwarnings("ignore")
from honestdid import constructOriginalCS, createSensitivityResults_relativeMagnitudes

ev = pd.read_csv(os.path.join(OUT, "event_study.csv"), index_col=0)
V  = np.load(os.path.join(OUT, "event_vcov.npy"))
months = list(ev.index)
pre  = [m for m in months if m <= 202001]
post = [m for m in months if m >= 202003]
nP, nQ = len(pre), len(post)
beta = ev.coef.values
print(f"pre periods {nP}, post periods {nQ}")
print("off-diagonal correlation of event-study coefficients:")
D = np.sqrt(np.diag(V)); R = V / np.outer(D, D)
print(f"  mean |corr| off-diagonal = {np.abs(R[~np.eye(len(R), dtype=bool)]).mean():.3f}")
print(f"  mean corr among post-period coefficients = "
      f"{R[nP:, nP:][~np.eye(nQ, dtype=bool)].mean():.3f}")

def val(x):
    v = x
    for _ in range(4):
        if hasattr(v, "iloc"): v = v.iloc[0]
        elif isinstance(v, (list, tuple, np.ndarray)): v = np.asarray(v).ravel()[0]
        else: break
    return float(v)

res = {}
targets = {"june2020": 202006, "sep2020": 202009, "avg_post": None}
GRID = 200
for lab, tgt in targets.items():
    l = np.zeros(nQ)
    if tgt is None: l[:] = 1.0 / nQ
    else: l[post.index(tgt)] = 1.0
    pt = float(l @ beta[nP:])
    row = {"point": round(pt, 4)}
    cs0 = constructOriginalCS(beta, V, nP, nQ, l_vec=l)
    row["parallel_trends"] = [round(val(cs0["lb"]), 3), round(val(cs0["ub"]), 3)]
    print(f"\n=== {lab}  point estimate {pt:+.3f} ===")
    print(f"  parallel trends            [{row['parallel_trends'][0]:+.3f}, "
          f"{row['parallel_trends'][1]:+.3f}]")
    for bd, bl in [(None, "unrestricted"), ("positive", "sign-restricted")]:
        for M in [0.25, 0.5, 1.0, 1.5, 2.0]:
            try:
                r = createSensitivityResults_relativeMagnitudes(
                    beta, V, nP, nQ, Mbarvec=[M], l_vec=l,
                    gridPoints=GRID, biasDirection=bd)
                lb, ub = val(r["lb"]), val(r["ub"])
                inc0 = "  <- includes 0" if (lb <= 0 <= ub) else ""
                print(f"  {bl:<16} Mbar={M:<4} [{lb:+.3f}, {ub:+.3f}]{inc0}")
                row[f"{bl}_M{M}"] = [round(lb, 3), round(ub, 3)]
            except Exception as e:
                print(f"  {bl:<16} Mbar={M:<4} FAILED: {type(e).__name__}")
    res[lab] = row

with open(os.path.join(OUT, "results_honest.json"), "w") as fh:
    json.dump(res, fh, indent=1)
print("\nwrote results_honest.json")
