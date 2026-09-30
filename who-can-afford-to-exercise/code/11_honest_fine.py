"""Pin down the breakdown value for the unrestricted relative-magnitudes set."""
import numpy as np, pandas as pd, os, json, warnings
from config import OUT
warnings.filterwarnings("ignore")
from honestdid import createSensitivityResults_relativeMagnitudes

ev = pd.read_csv(os.path.join(OUT, "event_study.csv"), index_col=0)
V = np.load(os.path.join(OUT, "event_vcov.npy"))
months = list(ev.index)
pre = [m for m in months if m <= 202001]; post = [m for m in months if m >= 202003]
nP, nQ = len(pre), len(post)
beta = ev.coef.values
l = np.zeros(nQ); l[post.index(202006)] = 1.0

def val(x):
    v = x
    for _ in range(4):
        if hasattr(v, "iloc"): v = v.iloc[0]
        elif isinstance(v, (list, tuple, np.ndarray)): v = np.asarray(v).ravel()[0]
        else: break
    return float(v)

res = json.load(open(os.path.join(OUT, "results_honest.json")))
out = {}
for M in [0.6, 0.7, 0.75, 0.8, 0.9]:
    r = createSensitivityResults_relativeMagnitudes(beta, V, nP, nQ, Mbarvec=[M],
                                                    l_vec=l, gridPoints=200)
    lb, ub = val(r["lb"]), val(r["ub"])
    inc = lb <= 0 <= ub
    print(f"  Mbar={M:<5} [{lb:+.3f}, {ub:+.3f}]{'  <- includes 0' if inc else ''}",
          flush=True)
    out[str(M)] = [round(lb, 3), round(ub, 3), bool(inc)]
    res["june2020"][f"unrestricted_M{M}"] = [round(lb, 3), round(ub, 3)]

bd = [float(k) for k, v in out.items() if v[2]]
res["june2020"]["breakdown"] = min(bd) if bd else ">0.9"
print("\nbreakdown value (first Mbar whose unrestricted set contains zero):",
      res["june2020"]["breakdown"])
json.dump(res, open(os.path.join(OUT, "results_honest.json"), "w"), indent=1)
