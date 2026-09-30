"""Treat the cutoff as located by the data and price the search.  (1) sup-|t| over every admissible weekday cutoff, with a
wild-cluster bootstrap of its null distribution around a smooth (cubic) trend with no break; (2) least-squares break date with a
likelihood-ratio confidence set and a bootstrap distribution; (3) split-sample: locate the break in one random half of loans,
estimate the jump at that date in the other half.  Conventional loans, government-backed as a check that the procedure does not
manufacture breaks.  Input: the loan-level frame written by contemporaneous_records.py.  Output: aggregate JSON."""
import json, numpy as np, pandas as pd
rng = np.random.default_rng(20200407)
import os
TMP = os.environ.get("AING_SECURE_TMP", "<SECURE_TMP>"); F = pd.read_parquet(os.path.join(TMP, "frame_contemp.parquet"))
OUTJ = os.environ.get("AING_OUT2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "validation", "cutoff_search.json"))
LO, HI = pd.Timestamp("2020-03-16"), pd.Timestamp("2020-05-29"); CUT = pd.Timestamp("2020-04-07")
def ll_fit(day, y, c, h):
    """local linear jump at cutoff day c (days), triangular kernel, bandwidth h; returns est, cluster-robust se (by day)."""
    r = day - c; m = np.abs(r) <= h
    if m.sum() < 10: return np.nan, np.nan
    r, yy, dd = r[m], y[m], day[m]; post = (r >= 0).astype(float); w = 1 - np.abs(r) / (h + 1)
    X = np.column_stack([np.ones_like(r), post, r, post * r]).astype(float)
    XtW = X.T * w; A = XtW @ X
    try: Ai = np.linalg.inv(A)
    except np.linalg.LinAlgError: return np.nan, np.nan
    b = Ai @ (XtW @ yy); e = yy - X @ b
    u = X * (w * e)[:, None]; G = pd.DataFrame(u).groupby(dd).sum().values
    V = Ai @ (G.T @ G) @ Ai; G_ = len(G); V *= G_ / max(G_ - 1, 1)
    return b[1], np.sqrt(V[1, 1])
def candidates(day, h, minside=25):
    out = []
    for c in pd.bdate_range(LO + pd.Timedelta(days=h), HI - pd.Timedelta(days=h)):
        k = (c - CUT).days; r = day - k
        if ((r >= -h) & (r < 0)).sum() >= minside and ((r >= 0) & (r <= h)).sum() >= minside: out.append(k)
    return out
res = {}
for g, lab in [(0, "conv"), (1, "gov")]:
    d = F[(F.Gov == g) & (F.inq >= LO) & (F.inq <= HI)]
    day = d.r.values.astype(float); y = d.fb.values.astype(float)
    for h in (7, 14):
        C = candidates(day, h)
        T = np.array([ll_fit(day, y, c, h) for c in C]); t = np.abs(T[:, 0] / T[:, 1])
        i = int(np.nanargmax(t)); obs = float(t[i])
        # null: cubic trend in calendar day, no break; wild cluster (Rademacher by day) residual bootstrap
        Xn = np.column_stack([np.ones_like(day), day, day**2, day**3]); bn = np.linalg.lstsq(Xn, y, rcond=None)[0]; fit = Xn @ bn; e = y - fit
        ud = np.unique(day); idx = np.searchsorted(ud, day); B = 999; sup = np.empty(B)
        for bdraw in range(B):
            v = rng.choice([-1.0, 1.0], size=len(ud))[idx]; ys = fit + e * v
            Tb = np.array([ll_fit(day, ys, c, h) for c in C]); sup[bdraw] = np.nanmax(np.abs(Tb[:, 0] / Tb[:, 1]))
        res[f"{lab}_h{h}"] = {"n_candidates": len(C), "argmax_date": str((CUT + pd.Timedelta(days=C[i])).date()), "sup_t": obs,
                              "est_at_argmax": 100 * T[i, 0], "t_at_7apr": float(t[C.index(0)]) if 0 in C else None,
                              "boot_p": float((sup >= obs).mean()), "boot_crit95": float(np.quantile(sup, .95)), "boot_crit99": float(np.quantile(sup, .99)),
                              "second_largest_t": float(np.sort(t)[-2]), "second_date": str((CUT + pd.Timedelta(days=C[int(np.argsort(t)[-2])])).date()),
                              "est_at_second": 100 * T[int(np.argsort(t)[-2]), 0],
                              "n_left_of_second": int(((day - C[int(np.argsort(t)[-2])] >= -h) & (day - C[int(np.argsort(t)[-2])] < 0)).sum()),
                              "days_left_of_second": int(len(np.unique(day[(day - C[int(np.argsort(t)[-2])] >= -h) & (day - C[int(np.argsort(t)[-2])] < 0)]))),
                              "first_candidate": str((CUT + pd.Timedelta(days=C[0])).date()), "last_candidate": str((CUT + pd.Timedelta(days=C[-1])).date())}
        print(lab, h, res[f"{lab}_h{h}"], flush=True)
    # least-squares break date over the admissible dates of the seven-day search: piecewise-linear with a jump at k, SSR(k)
    ks = [k for k in candidates(day, 7)]
    def ssr(k, yy=y, dd=day):
        post = (dd >= k).astype(float); X = np.column_stack([np.ones_like(dd), post, dd, post * (dd - k)])
        b = np.linalg.lstsq(X, yy, rcond=None)[0]; return float(((yy - X @ b) ** 2).sum())
    S = np.array([ssr(k) for k in ks]); j = int(S.argmin()); s2 = S[j] / len(y)
    LR = (S - S[j]) / s2; cs = [str((CUT + pd.Timedelta(days=ks[q])).date()) for q in range(len(ks)) if LR[q] <= 7.35]
    # bootstrap the break date: resample loans within inquiry day
    bd = []
    grp = pd.Series(np.arange(len(y))).groupby(day).apply(lambda s: s.values).to_dict()
    for bdraw in range(500):
        ii = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in grp.values()])
        Sb = np.array([ssr(k, y[ii], day[ii]) for k in ks]); bd.append(ks[int(Sb.argmin())])
    bd = pd.Series(bd).map(lambda k: str((CUT + pd.Timedelta(days=k)).date())).value_counts(normalize=True)
    res[f"{lab}_lsbreak"] = {"ls_date": str((CUT + pd.Timedelta(days=ks[j])).date()), "lr_confset_7.35": cs, "boot_dates": bd.round(3).to_dict()}
    print(lab, res[f"{lab}_lsbreak"], flush=True)
# split sample (conventional): locate the break by least squares in a random half A, estimate the jump at that date in half B
d = F[(F.Gov == 0) & (F.inq >= LO) & (F.inq <= HI)]; day = d.r.values.astype(float); y = d.fb.values.astype(float)
ks = candidates(day, 7); found, estB = [], []
def ssr2(k, yy, dd):
    post = (dd >= k).astype(float); X = np.column_stack([np.ones_like(dd), post, dd, post * (dd - k)])
    b = np.linalg.lstsq(X, yy, rcond=None)[0]; return float(((yy - X @ b) ** 2).sum())
for s in range(500):
    a = rng.random(len(y)) < 0.5
    k = ks[int(np.argmin([ssr2(k, y[a], day[a]) for k in ks]))]
    found.append(k); estB.append(100 * ll_fit(day[~a], y[~a], k, 14)[0])
found = pd.Series(found).map(lambda k: str((CUT + pd.Timedelta(days=k)).date())).value_counts(normalize=True)
estB = np.array(estB)
res["split"] = {"draws": 500, "found_dates": found.round(3).to_dict(), "estB_median": float(np.nanmedian(estB)),
                "estB_p05": float(np.nanpercentile(estB, 5)), "estB_p95": float(np.nanpercentile(estB, 95)), "share_estB_below_minus50": float(np.nanmean(estB < -50))}
print(res["split"])
os.makedirs(os.path.dirname(OUTJ), exist_ok=True); json.dump(res, open(OUTJ, "w"), indent=1, default=float)
