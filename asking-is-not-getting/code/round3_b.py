"""journal round 3: (1) inference that does not lean on ~29 day-clusters: wild cluster bootstrap + randomization inference.
(2) who was screened out: heterogeneity of the jump. (3) targeting: who completes the package. Uses stated-need flags if present."""
import os, json, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore"); rng = np.random.default_rng(20200407)
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out")
D = pd.read_parquet(os.path.join(OUT, "rd_frame.parquet"))
e = pd.read_csv(r"<DATA>/panel\Forbearance_Epsilon_Claritas_updated_IB_OB_Latest2023.csv", dtype=str, low_memory=False,
                usecols=["LoanID", "Liquid.Resources.2.0", "Advantage.Target.Income.3.0", "Advantage.Household.Age..Enhanced.", "Target.Net.Worth.3.0.Code"]).drop_duplicates("LoanID").set_index("LoanID")
e.columns = ["liq", "inc", "age", "nw"]; e = e.apply(pd.to_numeric, errors="coerce"); D = D.join(e)
D["lowliq"] = np.where(D.liq.isna(), np.nan, (D.liq <= 2).astype(float)); D["lowinc"] = np.where(D.inc.isna(), np.nan, (D.inc <= 4).astype(float)); D["lowfico"] = np.where(D.fico.isna(), np.nan, (D.fico < 600).astype(float))
nf = os.path.join(OUT, "inq_need_flags.parquet"); HAVE_NEED = os.path.exists(nf)
if HAVE_NEED:
    N = pd.read_parquet(nf); D = D.join(N.add_prefix("nd_"))
    D["said_jobloss"] = (D.nd_jobloss > 0).astype(float); D["said_cantpay"] = (D.nd_cantpay > 0).astype(float); D["said_precaution"] = ((D.nd_precaution > 0) & (D.nd_jobloss == 0) & (D.nd_cantpay == 0)).astype(float)
    D.loc[D.nd_n_call_notes == 0, ["said_jobloss", "said_cantpay", "said_precaution"]] = np.nan
Cv, Gv = D[D.Gov == 0].copy(), D[D.Gov == 1].copy(); res = {"have_need": HAVE_NEED}
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""

def design(x):
    X = np.column_stack([np.ones(len(x)), x.post, x.r, x.post * x.r]); w = (1 - x.r.abs() / 15).values; return X, w
def wls(y, X, w):
    W = np.sqrt(w)[:, None]; b, *_ = np.linalg.lstsq(X * W, y * W[:, 0], rcond=None); return b
def cl_se(y, X, w, b, g):
    Xw = X * w[:, None]; bread = np.linalg.inv(X.T @ Xw); u = y - X @ b; meat = np.zeros((X.shape[1],) * 2)
    for k in np.unique(g): s = (Xw[g == k] * u[g == k, None]).sum(0); meat += np.outer(s, s)
    return np.sqrt(np.diag(bread @ meat @ bread))
def wild(d, y, h=14, B=1999):
    x = d[d.r.abs() <= h].dropna(subset=[y]); X, w = design(x); yy = x[y].values.astype(float); g = pd.factorize(x.r)[0]
    b = wls(yy, X, w); t0 = b[1] / cl_se(yy, X, w, b, g)[1]
    Xr = X[:, [0, 2, 3]]; br = wls(yy, Xr, w); ur = yy - Xr @ br; G = g.max() + 1; ts = []
    for _ in range(B):
        v = rng.choice([-1.0, 1.0], G)[g]; ys = Xr @ br + ur * v; bs = wls(ys, X, w); ts.append(bs[1] / cl_se(ys, X, w, bs, g)[1])
    return b[1], float((np.abs(ts) >= abs(t0)).mean()), G
def randinf(d, y, win=7, B=4999, level="day"):
    x = d[(d.r >= -win) & (d.r < win)].dropna(subset=[y]); yy = x[y].values.astype(float); T = x.post.values; obs = yy[T == 1].mean() - yy[T == 0].mean()
    if level == "unit": draws = [(lambda p: yy[p == 1].mean() - yy[p == 0].mean())(rng.permutation(T)) for _ in range(B)]
    else:
        days = x.r.values; ud = np.unique(days); k = int((ud >= 0).sum()); draws = []
        for _ in range(B):
            tr = set(rng.choice(ud, k, replace=False)); p = np.array([dd in tr for dd in days]); draws.append(yy[p].mean() - yy[~p].mean())
    return obs, float((np.abs(draws) >= abs(obs)).mean()), len(x)

print("=== (1) inference: local-linear jump, conventional, h=14 ===")
for y, lab, sc in [("fb", "forbearance", 100), ("dq_sep20", "delinquent Sep-20", 100), ("mod_post", "modified", 100), ("performing", "performing Apr-21", 100), ("fc_path", "foreclosure path", 100)]:
    b, pw, G = wild(Cv, y); o1, p1, n1 = randinf(Cv, y, 7, level="unit"); o2, p2, _ = randinf(Cv, y, 7, level="day")
    res["inf_" + y] = {"b": b, "p_wild": pw, "clusters": int(G), "dm7": o1, "p_ri_unit": p1, "p_ri_day": p2, "n7": n1}
    print(f"  {lab:<20} jump {sc*b:7.1f}  wild-cluster p={pw:.3f} (G={G}) | +-7d diff in means {sc*o1:7.1f}  RI p unit={p1:.3f} day={p2:.3f}  n={n1}")

def dm(d, y, h=14):
    x = d[d.r.abs() <= h].dropna(subset=[y]).reset_index(drop=True)
    if len(x) < 40 or x.post.nunique() < 2: return (np.nan, np.nan, np.nan, len(x))
    m = smf.ols(f"{y} ~ post", x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); return m.params["post"], m.bse["post"], m.pvalues["post"], len(x)
def inter(d, y, s, h=14):
    x = d[d.r.abs() <= h].dropna(subset=[y, s]).reset_index(drop=True); m = smf.ols(f"{y} ~ post*{s}", x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); k = f"post:{s}"; return m.params[k], m.bse[k], m.pvalues[k]
print("\n=== (2) who was screened out: difference in means (+-14d) by subgroup, conventional ===")
SPL = [("dq_feb20", "delinquent Feb-20"), ("hard", "prior hardship flag"), ("lowfico", "FICO < 600"), ("lowliq", "liquid savings < $2,500 (modeled)"), ("lowinc", "income < $40k (modeled)")]
if HAVE_NEED: SPL += [("said_jobloss", "said job/income loss on call"), ("said_precaution", "precautionary / could still pay")]
OUTS = [("fb", "FB"), ("dq_sep20", "DQ Sep-20"), ("mod_post", "modified"), ("performing", "performing"), ("fc_path", "FC path")]
for s, sl in SPL:
    print(f"\n  -- {sl}: share = {Cv[Cv.r.abs() <= 14][s].mean():.2f}")
    for y, yl in OUTS:
        a1, a0, it = dm(Cv[Cv[s] == 1], y), dm(Cv[Cv[s] == 0], y), inter(Cv, y, s); res[f"het_{s}_{y}"] = {"yes": a1, "no": a0, "int": it}
        print(f"     {yl:<11} yes: {100*a1[0]:6.1f} ({100*a1[1]:.1f}){star(a1[2]):<3} n={a1[3]:<4} | no: {100*a0[0]:6.1f} ({100*a0[1]:.1f}){star(a0[2]):<3} n={a0[3]:<4} | diff {100*it[0]:6.1f} ({100*it[1]:.1f}){star(it[2])}")
if HAVE_NEED:
    print("\n  balance of stated need at the cutoff (diff in means +-14d):")
    for s in ["said_jobloss", "said_cantpay", "said_precaution"]:
        a = dm(Cv, s); g = dm(Gv, s); res["bal_" + s] = {"conv": a, "gov": g}; print(f"     {s:<16} conv {100*a[0]:6.1f} ({100*a[1]:.1f}){star(a[2])}  mean {100*Cv[Cv.r.abs()<=14][s].mean():.1f} | gov {100*g[0]:6.1f} ({100*g[1]:.1f}){star(g[2])}")

print("\n=== (3) targeting: who got forbearance under each regime (conventional, first inquiry 16 Mar-31 May) ===")
W = Cv[(Cv.inq >= "2020-03-16") & (Cv.inq <= "2020-05-31")]
cols = ["dq_feb20", "hard", "fico", "pre_in", "lowliq", "lowinc", "tenure"] + (["said_jobloss", "said_cantpay", "said_precaution"] if HAVE_NEED else [])
grp = {"easy regime: granted": W[(W.post == 0) & (W.fb == 1)], "easy regime: not granted": W[(W.post == 0) & (W.fb == 0)], "paperwork regime: granted": W[(W.post == 1) & (W.fb == 1)], "paperwork regime: not granted": W[(W.post == 1) & (W.fb == 0)]}
T = pd.DataFrame({k: v[cols].mean() for k, v in grp.items()}).T; T.insert(0, "n", [len(v) for v in grp.values()]); print(T.round(3).to_string()); res["targeting"] = T.round(4).to_dict()
# are paperwork-regime recipients needier than easy-regime recipients?
R = W[W.fb == 1].copy()
for c in cols:
    x = R.dropna(subset=[c]).reset_index(drop=True); m = smf.ols(f"{c} ~ post", x).fit(cov_type="HC1"); res["tgt_" + c] = (m.params["post"], m.bse["post"], m.pvalues["post"]); print(f"   recipients, paperwork minus easy: {c:<16} {m.params['post']:8.3f} ({m.bse['post']:.3f}){star(m.pvalues['post'])}")
# outcomes of recipients under each regime
print("\n   later outcomes of recipients: easy vs paperwork regime")
print(R.groupby("post")[["dq_sep20", "mod_post", "performing", "fc_path"]].mean().round(3).assign(n=R.groupby("post").size()).to_string())
json.dump(res, open(os.path.join(OUT, "round3b.json"), "w"), indent=1, default=float)
