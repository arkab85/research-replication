"""Regression discontinuity in time at 7 April 2020 for conventional hardship inquiries; government-backed as placebo.
Outcomes: forbearance (first stage); covariates (balance); payments Apr-2020..Mar-2021 (RemittanceGAAP); status Sep-2020; disposition Apr-2021."""
import os, json, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
from statsmodels.sandbox.regression.gmm import IV2SLS
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out")
CUT = pd.Timestamp("2020-04-07")
D = pd.read_parquet(os.path.join(OUT, "inq_timing.parquet")); L = pd.read_parquet(os.path.join(OUT, "loan_frame.parquet"))
D = D.join(L[["status_feb20", "AssetType", "pre_in", "pre_out", "fico", "ltv", "bal", "Investor", "pre_unemp", "pre_curtail", "board", "disp", "dq_apr21", "moddate", "fb_start", "fb_end", "optout"]])
D["r"] = (D.inq.dt.normalize() - CUT).dt.days; D["post"] = (D.r >= 0).astype(int); D["conv"] = 1 - D.Gov
D["npl"] = D.AssetType.eq("NPL").astype(int); D["hard"] = ((D.pre_unemp.fillna(0) + D.pre_curtail.fillna(0)) > 0).astype(int); D["tenure"] = (pd.Timestamp("2020-03-01") - D.board).dt.days / 30.44
D["lbal"] = np.log(D.bal.where(D.bal > 0))
# ---- remittance: monthly P&I payments ----
rm = pd.read_csv(r"<DATA>/relief\RemittanceGAAP.csv", dtype={"LoanID": str}, usecols=["LoanID", "RemittanceDate", "PrincipalPayment", "InterestPayment", "ProceedsOnClosedPositions"])
rm["m"] = pd.to_datetime(rm.RemittanceDate, errors="coerce").dt.to_period("M"); rm = rm.dropna(subset=["m"])
for c in ["PrincipalPayment", "InterestPayment", "ProceedsOnClosedPositions"]: rm[c] = pd.to_numeric(rm[c], errors="coerce").fillna(0)
g = rm.groupby(["LoanID", "m"]).agg(pi=("PrincipalPayment", "sum"), it=("InterestPayment", "sum"), pr=("ProceedsOnClosedPositions", "sum")).reset_index()
g["paid"] = ((g.pi + g.it) > 0).astype(int); g["closed"] = (g.pr > 0).astype(int)
print("remittance months:", g.m.min(), "..", g.m.max(), "| loans:", g.LoanID.nunique(), "| inquirers covered:", D.index.isin(g.LoanID.unique()).mean().round(3))
P = lambda s: pd.Period(s, "M")
def win(a, b, col, how):
    x = g[(g.m >= P(a)) & (g.m <= P(b))].groupby("LoanID")[col]; return (x.sum() if how == "sum" else x.max())
D["inrem"] = D.index.isin(g.LoanID.unique()).astype(int)
D["pay_pre"] = win("2020-01", "2020-03", "paid", "sum").reindex(D.index).fillna(0)          # placebo outcome (pre-period)
D["pay_aprsep"] = win("2020-04", "2020-09", "paid", "sum").reindex(D.index).fillna(0)
D["pay_octmar"] = win("2020-10", "2021-03", "paid", "sum").reindex(D.index).fillna(0)
D["pay_any_q1_21"] = (win("2021-01", "2021-03", "paid", "max").reindex(D.index).fillna(0) > 0).astype(int)
D["dollars_octmar"] = (win("2020-10", "2021-03", "pi", "sum").reindex(D.index).fillna(0) + win("2020-10", "2021-03", "it", "sum").reindex(D.index).fillna(0))
D["closed"] = (win("2020-04", "2021-03", "closed", "max").reindex(D.index).fillna(0) > 0).astype(int)
D["performing"] = D.disp.eq("Performing").astype(int); D["cur_apr21"] = D.dq_apr21.eq("0-29").astype(int)
D["mod_post"] = (D.moddate >= "2020-04-01").astype(int); D["fc_path"] = D.disp.isin(["Pending Foreclosure Completion", "REO"]).astype(int)
D["fb_months"] = ((D.fb_end - D.fb_start).dt.days / 30.44).where(D.fb == 1)
ib = pd.read_csv(r"<DATA>/panel\IB_OB_Flags_Full_Nov3.csv", usecols=["LoanId", "Month t", "LoanStatus"], dtype=str)
s9 = ib[ib["Month t"] == "9/30/2020"].drop_duplicates("LoanId").set_index("LoanId").LoanStatus
D["dq_sep20"] = D.index.map(s9).to_series(index=D.index).map(lambda v: np.nan if pd.isna(v) else float(v != "Active - Current"))
D.to_parquet(os.path.join(OUT, "rd_frame.parquet"))

star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
def ll(d, y, h, donut=0):
    """local linear, separate slopes, triangular kernel, SE clustered on inquiry day."""
    x = d[(d.r.abs() <= h) & ~((d.r >= -donut) & (d.r < donut))].dropna(subset=[y]).copy()
    x["w"] = 1 - x.r.abs() / (h + 1)
    m = smf.wls(f"{y} ~ post + r + post:r", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]})
    return m.params["post"], m.bse["post"], m.pvalues["post"], len(x), x.loc[x.post == 0, y].mean()
def did_disc(d, y, h):
    x = d[d.r.abs() <= h].dropna(subset=[y]).copy(); x["w"] = 1 - x.r.abs() / (h + 1)
    m = smf.wls(f"{y} ~ post*conv + r*conv + post:r + post:r:conv", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]})
    return m.params["post:conv"], m.bse["post:conv"], m.pvalues["post:conv"], len(x)
def fuzzy(d, y, h):
    x = d[d.r.abs() <= h].dropna(subset=[y]).copy()
    X = np.column_stack([np.ones(len(x)), x.fb, x.r, x.post * x.r]); Z = np.column_stack([np.ones(len(x)), x.post, x.r, x.post * x.r])
    m = IV2SLS(x[y].values, X, Z).fit(); return m.params[1], m.bse[1], m.pvalues[1], len(x)

C, G = D[D.Gov == 0], D[D.Gov == 1]; res = {}
print("\n=== FIRST STAGE: forbearance at the 7-April cutoff ===")
for h in (7, 10, 14, 21):
    a, b = ll(C, "fb", h), ll(G, "fb", h); dd = did_disc(D, "fb", h); res[f"fs_{h}"] = {"conv": a, "gov": b, "dd": dd}
    print(f" h={h:>2}: conventional {100*a[0]:6.1f} ({100*a[1]:.1f}){star(a[2])} n={a[3]} | gov placebo {100*b[0]:5.1f} ({100*b[1]:.1f}){star(b[2])} n={b[3]} | diff-in-disc {100*dd[0]:6.1f} ({100*dd[1]:.1f}){star(dd[2])}")
a = ll(C, "fb", 14, donut=1); print(f" h=14, donut excluding 6-7 Apr: {100*a[0]:.1f} ({100*a[1]:.1f}) n={a[3]}"); res["fs_donut"] = a
print("\n=== BALANCE at the cutoff, conventional (h=14) ===")
for y in ["dq_feb20", "npl", "pre_in", "pre_out", "hard", "fico", "ltv", "lbal", "tenure", "pay_pre", "inrem"]:
    a = ll(C, y, 14); res["bal_" + y] = a; print(f"  {y:<10} jump {a[0]:8.3f} ({a[1]:.3f}){star(a[2])}  left mean {a[4]:.3f}  n={a[3]}")
print("  density: conventional first inquiries in the 10 weekdays before / after:", int(((C.r >= -14) & (C.r < 0)).sum()), "/", int(((C.r >= 0) & (C.r < 14)).sum()),
      "| government-backed:", int(((G.r >= -14) & (G.r < 0)).sum()), "/", int(((G.r >= 0) & (G.r < 14)).sum()))
print("\n=== OUTCOMES: reduced form (conv), placebo (gov), diff-in-disc, fuzzy-RD effect of forbearance (conv), h=14 and h=21 ===")
OUTC = [("pay_aprsep", "months with P&I, Apr-Sep 20"), ("pay_octmar", "months with P&I, Oct 20-Mar 21"), ("pay_any_q1_21", "any P&I in 2021Q1"), ("dollars_octmar", "P&I dollars, Oct 20-Mar 21"),
        ("dq_sep20", "delinquent Sep 20"), ("performing", "performing Apr 21"), ("cur_apr21", "0-29 days Apr 21"), ("mod_post", "modified after Apr 20"), ("fc_path", "foreclosure path Apr 21"), ("closed", "closed/liquidated by Mar 21")]
for y, lab in OUTC:
    for h in (14, 21):
        a, b, dd, fz = ll(C, y, h), ll(G, y, h), did_disc(D, y, h), fuzzy(C, y, h); res[f"{y}_{h}"] = {"rf": a, "gov": b, "dd": dd, "iv": fz}
        print(f"  {lab:<30} h={h}: RF {a[0]:8.3f} ({a[1]:.3f}){star(a[2]):<3} | gov {b[0]:7.3f} ({b[1]:.3f}){star(b[2]):<3} | DiD {dd[0]:8.3f} ({dd[1]:.3f}){star(dd[2]):<3} | IV(FB) {fz[0]:8.3f} ({fz[1]:.3f}){star(fz[2]):<3} | mean left {a[4]:.3f} n={a[3]}")
print("\nforbearance length (months), conventional completers: before cutoff", C[(C.post == 0)].fb_months.median().round(1), "after", C[(C.post == 1)].fb_months.median().round(1), "| gov", G.fb_months.median().round(1))
json.dump(res, open(os.path.join(OUT, "rd_results.json"), "w"), indent=1, default=float)
