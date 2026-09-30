"""Long-horizon outcomes at the 7-April-2020 cutoff, to run when the updated servicing files arrive.
Usage:  python post2021.py --remit <updated RemittanceGAAP-format csv> [--relief <updated COVID_Inquiry_FB-format csv>] [--tag v]
Needs only the columns already used: remittance (LoanID, RemittanceDate, PrincipalPayment, InterestPayment, ProceedsOnClosedPositions);
relief (LoanID, DispositionPath, ModificationDate). Run with the existing files it reproduces the 12-month horizon as a check.
Writes tex/<tag>/tL_longrun.tex, fL_longrun.pdf, numbers_long.tex. No borrower identifiers are written."""
import os, argparse, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--remit", default=r"<DATA>/relief\RemittanceGAAP.csv"); ap.add_argument("--relief", default=None); ap.add_argument("--tag", default="v"); A = ap.parse_args()
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out"); TEX = os.path.join(H, "tex", A.tag); os.makedirs(TEX, exist_ok=True)
D = pd.read_parquet(os.path.join(OUT, "rd_frame.parquet")); D = D[D.r.abs() <= 14].copy(); D.index = D.index.astype(str)
rm = pd.read_csv(A.remit, dtype={"LoanID": str}, usecols=["LoanID", "RemittanceDate", "PrincipalPayment", "InterestPayment", "ProceedsOnClosedPositions"]); rm = rm[rm.LoanID.isin(D.index)]
rm["m"] = pd.to_datetime(rm.RemittanceDate, errors="coerce").dt.to_period("M"); rm = rm.dropna(subset=["m"])
for c in ["PrincipalPayment", "InterestPayment", "ProceedsOnClosedPositions"]: rm[c] = pd.to_numeric(rm[c], errors="coerce").fillna(0)
g = rm.groupby(["LoanID", "m"]).agg(pi=("PrincipalPayment", "sum"), it=("InterestPayment", "sum"), pr=("ProceedsOnClosedPositions", "sum")).reset_index(); g["paid"] = ((g.pi + g.it) > 0).astype(int); g["closed"] = (g.pr > 0).astype(int)
last = g.m.max(); print("remittance through", last, "| loans in window covered:", D.index.isin(g.LoanID.unique()).mean().round(3))
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
def dm(d, y):
    x = d.dropna(subset=[y]); m = smf.ols(f"{y} ~ post", x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); return m.params["post"], m.bse["post"], m.pvalues["post"], x.loc[x.post == 0, y].mean(), len(x)
def dd(d, y):
    x = d.dropna(subset=[y]).copy(); x["conv"] = 1 - x.Gov; m = smf.ols(f"{y} ~ post*conv", x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); return m.params["post:conv"], m.bse["post:conv"], m.pvalues["post:conv"]
P = lambda s: pd.Period(s, "M"); rows, M = [], {}
hor = [(12, "2021-03"), (24, "2022-03"), (36, "2023-03"), (48, "2024-03"), (60, "2025-03")]
for h, end in hor:
    if P(end) > last: continue
    w = g[(g.m >= P("2020-04")) & (g.m <= P(end))].groupby("LoanID"); D[f"pay{h}"] = w.paid.sum().reindex(D.index).fillna(0); D[f"liq{h}"] = (w.closed.max().reindex(D.index).fillna(0) > 0).astype(float)
    q = g[(g.m > P(end) - 3) & (g.m <= P(end))].groupby("LoanID").paid.max(); D[f"paying{h}"] = (q.reindex(D.index).fillna(0) > 0).astype(float)
    for y, lab, sc in [(f"pay{h}", f"Monthly payments made, Apr 2020 to month {h}", 1), (f"paying{h}", f"Paying in the last quarter of month {h} (pp)", 100), (f"liq{h}", f"Liquidated by month {h} (pp)", 100)]:
        c = dm(D[D.Gov == 0], y); gv = dm(D[D.Gov == 1], y); x = dd(D, y)
        rows.append([lab, f"{sc*c[3]:.1f}", f"{sc*c[0]:.1f}{star(c[2])}", f"({sc*c[1]:.1f})", f"{sc*gv[0]:.1f}", f"{sc*x[0]:.1f}{star(x[2])}", f"({sc*x[1]:.1f})", f"{c[4]:,}"])
        M[f"L{y}".replace("1", "One").replace("2", "Two").replace("3", "Three").replace("4", "Four").replace("6", "Six").replace("0", "Zero").replace("8", "Eight")] = f"{sc*c[0]:.1f}"
    rows.append("MID")
if A.relief:
    rv = pd.read_csv(A.relief, dtype=str, na_values=["NULL"], usecols=lambda c: c in ["LoanID", "DispositionPath", "ModificationDate"]).drop_duplicates("LoanID").set_index("LoanID")
    D["disp_new"] = rv.DispositionPath.reindex(D.index)
    for lab, vals in [("Performing at latest date (pp)", ["Performing"]), ("Foreclosure completed or REO (pp)", ["REO", "Pending Foreclosure Completion"]), ("Paid in full (pp)", ["Paid in Full", "PIF", "Payoff"])]:
        D["y"] = D.disp_new.isin(vals).astype(float).where(D.disp_new.notna()); c = dm(D[D.Gov == 0], "y"); gv = dm(D[D.Gov == 1], "y"); x = dd(D, "y")
        rows.append([lab, f"{100*c[3]:.1f}", f"{100*c[0]:.1f}{star(c[2])}", f"({100*c[1]:.1f})", f"{100*gv[0]:.1f}", f"{100*x[0]:.1f}{star(x[2])}", f"({100*x[1]:.1f})", f"{c[4]:,}"])
    print("disposition values in the new file:", D.disp_new.value_counts().to_dict())
while rows and rows[-1] == "MID": rows.pop()
s = ["\\begin{tabular}{lccccccc}", "\\toprule", "Outcome & Mean before & Conv. diff. & s.e. & Gov. diff. & Difference & s.e. & Loans \\\\", "\\midrule"] + ["\\midrule" if r == "MID" else " & ".join(r) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(TEX, "tL_longrun.tex"), "w", encoding="utf-8").write("\n".join(s)); open(os.path.join(TEX, "numbers_long.tex"), "w", encoding="utf-8").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
# month-by-month payment difference through the end of the data
ms = [m for m in sorted(g.m.unique()) if m >= P("2020-01")]; est = {0: [], 1: []}
for m in ms:
    paid = g[g.m == m].set_index("LoanID").paid; D["y"] = paid.reindex(D.index).fillna(0)
    for k in (0, 1): c = dm(D[D.Gov == k], "y"); est[k].append((100 * c[0], 196 * c[1]))
fig, ax = plt.subplots(figsize=(6.5, 2.8)); xs = np.arange(len(ms))
for k, col, lab in [(0, "#c8551f", "Conventional"), (1, "#1f5fa8", "Government-backed (placebo)")]:
    e = np.array(est[k]); ax.errorbar(xs + (.15 if k else -.15), e[:, 0], yerr=e[:, 1], fmt="o-", ms=2.5, lw=1, capsize=1.5, color=col, label=lab)
ax.axhline(0, color="#777", lw=.6); step = max(1, len(ms) // 12); ax.set_xticks(xs[::step]); ax.set_xticklabels([str(m) for m in ms][::step], rotation=45, fontsize=7); ax.set_ylabel("Difference in share paying (pp)", fontsize=8); ax.legend(frameon=False, fontsize=7)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(TEX, "fL_longrun.pdf")); fig.savefig(os.path.join(TEX, "fL_longrun.png"), dpi=110)
for r in rows: print(r)

