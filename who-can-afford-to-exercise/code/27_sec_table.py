"""Emit the FY2019 balance-sheet table and its macros."""
import pandas as pd, numpy as np, os, json, re
from config import OUT, PAPER as PAP

TAB = os.path.join(PAP, "tables")
m = pd.read_csv(os.path.join(OUT, "sec_merged.csv"), index_col=0)
m = m.sort_values("cash_msr")

def f(x, d=2, pct=False):
    if pd.isna(x): return "---"
    return f"{x*100:.1f}\\%" if pct else f"{x:.{d}f}"

rows = ""
for i, r in m.iterrows():
    act = r.ebo19 >= .10
    nm = str(r.parent)
    if act: nm = r"\textbf{" + nm + "}"
    rows += (f"{nm} & {f(r.assets/1e9,1)} & {f(r.cash/1e9,2)} & {f(r.msr/1e9,2)} & "
             f"{f(r.cash_assets,pct=True)} & {f(r.cash_msr,pct=True)} & "
             f"{r.ebo19*100:.1f}\\% & {r.d_ebo*100:+.1f} \\\\\n")

body = r"""\begin{tabular}{lrrrrrrr}
\toprule
& \multicolumn{3}{c}{FY2019 balance sheet (\$bn)} & \multicolumn{2}{c}{Liquidity ratios} & \multicolumn{2}{c}{Buyout exercise}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
SEC registrant & Assets & Cash & MSR & Cash/assets & Cash/MSR & 2019 & Change (pp)\\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}"""
open(os.path.join(TAB, "TA2.tex"), "w", encoding="utf-8").write(
    re.compile(r"(?<=[\s(\[&{,])-(?=\d)").sub("$-$", body))

# macros
act = m[m.ebo19 >= .10]
mac = {}
mac["SecN"] = str(len(m))
mac["SecNAct"] = str(len(act))
mac["SecCorrMSR"] = f"{m[['S','log_msr']].dropna().corr().iloc[0,1]:+.2f}"
mac["SecCorrMSRrank"] = f"{m[['S','log_msr']].dropna().corr(method='spearman').iloc[0,1]:+.2f}"
mac["SecCorrCashMSR"] = f"{m[['S','cash_msr']].dropna().corr().iloc[0,1]:+.2f}"
mac["SecCorrCashMSRrank"] = f"{m[['S','cash_msr']].dropna().corr(method='spearman').iloc[0,1]:+.2f}"
pm = m[m.parent.str.contains("PennyMac")].iloc[0]
mc = m[m.parent.str.contains("Cooper")].iloc[0]
oc = m[m.parent.str.contains("Ocwen")].iloc[0]
rk = m[m.parent.str.contains("Rocket")].iloc[0]
for k, r in [("PM", pm), ("MC", mc), ("OC", oc), ("RK", rk)]:
    mac["Sec" + k + "CashMSR"] = f"{r.cash_msr*100:.1f}"
    mac["Sec" + k + "CashAssets"] = f"{r.cash_assets*100:.1f}"
    mac["Sec" + k + "Drop"] = f"{abs(r.d_ebo)*100:.1f}"
# is the active-issuer ranking monotone?
a = act.sort_values("cash_msr")
mac["SecRankMono"] = "yes" if list(a.d_ebo.rank()) == sorted(a.d_ebo.rank()) else "no"
json.dump(mac, open(os.path.join(OUT, "sec_macros.json"), "w"), indent=1)
print(json.dumps(mac, indent=1))
print("\nactive issuers ordered by cash/MSR (low to high):")
print(a[["parent", "cash_msr", "d_ebo"]].round(3).to_string(index=False))
