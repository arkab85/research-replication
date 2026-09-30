"""The depository side of the audited evidence, and the pooled wedge.

Three depositories in the size-gradient test disclose enough to place them:

  Flagstar   reports the unexercised balance as its own balance-sheet line, "loans
             with government guarantees repurchase options" (FY2021: "...repurchase
             liability"): 70 / 1,851 / 200 ($m, 2019/2020/2021). It is the one
             depository in the sample that stopped exercising.
  Fifth Third discloses the 2020 decomposition outright: $921m of options vested
             through forbearance, $882m repurchased, $39m left unexercised. No 2019
             comparable exists, so it enters as a composition fact, not a growth rate.
  JPMorgan   reports only the combined "loans repurchased or option to repurchase",
             which is predominantly repurchased loans: 7,021 / 2,941 / 1,413 / 1,022
             ($m, 2018-2021). It enters as a level, falling.

Sources are verbatim in the table note. Everything here is read from the filing, not
scraped by position.
"""
import pandas as pd, numpy as np, os, json
from config import OUT, PAPER as PAP

REPORTED = {                                     # $bn, 31 December; key is a name fragment
    "FLAGSTAR BANK":  {"entity": "Flagstar Bank", "qty": "C",
                       2019: 0.070, 2020: 1.851, 2021: 0.200},
    "FIFTH THIRD":    {"entity": "Fifth Third Bank", "qty": "C",
                       2019: np.nan, 2020: 0.039, 2021: 0.000},
    "JP MORGAN CHASE": {"entity": "JPMorgan Chase", "qty": "B+C",
                        2019: 2.941, 2020: 1.413, 2021: 1.022},
}

p = pd.read_parquet(os.path.join(OUT, "panel.parquet")).dropna(subset=["buyout"])
p["yr"] = (p.ym // 100).astype(int)
nm = pd.read_csv(os.path.join(OUT, "issuer_id_names.csv"), index_col=0)["name"].str.upper().str.strip()

rows = []
for key, r in REPORTED.items():
    iid = nm[nm.str.startswith(key)].index
    cand = [i for i in iid if (p.issuer_id == i).sum() > 300]
    assert len(cand) == 1, (key, [(i, nm[i], int((p.issuer_id == i).sum())) for i in iid])
    g = p[p.issuer_id == cand[0]]
    rec = {"entity": r["entity"], "qty": r["qty"], "type": "depository"}
    for y in (2019, 2020):
        h = g[g.yr == y]
        rec[f"vested{y}"] = len(h)
        rec[f"rate{y}"] = h.buyout.mean()
        rec[f"rep{y}"] = r[y]
    rec["rep2021"] = r[2021]
    rec["flow_mult"] = rec["vested2020"] / rec["vested2019"]
    rec["rep_mult"] = rec["rep2020"] / rec["rep2019"] if rec["rep2019"] else np.nan
    rec["wedge"] = rec["rep_mult"] / rec["flow_mult"]
    rec["d_rate"] = (rec["rate2020"] - rec["rate2019"]) * 100
    rows.append(rec)
d = pd.DataFrame(rows)
nb = pd.read_csv(os.path.join(OUT, "four_firm.csv")); nb["type"] = "nonbank"

print("DEPOSITORIES")
for _, r in d.iterrows():
    mm = "n/a" if not np.isfinite(r.rep_mult) else f"{r.rep_mult:5.1f}x"
    wg = "n/a" if not np.isfinite(r.wedge) else f"{r.wedge:4.1f}"
    print(f"  {r.entity:<18} {r.qty:<4} balance {mm}  flow {r.flow_mult:4.2f}x  "
          f"wedge {wg}   exercise {r.rate2019*100:5.1f}% -> {r.rate2020*100:5.1f}% "
          f"({r.d_rate:+6.1f}pp)")

# the relationship that matters: wedge against the change in exercise, all firms with both
al = pd.concat([nb[["entity", "wedge", "d_rate", "rep_mult", "flow_mult", "vested2020"]].assign(type="nonbank"),
                d[["entity", "wedge", "d_rate", "rep_mult", "flow_mult", "vested2020"]].assign(type="depository")])
al = al[np.isfinite(al.wedge)]
c = np.corrcoef(np.log(al.wedge), -al.d_rate)[0, 1]
print(f"\n  {len(al)} firms with a computable wedge (4 nonbank, {len(al)-4} depository)")
print(f"  corr(log wedge, fall in exercise rate) = {c:+.3f}")
print(f"  every firm's balance outran its own flow: {bool((al.wedge > 1).all())}")
b = np.polyfit(-al.d_rate, np.log(al.wedge), 1)
print(f"  slope: a 10pp larger fall in exercise -> {np.exp(b[0]*10):.2f}x larger wedge")

# ---------------------------------------------------------------- table P9
L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
     r"& & \multicolumn{3}{c}{Reported balance (\$bn)} & \multicolumn{2}{c}{Options vesting} "
     r"& \multicolumn{2}{c}{Exercise rate}\\",
     r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
     r"Issuer & Item & 2019 & 2020 & 2021 & 2019 & 2020 & 2019 & 2020\\", r"\midrule"]
L[2] = (r"& & \multicolumn{3}{c}{Reported balance (\$bn)} & \multicolumn{2}{c}{Options vesting}"
        r" & \multicolumn{2}{c}{Exercise rate}\\")
L[4] = r"Issuer & Item & 2019 & 2020 & 2021 & 2019 & 2020 & 2019 & 2020\\"
L[0] = r"\begin{tabular}{llrrrrrrr}"


def f(x, k=3):
    return "---" if not np.isfinite(x) else f"{x:.{k}f}"


for _, r in d.iterrows():
    L.append(f"{r.entity} & {r.qty} & {f(r.rep2019)} & {f(r.rep2020)} & {f(r.rep2021)} & "
             f"{r.vested2019:,} & {r.vested2020:,} & "
             f"{r.rate2019*100:.1f}\\% & {r.rate2020*100:.1f}\\% \\\\")
L += [r"\midrule",
      r"\multicolumn{9}{l}{\textit{Growth 2019--2020, and the wedge between them}}\\"]
for _, r in d.iterrows():
    mm = "---" if not np.isfinite(r.rep_mult) else f"{r.rep_mult:.1f}$\\times$"
    wg = "---" if not np.isfinite(r.wedge) else f"{r.wedge:.1f}"
    L.append(f"\\quad {r.entity} & & \\multicolumn{{3}}{{c}}{{{mm}}} & "
             f"\\multicolumn{{2}}{{c}}{{{r.flow_mult:.1f}$\\times$}} & "
             f"\\multicolumn{{2}}{{c}}{{{wg}}}\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(PAP, "tables", "P9.tex"), "w", encoding="utf-8").write("\n".join(L))

fl = d[d.entity == "Flagstar Bank"].iloc[0]
ft = d[d.entity == "Fifth Third Bank"].iloc[0]
jp = d[d.entity == "JPMorgan Chase"].iloc[0]
mac = {
    "FlagPre": f"{fl.rep2019:.3f}", "FlagPeak": f"{fl.rep2020:.2f}",
    "FlagPost": f"{fl.rep2021:.2f}", "FlagMult": f"{fl.rep_mult:.0f}",
    "FlagFlow": f"{fl.flow_mult:.1f}", "FlagWedge": f"{fl.wedge:.1f}",
    "FlagEx": f"{fl.rate2019*100:.1f}", "FlagExPost": f"{fl.rate2020*100:.1f}",
    "FitbVested": "921", "FitbRepurch": "882", "FitbLeft": "39",
    "FitbPct": f"{882/921*100:.0f}", "FitbEx": f"{ft.rate2020*100:.1f}",
    "JpmPre": f"{jp.rep2019:.2f}", "JpmPeak": f"{jp.rep2020:.2f}",
    "JpmPost": f"{jp.rep2021:.2f}", "JpmEx": f"{jp.rate2020*100:.1f}",
    "JpmEighteen": "7.02",
    "AllWedgeN": f"{len(al)}", "AllWedgeCorr": f"{c:+.2f}".replace("-", "$-$"),
    "AllWedgeSlope": f"{np.exp(b[0]*10):.1f}",
}
pth = os.path.join(PAP, "numbers_pool.tex")
have = open(pth, encoding="utf-8").read()
with open(pth, "a", encoding="utf-8") as fh:
    fh.write("\n")
    for k, v in sorted(mac.items()):
        if "\\p" + k + "}" in have:
            continue
        fh.write("\\newcommand{\\p" + k + "}{" + v + "}\n")
d.to_csv(os.path.join(OUT, "depository_panel.csv"), index=False)
al.to_csv(os.path.join(OUT, "wedge_all.csv"), index=False)
print(f"\nwrote P9.tex and {len(mac)} macros")
