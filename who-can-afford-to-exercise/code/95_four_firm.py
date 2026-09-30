"""Four nonbank issuers, the same measure, from audited filings.

Balances below were read line by line out of the filings (10-K note or S-1 balance
sheet), not scraped: the extraction scripts printed the surrounding table and the
column dates, and these are what the columns say. Sources are in the table note.

The test does not need a depository comparison to work. If exercise behaviour had
not changed, the stock of vested-but-unexercised options would have grown roughly in
proportion to the flow of options vesting, which the loan-level data measures directly
for each of these firms. The wedge between the two is non-exercise.
"""
import pandas as pd, numpy as np, os, json, warnings
from config import OUT, PAPER as PAP
warnings.filterwarnings("ignore")


# ($bn) read from the filing; key = (issuer name in the loan-level data)
REPORTED = {
    "Pennymac Loan Services, LLC": {
        "entity": "PennyMac Financial Services", "src": "10-K, balance sheet",
        2018: 1.103, 2019: 1.047, 2020: 14.625, 2021: 3.026},
    "Nationstar Mortgage, LLC": {
        "entity": "Mr.\\ Cooper Group", "src": "10-K, Notes 10 and 13",
        2018: 0.266, 2019: 0.560, 2020: 6.159, 2021: 1.496},
    "Caliber Home Loans, Inc.": {
        "entity": "Caliber Home Loans", "src": "S-1/A, combined balance sheets",
        2018: 0.418, 2019: 0.195, 2020: 1.918},
    "Amerihome Mortgage Company,LLC": {
        "entity": "AmeriHome", "src": "S-1/A, consolidated balance sheets",
        2018: 0.140, 2019: 0.383, 2020: 2.426},
}
# Caliber and AmeriHome report 30 September 2020, the last month of the loan-level sample.
ASOF = {"Caliber Home Loans, Inc.": "30 Sep", "Amerihome Mortgage Company,LLC": "30 Sep",
        "Pennymac Loan Services, LLC": "31 Dec", "Nationstar Mortgage, LLC": "31 Dec"}

p = pd.read_parquet(os.path.join(OUT, "panel.parquet")).dropna(subset=["buyout"])
p["yr"] = (p.ym // 100).astype(int)
p["mo"] = (p.ym % 100).astype(int)
nm_map = pd.read_csv(os.path.join(OUT, "issuer_id_names.csv"), index_col=0)
names = nm_map["name"].str.upper().str.strip()

rows, mac = [], {}
print(f"{'issuer':<32} {'yr':>4} {'vested':>9} {'exer%':>6} {'unexUPB':>9} {'reported':>9}")
for nm, r in REPORTED.items():
    iid = names[names == nm.upper().strip()].index
    assert len(iid) == 1, (nm, list(iid))
    g = p[p.issuer_id == iid[0]]
    cut = 9 if ASOF[nm] == "30 Sep" else 12          # match the filing's as-of date
    rec = {"issuer": nm, "entity": r["entity"], "src": r["src"], "asof": ASOF[nm]}
    for y in (2019, 2020):
        h = g[(g.yr == y) & ((y < 2020) | (g.mo <= cut))]
        # for 2019 use the same months so the growth rates are like-for-like
        h = g[(g.yr == y) & (g.mo <= cut)]
        rec[f"vested{y}"] = len(h)
        rec[f"rate{y}"] = h.buyout.mean()
        rec[f"unex{y}"] = h.loc[h.buyout == 0, "upb"].sum() / 1e9
        rec[f"rep{y}"] = r.get(y, np.nan)
        print(f"{nm[:32]:<32} {y:>4} {len(h):>9,} {h.buyout.mean()*100:>5.1f}% "
              f"{rec[f'unex{y}']:>9.2f} {rec[f'rep{y}']:>9.2f}")
    rec["flow_mult"] = rec["vested2020"] / rec["vested2019"]
    rec["rep_mult"] = rec["rep2020"] / rec["rep2019"]
    rec["wedge"] = rec["rep_mult"] / rec["flow_mult"]
    rec["d_rate"] = (rec["rate2020"] - rec["rate2019"]) * 100
    rows.append(rec)

d = pd.DataFrame(rows)
print("\n" + "=" * 96)
print("THE WEDGE: reported balance growth vs growth in the flow of options vesting")
print("=" * 96)
for _, r in d.iterrows():
    print(f"  {r.issuer[:34]:<34} balance x{r.rep_mult:5.1f}   vesting flow x{r.flow_mult:4.2f}"
          f"   wedge {r.wedge:4.1f}   exercise {r.d_rate:+6.1f}pp")

# does the wedge line up with the fall in the exercise rate?
c = np.corrcoef(d.wedge, -d.d_rate)[0, 1]
print(f"\n  correlation across the four firms, wedge vs fall in exercise rate: {c:+.3f}")
print(f"  every firm: balance grew faster than flow  -> {bool((d.wedge > 1).all())}")

# ------------------------------------------------------------------ table
body = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"& & \multicolumn{2}{c}{Reported balance (\$bn)} & "
        r"\multicolumn{2}{c}{Options vesting} & \multicolumn{2}{c}{Exercise rate}\\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Issuer & As of & 2019 & 2020 & 2019 & 2020 & 2019 & 2020\\", r"\midrule"]
for _, r in d.iterrows():
    body.append(f"{r.entity} & {r['asof']} & {r.rep2019:.2f} & {r.rep2020:.2f} & "
                f"{r.vested2019:,} & {r.vested2020:,} & "
                f"{r.rate2019*100:.1f}\\% & {r.rate2020*100:.1f}\\% \\\\")
body += [r"\midrule",
         r"\multicolumn{2}{l}{\textit{Growth, 2019 to 2020}} & "
         + " & ".join([r"\multicolumn{2}{c}{}"] * 0) + r"&&&&&\\"]
body[-1] = (r"\multicolumn{2}{l}{\textit{Growth 2019--2020}} & "
            + r"\multicolumn{2}{c}{balance} & \multicolumn{2}{c}{vesting flow} & "
            + r"\multicolumn{2}{c}{ratio}\\")
for _, r in d.iterrows():
    body.append(f"\\quad {r.entity} & & \\multicolumn{{2}}{{c}}{{{r.rep_mult:.1f}$\\times$}} & "
                f"\\multicolumn{{2}}{{c}}{{{r.flow_mult:.1f}$\\times$}} & "
                f"\\multicolumn{{2}}{{c}}{{{r.wedge:.1f}}}\\\\")
body += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(PAP, "tables", "P8.tex"), "w", encoding="utf-8").write("\n".join(body))

mac = {"WedgeN": f"{len(d)}",
       "WedgeMin": f"{d.wedge.min():.1f}", "WedgeMax": f"{d.wedge.max():.1f}",
       "RepMultMin": f"{d.rep_mult.min():.1f}", "RepMultMax": f"{d.rep_mult.max():.0f}",
       "FlowMultMin": f"{d.flow_mult.min():.1f}", "FlowMultMax": f"{d.flow_mult.max():.1f}",
       "WedgeCorr": f"{c:+.2f}".replace("-", "$-$"),
       "CoopPre": f"{REPORTED['Nationstar Mortgage, LLC'][2019]:.2f}",
       "CoopPeak": f"{REPORTED['Nationstar Mortgage, LLC'][2020]:.2f}",
       "CoopPost": f"{REPORTED['Nationstar Mortgage, LLC'][2021]:.2f}",
       "CoopCares": "5.88", "CoopCaresPct": "95",
       "CalPre": "0.20", "CalPeak": "1.92", "CalTwoK": "0.42",
       "AmhPre": "0.38", "AmhPeak": "2.43"}
pth = os.path.join(PAP, "numbers_pool.tex")
have = open(pth, encoding="utf-8").read()
with open(pth, "a", encoding="utf-8") as fh:
    fh.write("\n")
    for k, v in sorted(mac.items()):
        if "\\p" + k + "}" in have:
            continue
        fh.write("\\newcommand{\\p" + k + "}{" + v + "}\n")
d.to_csv(os.path.join(OUT, "four_firm.csv"), index=False)
json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
           for k, v in mac.items()},
          open(os.path.join(OUT, "results_four_firm.json"), "w"), indent=1, default=str)
print(f"\nwrote P8.tex and {len(mac)} macros")
