"""External-validity table: this portfolio vs published national figures (MBA). National numbers are hard-coded with sources in the table note."""
import os, pandas as pd, numpy as np
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out"); TEX = os.path.join(HERE, "tex", "v")
L = pd.read_parquet(os.path.join(OUT, "loan_frame.parquet")); K = L[L.in_covid_file == 1].copy()
def stock(d, date): t = pd.Timestamp(date); return 100 * ((d.fb_start <= t) & (d.fb_end >= t)).mean()
def ever(d): return 100 * (d.fb_agree.notna() & (d.fb_agree <= "2020-09-30")).mean()
g, c = K[K.Gov == 1], K[K.Gov == 0]; fha, va = K[K.loan_type == "FHA"], K[K.loan_type == "VA"]
rows = [
 ["\\textit{30+ days delinquent, pre-pandemic (\\%)}", "", "", ""],
 ["\\quad FHA", f"{100*fha.dq_feb20.mean():.1f}", "8.4", "MBA NDS, 2019Q4"],
 ["\\quad VA", f"{100*va.dq_feb20.mean():.1f}", "3.6", "MBA NDS, 2019Q4"],
 ["\\quad Conventional", f"{100*c.dq_feb20.mean():.1f}", "2.8", "MBA NDS, 2019Q4"],
 "MID",
 ["\\textit{Loans in forbearance, first week of June 2020 (\\%)}", "", "", ""],
 ["\\quad Government-backed / Ginnie Mae", f"{stock(g,'2020-06-07'):.1f}", "--", ""],
 ["\\quad Conventional, this portfolio / all loans, national", f"{stock(c,'2020-06-07'):.1f}", "8.6", "MBA survey, peak, 7 June 2020"],
 "MID",
 ["\\textit{Loans in forbearance, December 2020 (\\%)}", "", "", ""],
 ["\\quad Government-backed / Ginnie Mae", f"{stock(g,'2020-12-13'):.1f}", "7.8", "MBA survey, Dec 2020"],
 ["\\quad Conventional, this portfolio / portfolio and private-label, national", f"{stock(c,'2020-12-13'):.1f}", "8.8", "MBA survey, Dec 2020"],
 "MID",
 ["\\textit{Ever in forbearance by 30 September 2020 (\\%)}", "", "", ""],
 ["\\quad Government-backed", f"{ever(g):.1f}", "--", ""], ["\\quad Conventional", f"{ever(c):.1f}", "--", ""],
 "MID",
 ["Median credit score at origination", f"{K.fico.median():.0f}", "--", ""], ["Median balance (\\$ thousand)", f"{K.bal.median()/1000:.0f}", "--", ""],
]
s = ["\\begin{tabular}{lccl}", "\\toprule", " & This portfolio & National & National source \\\\", "\\midrule"]
for r in rows: s.append("\\midrule" if r == "MID" else " & ".join(r) + " \\\\")
open(os.path.join(TEX, "t0_benchmark.tex"), "w", encoding="utf-8").write("\n".join(s + ["\\bottomrule", "\\end{tabular}"]))
M = {"bmDQGov": f"{100*g.dq_feb20.mean():.0f}", "bmDQConv": f"{100*c.dq_feb20.mean():.0f}", "bmStockGovJune": f"{stock(g,'2020-06-07'):.0f}", "bmStockConvJune": f"{stock(c,'2020-06-07'):.1f}", "bmStockConvDec": f"{stock(c,'2020-12-13'):.1f}", "bmStockGovDec": f"{stock(g,'2020-12-13'):.0f}"}
open(os.path.join(TEX, "numbers_bm.tex"), "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()))
print("\n".join(s[4:])); print(M)
