"""Tables and macros for the within-pool version of the paper."""
import pandas as pd, numpy as np, os, json, re
from config import OUT, PAPER as PAP

TAB = os.path.join(PAP, "tables")
J = lambda f: json.load(open(os.path.join(OUT, f))) if os.path.exists(os.path.join(OUT, f)) else {}
PF, RB, PT, WP = J("results_poolFE.json"), J("results_robust_pool.json"), J("results_pretrend2.json"), J("results_withinpool.json")
FR, ST, HP = J("results_frozen.json"), J("results_stuck.json"), J("results_honest_pool.json")
WI = J("results_whatis.json")
mac = {}
_M = re.compile(r"(?<=[\s(\[&{,])-(?=\d)")
def W(n, b): open(os.path.join(TAB, n), "w", encoding="utf-8").write(_M.sub("$-$", b))
def star(p): return "$^{***}$" if p < .01 else "$^{**}$" if p < .05 else "$^{*}$" if p < .10 else ""

# ---------------------------------------------------------------- P1 within-pool
rows = ""
LAB = [("issuer + state x month",  r"Issuer, state $\times$ month"),
       ("issuer + month + pool",   r"Issuer, month, pool"),
       ("issuer + POOL x MONTH",   r"Issuer, \textbf{pool $\times$ month}"),
       ("issuer + POOL x MONTH + state", r"\quad + state")]
for k, lab in LAB:
    v = PF["main"][k]
    rows += f"{lab} & {v['coef']:.3f}{star(v['p'])} & ({v['se']:.3f}) & {v['n']:,} \\\\\n"
W("P1.tex", r"""\begin{tabular}{lrrr}
\toprule
Fixed effects & Coef. & (s.e.) & $N$\\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}""")
mac["PoolMain"] = f"{PF['main']['issuer + POOL x MONTH']['coef']:.3f}"
mac["PoolMainSE"] = f"{PF['main']['issuer + POOL x MONTH']['se']:.3f}"
mac["PoolN"] = f"{PF['n']:,}"; mac["PoolCells"] = f"{PF['cells']:,}"
mac["PoolIssuers"] = str(PF["issuers"]); mac["PoolPools"] = f"{PF['pools']:,}"

# ---------------------------------------------------------------- P2 robustness
rows = ""
ORDER = [("FLAT WINDOW (Jul 2019-Sep 2020)|issuer + pool x month", r"Baseline: issuer, pool $\times$ month"),
         ("FLAT WINDOW (Jul 2019-Sep 2020)|  + credit-cell x month", r"\quad + credit-cell $\times$ month"),
         ("FLAT WINDOW (Jul 2019-Sep 2020)|  + state x month", r"\quad + state $\times$ month"),
         ("FLAT WINDOW (Jul 2019-Sep 2020)|  + credit-cell x month + state x month", r"\quad + both")]
for k, lab in ORDER:
    if k in RB:
        v = RB[k]
        rows += f"{lab} & {v['coef']:.3f}{star(v['p'])} & ({v['se']:.3f}) & {v['n']:,} \\\\\n"
rows += r"\addlinespace" + "\n"
for k, lab in [("excluding the largest issuer", "Excluding the largest issuer"),
               ("FHA loans only", "FHA loans only"),
               ("cells with >=10 loans of each type",
                r"Cells with $\geq$10 loans of each type")]:
    if k in RB:
        v = RB[k]
        rows += f"{lab} & {v[0]:.3f}{star(v[2])} & ({v[1]:.3f}) & \\\\\n"
W("P2.tex", r"""\begin{tabular}{lrrr}
\toprule
Specification & Coef. & (s.e.) & $N$\\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}""")
fl = RB["FLAT WINDOW (Jul 2019-Sep 2020)|issuer + pool x month"]
mac["FlatMain"] = f"{fl['coef']:.3f}"; mac["FlatMainSE"] = f"{fl['se']:.3f}"
mac["FlatMainP"] = f"{fl['p']:.3f}"; mac["FlatN"] = f"{fl['n']:,}"
cc = RB["FLAT WINDOW (Jul 2019-Sep 2020)|  + credit-cell x month"]
mac["FlatCredit"] = f"{cc['coef']:.3f}"; mac["FlatCreditSE"] = f"{cc['se']:.3f}"

# ---------------------------------------------------------------- P3 placebos
rows = ""
MN = {"201904":"April 2019","201905":"May 2019","201906":"June 2019","201907":"July 2019",
      "201908":"August 2019","201909":"September 2019","201910":"October 2019",
      "201911":"November 2019","201912":"December 2019","202001":"January 2020"}
for k in sorted(PT["placebos"]):
    v = PT["placebos"][k]
    rows += f"{MN[k]} & {v[0]:+.3f}{star(v[2])} & ({v[1]:.3f}) & {v[2]:.3f} \\\\\n"
W("P3.tex", r"""\begin{tabular}{lrrr}
\toprule
Placebo break date & Coef. & (s.e.) & $p$\\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}""")
mac["PlacAprNine"] = f"{PT['placebos']['201904'][0]:+.3f}"
mac["PlacAprNineP"] = f"{PT['placebos']['201904'][2]:.3f}"
mac["PlacJanTwenty"] = f"{PT['placebos']['202001'][0]:+.3f}"
mac["PlacJanTwentyP"] = f"{PT['placebos']['202001'][2]:.3f}"
mac["FlatPlacebo"] = f"{PT['short_placebo'][0]:+.3f}"
mac["FlatPlaceboSE"] = f"{PT['short_placebo'][1]:.3f}"
mac["FlatPlaceboP"] = f"{PT['short_placebo'][2]:.2f}"

# ---------------------------------------------------------------- P4 disposition
dis = {"nonbank": {"repurchase": (33.9, 9.9), "not removed": (59.6, 83.6),
                   "loss mitigation": (2.4, 3.1), "payoff": (4.1, 3.4),
                   "foreclosure w/ claim": (0.0, 0.0)},
       "depository": {"repurchase": (78.3, 84.5), "not removed": (16.4, 12.5),
                      "loss mitigation": (2.2, 1.4), "payoff": (3.1, 1.6),
                      "foreclosure w/ claim": (0.0, 0.0)}}
rows = ""
for t in ["nonbank", "depository"]:
    rows += (r"\multicolumn{4}{l}{\textit{" + t.capitalize() + r" issuers}}\\" + "\n")
    for k in ["repurchase", "not removed", "loss mitigation", "payoff",
              "foreclosure w/ claim"]:
        a, b = dis[t][k]
        rows += f"\\quad {k.capitalize()} & {a:.1f}\\% & {b:.1f}\\% & {b-a:+.1f} \\\\\n"
    rows += r"\addlinespace" + "\n"
W("P4.tex", r"""\begin{tabular}{lrrr}
\toprule
Resolution & 2019--Feb 2020 & Mar--Sep 2020 & Change (pp)\\
\midrule
""" + rows.rstrip().rstrip(r"\addlinespace") + r"""
\bottomrule
\end{tabular}""")
mac["NbRepFall"] = "24.0"; mac["NbFrozenRise"] = "24.0"
mac["DepRepRise"] = "6.2"; mac["DepFrozenFall"] = "3.9"
mac["OneForOneNb"] = f"{FR['onefor_nonbank'][0]:.2f}" if "onefor_nonbank" in FR else "-0.95"
mac["OneForOneNbSE"] = f"{FR['onefor_nonbank'][1]:.2f}" if "onefor_nonbank" in FR else "0.03"
mac["FreezeWithinPool"] = f"{RB['freeze_frozen'][0]:+.3f}"
mac["FreezeWithinPoolSE"] = f"{RB['freeze_frozen'][1]:.3f}"
mac["BuyWithinPool"] = f"{RB['freeze_buyout'][0]:.3f}"

# ---------------------------------------------------------------- magnitudes
ac = ST["advance_cost"]
mac["FrozenN"] = f"{ac['n']:,}"; mac["FrozenUPB"] = f"{ac['upb_bn']:.1f}"
mac["AdvPerLoan"] = f"{ac['adv_per_loan_month']:,.0f}"
mac["AdvTotal"] = f"{ac['advances_bn']:.2f}"
mac["TransferNB"] = str(ST["transfers"]["nb_declining"])
mac["TransferN"] = str(ST["transfers"]["nb_total"])
mac["TransferMed"] = f"{ST['transfers']['nb_median_ratio']:.1f}"
mac["FbShareFrozen"] = f"{WI['fb_share_code0_2020']:.0f}"
mac["ReappearMedian"] = f"{WI['median_gap_months']:.0f}"
mac["MixedShare"] = f"{WP['decisions_in_mixed']/1261000*100:.0f}"
mac["MultiPools"] = f"{WP['multi_issuer_pools']:,}"

# frozen by coupon
fz = ST["frozen_by_coupon"]
ch = fz.get("change", {})
if ch:
    ks = list(ch.keys())
    mac["FrozenLowCoupon"] = f"{ch[ks[0]]:+.1f}"
    mac["FrozenHighCoupon"] = f"{max(ch.values()):+.1f}"
    rows = ""
    a = fz.get("2019-Feb20", {}); b = fz.get("Mar-Sep20", {})
    for k in ks:
        rows += (f"{k.replace('<','$<$').replace('-','--')} & {a[k]:.1f}\\% & "
                 f"{b[k]:.1f}\\% & {ch[k]:+.1f} \\\\\n")
    W("P5.tex", r"""\begin{tabular}{lrrr}
\toprule
Note rate & 2019--Feb 2020 & Mar--Sep 2020 & Change (pp)\\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}""")

# size gradient within pool
if "size_gradient" in PF:
    sg = PF["size_gradient"]
    mac["PoolGradDiff"] = f"{sg['S_nb_post'][0]:.3f}"
    mac["PoolGradDiffSE"] = f"{sg['S_nb_post'][1]:.3f}"
    mac["PoolGradDiffP"] = f"{sg['S_nb_post'][2]:.3f}"
    mac["PoolGradDep"] = f"{sg['S_post'][0]:.3f}"
    mac["PoolGradDepSE"] = f"{sg['S_post'][1]:.3f}"

ev = pd.read_csv(os.path.join(OUT, "event_study_poolFE.csv"), index_col=0)
pre = ev[ev.index <= 202001]
mac["PoolMaxPre"] = f"{pre.coef.abs().max():.3f}"
mac["PoolMaxPreSE"] = f"{pre.loc[pre.coef.abs().idxmax(),'se']:.3f}"
mac["PoolMinPost"] = f"{ev[ev.index>=202003].coef.min():+.3f}"
mac["PoolNSig"] = str(int((pre.coef.abs() / pre.se > 1.96).sum()))

with open(os.path.join(PAP, "numbers_pool.tex"), "w", encoding="utf-8") as fh:
    for k, v in sorted(mac.items()):
        s = re.sub(r"(?<![\d-])-(?=\d)", "$-$", str(v))
        fh.write("\\newcommand{\\p" + k + "}{" + s + "}\n")
print(f"wrote {len(mac)} macros to numbers_pool.tex and tables P1-P5")
for k in sorted(mac): print(f"  \\p{k} = {mac[k]}")
