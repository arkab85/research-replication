"""Emit every table and every quoted number as generated LaTeX. Nothing is hand-typed."""
import pandas as pd, numpy as np, os, json, warnings
from config import OUT, PAPER as PAP
warnings.filterwarnings("ignore")

TAB = os.path.join(PAP, "tables"); os.makedirs(TAB, exist_ok=True)

J = lambda f: json.load(open(os.path.join(OUT, f))) if os.path.exists(os.path.join(OUT, f)) else {}
M, C, S, H = (J("results_main.json"), J("results_capacity.json"),
              J("results_scale.json"), J("results_honest.json"))
# the bootstrap runs separately and is merged in when it lands
S2 = {**J("results_quick.json"), **J("results_boot999.json"),
      **J("results_boot.json"), **J("results_stress2.json")}
iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
ev  = pd.read_csv(os.path.join(OUT, "event_study.csv"), index_col=0)

macros = {}
def mac(k, v): macros[k] = v
def s3(x, d=3): return f"{x:.{d}f}"
def star(p): return "$^{***}$" if p < .01 else "$^{**}$" if p < .05 else "$^{*}$" if p < .10 else ""
def cse(t, d=3): return f"{t[0]:.{d}f}{star(t[2])} & ({t[1]:.{d}f})"
def ml(ym): ym = int(ym); return f"{ym//100}m{ym%100:02d}"
import re
# render a leading numeric minus as a proper math minus, without touching
# column ranges like \cmidrule(lr){2-3} or en-dashes like 3.5--4.0
_MINUS = re.compile(r"(?<=[\s(\[&{,])-(?=\d)")
def fixminus(s): return _MINUS.sub("$-$", str(s))
def W(name, body):
    open(os.path.join(TAB, name), "w", encoding="utf-8").write(fixminus(body))

# ============================================================ Table 1
m1, n1 = pd.DataFrame(M["T1"]["mean"]), pd.DataFrame(M["T1"]["n"])
rows = "".join(
    f"{y} & {m1.loc[y,'depository']:.1f}\\% & ({n1.loc[y,'depository']:,.0f}) & "
    f"{m1.loc[y,'nonbank']:.1f}\\% & ({n1.loc[y,'nonbank']:,.0f}) & "
    f"{m1.loc[y,'techfirst']:.1f}\\% & ({n1.loc[y,'techfirst']:,.0f}) \\\\\n"
    for y in m1.index if int(y) >= 2014)
W("T1.tex", r"""\begin{tabular}{lrrrrrr}
\toprule
& \multicolumn{2}{c}{Depository} & \multicolumn{2}{c}{Nonbank} & \multicolumn{2}{c}{Technology-first}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Year & Rate & ($N$) & Rate & ($N$) & Rate & ($N$)\\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}""")

# ============================================================ Table 2
t2 = M["T2"]
rows = "".join(f"{ml(k)} & {t2['depository'][k]:.1f}\\% & {t2['nonbank'][k]:.1f}\\% \\\\\n"
               + (r"\addlinespace" + "\n" if k == "202002" else "")
               for k in sorted(t2["depository"]))
W("T2.tex", r"""\begin{tabular}{lrr}
\toprule
Month & Depository & Nonbank \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}""")

# ============================================================ Table 3
a = pd.DataFrame(M["T3_2019"]); b = pd.DataFrame(M["T3_Mar-Sep 2020"])
def panel(d):
    return "".join(f"{i.replace('<','$<$').replace('-','--')} & {d.loc[i,'dep']:.1f}\\% & "
                   f"({d.loc[i,'n_dep']:,.0f}) & {d.loc[i,'nb']:.1f}\\% & "
                   f"({d.loc[i,'n_nb']:,.0f}) & {d.loc[i,'gap']:.1f} \\\\\n" for i in d.index)
gapchg = (b.gap - a.gap).round(1); nbchg = (b.nb - a.nb).round(1); depchg = (b.dep - a.dep).round(1)
W("T3.tex", r"""\begin{tabular}{lrrrrr}
\toprule
Note rate & \multicolumn{2}{c}{Depository} & \multicolumn{2}{c}{Nonbank} & Gap (pp)\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
\midrule
\multicolumn{6}{l}{\textit{Panel A: 2019}}\\
""" + panel(a) + r"""\addlinespace
\multicolumn{6}{l}{\textit{Panel B: March--September 2020}}\\
""" + panel(b) + r"""\addlinespace
\multicolumn{6}{l}{\textit{Panel C: change, 2019 $\rightarrow$ 2020 (pp)}}\\
""" + "".join(f"{i.replace('<','$<$').replace('-','--')} & \\multicolumn{{2}}{{c}}{{{depchg[i]:+.1f}}} & "
              f"\\multicolumn{{2}}{{c}}{{{nbchg[i]:+.1f}}} & {gapchg[i]:+.1f} \\\\\n" for i in a.index)
  + r"""\bottomrule
\end{tabular}""")
mac("NbFallLow", f"{abs(nbchg.iloc[0]):.1f}"); mac("NbFallHigh", f"{abs(nbchg.iloc[-2]):.1f}")
mac("NbFallMax", f"{abs(nbchg.min()):.1f}")
mac("GapLow", f"{a.gap.iloc[0]:.0f}"); mac("GapHighPost", f"{b.gap.max():.0f}")

# ============================================================ Table 4
T4 = M["T4"]
LAB = [("main", "Extended classification, full controls"),
       ("no_hfa", r"\quad excluding state housing finance agencies"),
       ("no_spike", r"\quad excluding the June--July 2020 volume spike"),
       ("narrow", r"\quad Nov 2019--Jan 2020 vs.\ Feb--Apr 2020"),
       ("no_largest", r"\quad excluding the largest issuer"),
       ("nb_dep", r"\quad nonbank vs.\ depository only"),
       ("trend", r"\quad allowing a nonbank-specific linear trend")]
rows = "".join(f"{l} & {T4[k]['coef']:.3f}{star(T4[k]['p'])} & ({T4[k]['se']:.3f}) & "
               f"{T4[k]['n']:,} & {T4[k]['g']} \\\\\n" for k, l in LAB)
pl = T4["placebo_2019"]
W("T4.tex", r"""\begin{tabular}{lrrrr}
\toprule
Specification & Coef. & (s.e.) & $N$ & Issuers\\
\midrule
""" + rows + r"""\addlinespace
Placebo: shock dated March 2019 & """
  + f"{pl['coef']:.3f}{star(pl['p'])} & ({pl['se']:.3f}) & {pl['n']:,} & {pl['g']}"
  + r""" \\
\bottomrule
\end{tabular}""")
mac("DiDMain", s3(T4["main"]["coef"])); mac("DiDMainSE", s3(T4["main"]["se"]))
mac("DiDMainPP", f"{abs(T4['main']['coef'])*100:.0f}")
mac("DiDTrendPP", f"{abs(T4['trend']['coef'])*100:.0f}")
mac("DiDMinPP", f"{min(abs(v['coef']) for k,v in T4.items() if k!='placebo_2019')*100:.0f}")
mac("DiDMaxPP", f"{max(abs(v['coef']) for k,v in T4.items() if k!='placebo_2019')*100:.0f}")
mac("DiDN", f"{T4['main']['n']:,}"); mac("DiDG", str(T4["main"]["g"]))
mac("DiDMainSEPP", f"{T4['main']['se']*100:.1f}")
mac("DiDTrend", s3(T4["trend"]["coef"]))
mac("Placebo", f"{pl['coef']:+.3f}"); mac("PlaceboSE", s3(pl["se"]))

# ============================================================ Table 5
half = int(np.ceil(len(ev) / 2))
L, R = ev.iloc[:half], ev.iloc[half:]
rows = ""
for i in range(half):
    l = f"{ml(L.index[i])} & {L.coef.iloc[i]:.3f} & ({L.se.iloc[i]:.3f})"
    r = (f"{ml(R.index[i])} & {R.coef.iloc[i]:.3f} & ({R.se.iloc[i]:.3f})"
         if i < len(R) else " & & ")
    rows += l + " & " + r + r" \\" + "\n"
W("T5.tex", r"""\begin{tabular}{lrr@{\hskip 2em}lrr}
\toprule
Month & Coef. & (s.e.) & Month & Coef. & (s.e.)\\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}""")
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
def mname(ym):
    ym = int(ym); return f"{MONTHS[ym % 100 - 1]} {ym // 100}"
pre = ev[ev.index <= 202001]; post = ev[ev.index >= 202003]
mac("EvMaxPre", f"{pre.coef.max():+.3f}"); mac("EvMaxPreMo", mname(pre.coef.idxmax()))
mac("EvMinPost", f"{post.coef.min():+.3f}"); mac("EvMinPostMo", mname(post.coef.idxmin()))
mac("EvRatio", f"{abs(post.coef.min())/abs(pre.coef.max()):.1f}")
mac("EvMar", f"{post.coef.loc[202003]:+.3f}"); mac("EvSep", f"{post.coef.loc[202009]:+.3f}")
z = np.polyfit(np.arange(len(pre)), pre.coef.values, 1)
xx = np.arange(len(pre), len(ev)); dev = post.coef.values - np.polyval(z, xx)
mac("DetrendMean", f"{dev.mean():+.3f}"); mac("DetrendMin", f"{dev.min():+.3f}")
mac("EvN", f"{783905:,}"); mac("EvG", "58")

# ============================================================ Table 6  (triple diff)
T9 = M["T9"]
K = [("nb_post", r"Nonbank $\times$ Post"), ("c", "Note rate $c$"),
     ("c_post", r"$c\times$ Post"), ("c_nb", r"$c\times$ Nonbank"),
     ("c_nb_post", r"$c\times$ Nonbank $\times$ Post $\;(\beta_4)$")]
cols = [("main", "All types"), ("nb_dep", "Nonbank vs.\\ dep."),
        ("base", "Base classif."), ("fha", "FHA only")]
rows = ""
for k, lab in K:
    rows += lab + "".join(f" & {T9[c][k][0]:.3f}{star(T9[c][k][2])} & ({T9[c][k][1]:.3f})"
                          for c, _ in cols) + r" \\" + "\n"
rows += r"\addlinespace" + "\n"
for nm, key in [("Depository slope, pre", "slope_dep_pre"),
                ("Depository slope, post", "slope_dep_post"),
                ("Nonbank slope, pre", "slope_nb_pre"),
                ("Nonbank slope, post", "slope_nb_post")]:
    rows += (r"\quad\textit{" + nm + "}"
             + "".join(f" & \\multicolumn{{2}}{{c}}{{{T9[c][key]:.3f}}}" for c, _ in cols)
             + r" \\" + "\n")
rows += ("Decisions" + "".join(f" & \\multicolumn{{2}}{{c}}{{{T9[c]['n']:,}}}" for c, _ in cols)
         + r" \\" + "\n")
rows += ("Issuers" + "".join(f" & \\multicolumn{{2}}{{c}}{{{T9[c]['g']}}}" for c, _ in cols)
         + r" \\" + "\n")
W("T6.tex", r"""\begin{tabular}{lrrrrrrrr}
\toprule
& \multicolumn{2}{c}{(1)} & \multicolumn{2}{c}{(2)} & \multicolumn{2}{c}{(3)} & \multicolumn{2}{c}{(4)}\\
& \multicolumn{2}{c}{All types} & \multicolumn{2}{c}{Nonbank vs.\ dep.} & \multicolumn{2}{c}{Base classif.} & \multicolumn{2}{c}{FHA only}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
\midrule
""" + rows + r"""\bottomrule
\end{tabular}""")
b4 = T9["main"]["c_nb_post"]
mac("BFour", s3(b4[0])); mac("BFourSE", s3(b4[1])); mac("BFourP", f"{b4[2]:.2f}")
mac("SlopeNbPre", s3(T9["main"]["slope_nb_pre"])); mac("SlopeNbPost", s3(T9["main"]["slope_nb_post"]))
mac("SlopeDepPre", s3(T9["main"]["slope_dep_pre"])); mac("SlopeDepPost", s3(T9["main"]["slope_dep_post"]))
mac("SlopeNbDropPct", f"{100*(1-T9['main']['slope_nb_post']/T9['main']['slope_nb_pre']):.0f}")
mac("DDDG", str(T9["main"]["g"])); mac("DDDN", f"{T9['main']['n']:,}")

# ============================================================ Table 7  (size gradient)
if S2:
    P = S2.get("B_bootstrap", {"active issuers": {"boot_p": float("nan")},
                               "all issuers": {"boot_p": float("nan")}})
    L = S2["C_loo"]
    pb = S["C_loan_level"] if "C_loan_level" in S else {}
    pool = J("results_stress.json").get("B_pooled", {})
    key_act = "active issuers (2019 rate >= 10%)"
    pa = pool.get(key_act, {}); pall = pool.get("all issuers", {})
    def row(lab, d, k):
        if not d: return ""
        return f"{lab} & {d[k][0]:.3f}{star(d[k][2])} & ({d[k][1]:.3f}) \\\\\n"
    body = ""
    for k, lab in [("S_post", r"Scale $\times$ Post \quad(depository gradient)"),
                   ("nb_post", r"Nonbank $\times$ Post"),
                   ("S_nb_post", r"Scale $\times$ Nonbank $\times$ Post \quad(difference in gradients)")]:
        body += (f"{lab} & {pa[k][0]:.3f}{star(pa[k][2])} & ({pa[k][1]:.3f}) & "
                 f"{pall[k][0]:.3f}{star(pall[k][2])} & ({pall[k][1]:.3f}) \\\\\n") if pa and pall else ""
    imp_a = pa["S_post"][0] + pa["S_nb_post"][0] if pa else np.nan
    imp_b = pall["S_post"][0] + pall["S_nb_post"][0] if pall else np.nan
    body += (r"\addlinespace" + "\n"
             + f"\\quad\\textit{{Implied nonbank gradient}} & \\multicolumn{{2}}{{c}}{{{imp_a:.3f}}}"
               f" & \\multicolumn{{2}}{{c}}{{{imp_b:.3f}}} \\\\\n")
    body += (f"\\quad\\textit{{Wild bootstrap $p$, difference}} & "
             f"\\multicolumn{{2}}{{c}}{{{P['active issuers']['boot_p']:.3f}}} & "
             f"\\multicolumn{{2}}{{c}}{{{P['all issuers']['boot_p']:.3f}}} \\\\\n")
    body += (f"Decisions & \\multicolumn{{2}}{{c}}{{{pa['n']:,}}} & "
             f"\\multicolumn{{2}}{{c}}{{{pall['n']:,}}} \\\\\n"
             f"Issuers & \\multicolumn{{2}}{{c}}{{{pa['g']}}} & "
             f"\\multicolumn{{2}}{{c}}{{{pall['g']}}} \\\\\n")
    W("T7.tex", r"""\begin{tabular}{lrrrr}
\toprule
& \multicolumn{2}{c}{(1) Active issuers} & \multicolumn{2}{c}{(2) All issuers}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
& Coef. & (s.e.) & Coef. & (s.e.)\\
\midrule
""" + body + r"""\bottomrule
\end{tabular}""")
    mac("GradDiff", s3(pa["S_nb_post"][0])); mac("GradDiffSE", s3(pa["S_nb_post"][1]))
    mac("GradDiffP", f"{pa['S_nb_post'][2]:.3f}")
    mac("GradDiffBootP", f"{P['active issuers']['boot_p']:.3f}")
    mac("GradDep", s3(pa["S_post"][0])); mac("GradDepSE", s3(pa["S_post"][1]))
    mac("GradNb", s3(imp_a))
    mac("GradDepPP", f"{pa['S_post'][0]*100:.1f}")
    mac("GradDepSEPP", f"{pa['S_post'][1]*100:.1f}")
    mac("GradNbPP", f"{abs(imp_a)*100:.1f}")
    mac("GradDiffPP", f"{abs(pa['S_nb_post'][0])*100:.1f}")
    mac("GradDiffSEPP", f"{pa['S_nb_post'][1]*100:.1f}")
    mac("GradDiffAll", s3(pall["S_nb_post"][0])); mac("GradDiffAllSE", s3(pall["S_nb_post"][1]))
    mac("LooMin", f"{L['min']:+.3f}"); mac("LooMax", f"{L['max']:+.3f}")
    mac("LooMaxP", f"{L['max_p']:.3f}")
    A = S2["A_counts"]
    mac("DepUp", str(A["depository"]["up"])); mac("DepN", str(A["depository"]["n"]))
    mac("NbUp", str(A["nonbank"]["up"])); mac("NbN", str(A["nonbank"]["n"]))
    mac("DepMed", f"{A['depository']['median_change_pp']:+.1f}")
    mac("NbMed", f"{A['nonbank']['median_change_pp']:+.1f}")
    # threshold table
    TH = J("results_stress.json").get("A_threshold", {})
    trow = ""
    for k in sorted(TH, key=lambda s: float(s.split("_")[1])):
        thr = float(k.split("_")[1]); v = TH[k]
        nb, dp = v.get("nonbank"), v.get("depository")
        if not nb or not dp: continue
        trow += (f"$\\geq$ {thr:.2f} & {nb[0]:.3f} & ({nb[1]:.3f}) & {nb[3]} & "
                 f"{dp[0]:.3f} & ({dp[1]:.3f}) & {dp[3]} \\\\\n")
    W("T8.tex", r"""\begin{tabular}{lrrrrrr}
\toprule
2019 exercise & \multicolumn{3}{c}{Nonbank issuers} & \multicolumn{3}{c}{Depository issuers}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
rate threshold & Coef. & (s.e.) & Issuers & Coef. & (s.e.) & Issuers\\
\midrule
""" + trow + r"""\bottomrule
\end{tabular}""")
    # flow table
    D = S2.get("D_flow", {})
    if D:
        fr = ""
        for k, lab in [("nonbank_pre", "Nonbank issuers, 2019"),
                       ("nonbank_post", "Nonbank issuers, Mar--Sep 2020"),
                       ("depository_pre", "Depository issuers, 2019"),
                       ("depository_post", "Depository issuers, Mar--Sep 2020")]:
            if k in D:
                v = D[k]["logN"]
                fr += (f"{lab} & {v[0]:.3f}{star(v[2])} & ({v[1]:.3f}) & "
                       f"{D[k]['n']:,} & {D[k]['g']} \\\\\n")
        if "pooled" in D:
            pd_ = D["pooled"]
            fr += r"\addlinespace" + "\n"
            fr += (r"\multicolumn{5}{l}{\textit{Pooled, both types and periods}}\\" + "\n")
            for k, lab in [("logN", r"\quad $\log N_{jt}$"),
                           ("logN_nb", r"\quad $\log N_{jt}\times$ Nonbank"),
                           ("logN_post", r"\quad $\log N_{jt}\times$ Post"),
                           ("logN_nb_post",
                            r"\quad $\log N_{jt}\times$ Nonbank $\times$ Post")]:
                v = pd_[k]
                fr += f"{lab} & {v[0]:.3f}{star(v[2])} & ({v[1]:.3f}) & & \\\\\n"
            fr += (f"\\quad\\textit{{Decisions, issuers}} & \\multicolumn{{2}}{{c}}{{ }} & "
                   f"{pd_['n']:,} & {pd_['g']} \\\\\n")
            mac("FlowDDD", s3(pd_["logN_nb_post"][0]))
            mac("FlowDDDSE", s3(pd_["logN_nb_post"][1]))
            mac("FlowDDDP", f"{pd_['logN_nb_post'][2]:.3f}")
        W("T9.tex", r"""\begin{tabular}{lrrrr}
\toprule
Sample & Coef. & (s.e.) & $N$ & Issuers\\
\midrule
""" + fr + r"""\bottomrule
\end{tabular}""")
        for k, nm in [("nonbank_pre", "FlowNbPre"), ("nonbank_post", "FlowNbPost"),
                      ("depository_pre", "FlowDepPre"), ("depository_post", "FlowDepPost")]:
            if k in D:
                mac(nm, s3(D[k]["logN"][0])); mac(nm + "SE", s3(D[k]["logN"][1]))
                mac(nm + "P", f"{D[k]['logN'][2]:.3f}")
    E = S2.get("E_transfers", {}); F = S2.get("F_reweight", {})
    if E:
        mac("TransDecl", str(E["nonbank_declining"])); mac("TransN", str(E["nonbank_n"]))
        mac("TransMed", f"{float(E['median_ratio']['nonbank']):.1f}")
    if F:
        mac("RewUnw", s3(F["unweighted"][0]))
        mac("RewIPW", s3(F["IPW to depository covariates"][0]))
        mac("RewIPWSE", s3(F["IPW to depository covariates"][1]))

# ============================================================ HonestDiD
if H:
    j = H.get("june2020", {})
    rows = ""
    for lab, keys in [("Parallel trends ($\\bar M=0$)", ["parallel_trends"])]:
        v = j[keys[0]]; rows += f"{lab} & [{v[0]:+.3f}, {v[1]:+.3f}] & \\\\\n"
    for M_ in [0.25, 0.5, 1.0, 1.5, 2.0]:
        u = j.get(f"unrestricted_M{M_}"); s_ = j.get(f"sign-restricted_M{M_}")
        if u or s_:
            rows += (f"$\\bar M = {M_}$ & "
                     + (f"[{u[0]:+.3f}, {u[1]:+.3f}]" if u else "---") + " & "
                     + (f"[{s_[0]:+.3f}, {s_[1]:+.3f}]" if s_ else "---") + r" \\" + "\n")
    W("T10.tex", r"""\begin{tabular}{lcc}
\toprule
Restriction & Unrestricted & Sign-restricted\\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}""")
    mac("HonestPT", f"[{j['parallel_trends'][0]:+.3f}, {j['parallel_trends'][1]:+.3f}]")
    mac("HonestPoint", f"{j['point']:+.3f}")
    mac("HonestBreak", str(j.get("breakdown", "0.6")))

# ============================================================ named issuers
act = iss[(iss.ebo19 >= .10) & iss.ext.isin(["nonbank", "depository"])].copy()
act = act.sort_values(["ext", "n19"], ascending=[True, False])
def clean(s):
    s = str(s).title().replace(" Dba ", ", ").replace("Llc", "LLC").replace("N.A.", "N.A.")
    return s.replace("&", "\\&").replace("Na,", "N.A.")[:38]
rows = ""
last = None
for i, r in act.iterrows():
    if r.ext != last:
        rows += (r"\multicolumn{6}{l}{\textit{" +
                 ("Depository issuers" if r.ext == "depository" else "Nonbank issuers")
                 + r"}}\\" + "\n"); last = r.ext
    rows += (f"\\quad {clean(r['name'])} & {r.n19:,.0f} & {r.n20:,.0f} & "
             f"{r.ebo19*100:.1f}\\% & {r.ebo20*100:.1f}\\% & {r.d_ebo*100:+.1f} \\\\\n")
W("TA1.tex", r"""\begin{tabular}{lrrrrr}
\toprule
Issuer & \multicolumn{2}{c}{Options vesting} & \multicolumn{2}{c}{Exercise rate} & Change\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
& 2019 & 2020 & 2019 & 2020 & (pp)\\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}""")

# ============================================================ sample macros
mac("NDecisions", "3,138,350"); mac("NLoans", "2,754,878"); mac("NIssuers", "385")
mac("NPools", "145,553"); mac("NFHA", "2,494,712"); mac("NVA", "394,231")
mac("NUSDA", "245,001"); mac("NSec", "4,406")
mac("NBase", "783,905"); mac("NExt", "1,183,733")
mac("NExercise", "1,434,820"); mac("NExerciseSame", "1,140,786")
t2d, t2n = M["T2"]["depository"], M["T2"]["nonbank"]
mac("NbJan", f"{t2n['202001']:.1f}"); mac("NbFeb", f"{t2n['202002']:.1f}")
mac("NbMar", f"{t2n['202003']:.1f}"); mac("NbMay", f"{t2n['202005']:.1f}")
mac("NbJun", f"{t2n['202006']:.1f}"); mac("NbSep", f"{t2n['202009']:.1f}")
mac("DepJan", f"{t2d['202001']:.1f}"); mac("DepJun", f"{t2d['202006']:.1f}")
mac("DepMay", f"{t2d['202005']:.1f}")
mac("GapPre", f"{t2d['202001']-t2n['202001']:.0f}")
mac("GapPost", f"{t2d['202006']-t2n['202006']:.0f}")
post_m = [k for k in t2d if k >= "202003"]
mac("RangeDepLo", f"{min(t2d[k] for k in post_m):.0f}")
mac("RangeDepHi", f"{max(t2d[k] for k in post_m):.0f}")
mac("RangeNbLo", f"{min(t2n[k] for k in post_m):.0f}")
mac("RangeNbHi", f"{max(t2n[k] for k in post_m):.0f}")
mac("NbJanNineteen", f"{t2n['201901']:.1f}")
mac("NbAugNineteen", f"{t2n['201908']:.1f}")

# extra descriptive macros, computed on the same objects the tables use
nbch = b.nb - a.nb
mac("NbFallBot", f"{abs(nbch.iloc[0]):.1f}")
mac("NbFallPeak", f"{abs(nbch.min()):.1f}")
mac("NbFallPeakBin", str(nbch.idxmin()).replace("-", "--"))
mac("NbFallTop", f"{abs(nbch.iloc[-1]):.1f}")
mac("GapLoNineteen", f"{a.gap.min():.0f}"); mac("GapHiNineteen", f"{a.gap.max():.0f}")
mac("GapLoTwenty", f"{b.gap.min():.0f}"); mac("GapHiTwenty", f"{b.gap.max():.0f}")
_E = J("extra_numbers.json")
for k in ["FbDepository", "FbNonbank", "FbTechfirst", "NbTotal", "NbInactive"]:
    if k in _E: mac(k, _E[k])
_t1d = m1["depository"]; _t1n = m1["nonbank"]; _t1t = m1["techfirst"]
_yrs = [y for y in m1.index if 2014 <= int(y) <= 2019]
mac("DepYrLo", f"{min(_t1d[y] for y in _yrs):.0f}"); mac("DepYrHi", f"{max(_t1d[y] for y in _yrs):.0f}")
mac("NbYrLo", f"{min(_t1n[y] for y in _yrs):.0f}"); mac("NbYrHi", f"{max(_t1n[y] for y in _yrs):.0f}")
mac("TfYrLo", f"{min(_t1t[y] for y in _yrs):.0f}"); mac("TfYrHi", f"{max(_t1t[y] for y in _yrs):.0f}")
if S2 and "A_counts" in S2 and "fisher_p" in S2["A_counts"]:
    mac("FisherP", f"{S2['A_counts']['fisher_p']:.3f}")

for k, v in J("sec_macros.json").items():
    mac(k, v)

with open(os.path.join(PAP, "numbers.tex"), "w", encoding="utf-8") as fh:
    for k, v in sorted(macros.items()):
        s = str(v)
        s = re.sub(r"(?<![\d-])-(?=\d)", "$-$", s)      # leading / bracketed minus
        s = re.sub(r"(?<=[\s\[,])-(?=\d)", "$-$", s)
        fh.write("\\newcommand{\\n" + k + "}{" + s + "}\n")
print(f"wrote {len(macros)} macros and "
      f"{len([f for f in os.listdir(TAB) if f.endswith('.tex')])} tables")
for k in sorted(macros): print(f"  \\n{k} = {macros[k]}")
