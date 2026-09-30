"""numbers.tex macros for journal; tables + figures for journal and JF."""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "savefig.bbox": "tight"})
def save(fig, path): fig.savefig(path); fig.savefig(path[:-4] + ".png", dpi=110)
BLUE, ORANGE, GREEN, GREY = "#1f5fa8", "#c8551f", "#2f8f5b", "#777777"
def tab(path, header, rows, align=None):
    align = align or ("l" + "c" * (len(header) - 1))
    s = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows: s.append("\\midrule" if r == "MID" else " & ".join(str(x) for x in r) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]; open(path, "w", encoding="utf-8").write("\n".join(s))

# ---------------- journal numbers ----------------
R = json.load(open(os.path.join(OUT, "v_numbers.json"))); D = os.path.join(HERE, "tex", "v")
p1 = lambda x: f"{100*x:.1f}"; i0 = lambda x: f"{int(round(x)):,}"
M = {"nFrame": i0(R["n_frame"]), "nLinked": i0(R["n_linked"]), "nGov": i0(R["n_gov"]), "nConv": i0(R["n_conv"]),
     "linkGov": p1(R["link_gov"]), "linkConv": p1(R["link_conv"]), "inqG": p1(R["inq_g"]), "inqC": p1(R["inq_c"]),
     "convG": p1(R["conv_g"]), "convC": p1(R["conv_c"]), "fbG": p1(R["fb_g"]), "fbC": p1(R["fb_c"]), "gapFB": p1(R["gap"]),
     "partInq": p1(R["part_inq"]), "partConv": p1(R["part_conv"]), "shareConv": f"{100*R['part_conv']/R['gap']:.0f}", "shareInq": f"{100*R['part_inq']/R['gap']:.0f}",
     "fbNoInq": f"{100*R['fb_noinq']:.2f}", "daysG": i0(R["days_g"]), "daysC": i0(R["days_c"]),
     "cfConv": p1(R["cf_conv_with_govconv"]), "cfGov": p1(R["cf_gov_with_convconv"]),
     "convGapCtrl": p1(R["conv_gap_ctrl"]), "convGapCtrlSE": p1(R["conv_gap_ctrl_se"]), "inqGapCtrl": p1(R["inq_gap_ctrl"]), "inqGapCtrlSE": p1(R["inq_gap_ctrl_se"]),
     "fbGapCtrl": p1(R["fb_gap_ctrl"]), "fbGapCtrlSE": p1(R["fb_gap_ctrl_se"]),
     "didMain": p1(R["did_Linked loans"][0]), "didMainSE": p1(R["did_Linked loans"][1]), "didAll": p1(R["did_All typed lo"][0]),
     "esLast": p1(R["es_last"][0]), "esLastSE": p1(R["es_last"][1]),
     "nResp": i0(R["n_resp"]), "respSlopeConv": p1(R["resp_slope_conv"]), "respSlopeConvP": f"{R['resp_slope_conv_p']:.2f}", "respIntP": f"{R['resp_slope_int_p']:.2f}",
     "cellsN": i0(R["cells_n"]), "cellsStates": i0(R["cells_states"]), "cellsGapMin": f"{100*R['cells_gap_min']:.0f}", "cellsGapMax": f"{100*R['cells_gap_max']:.0f}",
     "cellsGMin": f"{100*R['cells_g_min']:.0f}", "cellsCMax": f"{100*R['cells_c_max']:.0f}",
     "deniedN": i0(R["denied_n"]), "deniedPerf": p1(R["denied_perf"]), "deniedFC": p1(R["denied_fc"]),
     "didDQ": p1(R["did_dq30"]), "didInbound": p1(R["did_inbound"]), "preGovDQ": p1(R["pre_gov_dq"]), "preConvDQ": p1(R["pre_conv_dq"]),
     "inqNeverG": p1(R["inq_Never_g"]), "inqNeverC": p1(R["inq_Never_c"]), "inqHighG": p1(R["inq_High_g"]), "inqHighC": p1(R["inq_High_c"]),
     "convNeverC": p1(R["conv_Never_c"]), "convLowC": p1(R["conv_Low_c"]), "convHighC": p1(R["conv_High_c"])}
open(os.path.join(D, "numbers.tex"), "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()))
print("journal macros:", len(M))

# ---------------- journal assets ----------------
J = json.load(open(os.path.join(OUT, "v_results.json"))); D = os.path.join(HERE, "tex", "v"); os.makedirs(D, exist_ok=True)
f1 = lambda x: f"{x:.1f}"
coh = [r for r in J["cohort"] if r["loans"] >= 100]
tab(os.path.join(D, "t_cohort.tex"), ["Boarding quarter", "Loans", "Months of history", "Median notes", "Employment", "Health", "Family", "DQ Feb-20", "Forbearance", "Modified", "FC referral"],
    [[r["cohort"], i0(r["loans"]), f1(r["months_hist_med"]), i0(r["notes_med"]), f1(r["emp_pct"]), f1(r["health_pct"]), f1(r["family_pct"]), f1(r["dq_feb20_pct"]), f1(r["fb_pct"]), f1(r["mod_post_pct"]), f1(r["fc_post_pct"])] for r in coh])
dep = J["depth_by_status_withnotes"]; lab = {False: "Current", True: "Delinquent"}
rows = []
for st in [False, True]:
    for r in [x for x in dep if x["dq_feb20"] == st]:
        rows.append([f"{lab[st]}, {r['depth_tercile']}", i0(r["loans"]), f1(r["months_hist_med"]), i0(r["notes_med"]), f1(r["emp_pct"]), f1(r["health_pct"]), f1(r["family_pct"]), f1(r["fb_pct"]), f1(r["mod_post_pct"]), f1(r["fc_post_pct"]), f1(r["performing_pct"])])
    if st is False: rows.append("MID")
tab(os.path.join(D, "t_depth.tex"), ["Feb-20 status, history", "Loans", "Months", "Median notes", "Employment", "Health", "Family", "Forbearance", "Modified", "FC referral", "Performing"], rows)
pre_types = [("Collateral checklist (categories 1--3)", 5345+2583+1482), ("Legal due diligence and review", 5135+2668), ("Property report", 3219), ("Compliance and collateral testing", 1204+711),
             ("Title due diligence", 1062), ("Curative monitoring", 856), ("New file intake", 830), ("Untyped", 4506)]
tot = 32024; rest = tot - sum(v for _, v in pre_types)
tab(os.path.join(D, "t_pretypes.tex"), ["Record type", "Records", "Share (\\%)"], [[a, i0(b), f1(100*b/tot)] for a, b in pre_types] + [["Other", i0(rest), f1(100*rest/tot)], "MID", ["All records dated before boarding", i0(tot), "100.0"]], align="lcc")
q = J["hist_quantiles"]; nq = J["notes_quantiles"]
JM = {"nBoardPre": i0(J["n_boarded_pre"]), "nBoardPost": i0(J["n_boarded_post"]), "histMed": f1(q["0.5"]), "histPten": f1(q["0.1"]), "histPninety": f1(q["0.9"]),
      "notesMed": i0(nq["0.5"]), "notesPten": i0(nq["0.1"]), "notesPninety": i0(nq["0.9"]), "zeroNotes": f1(J["zero_notes_pct"]), "corrHistNotes": f"{J['corr_hist_notes']:.2f}"}
sp = os.path.join(OUT, "preboard_sn.json")
if os.path.exists(sp):
    S = json.load(open(sp)); g = S["gaps"]; n = max(1, S["pre_servicer_notes"])
    JM.update({"preSN": i0(S["pre_servicer_notes"]), "preSNloans": i0(S["pre_sn_loans"]), "preSNweek": f1(100*(g.get("1-3", 0)+g.get("4-7", 0))/n), "preSNmonth": f1(100*(g.get("1-3", 0)+g.get("4-7", 0)+g.get("8-30", 0))/n),
               "preTotal": i0(S["pre_total"]), "postTotal": i0(S["post_total"]), "postSNshare": f1(100*S["post_servicer_notes"]/S["post_total"])})
open(os.path.join(D, "numbers.tex"), "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in JM.items())); print("journal macros:", JM)
fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.5))
x = np.arange(len(coh)); axs[0].bar(x, [r["loans"] for r in coh], color=GREY, width=.6); axs[0].set_xticks(x); axs[0].set_xticklabels([r["cohort"] for r in coh], rotation=60, fontsize=7); axs[0].set_ylabel("Loans boarded"); axs[0].set_title("Boarding is lumpy", fontsize=9)
ax2 = axs[1]; ax2.scatter([r["months_hist_med"] for r in coh], [r["notes_med"] for r in coh], s=[max(8, r["loans"]/40) for r in coh], color=BLUE, alpha=.8)
ax2.set_xlabel("Months of narrative at March 2020"); ax2.set_ylabel("Median servicer notes per loan"); ax2.set_title("Narrative accumulates with tenure", fontsize=9)
fig.tight_layout(); save(fig, os.path.join(D, "f_boarding.pdf")); plt.close(fig)
fig, ax = plt.subplots(figsize=(5.6, 2.5)); w = .26
for k, (key, col, lb) in enumerate([("emp_pct", BLUE, "Employment"), ("health_pct", ORANGE, "Health"), ("family_pct", GREEN, "Family")]):
    ax.bar(x + (k-1)*w, [r[key] for r in coh], w, color=col, label=lb)
ax.set_xticks(x); ax.set_xticklabels([f"{r['cohort']}\n{r['months_hist_med']:.0f} mo" for r in coh], fontsize=7); ax.set_ylabel("% of loans with a marked note"); ax.legend(frameon=False, fontsize=8, ncol=3)
save(fig, os.path.join(D, "f_markers.pdf")); plt.close(fig)

# ---------------- JF assets (verified Dashboard scan, 4M rows) ----------------
D = os.path.join(HERE, "tex", "jf"); os.makedirs(D, exist_ok=True)
PJ = json.load(open(os.path.join(OUT, "jf_pairs.json")))
tab(os.path.join(D, "t_dashboard.tex"), ["Measure", "Value"],
    [["Dashboard loan-date rows scanned", "4,000,000"], ["Rows with both a `high' and a `new' value", "3,165,368"], ["\\quad identical values", "2,956,337 (93.4\\%)"],
     ["\\quad identical valuation dates", "2,937,157 (93.4\\%)"], "MID", ["Type pair: Seller BPO / Seller BPO", "1,464,205"], ["Type pair: BPO / BPO", "1,121,975"], ["Type pair: Servicer BPO / Servicer BPO", "421,674"],
     ["Type pair: BPO / Servicer BPO", "82,496"], ["Type pair: Seller BPO / Servicer BPO", "75,018"],
     ["Net-asset-value field defined as 93.5\\% of servicer BPO (rows)", "290,976"], "MID",
     ["\\textit{Full file, 6,046,788 rows: seller--servicer pairs}", ""], ["Distinct dated pairs", f"{PJ['distinct_pairs']:,}"], ["Loans", f"{PJ['loans']:,}"],
     ["Pairs dated the same day / within 90 days", f"{PJ['same_date']} / {PJ['within_90d']}"],
     ["Days between the two opinions, 10th / 50th / 90th percentile", f"{PJ['gap_p10']:,} / {PJ['gap_p50']:,} / {PJ['gap_p90']:,}"]], align="lc")
print("assets done")
