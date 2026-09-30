"""journal memory macros; JF model figure/table."""
import os, json, numpy as np
from scipy import integrate, stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "savefig.bbox": "tight"})
M = json.load(open(os.path.join(OUT, "memory_analysis.json"))); D = os.path.join(HERE, "tex", "v")
c, g = M["Conventional (discretion)"], M["Government-backed (right; placebo)"]
p1 = lambda x: f"{100*x:.1f}"
mac = {"memN": f"{c['n']:,}", "memGovN": f"{g['n']:,}", "memConv": p1(c["A"]["anyhard"][0]), "memConvSE": p1(c["A"]["anyhard"][1]), "memGov": p1(g["A"]["anyhard"][0]), "memGovSE": p1(g["A"]["anyhard"][1]),
       "memEmp": p1(c["B"]["emp"][0]), "memEmpSE": p1(c["B"]["emp"][1]), "memFam": p1(c["B"]["family"][0]), "memFamSE": p1(c["B"]["family"][1]), "memHealth": p1(c["B"]["health"][0]),
       "memRawNo": p1(M["raw_anyhard"]["mean"]["0"]), "memRawYes": p1(M["raw_anyhard"]["mean"]["1"]), "memRawNoN": f"{M['raw_anyhard']['size']['0']:,}", "memRawYesN": f"{M['raw_anyhard']['size']['1']:,}",
       "memShareConv": p1(c["anyhard"]), "memShareGov": p1(g["anyhard"]), "memNotesConv": f"{c['notes_med']:.0f}", "memNotesGov": f"{g['notes_med']:.0f}", "memHistConv": f"{c['hist_med']:.1f}", "memHistGov": f"{g['hist_med']:.1f}",
       "memNoCtrl": p1(M["rob_no_controls"][0]), "memNoCtrlSE": p1(M["rob_no_controls"][1]), "memCur": p1(M["rob_within_current"][0]), "memCurSE": p1(M["rob_within_current"][1]), "memCurN": f"{M['rob_within_current'][3]:,}",
       "memInt": p1(M["rob_intensity"][0]), "memIntSE": p1(M["rob_intensity"][1]), "altMod": p1(M["alt"]["mod_post"][0]), "altModSE": p1(M["alt"]["mod_post"][1]), "altPerf": p1(M["alt"]["perf"][0]), "altPerfSE": p1(M["alt"]["perf"][1]),
       "altFC": p1(M["alt"]["fcpath"][0]), "altFCSE": p1(M["alt"]["fcpath"][1]), "tercLow": p1(M["raw_notes_terc"][0][1]), "tercMid": p1(M["raw_notes_terc"][1][1]), "tercHigh": p1(M["raw_notes_terc"][2][1]),
       "askNotes": p1(M["ask_Conventional"]["lnotes"][0]), "askNotesSE": p1(M["ask_Conventional"]["lnotes"][1])}
open(os.path.join(D, "numbers.tex"), "a").write("\n" + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items())); print("journal memory macros", len(mac))

# JF: Q_K(rho) with marginal denial .30
D = os.path.join(HERE, "tex", "jf"); a = stats.norm.ppf(.30)
def Q(rho, K):
    if rho == 0: return .30 ** K
    if rho == 1: return .30
    f = lambda u: stats.norm.pdf(u) * stats.norm.cdf((a - np.sqrt(rho) * u) / np.sqrt(1 - rho)) ** K
    return integrate.quad(f, -10, 10, epsabs=1e-10)[0]
rhos = [0, .25, .5, .75, .9, 1]
rows = [f"{r:.3f} & 0.300 & {Q(r,3):.3f} & {Q(r,5):.3f} \\\\" for r in rhos]
open(os.path.join(D, "t_model.tex"), "w").write("\\begin{tabular}{cccc}\n\\toprule\nCorrelation & One lender & Three lenders & Five lenders \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
fig, ax = plt.subplots(figsize=(5.2, 2.7)); xs = np.linspace(0, 1, 41)
ax.plot(xs, [100*Q(r, 3) for r in xs], color="#1f5fa8", lw=1.6, label="Three lenders"); ax.plot(xs, [100*Q(r, 5) for r in xs], color="#c8551f", lw=1.6, label="Five lenders")
ax.set_xlabel("Common-error correlation"); ax.set_ylabel("Probability all lenders deny (%)"); ax.legend(frameon=False, fontsize=8); ax.set_ylim(0, 32)
fig.savefig(os.path.join(D, "f_model.pdf")); print("JF model assets", [round(Q(r, 3), 3) for r in rhos])
