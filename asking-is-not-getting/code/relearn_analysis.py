"""journal: how long does the receiving servicer take to re-learn a borrower's circumstances after boarding?
Monthly hazard of the FIRST recorded hardship marker, by months since boarding. Front-loaded hazard = rediscovery."""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out"); TJ = os.path.join(HERE, "tex", "v")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "savefig.bbox": "tight"})
F = pd.read_parquet(os.path.join(OUT, "v_first_marker.parquet"))
for c in ["board", "note", "employment", "health", "family"]: F[c] = pd.to_datetime(F[c], errors="coerce")
END = pd.Timestamp("2020-02-29")                      # stop before the pandemic so markers are not COVID-driven
F = F[(F.board >= "2016-01-01") & (F.board <= "2019-06-30") & F.note.notna()].copy()
F["any"] = F[["employment", "health", "family"]].min(axis=1)
F["followup"] = ((END - F.board).dt.days / 30.44).clip(lower=0)
L = pd.read_parquet(os.path.join(OUT, "loan_frame.parquet"))
print("loans boarded 2016-01..2019-06 with notes:", len(F), "| median follow-up (months):", round(F.followup.median(), 1))
res = {"n": int(len(F))}; curves = {}
for k in ["any", "employment", "health", "family"]:
    t = ((F[k] - F.board).dt.days / 30.44); ev = t.notna() & (F[k] <= END); dur = np.where(ev, t, F.followup); dur = np.clip(dur, 0, None)
    H, S, surv = [], [], 1.0
    for m in range(0, 30):
        at = (dur >= m).sum(); d = (ev & (dur >= m) & (dur < m + 1)).sum(); h = d / at if at else np.nan; H.append(h); surv *= (1 - h) if at else 1; S.append(1 - surv)
    curves[k] = {"hazard": H, "cum": S}
    res[k] = {"ever_pct": float(100 * ev.mean()), "cum3": 100 * S[2], "cum6": 100 * S[5], "cum12": 100 * S[11], "cum24": 100 * S[23], "haz_m1": 100 * H[0], "haz_m2_3": 100 * np.mean(H[1:3]), "haz_m4_6": 100 * np.mean(H[3:6]), "haz_m7_12": 100 * np.mean(H[6:12]), "haz_m13_24": 100 * np.mean(H[12:24]),
              "median_months_among_marked": float(np.median(t[ev])) if ev.any() else None}
    print(f"  {k:<11} ever (pre-pandemic) {res[k]['ever_pct']:.1f}% | cumulative by 3/6/12/24 mo: {res[k]['cum3']:.1f}/{res[k]['cum6']:.1f}/{res[k]['cum12']:.1f}/{res[k]['cum24']:.1f} | monthly hazard m1 {res[k]['haz_m1']:.2f}, m2-3 {res[k]['haz_m2_3']:.2f}, m4-6 {res[k]['haz_m4_6']:.2f}, m7-12 {res[k]['haz_m7_12']:.2f}, m13-24 {res[k]['haz_m13_24']:.2f} | median months among marked {res[k]['median_months_among_marked']:.1f}")
# by boarding cohort (is front-loading a cohort artifact?)
F["cohort"] = F.board.dt.year
for y, g in F.groupby("cohort"):
    t = ((g["any"] - g.board).dt.days / 30.44); ev = t.notna() & (g["any"] <= END); dur = np.where(ev, t, g.followup)
    hz = lambda a, b: 100 * np.mean([((ev & (dur >= m) & (dur < m + 1)).sum() / max(1, (dur >= m).sum())) for m in range(a, b)])
    res[f"cohort_{y}"] = {"n": int(len(g)), "h1_3": hz(0, 3), "h4_6": hz(3, 6), "h7_12": hz(6, 12)}; print(f"   cohort {y}: n={len(g):>5}  hazard m1-3 {hz(0,3):.2f}  m4-6 {hz(3,6):.2f}  m7-12 {hz(6,12):.2f}")
# notes volume is also front-loaded? contact intensity control: hazard per note is not available here; report first-note timing
t0 = (F.note - F.board).dt.days; res["first_note_days_median"] = float(t0.median()); res["first_note_within30"] = float(100 * (t0 <= 30).mean()); print("first servicer note: median days after boarding", t0.median(), "| within 30 days:", round(100 * (t0 <= 30).mean(), 1), "%")
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.5)); m = np.arange(1, 31)
for k, col, lb in [("employment", "#1f5fa8", "Employment"), ("health", "#c8551f", "Health"), ("family", "#2f8f5b", "Family")]:
    axs[0].plot(m, 100 * np.array(curves[k]["hazard"]), color=col, lw=1.4, label=lb); axs[1].plot(m, 100 * np.array(curves[k]["cum"]), color=col, lw=1.4, label=lb)
axs[0].set_title("Monthly hazard of first marker (%)", fontsize=9); axs[1].set_title("Cumulative share with a marker (%)", fontsize=9)
for ax in axs: ax.set_xlabel("Months since boarding")
axs[1].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(TJ, "f_relearn.pdf")); fig.savefig(os.path.join(TJ, "f_relearn.png"), dpi=110)
mac = {"rlN": f"{len(F):,}", "rlHazOne": f"{res['any']['haz_m1']:.1f}", "rlHazTwoThree": f"{res['any']['haz_m2_3']:.1f}", "rlHazLate": f"{res['any']['haz_m13_24']:.1f}", "rlCumThree": f"{res['any']['cum3']:.0f}", "rlCumSix": f"{res['any']['cum6']:.0f}",
       "rlCumTwelve": f"{res['any']['cum12']:.0f}", "rlCumTwentyFour": f"{res['any']['cum24']:.0f}", "rlMedian": f"{res['any']['median_months_among_marked']:.1f}", "rlFirstNote": f"{res['first_note_days_median']:.0f}",
       "rlRatio": f"{res['any']['haz_m1'] / max(1e-9, res['any']['haz_m13_24']):.1f}"}
open(os.path.join(TJ, "numbers_rl.tex"), "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()))
json.dump(res, open(os.path.join(OUT, "relearn_analysis.json"), "w"), indent=1, default=float); print(mac)
