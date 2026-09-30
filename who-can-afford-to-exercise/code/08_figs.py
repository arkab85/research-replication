"""Figures for the job market paper."""
import pandas as pd, numpy as np, os, matplotlib
from config import OUT, FIGS as FIG
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
DEP, NB, TF = "#1f3a68", "#c8781f", "#7a7a7a"

p = pd.read_parquet(os.path.join(OUT, "panel.parquet"))
p = p.dropna(subset=["coupon", "fico", "cltv", "age", "buyout"])
p = p[p.itype_ext.isin(["depository", "nonbank", "techfirst"])].copy()
iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)

def mlab(ym):
    ym = int(ym); return f"{ym//100}m{ym%100:02d}"

# ---------------------------------------------------------------- Fig 1
g = (p.groupby(["ym", "itype_ext"]).buyout.mean().unstack() * 100).sort_index()
n = p.groupby("ym").size().sort_index()
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.4), height_ratios=[3, 1], sharex=True)
x = np.arange(len(g))
ax.plot(x, g["depository"], "-o", color=DEP, ms=4, lw=1.8, label="Depository issuers")
ax.plot(x, g["nonbank"], "-s", color=NB, ms=4, lw=1.8, ls="--", label="Nonbank issuers")
ax.plot(x, g["techfirst"], "-^", color=TF, ms=3, lw=1.1, ls=":", label="Technology-first")
k = list(g.index).index(202003)
ax.axvline(k - .5, color="k", lw=.9, ls=(0, (4, 3)))
ax.annotate("COVID-19 liquidity shock\nMarch 2020", xy=(k - .4, 62), fontsize=8.5,
            ha="left", va="top")
ax.set_ylabel("Buyout rate (\\% of vested options exercised)" if False
              else "Buyout rate (% of vested options)")
ax.set_ylim(0, 100); ax.legend(frameon=False, loc="center left", fontsize=9)
ax2.bar(x, n.values / 1000, color="#b9c2d0", width=.72)
ax2.set_ylabel("Options\nvesting (000s)", fontsize=8.5)
ax2.set_xticks(x[::2]); ax2.set_xticklabels([mlab(m) for m in g.index][::2],
                                            rotation=45, ha="right", fontsize=8)
fig.savefig(os.path.join(FIG, "fig1_monthly.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Fig 2
ev = pd.read_csv(os.path.join(OUT, "event_study.csv"), index_col=0)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
xs = np.arange(len(ev)); base = list(ev.index).index(202001) + .5
pre = ev.index <= 202001
ax.errorbar(xs[pre], ev.coef[pre], yerr=1.96 * ev.se[pre], fmt="o", color="#555",
            ms=4, lw=1, capsize=2, label="Pre-period")
ax.errorbar(xs[~pre], ev.coef[~pre], yerr=1.96 * ev.se[~pre], fmt="s", color=NB,
            ms=4.5, lw=1.2, capsize=2, label="Post-period")
z = np.polyfit(xs[pre], ev.coef[pre], 1)
ax.plot(xs, np.polyval(z, xs), color="#888", lw=.9, ls=(0, (5, 4)),
        label="Pre-trend, extrapolated")
ax.axhline(0, color="k", lw=.7); ax.axvline(base, color="k", lw=.9, ls=(0, (4, 3)))
ax.set_xticks(xs); ax.set_xticklabels([mlab(m) for m in ev.index], rotation=45,
                                      ha="right", fontsize=8)
ax.set_ylabel("Nonbank $\\times$ month coefficient")
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
fig.savefig(os.path.join(FIG, "fig2_event.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Fig 3
B = [0, 3.5, 4, 4.5, 5, 5.5, 99]
L = ["<3.5", "3.5–4.0", "4.0–4.5", "4.5–5.0", "5.0–5.5", "5.5+"]
p["cbin"] = pd.cut(p.coupon, B, labels=L, right=False)
q = p[p.itype_ext.isin(["depository", "nonbank"])].copy()
q["per"] = np.where(q.ym.between(201901, 201912), "2019",
                    np.where(q.ym.between(202003, 202009), "2020", None))
t = (q[q.per.notna()].pivot_table(index="cbin", columns=["itype_ext", "per"],
                                  values="buyout", observed=True) * 100)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.4))
xs = np.arange(6)
a1.plot(xs, t[("depository", "2019")], "-o", color=DEP, ms=4, label="Depository, 2019")
a1.plot(xs, t[("depository", "2020")], "--o", color=DEP, ms=4, mfc="w",
        label="Depository, 2020")
a1.plot(xs, t[("nonbank", "2019")], "-s", color=NB, ms=4, label="Nonbank, 2019")
a1.plot(xs, t[("nonbank", "2020")], "--s", color=NB, ms=4, mfc="w", label="Nonbank, 2020")
a1.set_xticks(xs); a1.set_xticklabels(L, rotation=45, ha="right", fontsize=8)
a1.set_ylabel("Buyout rate (%)"); a1.set_xlabel("Note rate (%)"); a1.set_ylim(0, 100)
a1.legend(frameon=False, fontsize=7.5, ncol=1, loc="center left")
a1.set_title("a. Exercise rises with coupon in both years", fontsize=9.5, loc="left")
# difference the ROUNDED levels, exactly as Table T3 Panel C does, so the bar
# labels and the table cannot disagree at the first decimal
d = t[("nonbank", "2020")].round(1) - t[("nonbank", "2019")].round(1)
a2.bar(xs, d.values, color=NB, width=.62)
for i, v in enumerate(d.values):
    a2.text(i, v - 1.6, f"{v:.1f}", ha="center", va="top", fontsize=7.5)
a2.axhline(0, color="k", lw=.7)
a2.set_xticks(xs); a2.set_xticklabels(L, rotation=45, ha="right", fontsize=8)
a2.set_ylabel("Change in nonbank rate (pp)"); a2.set_xlabel("Note rate (%)")
a2.set_title("b. Retrenchment is largest where the trade is best",
             fontsize=9.5, loc="left")
fig.savefig(os.path.join(FIG, "fig3_coupon.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Fig 4  (new)
SHORT = {3355: "Wells Fargo", 2094: "U.S. Bank", 3975: "JPMorgan Chase",
         3162: "MidFirst Bank", 3345: "Flagstar", 2936: "Bank of America",
         3663: "M&T Bank", 4413: "Fifth Third", 2559: "Truist", 2572: "PNC",
         3886: "CitiMortgage", 3774: "BOKF", 4033: "Gateway First",
         4094: "PennyMac", 4150: "Lakeview", 4052: "Nationstar",
         4143: "Carrington", 4213: "Caliber", 3359: "AmeriHome",
         4102: "The Money Source", 4034: "American Fin. Res.",
         4135: "Planet Home", 1990: "Mid America", 2397: "PHH"}
act = iss[(iss.ebo19 >= .10) & iss.ext.isin(["nonbank", "depository"])].copy()
fig, ax = plt.subplots(figsize=(7.2, 4.8))
for t_, col, mk, lab in [("depository", DEP, "o", "Depository issuers"),
                         ("nonbank", NB, "s", "Nonbank issuers")]:
    d = act[act.ext == t_]
    sz = 26 + 260 * (d.n19 / act.n19.max()) ** .5
    ax.scatter(d.S, d.d_ebo * 100, s=sz, c=col, marker=mk, alpha=.85,
               edgecolor="w", linewidth=.7, label=lab, zorder=3)
    w = d.n19 + d.n20
    b = np.polyfit(d.S, d.d_ebo * 100, 1, w=np.sqrt(w))
    xx = np.linspace(act.S.min() - .15, act.S.max() + .15, 20)
    ax.plot(xx, np.polyval(b, xx), color=col, lw=1.8,
            ls="-" if t_ == "depository" else "--", zorder=2, alpha=.9)
ax.axhline(0, color="k", lw=.7)
ax.set_xlim(act.S.min() - .45, act.S.max() + .95)
ax.set_ylim(act.d_ebo.min() * 100 - 10, act.d_ebo.max() * 100 + 10)
NUDGE = {3355: (-14, 9), 4094: (-16, -4), 4150: (-16, 9), 4052: (-14, 8),
         3162: (8, -10), 4143: (10, 2), 3359: (-6, -11), 4213: (9, 2)}
for i, r in act.iterrows():
    nm = SHORT.get(i)
    if nm is None or r.n19 < 1000: continue
    dx, dy = NUDGE.get(i, (9, 6))
    ax.annotate(nm, (r.S, r.d_ebo * 100), fontsize=6.8, color="#3a3a3a",
                ha="left" if dx > 0 else "right",
                xytext=(dx, dy), textcoords="offset points")
ax.set_xlabel("Issuer scale: log(number of options vesting in 2019)")
ax.set_ylabel("Change in exercise rate,\n2019 to Mar–Sep 2020 (pp)")
ax.legend(frameon=False, fontsize=9, loc="lower left")
fig.savefig(os.path.join(FIG, "fig4_scale.pdf")); plt.close(fig)

print("figures written to", FIG)
for f in sorted(os.listdir(FIG)): print("  ", f)
