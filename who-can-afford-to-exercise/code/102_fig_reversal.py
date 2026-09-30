"""Figure 5: the audited series, and the wedge.

Panel (a): PennyMac's reported stock of vested-but-unexercised options, quarterly
2017-2025, with the loan-level measure over the quarters where both exist.
Panel (b): for each of the four filers, growth in the reported balance against
growth in the flow of newly vesting options. The 45-degree line is the null:
no change in exercise behaviour.
"""
import pandas as pd, numpy as np, os
import matplotlib
from config import OUT, FIGS as FIG
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})

a = pd.read_csv(os.path.join(OUT, "pennymac_recovery.csv"), index_col=0, parse_dates=True)
d = pd.read_csv(os.path.join(OUT, "four_firm.csv"))

# loan-level quarterly measure for PennyMac, over the overlap
p = pd.read_parquet(os.path.join(OUT, "panel.parquet")).dropna(subset=["buyout"])
nm = pd.read_csv(os.path.join(OUT, "issuer_id_names.csv"), index_col=0)["name"].str.upper().str.strip()
pid = nm[nm == "PENNYMAC LOAN SERVICES, LLC"].index[0]
g = p[p.issuer_id == pid].copy()
g["q"] = pd.PeriodIndex(pd.to_datetime(g.ym.astype(str), format="%Y%m"), freq="Q")
ll = (g[g.buyout == 0].groupby("q").upb.sum() / 1e9)
ll.index = ll.index.to_timestamp(how="end")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.7))

# ---------------------------------------------------------------- panel (a)
ax1.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30"),
            color="0.90", zorder=0)
ax1.plot(a.index, a.elig, color="#1f3b73", lw=1.6, marker="o", ms=2.6,
         label="Reported, 10-K/10-Q (ASC 860-50)")
ov = ll[(ll.index >= a.index.min()) & (ll.index <= a.index.max())]
ax1.plot(ov.index, ov.values, color="#b3402f", lw=1.3, ls="--", marker="s",
         ms=2.6, label="Computed from loan-level disclosure")
ax1.axhline(a.loc[(a.index >= "2018-01-01") & (a.index <= "2020-02-29"), "elig"].mean(),
            color="0.45", lw=0.8, ls=":")
ax1.annotate("pre-shock mean", xy=(pd.Timestamp("2017-06-30"), 1.6),
             fontsize=7.5, color="0.35")
ax1.annotate("CARES forbearance\nwave", xy=(pd.Timestamp("2020-06-30"), 15.6),
             xytext=(pd.Timestamp("2021-09-30"), 15.6), fontsize=7.5,
             color="0.25", va="center",
             arrowprops=dict(arrowstyle="->", lw=0.7, color="0.45"))
ax1.annotate("loan-level sample ends", xy=(pd.Timestamp("2020-09-30"), 0.4),
             xytext=(pd.Timestamp("2016-10-01"), 8.4), fontsize=7.5, color="0.35",
             arrowprops=dict(arrowstyle="->", lw=0.7, color="0.55"))
ax1.set_ylabel("Vested, unexercised buyout options (\\$bn)")
ax1.set_title("(a) PennyMac, quarterly", fontsize=9.5, loc="left")
ax1.legend(frameon=False, fontsize=7.6, loc="upper left")
ax1.set_ylim(0, 27)

# ---------------------------------------------------------------- panel (b)
al = pd.read_csv(os.path.join(OUT, "wedge_matched.csv"))
al["fall"] = -al.d_rate
w = al[al.net].copy()          # the statistic uses net-of-repurchase issuers only
gross = al[~al.net].copy()     # shown, but not in the fit or the correlation
ax2.axhline(1, color="0.45", lw=0.9, ls=":")
ax2.text(-2, 1.06, "balance grew in step with the flow", fontsize=7.2, color="0.4")
x = np.linspace(-12, 92, 50)
bb = np.polyfit(w.fall, np.log(w.wedge), 1)
ax2.plot(x, np.exp(np.polyval(bb, x)), color="0.62", lw=0.9)
for t, mk, col in (("nonbank", "o", "#1f3b73"), ("depository", "^", "#b3402f")):
    s = w[w.type == t]
    ax2.scatter(s.fall, s.wedge, marker=mk, s=54, color=col, zorder=3,
                edgecolor="white", linewidth=0.6, label=t.capitalize())
ax2.scatter(gross.fall, gross.wedge, marker="D", s=44, facecolor="white",
            edgecolor="#b3402f", linewidth=1.1, zorder=3,
            label="Depository, combined measure")
off = {"PennyMac Financial": (-3, 1.16, "right"),
       "Mr.\\ Cooper Group": (2.5, 0.92, "left"),
       "Caliber Home Loans": (-2.5, 1.04, "right"),
       "AmeriHome": (2.5, 1.02, "left"),
       "Flagstar Bank": (-3, 1.12, "right"),
       "JPMorgan Chase": (3.5, 1.00, "left")}
for _, r in al.iterrows():
    dx, dy, ha = off.get(r.entity, (2.5, 1.05, "left"))
    ax2.annotate(r.entity.replace("\\ ", " "), xy=(r.fall + dx, r.wedge * dy),
                 fontsize=7.4, ha=ha, va="center")
ax2.set_yscale("log")
ax2.set_ylim(0.3, 17)
ax2.set_xlim(-14, 100)
ax2.set_yticks([0.5, 1, 2, 4, 8, 16])
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax2.yaxis.set_minor_formatter(lambda v, _: "")
ax2.set_xlabel("Fall in the issuer's exercise rate, 2019 to 2020 (pp)")
ax2.set_ylabel("Wedge: balance growth $\\div$ flow growth")
ax2.set_title("(b) The wedge tracks retrenchment, across charter types",
              fontsize=9.5, loc="left")
ax2.legend(frameon=False, fontsize=7.4, loc="upper left")
ax2.annotate(f"$r={np.corrcoef(np.log(w.wedge), w.fall)[0,1]:.2f}$ (log wedge, "
             f"{len(w)} net-of-repurchase issuers)",
             xy=(0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=7.6,
             color="0.3")

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_audited.pdf"), bbox_inches="tight")
print("wrote fig5_audited.pdf")
print(f"  panel (a): {len(a)} reported quarters {a.index.min():%Y-%m} to {a.index.max():%Y-%m},"
      f" {len(ov)} overlap quarters")
print(f"  panel (b): {len(d)} firms, flow {d.flow_mult.min():.2f}-{d.flow_mult.max():.2f}x,"
      f" balance {d.rep_mult.min():.1f}-{d.rep_mult.max():.1f}x")
