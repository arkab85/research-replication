# Recreates Fig. 1 from the raw under-contract shares reported in Table 1 / Sect. 6.1.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 10})
bins   = ["< 0.90", "0.90–1.10", "1.10–1.25", "1.25–2.00", "> 2.00"]
shares = [4.62, 6.17, 4.18, 4.20, 2.85]
hatch  = ["//", "", "..", "..", "xx"]
fig, ax = plt.subplots(figsize=(6.85, 3.6))   # 174 mm wide
bars = ax.bar(bins, shares, color=["#bdbdbd", "#4d4d4d", "#9e9e9e", "#9e9e9e", "#d9d9d9"],
              edgecolor="black", linewidth=0.6)
for b, h in zip(bars, hatch):
    b.set_hatch(h)
for b, s in zip(bars, shares):
    ax.text(b.get_x() + b.get_width()/2, s + 0.12, f"{s:.2f}", ha="center", va="bottom", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
ax.axhline(4.91, color="black", linestyle="--", linewidth=0.8)
ax.text(4.45, 5.35, "Sample mean 4.91", ha="right", va="bottom", fontsize=8.5)
ax.set_xlabel("Asking price / AVM point estimate")
ax.set_ylabel("Pending or contingent (%)")
ax.set_ylim(0, 7.2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("Fig1.pdf")
fig.savefig("Fig1.png", dpi=600)
