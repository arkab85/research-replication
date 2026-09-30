"""(A) Anonymisation audit of everything in the delivery folder.
(B) The shadow price of balance sheet, in the model's own units.

(B) works by revealed preference. If an issuer ranks vested options by value and value is
increasing in the note rate, then an exercise rate of x% means the marginal loan it could
afford sits at the (1-x) quantile of the coupon distribution it faced. Watching that
cutoff move tells us how much value the constraint forced issuers to decline, without
needing a price for the option.
"""
import pandas as pd, numpy as np, os, re, json, glob, warnings
from config import OUT, DATA
warnings.filterwarnings("ignore")

DEL = os.environ.get("JMP_DELIVERY", str(Path.home() / "JMP_2026-09-23"))
res = {}
def line(m): print("\n" + "=" * 86); print(m); print("=" * 86, flush=True)

# ---------------------------------------------------------------- (A) audit
line("A. ANONYMISATION AUDIT OF THE DELIVERY FOLDER")
PII = [(re.compile(r"\b\d{1,5}\s+[NSEW]?\s*[A-Z][A-Za-z]{2,}\s+(ST|AVE|RD|DR|LN|CT|BLVD|PL|WAY|CIR)\b"), "street address"),
       (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN-like"),
       (re.compile(r"(?i)\bborrower[_ ]?name\b"), "borrower name field"),
       (re.compile(r"(?i)\bcape_parcel_id\b|cape_primary_structure_lat"), "parcel/geo id")]
bad = 0
SELF = {"70_audit_shadow.py", "61_repool_small.py", "62_postbuyout.py"}  # contain the patterns or the paths, not the data
for f in glob.glob(os.path.join(DEL, "**", "*"), recursive=True):
    if not os.path.isfile(f): continue
    if os.path.basename(f) in SELF: continue
    if os.path.splitext(f)[1].lower() not in (".tex", ".csv", ".json", ".md", ".py"): continue
    try:
        t = open(f, encoding="utf-8", errors="replace").read()
    except Exception: continue
    for rx, lab in PII:
        m = rx.search(t)
        if m:
            print(f"   FLAG {lab:<18} {os.path.relpath(f, DEL)}  -> {m.group(0)[:40]!r}")
            bad += 1
print(f"   {'CLEAN - no borrower identifiers found' if bad == 0 else str(bad)+' flags'}")
res["pii_flags"] = bad

# ---------------------------------------------------------------- (B) shadow price
line("B. WHERE THE MARGINAL EXERCISED LOAN SAT, BEFORE AND AFTER")
p = pd.read_parquet(os.path.join(OUT, "panel.parquet"))
p = p.dropna(subset=["coupon", "buyout"])
p = p[p.itype_ext.isin(["depository", "nonbank"])].copy()
p["per"] = np.where(p.ym.between(201907, 202002), "pre",
           np.where(p.ym.between(202003, 202009), "post", None))
p = p[p.per.notna()]

rows = []
for t in ["nonbank", "depository"]:
    for per in ["pre", "post"]:
        x = p[(p.itype_ext == t) & (p.per == per)]
        rate = x.buyout.mean()
        # marginal loan sits at the (1-rate) quantile of the coupon faced
        cstar = x.coupon.quantile(max(0.0, min(1.0, 1 - rate)))
        rows.append(dict(type=t, period=per, n=len(x), exercise=rate * 100,
                         pctile=(1 - rate) * 100, cstar=cstar,
                         mean_coupon=x.coupon.mean()))
M = pd.DataFrame(rows)
print(M.round(2).to_string(index=False))
for t in ["nonbank", "depository"]:
    a = M[(M.type == t) & (M.period == "pre")].iloc[0]
    b = M[(M.type == t) & (M.period == "post")].iloc[0]
    print(f"\n   {t}: marginal exercised loan moves from the {a.pctile:.0f}th to the "
          f"{b.pctile:.0f}th percentile of the coupon distribution")
    print(f"      implied cutoff coupon {a.cstar:.2f}% -> {b.cstar:.2f}%  "
          f"({(b.cstar-a.cstar)*100:+.0f} bp)")
    res[f"cut_{t}"] = {"pctile_pre": round(float(a.pctile), 0),
                       "pctile_post": round(float(b.pctile), 0),
                       "cstar_pre": round(float(a.cstar), 2),
                       "cstar_post": round(float(b.cstar), 2),
                       "bp": round(float((b.cstar - a.cstar) * 100), 0)}

line("C. TRANSLATING THE CUTOFF SHIFT INTO POINTS OF FORGONE GAIN")
print("   A repurchased loan that cures is redelivered into a new pool. The gain per")
print("   dollar is the pool premium at that coupon, which rises with the coupon.")
print("   Using a linear price schedule P(c) = 100 + D x (c - c_cur) with the 2020")
print("   current coupon c_cur = 2.5% and effective duration D:")
nb = res["cut_nonbank"]
for D in [3.0, 4.0, 5.0]:
    g_pre = D * (nb["cstar_pre"] - 2.5)
    g_post = D * (nb["cstar_post"] - 2.5)
    print(f"     D={D}:  marginal trade worth {g_pre:5.2f} pts before, "
          f"{g_post:5.2f} pts after   (forgone at the margin {g_post-g_pre:+5.2f} pts)")
res["forgone_pts_D4"] = round(float(4.0 * (nb["cstar_post"] - nb["cstar_pre"])), 2)
print("\n   The shadow price of a dollar of balance sheet is the value of the marginal")
print("   trade the issuer had to decline. At D=4 that is "
      f"{res['forgone_pts_D4']:.2f} points per dollar repurchased.")

line("D. CALIBRATING THE CURE PROBABILITY FROM A PURCHASER'S BOOK")
try:
    x = pd.read_excel(DATA / "RT_EBO_Repool_LoanMod.xlsx")
    x["rr"] = x.Repool / x.EBO * 100
    pre = x[(x.YearMonth >= 201901) & (x.YearMonth <= 202002)].rr.mean()
    post = x[x.YearMonth >= 202003].rr.mean()
    print(f"   repool (cure-and-redeliver) rate: {pre:.1f}% pre, {post:.1f}% post")
    print("   -> the model's cure probability is observed, not assumed")
    res["repool_pre"] = round(float(pre), 1); res["repool_post"] = round(float(post), 1)
except Exception as e:
    print("   not available:", type(e).__name__)

json.dump(res, open(os.path.join(OUT, "results_shadow.json"), "w"), indent=1, default=str)
print("\nwrote results_shadow.json")
