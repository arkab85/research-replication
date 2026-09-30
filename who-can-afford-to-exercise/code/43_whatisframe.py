"""What does 'not removed' actually mean?

removal_in_n_mth_code = 0 bundles two very different loans: one that cured in place and
one that stayed delinquent while the issuer kept advancing. The paper's claim depends on
which. Four diagnostics:
  1. is the tracking window fixed, or 'until the data end'?
  2. do code-0 loans reappear later at three months delinquent?
  3. what was their forbearance status and delinquency trajectory?
  4. how does the code-0 share behave for cohorts with long vs short follow-up?
"""
import pandas as pd, numpy as np, os, json
from config import OUT, EXTRACT as DATA

USE = ["as_of_date", "issuer_id", "pool_id", "seq_num", "removal_in_n_mth_code",
       "removal_reason", "current_liquidation_flag", "fb_flag", "num_mth_fb",
       "months_dlq", "year", "identifier", "month_prepaid", "loan_age", "upb",
       "interest_rate"]
d = pd.read_csv(DATA, usecols=USE, low_memory=False)
d["ym"] = d.as_of_date.astype(int)
d["code"] = d.removal_in_n_mth_code
def line(m): print("\n" + "=" * 84); print(m); print("=" * 84, flush=True)

line("1. IS THE TRACKING WINDOW FIXED? code-0 share by vesting year")
g = d.groupby("year").code.apply(lambda s: (s == 0).mean() * 100)
n = d.groupby("year").size()
for y in g.index:
    print(f"   {y}  not-removed {g[y]:5.1f}%   N={n[y]:>9,}")
print("\n   If the window were 'until the data end', early cohorts would show a much")
print("   lower not-removed share than late ones. Read the pattern above.")

line("2. DO 'NOT REMOVED' LOANS COME BACK? repeat appearances of the same loan")
d["loan"] = d.identifier.astype(str)
cnt = d.loan.value_counts()
print(f"   distinct loans {len(cnt):,};  appearing once {int((cnt==1).sum()):,} "
      f"({(cnt==1).mean()*100:.1f}%);  2+ times {int((cnt>1).sum()):,}")
multi = set(cnt[cnt > 1].index)
d["repeat"] = d.loan.isin(multi)
print("\n   share of each disposition code that belongs to a loan seen more than once:")
t = d.groupby("code").repeat.mean() * 100
LAB = {0: "not removed", 1: "payoff", 2: "repurchase", 3: "foreclosure w/claim",
       4: "loss mitigation", 5: "substitution", 6: "other"}
for k, v in t.items(): print(f"     {LAB.get(k,k):<20} {v:5.1f}%")

# for loans seen more than once, what is the gap between appearances?
m = d[d.repeat].sort_values(["loan", "ym"])
m["nxt"] = m.groupby("loan").ym.shift(-1)
m["gap"] = ((m.nxt // 100 - m.ym // 100) * 12 + (m.nxt % 100 - m.ym % 100))
gg = m[(m.code == 0) & m.gap.notna()]
print(f"\n   code-0 records whose loan reappears later: {len(gg):,}")
print(f"   median months to reappearance: {gg.gap.median():.0f}   "
      f"25th {gg.gap.quantile(.25):.0f}  75th {gg.gap.quantile(.75):.0f}")
print("   (a loan that reappears at three months delinquent was never resolved)")

line("3. FORBEARANCE AND DELINQUENCY OF THE 'NOT REMOVED' GROUP")
w = d[d.ym.between(202003, 202009)]
z = w[w.code == 0]
print(f"   Mar-Sep 2020 not-removed: {len(z):,}")
print(f"     in forbearance at vesting: "
      f"{(z.fb_flag.astype(str).str.upper()=='Y').mean()*100:.1f}%")
print(f"     mean months already in forbearance: {pd.to_numeric(z.num_mth_fb, errors='coerce').mean():.2f}")
print(f"   comparison, repurchased in the same window: "
      f"{(w[w.code==2].fb_flag.astype(str).str.upper()=='Y').mean()*100:.1f}% in forbearance")

line("4. CODE-0 SHARE BY FOLLOW-UP LENGTH (2019 cohort only, window held fixed)")
c19 = d[d.year == 2019].copy()
c19["horizon"] = (202009 // 100 * 12 + 202009 % 100) - (c19.ym // 100 * 12 + c19.ym % 100)
print(c19.groupby("horizon").code.apply(lambda s: (s == 0).mean() * 100).round(1).to_string())

res = {"code0_by_year": g.round(1).to_dict(),
       "pct_once": round(float((cnt == 1).mean() * 100), 1),
       "median_gap_months": float(gg.gap.median()) if len(gg) else None,
       "fb_share_code0_2020": round(float((z.fb_flag.astype(str).str.upper()=='Y').mean()*100), 1)}
json.dump(res, open(os.path.join(OUT, "results_whatis.json"), "w"), indent=1, default=str)
print("\nwrote results_whatis.json")
