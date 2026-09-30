"""(1) Does the journal memory result survive inquiry timing? (2) Responsiveness within regime period.
(3) RD robustness: covariate-adjusted, placebo cutoffs, mechanism (note flags)."""
import os, json, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out")
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
D = pd.read_parquet(os.path.join(OUT, "rd_frame.parquet")); N = pd.read_parquet(os.path.join(OUT, "v_loan_level.parquet")); F = pd.read_parquet(os.path.join(OUT, "inq_note_flags.parquet"))
L = pd.read_parquet(os.path.join(OUT, "loan_frame.parquet"))
D = D.join(N).join(F.add_prefix("nf_")).join(L[["pre_in_n", "pre_out_n", "pre_months"]])
for c in ["notes_pre", "mk_employment", "mk_health", "mk_family"]: D[c] = D[c].fillna(0)
D["anyhard"] = ((D.mk_employment + D.mk_health + D.mk_family) > 0).astype(int); D["lnotes"] = np.log1p(D.notes_pre)
D["wk"] = D.inq.dt.to_period("W-MON").astype(str); D["state"] = D.state.fillna("NA")
for c, b in [("fico", [0, 580, 620, 660, 700, 900]), ("ltv", [0, 80, 90, 97, 105, 1000])]: D[c + "_b"] = pd.cut(D[c], b).astype(object).fillna("missing").astype(str)
D["bal_b"] = pd.qcut(D.bal, 5, duplicates="drop").astype(object).fillna("missing").astype(str)
def fit(f, d, cl="state"):
    d = d.reset_index(drop=True); m = smf.ols(f, d, missing="drop"); return m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.loc[m.data.row_labels, cl])[0]})
pr = lambda m, k: f"{100*m.params[k]:.1f} ({100*m.bse[k]:.1f}){star(m.pvalues[k])}"
HARD = " + C(status_feb20) + npl + C(fico_b) + C(ltv_b) + C(bal_b) + C(state)"
CV = D[(D.Gov == 0) & (D.board < "2020-03-01")]; res = {}

print("=== (1) journal memory result vs inquiry timing (conventional inquirers) ===")
print("share with marker, by period:", CV.groupby("post").anyhard.mean().round(3).to_dict(), "| n:", CV.groupby("post").size().to_dict())
for lab, f, d in [("baseline", "fb ~ anyhard + lnotes + tenure" + HARD, CV), ("+ post-cutoff dummy", "fb ~ anyhard + lnotes + tenure + post" + HARD, CV),
                  ("+ inquiry-week FE", "fb ~ anyhard + lnotes + tenure + C(wk)" + HARD, CV), ("pre-cutoff only", "fb ~ anyhard + lnotes + tenure" + HARD, CV[CV.post == 0]),
                  ("post-cutoff only", "fb ~ anyhard + lnotes + tenure" + HARD, CV[CV.post == 1])]:
    m = fit(f, d); res["mem_" + lab] = (m.params["anyhard"], m.bse["anyhard"], m.pvalues["anyhard"], int(m.nobs), d.fb.mean()); print(f"  {lab:<22} anyhard {pr(m, 'anyhard')}  N={int(m.nobs)}  mean fb={d.fb.mean():.3f}")
GV = D[(D.Gov == 1) & (D.board < "2020-03-01")]; m = fit("fb ~ anyhard + lnotes + tenure + C(wk)" + HARD, GV); print(f"  gov placebo + week FE  anyhard {pr(m, 'anyhard')}  N={int(m.nobs)}"); res["mem_gov_wk"] = (m.params["anyhard"], m.bse["anyhard"], m.pvalues["anyhard"], int(m.nobs))

print("\n=== (2) responsiveness and completion, within period (conventional; >=6 pre months, contacted) ===")
H = D[(D.Gov == 0) & (D.pre_months >= 6) & (D.pre_out_n >= 1)]
for lab, d in [("pre-cutoff", H[H.post == 0]), ("post-cutoff", H[H.post == 1]), ("pooled + week FE", H)]:
    f = "fb ~ pre_in + pre_out" + HARD + (" + C(wk)" if "week" in lab else ""); m = fit(f, d); res["resp_" + lab] = (m.params["pre_in"], m.bse["pre_in"], m.pvalues["pre_in"], int(m.nobs)); print(f"  {lab:<18} slope on responsiveness {pr(m, 'pre_in')}  N={int(m.nobs)}")

print("\n=== (3a) covariate-adjusted RD, conventional ===")
def ll(d, y, h, cov=""):
    x = d[d.r.abs() <= h].dropna(subset=[y]).copy(); x["w"] = 1 - x.r.abs() / (h + 1); x = x.reset_index(drop=True)
    mod = smf.wls(f"{y} ~ post + r + post:r" + cov, x, weights=x.w, missing="drop"); m = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.loc[mod.data.row_labels, "r"])[0]})
    return m.params["post"], m.bse["post"], m.pvalues["post"], int(m.nobs)
COV = " + dq_feb20 + npl + pre_in + pre_out + hard + C(fico_b) + C(ltv_b) + C(bal_b) + tenure"
Cc = D[D.Gov == 0]
for y in ["fb", "dq_sep20", "mod_post", "dollars_octmar", "performing", "fc_path", "pay_octmar"]:
    a, b = ll(Cc, y, 14), ll(Cc, y, 14, COV); res["adj_" + y] = {"raw": a, "adj": b}; sc = 1 if y == "dollars_octmar" or y == "pay_octmar" else 100
    print(f"  {y:<15} raw {sc*a[0]:9.2f} ({sc*a[1]:.2f}){star(a[2]):<3}  adjusted {sc*b[0]:9.2f} ({sc*b[1]:.2f}){star(b[2]):<3}  N={b[3]}")
print("\n=== (3b) placebo cutoffs: first-stage jump for conventional at every weekday 20 Mar-15 May (h=7) ===")
pl = []
for cut in pd.bdate_range("2020-03-20", "2020-05-15"):
    x = Cc.assign(r=(Cc.inq.dt.normalize() - cut).dt.days); x["post"] = (x.r >= 0).astype(int)
    if ((x.r.abs() <= 7) & (x.post == 0)).sum() < 25 or ((x.r.abs() <= 7) & (x.post == 1)).sum() < 25: continue
    a = ll(x, "fb", 7); pl.append((cut.strftime("%m-%d"), round(100 * a[0], 1)))
print("  ", pl); jumps = sorted(abs(v) for d, v in pl if d not in ("04-06", "04-07", "04-08")); res["placebo"] = pl
print("   largest |jump| away from 6-8 April:", jumps[-1] if jumps else None, "| at 04-07:", dict(pl).get("04-07"))
print("\n=== (3c) mechanism: jump in note content at the cutoff (share of inquirers with >=1 matching note, 60 days after inquiry) ===")
for k in ["nf_docs", "nf_incomplete", "nf_denied", "nf_repay", "nf_approved", "nf_notes60"]:
    D["_y"] = (D[k] > 0).astype(float) if k != "nf_notes60" else D[k]
    a, b = ll(D[D.Gov == 0], "_y", 14), ll(D[D.Gov == 1], "_y", 14); sc = 1 if k == "nf_notes60" else 100; res["mech_" + k] = {"conv": a, "gov": b}
    print(f"  {k:<14} conventional {sc*a[0]:7.1f} ({sc*a[1]:.1f}){star(a[2]):<3} | gov {sc*b[0]:6.1f} ({sc*b[1]:.1f}){star(b[2])}")
# by-period funnel for the paper
print("\n=== conventional completion by period ===")
for lab, d in [("first inquiry before 7 Apr", Cc[Cc.post == 0]), ("first inquiry 7 Apr-30 Sep", Cc[Cc.post == 1])]:
    print(f"  {lab}: n={len(d)}, completion {100*d.fb.mean():.1f}%, median days {d.loc[d.fb==1,'days'].median()}")
Gg = D[D.Gov == 1]
for lab, d in [("gov before 7 Apr", Gg[Gg.post == 0]), ("gov 7 Apr-30 Sep", Gg[Gg.post == 1])]: print(f"  {lab}: n={len(d)}, completion {100*d.fb.mean():.1f}%")
res["period"] = {"c_pre": (len(Cc[Cc.post == 0]), Cc[Cc.post == 0].fb.mean()), "c_post": (len(Cc[Cc.post == 1]), Cc[Cc.post == 1].fb.mean()), "g_pre": (len(Gg[Gg.post == 0]), Gg[Gg.post == 0].fb.mean()), "g_post": (len(Gg[Gg.post == 1]), Gg[Gg.post == 1].fb.mean())}
json.dump(res, open(os.path.join(OUT, "round2c.json"), "w"), indent=1, default=float)
