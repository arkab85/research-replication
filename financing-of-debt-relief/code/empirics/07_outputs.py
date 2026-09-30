"""Figures 4-5, the LaTeX macros used in the text, and the generated table rows."""
import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import OUT, FIGS, TABLES, WORK, PERIODS
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm", "font.size": 9.5,
  "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
BLACK, GRAY, LIGHT = "#111111", "#6b6b6b", "#b0b0b0"
R = json.load(open(OUT / "results.json")); S = pd.read_csv(OUT / "series.csv"); A = json.load(open(OUT / "index_auc.json"))
S["date"] = pd.to_datetime(S.ym.astype(str), format="%Y%m")
PER = [p for p, _, _ in PERIODS]; PK = dict(pre="Pre", freeze="Frz", release="Rel", recovery1="RecOne", recovery2="RecTwo", ratehike="Rate")
def shade(a):
    a.axvspan(pd.Timestamp("2020-02-16"), pd.Timestamp("2020-06-16"), color="#e6e6e6", zorder=0)
    a.axvspan(pd.Timestamp("2020-06-16"), pd.Timestamp("2020-12-16"), color="#f3f3f3", zorder=0)
    a.axvline(pd.Timestamp("2021-12-16"), color=LIGHT, lw=0.8, ls=":", zorder=0)
    a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    a.set_xticks([pd.Timestamp(x) for x in ["2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01"]])
# ---------- Figure 4 ----------
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.9))
for it, col, ls, lab in [("depository", GRAY, "--", "Depositories"), ("nonbank", BLACK, "-", "Nonbanks")]:
    s = S[S.itype == it]; ax[0].plot(s.date, s.m, color=col, ls=ls, lw=1.4, label=lab)
nb = S[S.itype == "nonbank"]
ax[1].plot(nb.date, nb.rel, "o", color=BLACK, ms=3.0, label="Actual")
ax[1].plot(nb.date, nb.rel_pred, color=GRAY, lw=1.4, label="Fixed ranking (model)")
ax[1].plot(nb.date, nb.rel_random, color=LIGHT, lw=1.3, ls="--", label="Random rationing")
for a in ax: shade(a)
ax[0].set(ylabel="Share of vested options exercised", ylim=(0, 1)); ax[0].legend(fontsize=8, loc="center left")
ax[0].set_title("A. Funded share", loc="left", fontsize=9.5)
ax[1].set(ylabel="Relative response to ranking index"); ax[1].legend(fontsize=7.6, loc="upper center")
ax[1].set_title("B. Nonbank relative response", loc="left", fontsize=9.5)
top = np.nanpercentile(nb.rel, 98); ax[1].set_ylim(0, max(2.2, top * 1.1))
fig.tight_layout(w_pad=2.2); fig.savefig(FIGS / "fig4_rationing_data.pdf"); plt.close(fig)
# ---------- Figure 5 ----------
d = pd.read_parquet(WORK / "scored.parquet", columns=["ym", "itype", "period", "x_ever", "t_ever"])
d = d[d.itype.isin(["depository", "nonbank"]) & d.period.isin(["pre", "freeze"])]
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.9))
K = np.arange(0, 25); curves = {}
for it, col in [("nonbank", BLACK), ("depository", GRAY)]:
    for p, ls in [("pre", "--"), ("freeze", "-")]:
        g = d[(d.itype == it) & (d.period == p)]
        c = [((g.x_ever == 1) & (g.t_ever <= k)).mean() for k in K]; curves[f"{it}_{p}"] = c
        EN = "\u2013"; lab = ("Nonbanks" if it == "nonbank" else "Depositories") + ", " + (f"Mar 2019{EN}Feb 2020" if p == "pre" else f"Mar{EN}Jun 2020")
        ax[0].plot(K, c, color=col, ls=ls, lw=1.4, label=lab)
ax[0].set(xlabel="Months since vesting", ylabel="Share exercised", ylim=(0, 1), xlim=(0, 24)); ax[0].legend(fontsize=7.0, loc="center", bbox_to_anchor=(0.58, 0.64))
ax[0].set_title("A. Exercise by months since vesting", loc="left", fontsize=9.5)
for it, col, ls, lab in [("depository", GRAY, "--", "Depositories"), ("nonbank", BLACK, "-", "Nonbanks")]:
    s = S[S.itype == it]; ax[1].plot(s.date, s.level, color=col, ls=ls, lw=1.4, label=lab)
shade(ax[1]); ax[1].axhline(0, color=LIGHT, lw=0.6)
ax[1].set(ylabel="Level response to ranking index"); ax[1].legend(fontsize=8, loc="upper center")
ax[1].set_title("B. Level response", loc="left", fontsize=9.5)
fig.tight_layout(w_pad=2.2); fig.savefig(FIGS / "fig5_queue_screening.pdf"); plt.close(fig)
# ---------- macros ----------
def f(x, dd=2):
    t = f"{x:.{dd}f}"
    if t.startswith("-") and float(t) == 0: t = t[1:]      # print 0.00, not -0.00
    return t.replace("-", "$-$") if t.startswith("-") else t
M = {}
M["EmpN"] = f"{R['N_decisions']:,}"; M["EmpIssuers"] = str(R["N_issuers"])
M["EmpNbN"] = f"{R['N_by_type']['nonbank']:,}"; M["EmpDepN"] = f"{R['N_by_type']['depository']:,}"
P = R["periods"]
M["NbFrzDrop"] = f(100 * (P["nonbank_pre"]["m"] - P["nonbank_freeze"]["m"]), 0); M["NbFrzRatio"] = f(P["nonbank_pre"]["m"] / P["nonbank_freeze"]["m"], 1)
for it, k in [("nonbank", "Nb"), ("depository", "Dep")]:
    for per in PER:
        p = P[f"{it}_{per}"]; pk = PK[per]
        M[f"{k}{pk}M"] = f(100 * p["m"], 1); M[f"{k}{pk}Lvl"] = f(p["level"], 3); M[f"{k}{pk}LvlPred"] = f(p["level_pred"], 3)
        M[f"{k}{pk}Rel"] = f(p["rel"]); M[f"{k}{pk}RelPred"] = f(p["rel_pred"]); M[f"{k}{pk}RelRand"] = f(p["rel_random"])
        if per == "pre": continue
        g = lambda c: R.get(f"{it}_boot_se_{per}_{c}", np.nan)
        M[f"{k}{pk}GapFix"] = f(p["rel"] - p["rel_pred"]); M[f"{k}{pk}GapFixSE"] = f(g("gap_fixed"))
        M[f"{k}{pk}GapRand"] = f(p["rel"] - p["rel_random"]); M[f"{k}{pk}GapRandSE"] = f(g("gap_random"))
        pre = P[f"{it}_pre"]; did = (p["rel"] - pre["rel"]) - (p["rel_random"] - pre["rel_random"])
        M[f"{k}{pk}DidRand"] = f(did); M[f"{k}{pk}DidRandSE"] = f(g("did_random"))
        M[f"{k}{pk}LvlSE"] = f(g("level"), 3)
        bt = pd.read_csv(OUT / f"boot_{it}.csv")
        for col, ck in [("gap_fixed", "GapFix"), ("gap_random", "GapRand"), ("did_random", "DidRand")]:
            q = np.percentile(bt[f"{per}_{col}"].dropna(), [2.5, 97.5]); M[f"{k}{pk}{ck}PLo"] = f(q[0]); M[f"{k}{pk}{ck}PHi"] = f(q[1])
        fl = R[f"{it}_felogit"]
        M[f"{k}BetaD{pk}"] = f(fl[f"d_{per}"]); M[f"{k}BetaD{pk}SE"] = f(fl[f"se_{per}"]); M[f"{k}BetaD{pk}Abs"] = f(abs(fl[f"d_{per}"]))
        M[f"{k}BetaD{pk}Lo"] = f(fl[f"d_{per}"] - 1.96 * fl[f"se_{per}"]); M[f"{k}BetaD{pk}Hi"] = f(fl[f"d_{per}"] + 1.96 * fl[f"se_{per}"])
    fl = R[f"{it}_felogit"]; M[f"{k}BetaPre"] = f(fl["beta_pre"]); M[f"{k}BetaPreSE"] = f(fl["se_pre"])
    lo = R[f"{it}_loo"]
    for a, b in [("rel_min", "LooRelMin"), ("rel_max", "LooRelMax"), ("gap_fixed_min", "LooGapFixMin"), ("gap_fixed_max", "LooGapFixMax"),
                 ("gap_random_min", "LooGapRandMin"), ("gap_random_max", "LooGapRandMax")]: M[f"{k}{b}"] = f(lo[a])
    M[f"{k}TopTwoShare"] = f(100 * R[f"{it}_top2_cov_share"], 0)
    ex = R[f"{it}_rest"]; M[f"{k}ExTwoRel"] = f(ex["freeze_rel"]); M[f"{k}ExTwoRelPred"] = f(ex["freeze_rel_pred"]); M[f"{k}ExTwoRelRand"] = f(ex["freeze_rel_random"])
    fb = R[f"{it}_forbear_lpm"]; fc = R[f"{it}_forbear_lpm_ctrl"]
    M[f"{k}FbRate"] = f(100 * fb["rate_forborne"], 1); M[f"{k}NoFbRate"] = f(100 * fb["rate_nonforborne"], 1)
    M[f"{k}FbCoef"] = f(100 * fb["coef"], 1); M[f"{k}FbSE"] = f(100 * fb["se"], 1); M[f"{k}FbCoefAbs"] = f(abs(100 * fb["coef"]), 1)
    M[f"{k}FbShare"] = f(100 * fb["share_forborne"], 0); M[f"{k}FbIssuers"] = str(fb["issuers"])
    M[f"{k}FbCoefCtrl"] = f(100 * fc["coef"], 1); M[f"{k}FbSECtrl"] = f(100 * fc["se"], 1); M[f"{k}FbCoefCtrlAbs"] = f(abs(100 * fc["coef"]), 1)
    M[f"{k}FbEver"] = f(100 * fb["ever_forborne"], 1); M[f"{k}NoFbEver"] = f(100 * fb["ever_nonforborne"], 1); M[f"{k}FbTwelve"] = f(100 * fb["x12_forborne"], 1)
    for per in ["pre", "freeze", "release", "recovery1"]:
        q = R[f"queue_{it}_{per}"]; pk = PK[per]
        M[f"{k}Q{pk}N"] = f"{q['n']:,}"; M[f"{k}Q{pk}Rate"] = f(100 * q["rate"], 1); M[f"{k}Q{pk}Beta"] = f(q["beta"]); M[f"{k}Q{pk}SE"] = f(q["se"]); M[f"{k}Q{pk}Ever"] = f(100 * q["ever"], 1)
        qa = R[f"queue_all_{it}_{per}"]; M[f"{k}QA{pk}Beta"] = f(qa["beta"]); M[f"{k}QA{pk}SE"] = f(qa["se"]); M[f"{k}QA{pk}Rate"] = f(100 * qa["rate"], 1); M[f"{k}QA{pk}Ever"] = f(100 * qa["ever"], 1)
    M[f"{k}TopTwoOptShare"] = f(100 * R[f"{it}_top2_option_share"], 0)
    for lab, lk in [("top2", "TopTwo"), ("rest", "Rest")]:
        t2 = R[f"{it}_{lab}"]; M[f"{k}{lk}BetaPre"] = f(t2["beta_pre"]); M[f"{k}{lk}BetaPreSE"] = f(t2["se_pre"]); M[f"{k}{lk}Issuers"] = str(t2["issuers"])
        for per in ["pre", "freeze", "release", "recovery1"]:
            pk = PK[per]; M[f"{k}{lk}{pk}M"] = f(100 * t2[f"{per}_m"], 1); M[f"{k}{lk}{pk}Rel"] = f(t2[f"{per}_rel"]); M[f"{k}{lk}{pk}RelPred"] = f(t2[f"{per}_rel_pred"]); M[f"{k}{lk}{pk}RelRand"] = f(t2[f"{per}_rel_random"])
    mo = R[f"monthly_{it}"]
    for ym, nm in [("202003", "Mar"), ("202004", "Apr"), ("202005", "May"), ("202006", "Jun")]:
        M[f"{k}{nm}M"] = f(100 * mo[ym]["m"], 1); M[f"{k}{nm}Rel"] = f(mo[ym]["rel"]); M[f"{k}{nm}RelPred"] = f(mo[ym]["rel_pred"]); M[f"{k}{nm}RelRand"] = f(mo[ym]["rel_random"])
    M[f"{k}FbByJune"] = f(100 * fb["byjune_forborne"], 1); M[f"{k}NoFbByJune"] = f(100 * fb["byjune_nonforborne"], 1)
    M[f"{k}BootB"] = str(R[f"{it}_boot_draws"]); M[f"{k}BootIssuers"] = str(R[f"{it}_boot_issuers"])
fo = R["freeze_outcomes"]
for it, k in [("nonbank", "Nb"), ("depository", "Dep")]:
    M[f"{k}FrzTwelve"] = f(100 * fo["x12"][it], 1); M[f"{k}FrzEver"] = f(100 * fo["x_ever"][it], 1)
po = pd.DataFrame(R["period_outcomes"]).set_index(["period", "itype"])
for it, k in [("nonbank", "Nb"), ("depository", "Dep")]:
    M[f"{k}PreTwelve"] = f(100 * po.loc[("pre", it), "x12"], 1); M[f"{k}PreEver"] = f(100 * po.loc[("pre", it), "x_ever"], 1)
M["QueueByJune"] = f(100 * R["queue_share_by_2021m6"], 0)
btn = pd.read_csv(OUT / "boot_nonbank.csv"); M["NbFrzRelFallPI"] = f(abs(np.percentile(btn.freeze_rel - btn.pre_rel, 2.5)))
M["QueueForgone"] = f(100 * (po.loc[("pre", "nonbank"), "x_ever"] - fo["x_ever"]["nonbank"]), 0)
cal = pd.Series(R["queue_calendar"]); cal.index = cal.index.astype(int)
M["QueueOctDec"] = f(100 * cal[(cal.index >= 202010) & (cal.index <= 202012)].sum() / cal.sum(), 0)
I = json.load(open(OUT / "issuers.json"))
M["TopFiveSharePre"] = f(100 * I["top_share_pre"], 0); M["TopFiveShareFrz"] = f(100 * I["top_share_freeze"], 0)
irows = []
for r in I["issuers"]:
    lab = r["label"] if r["label"] != "Others" else f"Other {r['issuers']} nonbanks"
    lk = r['label'] if r['label'] != 'Others' else 'Rest'
    M[f"Iss{lk}FrzM"] = f(100 * r["freeze_m"], 1); M[f"Iss{lk}PreM"] = f(100 * r["pre_m"], 1); M[f"Iss{lk}PreRel"] = f(r["pre_rel"])
    for p, pk in [("freeze", "Frz"), ("release", "Rel"), ("recovery1", "RecOne")]:
        for k, kk in [("rel", "Rel"), ("pred", "RelPred"), ("rand", "RelRand")]:
            M[f"Iss{r['label'] if r['label'] != 'Others' else 'Rest'}{pk}{kk}"] = f(r[f"{p}_{k}"])
    irows.append(f"{lab} & {r['pre_n']:,} & {f(r['beta_pre'])} & {f(100*r['pre_m'],1)} & {f(r['pre_rel'])} & {f(100*r['freeze_m'],1)} & {f(r['freeze_rel'])} & {f(r['freeze_pred'])} & {f(r['freeze_rand'])} & {f(100*r['recovery1_m'],1)} & {f(r['recovery1_rel'])} & {f(r['recovery1_pred'])} & {f(r['recovery1_rand'])}" + r"\\")
open(TABLES / "issuer_rows.tex", "w").write("\\newcommand{\\IssuerRows}{%\n" + "\n".join(irows) + "}\n")
for it, k in [("nonbank", "Nb"), ("depository", "Dep")]:
    for p, pk in [("pre", "Pre"), ("freeze", "Frz")]:
        for h, hk in [(3, "Hthree"), (12, "Htwelve"), (24, "Htwentyfour")]: M[f"{k}{pk}{hk}"] = f(100 * I[f"h{h}_{it}_{p}"], 1)
M["QueueForgoneHtwentyfour"] = f(round(100 * I["h24_nonbank_pre"], 1) - round(100 * I["h24_nonbank_freeze"], 1), 1)
M["QueueGapHthree"] = f(round(100 * I["h3_nonbank_pre"], 1) - round(100 * I["h3_nonbank_freeze"], 1), 1)
M["QueueMadeUpShare"] = f(100 * (1 - (I["h24_nonbank_pre"] - I["h24_nonbank_freeze"]) / (I["h3_nonbank_pre"] - I["h3_nonbank_freeze"])), 0)
# ---------- funding pressure (04c) and value per dollar (04d) ----------
X = json.load(open(OUT / "exposure.json")); PDJ = json.load(open(OUT / "perdollar.json"))
for it, k in [("nonbank", "Nb"), ("depository", "Dep")]:
    a = X["adv_summary"][it]
    for per, pk in [("pre", "Pre"), ("freeze", "Frz"), ("recovery1", "RecOne"), ("ratehike", "Rate")]: M[f"Adv{k}{pk}"] = f(a[per])
    M[f"Adv{k}WithinSD"] = f(a["sd_within"])
    pa = X["panel"][it]["ladv"]
    M[f"Adv{k}M"] = f(100 * pa["m"]["b"], 1); M[f"Adv{k}MSE"] = f(100 * pa["m"]["se"], 1); M[f"Adv{k}MAbs"] = f(abs(100 * pa["m"]["b"]), 1)
    M[f"Adv{k}MperSD"] = f(abs(100 * pa["m"]["b"] * pa["within_sd"]), 1); M[f"Adv{k}LogSD"] = f(pa["within_sd"])
    M[f"Adv{k}MCtrl"] = f(100 * pa["m_ctrl"]["b"], 1); M[f"Adv{k}MCtrlSE"] = f(100 * pa["m_ctrl"]["se"], 1)
    M[f"Adv{k}Stock"] = f(100 * pa["mx"]["b"], 1); M[f"Adv{k}StockSE"] = f(100 * pa["mx"]["se"], 1); M[f"Adv{k}StockAbs"] = f(abs(100 * pa["mx"]["b"]), 1)
    M[f"Adv{k}Rel"] = f(pa["rel"]["b"]); M[f"Adv{k}RelSE"] = f(pa["rel"]["se"]); M[f"Adv{k}Issuers"] = str(pa["m"]["issuers"]); M[f"Adv{k}N"] = f"{pa['m']['n']:,}"
    lg = X["logit"][it]; M[f"Adv{k}Beta"] = f(lg["beta_adv_within_month"]); M[f"Adv{k}BetaSE"] = f(lg["se_adv_within_month"])
    ss = X["shiftshare"][it]; M[f"SS{k}Issuers"] = str(ss["issuers"])
    M[f"SS{k}Lab"] = f(ss["dm_on_exp_lab"]["b"], 1); M[f"SS{k}LabSE"] = f(ss["dm_on_exp_lab"]["se"], 1); M[f"SS{k}Fb"] = f(ss["dm_on_exp_fb"]["b"], 1); M[f"SS{k}FbSE"] = f(ss["dm_on_exp_fb"]["se"], 1)
    M[f"SS{k}LabSD"] = f(ss["exp_lab_sd"], 1); M[f"SS{k}FbSD"] = f(ss["exp_fb_sd"], 1)
    pdp = PDJ[f"{it}_periods"]; M[f"Bal{k}Pre"] = f(pdp["lb_pre"]); M[f"Bal{k}PreSE"] = f(pdp["se_lb_pre"])
    for per in PER[1:]:
        M[f"Bal{k}D{PK[per]}"] = f(pdp[f"lb_{per}"]); M[f"Bal{k}D{PK[per]}SE"] = f(pdp[f"se_lb_{per}"]); M[f"Bal{k}D{PK[per]}Abs"] = f(abs(pdp[f"lb_{per}"]))
        M[f"IdxB{k}D{PK[per]}"] = f(pdp[f"r_{per}"]); M[f"IdxB{k}D{PK[per]}SE"] = f(pdp[f"se_r_{per}"])
    M[f"IdxB{k}Pre"] = f(pdp["r_pre"]); M[f"IdxB{k}PreSE"] = f(pdp["se_r_pre"])
    pda = PDJ[f"{it}_advance"]; M[f"Bal{k}Adv"] = f(pda["lb_adv"]); M[f"Bal{k}AdvSE"] = f(pda["se_lb_adv"])
    M[f"IdxB{k}Adv"] = f(pda["r_adv"]); M[f"IdxB{k}AdvSE"] = f(pda["se_r_adv"])
    M[f"SS{k}LabPerSD"] = f(ss["dm_on_exp_lab"]["b"] * ss["exp_lab_sd"], 1); M[f"SS{k}LabPerSDSE"] = f(ss["dm_on_exp_lab"]["se"] * ss["exp_lab_sd"], 1)
    M[f"SS{k}FbPerSD"] = f(ss["dm_on_exp_fb"]["b"] * ss["exp_fb_sd"], 1); M[f"SS{k}FbPerSDSE"] = f(ss["dm_on_exp_fb"]["se"] * ss["exp_fb_sd"], 1)
    if f"{it}_freeze_split" in PDJ:
        pds = PDJ[f"{it}_freeze_split"]
        for sp, sk in [("frz_early", "FrzEarly"), ("frz_late", "FrzLate")]:
            M[f"Bal{k}D{sk}"] = f(pds[f"lb_{sp}"]); M[f"Bal{k}D{sk}SE"] = f(pds[f"se_lb_{sp}"]); M[f"Bal{k}D{sk}Abs"] = f(abs(pds[f"lb_{sp}"]))
            M[f"IdxB{k}D{sk}"] = f(pds[f"r_{sp}"]); M[f"IdxB{k}D{sk}SE"] = f(pds[f"se_r_{sp}"])
M["AdvDiff"] = f(100 * X["panel"]["diff_ladv_m"]["b_diff"], 1); M["AdvDiffSE"] = f(100 * X["panel"]["diff_ladv_m"]["se_diff"], 1)
# rows: value per dollar table
brows = []
for it, k, lab in [("nonbank", "Nb", "Nonbanks"), ("depository", "Dep", "Depositories")]:
    brows.append(r"\multicolumn{7}{l}{\emph{" + lab + r"}}\\")
    brows.append("Weight on log balance & " + f"{M[f'Bal{k}Pre']} ({M[f'Bal{k}PreSE']})" + " & " + " & ".join(f"{M[f'Bal{k}D{PK[p]}']} ({M[f'Bal{k}D{PK[p]}SE']})" for p in PER[1:]) + r"\\")
    brows.append("Weight on ranking index & " + f"{M[f'IdxB{k}Pre']} ({M[f'IdxB{k}PreSE']})" + " & " + " & ".join(f"{M[f'IdxB{k}D{PK[p]}']} ({M[f'IdxB{k}D{PK[p]}SE']})" for p in PER[1:]) + r"\\")
open(TABLES / "balance_rows.tex", "w").write("\\newcommand{\\BalanceRows}{%\n" + "\n".join(brows) + "}\n")
# rows: funding pressure table (IA)
frows = []
for xv, lab in [("ladv", "Log advances outstanding (\\% of balance)"), ("adv", "Advances outstanding (\\% of balance)"), ("adv_pmt", "Advances outstanding (months of P\\&I)"), ("fbsh", "Share of portfolio in forbearance (\\%)")]:
    cells = []
    for it in ["nonbank", "depository"]:
        pa = X["panel"][it][xv]
        cells += [f"{f(100*pa['m']['b'],1)} ({f(100*pa['m']['se'],1)})", f"{f(100*pa['mx']['b'],1)} ({f(100*pa['mx']['se'],1)})", f"{f(pa['rel']['b'])} ({f(pa['rel']['se'])})"]
    frows.append(lab + " & " + " & ".join(cells) + r"\\")
open(TABLES / "funding_rows.tex", "w").write("\\newcommand{\\FundingRows}{%\n" + "\n".join(frows) + "}\n")
M["NTrain"] = f"{A['n_train']:,}"; M["NTrainCells"] = f"{A['cells_train']:,}"
M["CoefMspread"] = f(A["coef_main_raw"]["mspread"]); M["CoefVA"] = f(abs(A["coef_main_raw"]["agency_V"]))
for grp, gk in [("depository", "Dep"), ("nonbank", "Nb")]:
    for per in PER:
        M[f"WVar{gk}{PK[per]}"] = f(A[f"wvar_{grp}_{per}"]); M[f"NegSp{gk}{PK[per]}"] = f(100 * A[f"neg_mspread_{grp}_{per}"], 0); M[f"ShFb{gk}{PK[per]}"] = f(100 * A[f"share_forb_{grp}_{per}"], 0)
for nm, key in [("r_main", "Main"), ("r_t10", "TTen"), ("r_gb", "GB"), ("r_dep", "DepIdx"), ("r_pool", "Pool"), ("mspread", "SP")]:
    for grp, gk in [("depository", "Dep"), ("nonbank", "Nb")]:
        for yr in [2019, 2020, 2021, 2022]:
            M[f"AUC{key}{gk}{['Nineteen','Twenty','TwentyOne','TwentyTwo'][yr-2019]}"] = f(A[f"{nm}_{grp}_{yr}"])
try:
    ST = json.load(open(OUT / "stock_results.json")); sp = pd.DataFrame(ST["periods"]).set_index(["itype", "period"])
    M["StockN"] = f"{ST['N_loan_months']:,}"
    for it, k in [("nonbank", "Nb"), ("depository", "Dep")]:
        for per in PER:
            r = sp.loc[(it, per)]; pk = PK[per]
            M[f"St{k}{pk}M"] = f(100 * r.m, 1); M[f"St{k}{pk}Rel"] = f(r.rel); M[f"St{k}{pk}RelPred"] = f(r.rel_pred); M[f"St{k}{pk}RelRand"] = f(r.rel_random)
        for per in PER[1:]:
            M[f"St{k}BetaD{PK[per]}"] = f(ST[f"{it}_felogit"][f"d_{per}"]); M[f"St{k}BetaD{PK[per]}SE"] = f(ST[f"{it}_felogit"][f"se_{per}"])
    rows = []
    for it, lab in [("nonbank", "Nonbanks"), ("depository", "Depositories")]:
        rows.append(r"\multicolumn{7}{l}{\emph{" + lab + r"}}\\")
        for nm, key in [("Monthly exercise rate (\\%)", "m"), ("Relative response", "rel"), ("\\quad Fixed ranking", "rel_pred"), ("\\quad Random rationing", "rel_random")]:
            vals = [f(100 * sp.loc[(it, p), key], 1) if key == "m" else f(sp.loc[(it, p), key]) for p in PER]
            rows.append(nm + " & " + " & ".join(vals) + r"\\")
        vals = ["--"] + [f"{f(ST[f'{it}_felogit'][f'd_{p}'])} ({f(ST[f'{it}_felogit'][f'se_{p}'])})" for p in PER[1:]]
        rows.append("Change in latent weight & " + " & ".join(vals) + r"\\")
    open(TABLES / "stock_rows.tex", "w").write("\\newcommand{\\StockRows}{%\n" + "\n".join(rows) + "}\n")
except FileNotFoundError:
    pass
# ---------- Table 3 rows ----------
def panel(it, k):
    rows = []
    fl = R[f"{it}_felogit"]
    Nrow = [f"{int(pd.read_csv(OUT / f'cells_{it}.csv').query('period==@p').n.sum()):,}" for p in PER]
    rows.append("Options & " + " & ".join(Nrow) + r"\\")
    rows.append(r"Funded share (\%) & " + " & ".join(M[f"{k}{PK[p]}M"] for p in PER) + r"\\")
    rows.append(r"Latent weight on index$^{a}$ & " + f"{M[f'{k}BetaPre']} ({M[f'{k}BetaPreSE']})" + " & " + " & ".join(f"{M[f'{k}BetaD{PK[p]}']} ({M[f'{k}BetaD{PK[p]}SE']})" for p in PER[1:]) + r"\\")
    rows.append("Level response & " + " & ".join(M[f"{k}{PK[p]}Lvl"] for p in PER) + r"\\")
    rows.append("Relative response & " + " & ".join(M[f"{k}{PK[p]}Rel"] for p in PER) + r"\\")
    rows.append(r"\quad Fixed ranking & " + " & ".join(M[f"{k}{PK[p]}RelPred"] for p in PER) + r"\\")
    rows.append(r"\quad Random rationing & " + " & ".join(M[f"{k}{PK[p]}RelRand"] for p in PER) + r"\\")
    rows.append(r"\quad Actual minus fixed & -- & " + " & ".join(f"{M[f'{k}{PK[p]}GapFix']} ({M[f'{k}{PK[p]}GapFixSE']})" for p in PER[1:]) + r"\\")
    rows.append(r"\quad Actual minus random & -- & " + " & ".join(f"{M[f'{k}{PK[p]}GapRand']} ({M[f'{k}{PK[p]}GapRandSE']})" for p in PER[1:]) + r"\\")
    rows.append(r"\quad Same, change from pre & -- & " + " & ".join(f"{M[f'{k}{PK[p]}DidRand']} ({M[f'{k}{PK[p]}DidRandSE']})" for p in PER[1:]) + r"\\")
    return "\n".join(rows)
open(TABLES / "empirics_rows.tex", "w").write("\\newcommand{\\NbRows}{%\n" + panel("nonbank", "Nb") + "}\n\\newcommand{\\DepRows}{%\n" + panel("depository", "Dep") + "}\n")
# ---------- queue rows ----------
qrows = []
for per in ["pre", "freeze", "release", "recovery1"]:
    pk = PK[per]; lab = {"pre": "3/2019--2/2020", "freeze": "3/2020--6/2020", "release": "7/2020--12/2020", "recovery1": "1/2021--6/2021"}[per]
    qrows.append(f"{lab} & {M[f'NbQ{pk}N']} & {M[f'NbQ{pk}Rate']} & {M[f'NbQ{pk}Beta']} ({M[f'NbQ{pk}SE']}) & {M[f'DepQ{pk}N']} & {M[f'DepQ{pk}Rate']} & {M[f'DepQ{pk}Beta']} ({M[f'DepQ{pk}SE']})" + r"\\")
open(TABLES / "queue_rows.tex", "w").write("\\newcommand{\\QueueRows}{%\n" + "\n".join(qrows) + "}\n")
# ---------- robustness rows ----------
try:
    rob = pd.read_csv(OUT / "robust.csv"); rr = []
    for _, r in rob.iterrows():
        rr.append(f"{r.label} & {f(r.beta_pre)} & {f(r.d_freeze)} ({f(r.se_d_freeze)}) & {f(100*r.freeze_m,1)} & {f(r.pre_rel)} & {f(r.freeze_rel)} & "
                  f"{f(r.freeze_rel-r.freeze_rel_pred)} ({f(r.se_gap_fixed)}) & {f(r.freeze_rel-r.freeze_rel_random)} ({f(r.se_gap_random)})" + r"\\")
    open(TABLES / "robust_rows.tex", "w").write("\\newcommand{\\RobustRows}{%\n" + "\n".join(rr) + "}\n")
    M["RobSEFix"] = f(rob.iloc[0].se_gap_fixed); M["RobSERand"] = f(rob.iloc[0].se_gap_random)
    for i, key in enumerate(["Base", "IM", "TTenX", "GBx", "DepX", "PoolX", "SPx", "Twelve", "ExBig", "Tech", "NoFb"]):
        r = rob.iloc[i]; M[f"Rob{key}Rel"] = f(r.freeze_rel); M[f"Rob{key}Pred"] = f(r.freeze_rel_pred); M[f"Rob{key}Rand"] = f(r.freeze_rel_random)
        M[f"Rob{key}M"] = f(100 * r.freeze_m, 1); M[f"Rob{key}GapFix"] = f(r.freeze_rel - r.freeze_rel_pred); M[f"Rob{key}GapRand"] = f(r.freeze_rel - r.freeze_rel_random)
        M[f"Rob{key}GapFixSE"] = f(r.se_gap_fixed); M[f"Rob{key}GapRandSE"] = f(r.se_gap_random)
except FileNotFoundError:
    pass
with open(TABLES / "emp_macros.tex", "w") as fh:
    for k, v in M.items(): fh.write(f"\\newcommand{{\\e{k}}}{{{v}}}\n")
print("wrote", len(M), "macros")
