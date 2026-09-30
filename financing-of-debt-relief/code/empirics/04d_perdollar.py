"""Rationing by value per dollar (Proposition 1 with heterogeneous cash needs). The cash a repurchase
requires is the loan's unpaid balance. As the shadow value of cash rises, the cutoff g_i > lambda*l_i
tilts against high-balance loans, holding value fixed: the weight on log balance in the exercise
logit should fall at constrained issuers when their budgets contract, and not at unconstrained ones.
Two versions: (i) period interactions (the common 2020 shock and its reversal); (ii) the issuer's own
log advance burden, with index-by-month and balance-by-month interactions so that the advance terms
are identified from cross-issuer differences within a month."""
import pandas as pd, numpy as np, json
from felogit import fe_logit
from config import WORK, OUT, ISSUER_MONTHLY, PERIODS
PER = [p for p, _, _ in PERIODS]
d = pd.read_parquet(WORK / "scored.parquet", columns=["issuer_id", "itype", "ym", "period", "x", "forbear", "r_main", "upb"])
d = d[d.itype.isin(["nonbank", "depository"]) & d.ym.between(201901, 202209) & d.period.isin(PER)].copy()
d["r"] = (d.r_main - d.r_main.mean()) / d.r_main.std()
d["lb"] = np.log(d.upb.clip(lower=1000)); d["lb"] = (d.lb - d.lb.mean()) / d.lb.std()
im = pd.read_csv(ISSUER_MONTHLY); im["ym"] = pd.to_datetime(im.as_of_date).dt.strftime("%Y%m").astype(int)
im["adv"] = im.tot_port_advance / im.port_upb * 100
d = d.merge(im[["issuer_id", "ym", "adv"]], on=["issuer_id", "ym"], how="left").dropna(subset=["adv"])
d["cell"] = d.issuer_id.astype(str) + "_" + d.ym.astype(str) + "_" + d.forbear.astype(int).astype(str)
R = {}
for it in ["nonbank", "depository"]:
    g = d[d.itype == it]
    cols = [g.r.values, g.lb.values] + [(g.r * (g.period == p)).values for p in PER[1:]] + [(g.lb * (g.period == p)).values for p in PER[1:]]
    b, se, N, C, *_ = fe_logit(g.x.values, np.column_stack(cols), g.cell.values, g.issuer_id.values)
    R[f"{it}_periods"] = {"r_pre": float(b[0]), "se_r_pre": float(se[0]), "lb_pre": float(b[1]), "se_lb_pre": float(se[1]),
                          **{f"r_{p}": float(b[2 + i]) for i, p in enumerate(PER[1:])}, **{f"se_r_{p}": float(se[2 + i]) for i, p in enumerate(PER[1:])},
                          **{f"lb_{p}": float(b[7 + i]) for i, p in enumerate(PER[1:])}, **{f"se_lb_{p}": float(se[7 + i]) for i, p in enumerate(PER[1:])}, "n": int(N)}
    print(it, "periods", {k: round(v, 3) for k, v in R[f"{it}_periods"].items() if k.startswith(("lb", "se_lb"))}, flush=True)
    la = np.log(g.adv.clip(lower=0.01)); ad = (la - la.groupby(g.issuer_id).transform("mean")).values
    D = pd.get_dummies(g.ym, drop_first=True).astype(float).values
    Z = np.column_stack([g.r.values, g.lb.values, D * g.r.values[:, None], D * g.lb.values[:, None], g.r.values * ad, g.lb.values * ad])
    b2, se2, *_ = fe_logit(g.x.values, Z, g.cell.values, g.issuer_id.values)
    R[f"{it}_advance"] = {"r_adv": float(b2[-2]), "se_r_adv": float(se2[-2]), "lb_adv": float(b2[-1]), "se_lb_adv": float(se2[-1]), "adv_sd": float(ad.std())}
    print(it, "advance", R[f"{it}_advance"], flush=True)
    # (iii) the freeze split into March-April, when cells cannot separate loans in forbearance from the
    # rest, and May-June, when the forbearance flag is observed and enters the cell definition.
    sub = np.where(g.ym.between(202003, 202004), "frz_early", np.where(g.ym.between(202005, 202006), "frz_late", g.period))
    PS = ["frz_early", "frz_late"] + PER[2:]
    cols = [g.r.values, g.lb.values] + [(g.r * (sub == p)).values for p in PS] + [(g.lb * (sub == p)).values for p in PS]
    b3, se3, *_ = fe_logit(g.x.values, np.column_stack(cols), g.cell.values, g.issuer_id.values)
    k = len(PS)
    R[f"{it}_freeze_split"] = {"lb_pre": float(b3[1]), "se_lb_pre": float(se3[1]),
                               **{f"lb_{p}": float(b3[2 + k + i]) for i, p in enumerate(PS)}, **{f"se_lb_{p}": float(se3[2 + k + i]) for i, p in enumerate(PS)},
                               **{f"r_{p}": float(b3[2 + i]) for i, p in enumerate(PS)}, **{f"se_r_{p}": float(se3[2 + i]) for i, p in enumerate(PS)}}
    print(it, "freeze split", {k_: round(v, 3) for k_, v in R[f"{it}_freeze_split"].items() if "frz" in k_}, flush=True)
json.dump(R, open(OUT / "perdollar.json", "w"), indent=1)
