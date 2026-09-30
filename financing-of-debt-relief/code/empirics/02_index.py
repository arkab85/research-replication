"""Ranking index. By Proposition 1 a constrained intermediary funds the top share of its ranking of
opportunities, so its ranking is revealed by which options it exercises within a decision cell,
whatever the size of its budget. The main index is a conditional logit of exercise on hard loan
characteristics, estimated on the within-issuer-month variation of nonbank decisions on options that
vested in 2016-2018 (issuer-month dummies absorb each cell's budget, so only the ranking is learned),
and applied out of time to 2019-2022. The coupon enters as the note rate minus the contemporaneous
30-year mortgage rate, the margin that governs the redelivery value of a repurchased loan. Features
are standardized and imputed on the training sample. Alternatives: the same logit with the spread
over the ten-year Treasury rate entered piecewise linearly, a gradient-boosted ranker fit the same
way (cell log-odds as offsets), the conditional logit fit to depository decisions, a boosted classifier
fit to pooled depository decisions without offsets, and the mortgage spread alone."""
from config import WORK, OUT
import pandas as pd, numpy as np, time, json, scipy.sparse as sp
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
df = pd.read_parquet(WORK / "panel.parquet")
df = df[df.itype.isin(["depository", "nonbank", "techfirst"])].copy()
df["cell"] = df.issuer_id.astype(str) + "_" + df.ym.astype(str)

TRAIN = (df.itype == "nonbank") & df.year.between(2016, 2018)
CS_MED = float(df.loc[TRAIN, "credit_score"].median()); SP_MED = float(df.loc[TRAIN, "spread"].median())
def feats(d, coupon="mspread"):
    F = pd.DataFrame(index=d.index)
    if coupon == "mspread":
        F["mspread"] = d.mspread.clip(-4, 6)
    else:
        s = d.spread.fillna(SP_MED).clip(-2, 8); F["spread"] = s
        for k in [2.0, 3.0, 4.0]: F[f"spread_gt{k:g}"] = np.maximum(s - k, 0)
    F["T10_change"] = d.T10_change.fillna(0)
    F["log_age"] = np.log1p(d.loan_age.clip(lower=0))
    F["log_upb"] = np.log(d.upb.clip(lower=1000))
    cs = d.credit_score; F["credit_score"] = cs.fillna(CS_MED) / 100; F["credit_missing"] = cs.isna().astype(float)
    F["ltv_current"] = d.ltv_current.clip(0, 200).fillna(100) / 100
    F["judicial"] = d.Judicial.astype(float)
    for k in ["V", "R"]: F[f"agency_{k}"] = (d.agency == k).astype(float)
    for k in ["M", "C"]: F[f"issue_{k}"] = (d.issue_type == k).astype(float)
    F["first_time"] = d.first_timer.eq("Y").astype(float)
    F["refi"] = (d.loan_purpose == 2).astype(float); F["modified"] = d.loan_purpose.isin([3, 4]).astype(float)
    F["prior_missed"] = d.cum_missed_pmt_12.clip(0, 12).astype(float)
    F["annual_mip"] = d.annual_mip.fillna(0).astype(float)
    return F
def standardize(F):
    mu = F[TRAIN].mean(); sd = F[TRAIN].std().replace(0, 1); return (F - mu) / sd
F = feats(df, "mspread"); Fs = standardize(F)
F2 = feats(df, "t10"); Fs2 = standardize(F2)

def clogit(mask, Fs):
    tr = df[mask]; m = tr.groupby("cell").x.transform("mean"); idx = tr[(m > 0) & (m < 1)].index
    codes = pd.factorize(df.cell[idx])[0]
    D = sp.csr_matrix((np.ones(len(idx)), (np.arange(len(idx)), codes)))
    X = sp.hstack([sp.csr_matrix(Fs.loc[idx].values), D]).tocsr()
    lr = LogisticRegression(C=10, max_iter=5000, fit_intercept=False).fit(X, df.x[idx])
    return lr.coef_[0][:Fs.shape[1]], int(len(idx)), int(codes.max() + 1)

def lgb_ranker(mask, offset=True):
    Xg = F.copy(); Xg["state_c"] = df.state.astype("category").cat.codes
    tr = df[mask]; m = tr.groupby("cell").x.transform("mean")
    ok = ((m > 0) & (m < 1)) if offset else pd.Series(True, index=tr.index)
    idx = tr[ok].index; m = m[ok]
    init = np.log(m.clip(.005, .995) / (1 - m.clip(.005, .995))).values if offset else None
    rng = np.random.default_rng(7); cells = df.cell[idx].unique(); vc = set(rng.choice(cells, int(.15 * len(cells)), replace=False))
    v = df.cell[idx].isin(vc).values
    P = dict(objective="binary", num_leaves=15, min_data_in_leaf=1000, learning_rate=0.03, lambda_l2=5.0,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=7, verbose=-1, num_threads=2)
    dtr = lgb.Dataset(Xg.loc[idx[~v]], df.x[idx[~v]], init_score=None if init is None else init[~v], categorical_feature=["state_c"], free_raw_data=False)
    dva = lgb.Dataset(Xg.loc[idx[v]], df.x[idx[v]], init_score=None if init is None else init[v], categorical_feature=["state_c"], reference=dtr)
    bst = lgb.train(P, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return bst.predict(Xg, raw_score=True, num_iteration=bst.best_iteration)

tr_nb = TRAIN
tr_dep = (df.itype == "depository") & df.year.between(2016, 2018)
t = time.time()
b_nb, n_nb, c_nb = clogit(tr_nb, Fs); df["r_main"] = Fs.values @ b_nb
b_t10, _, _ = clogit(tr_nb, Fs2); df["r_t10"] = Fs2.values @ b_t10
b_dep, n_dep, c_dep = clogit(tr_dep, Fs); df["r_dep"] = Fs.values @ b_dep
df["r_gb"] = lgb_ranker(tr_nb, True)
df["r_pool"] = lgb_ranker(tr_dep, False)
print("fit", round(time.time() - t, 1))
res = {"n_train": n_nb, "cells_train": c_nb, "n_train_dep": n_dep,
       "coef_main": {k: round(float(v), 4) for k, v in zip(F.columns, b_nb)},
       "coef_main_raw": {k: round(float(v / F[TRAIN].std().replace(0, 1)[k]), 4) for k, v in zip(F.columns, b_nb)},
       "coef_t10": {k: round(float(v), 4) for k, v in zip(F2.columns, b_t10)},
       "coef_dep": {k: round(float(v), 4) for k, v in zip(F.columns, b_dep)}}
def wauc(sub, col):
    vals = [(len(gc), roc_auc_score(gc.x, gc[col].fillna(0))) for c, gc in sub.groupby("cell") if len(gc) >= 50 and gc.x.nunique() == 2]
    a = np.array(vals); return round(float(np.average(a[:, 1], weights=a[:, 0])), 4)
for nm in ["r_main", "r_t10", "r_gb", "r_dep", "r_pool", "mspread", "spread"]:
    for grp in ["depository", "nonbank"]:
        for yr in [2019, 2020, 2021, 2022]:
            res[f"{nm}_{grp}_{yr}"] = wauc(df[(df.itype == grp) & (df.year == yr)], nm)
# within-cell variance of the standardized main index by period and type (the dispersion the responses scale with)
z = (df.r_main - df.r_main[df.itype.isin(["depository", "nonbank"]) & df.ym.between(201901, 202209)].mean()) / df.r_main[df.itype.isin(["depository", "nonbank"]) & df.ym.between(201901, 202209)].std()
for grp in ["depository", "nonbank"]:
    for per in ["pre", "freeze", "release", "recovery1", "recovery2", "ratehike"]:
        sub = df[(df.itype == grp) & (df.period == per)]
        cc = sub.cell + "_" + sub.forbear.astype(int).astype(str)
        res[f"wvar_{grp}_{per}"] = round(float((z[sub.index] - z[sub.index].groupby(cc).transform("mean")).pow(2).mean()), 4)
        res[f"share_forb_{grp}_{per}"] = round(float(sub.forbear.mean()), 4)
        res[f"neg_mspread_{grp}_{per}"] = round(float((sub.mspread < 0).mean()), 4)
print(json.dumps(res, indent=1))
keep = ["identifier", "pool_id", "ym", "year", "period", "issuer_id", "itype", "x", "x12", "x_ever", "t_ex", "t_ever", "forbear", "gone3",
        "foreclosed12", "cured12", "spread", "mspread", "interest_rate", "T10", "pmms", "upb", "state", "agency", "loan_age", "ltv_current", "credit_score",
        "r_main", "r_t10", "r_gb", "r_dep", "r_pool"]
df[keep].to_parquet(WORK / "scored.parquet")
json.dump(res, open(OUT / "index_auc.json", "w"), indent=1)
