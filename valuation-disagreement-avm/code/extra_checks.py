"""Descriptive statistics and four pre-specified supplementary checks for the journal manuscript.

Run once on the licensed file, then recompile the manuscript:
    python code/extra_checks.py /path/to/FA_Luxury.csv code/public_county_2018.csv
    pdflatex manuscript.tex ; pdflatex manuscript.tex

Writes (all read automatically by manuscript.tex when present):
    descriptive_table.tex   Table: summary statistics
    extra_table.tex         Table: supplementary checks
    extra_macros.tex        numbers quoted in the text
    code/extra_results.json machine-readable outputs
Variable construction, samples, learner settings and folds are identical to strengthen.py.
The four checks were specified before being run; every result is written regardless of sign.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import sys, json, hashlib, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import pyfixest as pf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
src = Path(sys.argv[1]); county = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'public_county_2018.csv'
d = pd.read_csv(src, dtype=str, low_memory=False); d.columns = d.columns.str.strip()
num = lambda s: pd.to_numeric(s.str.replace(r'[$,\s]', '', regex=True), errors='coerce')
for c in ['CurrentListingPrice', 'FACurrentAVM', 'Low_Value', 'High_Value', 'HomeSize', 'LotSizeSqFt', 'YearBuilt', 'ConfidenceScore', 'DOM']:
    d[c] = num(d[c])
w = lambda x: x.clip(x.quantile(.01), x.quantile(.99))
P, A = d.CurrentListingPrice, d.FACurrentAVM; lr = np.log(P / A)
d['y'] = d.Status.isin(['Pending', 'Contingent']).astype(int); d['gap'] = lr
d['prem'] = w(lr.clip(lower=0)); d['disc'] = w((-lr).clip(lower=0)); d['conf10'] = d.ConfidenceScore / 10
d['hiconf'] = (d.ConfidenceScore >= 80).astype(int); d['lp'] = w(np.log(P))
for v, c in [('lsq', 'HomeSize'), ('llot', 'LotSizeSqFt')]:
    x = np.log(d[c].where(d[c] > 0)); d[v + '_m'] = x.isna().astype(int); d[v] = w(x).fillna(0)
age = 2019 - d.YearBuilt; d['age_m'] = age.isna().astype(int); d['age'] = w(age).fillna(0)
d['age2020'] = w(2020 - d.YearBuilt).fillna(0)
d['city'] = d['CITY/STATE'].fillna('NA'); d['ptype'] = d.PropertyType.fillna('NA'); d['county'] = d.FIPS.str.zfill(5); d['state'] = d.PropertyState
ld = pd.to_datetime(d.ListingDate, errors='coerce'); up = pd.to_datetime(d.FA_UpdateTimeStamp, errors='coerce'); av = pd.to_datetime(d.ValuationDate, errors='coerce')
ref = pd.Timestamp('2020-02-04'); d['month'] = ld.dt.to_period('M').astype(str)
d['recency'] = (ref - up).dt.days; d['listing_age'] = (ref - ld).dt.days; d['avm_lag'] = (ref - av).dt.days
d['logwidth'] = np.log(d.High_Value / d.Low_Value); d['plaus'] = (A / P).between(.5, 2)
R = {'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest()}

# ---------------- Descriptive statistics ----------------
full = d[d.ConfidenceScore.notna() & np.isfinite(d.gap) & np.isfinite(d.logwidth)].copy()
pl = d[d.plaus & d.ConfidenceScore.notna()].copy()
rows = [('Asking price (\\$000s)', lambda z: z.CurrentListingPrice / 1000), ('AVM value (\\$000s)', lambda z: z.FACurrentAVM / 1000),
        ('Log asking price minus log AVM', lambda z: z.gap), ('Asking price above AVM (share)', lambda z: (z.gap > 0).astype(float)),
        ('AVM confidence score', lambda z: z.ConfidenceScore), ('Confidence at least 80 (share)', lambda z: z.hiconf.astype(float)),
        ('Log high/low valuation range', lambda z: z.logwidth), ('Living area (sq. ft.)', lambda z: z.HomeSize.where(z.HomeSize > 0)),
        ('Lot size (sq. ft.)', lambda z: z.LotSizeSqFt.where(z.LotSizeSqFt > 0)), ('Age in 2020 (years)', lambda z: (2020 - z.YearBuilt)),
        ('Days since listing', lambda z: z.listing_age), ('Days since record update', lambda z: z.recency), ('Days since AVM', lambda z: z.avm_lag),
        ('Pending or contingent (share)', lambda z: z.y.astype(float))]
desc = []
for lab, f in rows:
    x = f(full); lo = f(pl[pl.hiconf == 0]).mean(); hi = f(pl[pl.hiconf == 1]).mean()
    desc.append({'label': lab, 'n': int(x.notna().sum()), 'mean': float(x.mean()), 'sd': float(x.std()), 'p25': float(x.quantile(.25)),
                 'p50': float(x.median()), 'p75': float(x.quantile(.75)), 'mean_low': float(lo), 'mean_high': float(hi)})
R['plausible_base_rate'] = float(pl.y.mean())
R['descriptives'] = {'full_n': len(full), 'plausible_n': len(pl), 'low_n': int((pl.hiconf == 0).sum()), 'high_n': int((pl.hiconf == 1).sum()), 'rows': desc}

def fmt(v):
    a = abs(v)
    return f"{v:,.0f}" if a >= 1000 else (f"{v:,.1f}" if a >= 10 else f"{v:.3f}")
t = r'''\begin{table}[tbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Descriptive statistics}\label{tab:desc}
\begin{tabular}{lrrrrrrr}\toprule
 & \multicolumn{5}{c}{Full confidence sample} & \multicolumn{2}{c}{Plausible-AVM means}\\\cmidrule(lr){2-6}\cmidrule(lr){7-8}
Variable & Mean & SD & P25 & Median & P75 & Conf.\ $<80$ & Conf.\ $\geq80$\\\midrule
'''
for r in desc:
    t += f"{r['label']} & {fmt(r['mean'])} & {fmt(r['sd'])} & {fmt(r['p25'])} & {fmt(r['p50'])} & {fmt(r['p75'])} & {fmt(r['mean_low'])} & {fmt(r['mean_high'])}" + r'\\' + '\n'
t += r'''\bottomrule\end{tabular}
\par\smallskip\parbox{.97\textwidth}{\footnotesize The full confidence sample has ''' + f"{len(full):,}" + r''' records; the plausible-AVM sample ($0.5\leq A/P\leq2$) has ''' + f"{len(pl):,}" + r''' records, ''' + f"{int((pl.hiconf==0).sum()):,}" + r''' with confidence below 80 and ''' + f"{int((pl.hiconf==1).sum()):,}" + r''' at or above 80. Physical attributes are summarized over nonmissing positive values. Day counts are measured to the February 4, 2020 reference date.}
\end{table}
'''
(ROOT.parent / 'descriptive_table.tex').write_text(t)

# ---------------- Check 1: continuous confidence ----------------
C = 'lp + lsq + llot + age + lsq_m + llot_m + age_m'
pl['cc'] = pl.conf10 - pl.conf10.mean()
m = pf.feols('y ~ prem + disc + prem:cc + disc:cc + cc + ' + C + ' | city + ptype + month', data=pl, vcov={'CRV1': 'city'}, fixef_rm='none', fixef_maxiter=100000)
tt = m.tidy()
R['continuous_confidence'] = {'n': int(m._N), 'prem_x_conf10': float(tt.loc['prem:cc', 'Estimate']), 'se_prem_x_conf10': float(tt.loc['prem:cc', 'Std. Error']),
                              'disc_x_conf10': float(tt.loc['disc:cc', 'Estimate']), 'se_disc_x_conf10': float(tt.loc['disc:cc', 'Std. Error'])}

# ---------------- Check 2: price tiers ----------------
tiers = [('\\$0.5--1.0 million', 5e5, 1e6), ('\\$1.0--2.0 million', 1e6, 2e6), ('above \\$2.0 million', 2e6, np.inf)]
R['price_tiers'] = []
for lab, a, b in tiers:
    z = pl[(pl.CurrentListingPrice >= a) & (pl.CurrentListingPrice < b)]
    mm = pf.feols('y ~ prem + disc + prem:hiconf + disc:hiconf + hiconf + ' + C + ' | city + ptype + month', data=z, vcov={'CRV1': 'city'}, fixef_rm='none', fixef_maxiter=100000)
    q = mm.tidy()
    R['price_tiers'].append({'tier': lab, 'n': int(mm._N), 'prem_x_high': float(q.loc['prem:hiconf', 'Estimate']), 'se': float(q.loc['prem:hiconf', 'Std. Error'])})

# ---------------- Checks 3 and 4: recency decomposition and top-decile lift ----------------
s = full.reset_index(drop=True)
X = pd.DataFrame({'logprice': np.log(s.CurrentListingPrice), 'loghome': np.log(s.HomeSize.where(s.HomeSize > 0)), 'loglot': np.log(s.LotSizeSqFt.where(s.LotSizeSqFt > 0)),
                  'age': s.age2020, 'listing_age': s.listing_age, 'lat': pd.to_numeric(s.SitusLatitude, errors='coerce'), 'lon': pd.to_numeric(s.SitusLongitude, errors='coerce')})
for k in ['state', 'ptype']:
    zz = s[k]; zz = zz.where(zz.map(zz.value_counts()) >= 100, 'OTHER'); X[k] = zz.astype('category').cat.codes
cm = pd.read_csv(county, dtype={'county': str}).set_index('county')
for v in cm.columns: X[v] = s.county.map(cm[v])
base = list(X)
X['gap'] = s.gap; X['confidence'] = s.conf10; X['logwidth'] = s.logwidth; X['report_age'] = s.recency; X['avm_age'] = s.avm_lag
sets = {'gap': base + ['gap'], 'full': base + ['gap', 'confidence', 'logwidth'],
        'gap_avmage': base + ['gap', 'avm_age'], 'full_avmage': base + ['gap', 'confidence', 'logwidth', 'avm_age'],
        'gap_reportage': base + ['gap', 'report_age'], 'full_reportage': base + ['gap', 'confidence', 'logwidth', 'report_age']}
Y = s.y.to_numpy(); g = s.county.to_numpy(); pred = {k: np.zeros(len(s)) for k in sets}
for tr, te in GroupKFold(5).split(X, Y, g):
    for name, cols in sets.items():
        cat = [cols.index(k) for k in ['state', 'ptype']]
        mdl = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, min_samples_leaf=100, learning_rate=.05, l2_regularization=1,
                                             categorical_features=cat, early_stopping=False, random_state=71)
        mdl.fit(X.iloc[tr][cols], Y[tr]); pred[name][te] = mdl.predict_proba(X.iloc[te][cols])[:, 1]
B = pd.DataFrame({'county': g})
for k, p in pred.items():
    p = np.clip(p, 1e-7, 1 - 1e-7); B[k] = -(Y * np.log(p) + (1 - Y) * np.log(1 - p))
Agg = B.groupby('county').sum(); rng = np.random.default_rng(31217); R['recency_decomposition'] = {}
for x, y in [('gap', 'full'), ('gap_avmage', 'full_avmage'), ('gap_reportage', 'full_reportage')]:
    bs = [100 * (1 - (z := Agg.iloc[rng.integers(0, len(Agg), len(Agg))])[y].sum() / z[x].sum()) for _ in range(2000)]
    R['recency_decomposition'][x + '_to_' + y] = {'logloss_reduction_pct': float(100 * (1 - Agg[y].sum() / Agg[x].sum())), 'ci95': [float(v) for v in np.quantile(bs, [.025, .975])]}
R['confidence_vs_age_spearman'] = {'confidence_avm_age': float(spearmanr(s.conf10, s.avm_lag, nan_policy='omit').statistic),
                                   'confidence_report_age': float(spearmanr(s.conf10, s.recency, nan_policy='omit').statistic),
                                   'width_avm_age': float(spearmanr(s.logwidth, s.avm_lag, nan_policy='omit').statistic)}
base_rate = float(Y.mean()); R['top_decile'] = {'base_rate': base_rate}
for k in ['gap', 'full']:
    cut = np.quantile(pred[k], .9); top = pred[k] >= cut
    R['top_decile'][k] = {'share_positive_in_top_decile': float(Y[top].mean()), 'lift': float(Y[top].mean() / base_rate),
                          'share_of_all_positives_captured': float(Y[top].sum() / Y.sum())}

(ROOT / 'extra_results.json').write_text(json.dumps(R, indent=2))

# ---------------- Supplementary table and text macros ----------------
cc = R['continuous_confidence']; rd = R['recency_decomposition']; td = R['top_decile']
e = r'''\begin{table}[tbp]\centering\small
\caption{Supplementary checks}\label{tab:extra}
\begin{tabular}{lrrr}\toprule
\multicolumn{4}{l}{\textit{Panel A. Premium $\times$ confidence interaction}}\\
Specification & Estimate & SE & $N$\\\midrule
''' + f"Continuous confidence (per 10 points) & {cc['prem_x_conf10']:.4f} & {cc['se_prem_x_conf10']:.4f} & {cc['n']:,}" + r'\\' + '\n'
for r in R['price_tiers']:
    e += f"Premium $\\times$ high, asking price {r['tier']} & {r['prem_x_high']:.4f} & {r['se']:.4f} & {r['n']:,}" + r'\\' + '\n'
e += r'''\midrule
\multicolumn{4}{l}{\textit{Panel B. Log-loss gain from uncertainty metadata (\%), county-held-out}}\\
Both models also include & Gain & \multicolumn{2}{r}{95\% interval}\\\midrule
'''
for k, lab in [('gap_to_full', 'Neither age'), ('gap_avmage_to_full_avmage', 'AVM age only'), ('gap_reportage_to_full_reportage', 'Report age only')]:
    e += f"{lab} & {rd[k]['logloss_reduction_pct']:.2f} & \\multicolumn{{2}}{{r}}{{[{rd[k]['ci95'][0]:.2f}, {rd[k]['ci95'][1]:.2f}]}}" + r'\\' + '\n'
e += r'''\midrule
\multicolumn{4}{l}{\textit{Panel C. Top decile of held-out scores}}\\
Model & Pending share & Lift & Positives captured\\\midrule
'''
for k, lab in [('gap', 'Gap model'), ('full', 'Gap plus metadata')]:
    e += f"{lab} & {td[k]['share_positive_in_top_decile']:.3f} & {td[k]['lift']:.2f} & {td[k]['share_of_all_positives_captured']:.3f}" + r'\\' + '\n'
e += r'''\bottomrule\end{tabular}
\par\smallskip\parbox{.97\textwidth}{\footnotesize Panel A uses the plausible-AVM sample with city, property-type and listing-month effects and city-clustered standard errors. Panel B uses the same learner, folds and 2,000-draw county bootstrap as Table~\ref{tab:gains}. Panel C reports the pending-or-contingent share among the 10 percent of records with the highest held-out scores, its ratio to the base rate of ''' + f"{base_rate:.3f}" + r''', and the share of all positive records in that decile.}
\end{table}
'''
(ROOT.parent / 'extra_table.tex').write_text(e)
sp = R['confidence_vs_age_spearman']
mac = '\n'.join([r'\newcommand{\XccEst}{' + f"{cc['prem_x_conf10']:.4f}" + '}', r'\newcommand{\XccSE}{' + f"{cc['se_prem_x_conf10']:.4f}" + '}',
                 r'\newcommand{\XavmGain}{' + f"{rd['gap_avmage_to_full_avmage']['logloss_reduction_pct']:.2f}" + '}',
                 r'\newcommand{\XrepGain}{' + f"{rd['gap_reportage_to_full_reportage']['logloss_reduction_pct']:.2f}" + '}',
                 r'\newcommand{\XrhoAvm}{' + f"{sp['confidence_avm_age']:.2f}" + '}', r'\newcommand{\XrhoRep}{' + f"{sp['confidence_report_age']:.2f}" + '}',
                 r'\newcommand{\XplBase}{' + f"{100 * R['plausible_base_rate']:.2f}" + '}', r'\newcommand{\XliftGap}{' + f"{td['gap']['lift']:.2f}" + '}', r'\newcommand{\XliftFull}{' + f"{td['full']['lift']:.2f}" + '}'])
(ROOT.parent / 'extra_macros.tex').write_text(mac + '\n')
print(json.dumps(R, indent=2))
