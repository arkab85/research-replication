"""Raw alignment pattern and pricing outside the valuation range (descriptive companion analyses).

Run once on the licensed file, after or instead of extra_checks.py, then recompile the manuscript:
    python code/companion_checks.py /path/to/FA_Luxury.csv code/public_county_2018.csv
    pdflatex manuscript.tex ; pdflatex manuscript.tex

Writes (read automatically by manuscript.tex when present):
    alignment_figure.pdf     Figure: pending-or-contingent share by asking-price gap bin and confidence group
    outside_table.tex        Table: pricing above or below the vendor's valuation range
    slopes_table.tex         Table: slope horse race (premium/discount x atypicality, rarity, county
                             market thickness) plus finer geography (ZIP, ZIP x quarter) and alternative
                             plausibility screens
    companion_macros.tex     numbers quoted in the text
    code/companion_results.json
Variable construction and samples are identical to strengthen.py and extra_checks.py.
Both analyses were specified before being run; every result is written regardless of sign.

Doubly robust estimates: within each confidence group (score < 80, >= 80) and for each contrast
(priced above the high valuation bound vs. inside the range; priced below the low bound vs. inside),
the augmented inverse-probability-weighted (AIPW) difference in the probability of pending or
contingent status. Propensity and outcome models are histogram gradient-boosting learners with the
settings of the validation exercise, cross-fitted over five county-blocked folds. Propensities are
clipped to [0.02, 0.98]. Standard errors cluster the influence function by county. The AIPW contrast
does not condition on the signed gap (crossing a bound is nearly determined by the gap and the range
width, so overlap would fail); whether crossing the bound carries information beyond the point gap is
checked with a linear probability model that also contains the premium and discount terms. The estimates are
covariate-adjusted average differences in recorded status, not causal effects of pricing.

Slope horse race: the baseline interaction model with the expanded physical controls of strengthen.py,
adding interactions of premium and discount with standardized moderators (an index of county-relative
size, lot and age atypicality; property-type rarity; 2018 county median days on market; 2018 county
new-to-active listings), each also entered as a main effect. If the confidence gradient reflected
atypicality or market thickness, the premium x high-confidence interaction should shrink once the
premium is allowed to vary with these moderators.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import sys, json, hashlib, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyfixest as pf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
src = Path(sys.argv[1]); county = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'public_county_2018.csv'
d = pd.read_csv(src, dtype=str, low_memory=False); d.columns = d.columns.str.strip()
num = lambda s: pd.to_numeric(s.str.replace(r'[$,\s]', '', regex=True), errors='coerce')
for c in ['CurrentListingPrice', 'FACurrentAVM', 'Low_Value', 'High_Value', 'HomeSize', 'LotSizeSqFt', 'YearBuilt', 'ConfidenceScore', 'DOM']:
    d[c] = num(d[c])
w = lambda x: x.clip(x.quantile(.01), x.quantile(.99))
P, A = d.CurrentListingPrice, d.FACurrentAVM; lr = np.log(P / A)
d['y'] = d.Status.isin(['Pending', 'Contingent']).astype(int); d['gap'] = lr
d['prem'] = w(lr.clip(lower=0)); d['disc'] = w((-lr).clip(lower=0))
d['hiconf'] = (d.ConfidenceScore >= 80).astype(int); d['lp'] = w(np.log(P))
for v, c in [('lsq', 'HomeSize'), ('llot', 'LotSizeSqFt')]:
    x = np.log(d[c].where(d[c] > 0)); d[v + '_m'] = x.isna().astype(int); d[v] = w(x).fillna(0)
age = 2019 - d.YearBuilt; d['age_m'] = age.isna().astype(int); d['age'] = w(age).fillna(0)
d['age2020'] = w(2020 - d.YearBuilt).fillna(0)
d['city'] = d['CITY/STATE'].fillna('NA'); d['ptype'] = d.PropertyType.fillna('NA'); d['county'] = d.FIPS.str.zfill(5); d['state'] = d.PropertyState
ld = pd.to_datetime(d.ListingDate, errors='coerce'); ref = pd.Timestamp('2020-02-04')
d['month'] = ld.dt.to_period('M').astype(str); d['listing_age'] = (ref - ld).dt.days
d['plaus'] = (A / P).between(.5, 2)
d['zip'] = d.PropertyZip.fillna('NA') if 'PropertyZip' in d else 'NA'; d['quarter'] = ld.dt.to_period('Q').astype(str)
# Observed atypicality proxies, identical to strengthen.py (computed without status outcomes).
for v in ['lsq', 'llot', 'age2020']:
    med = d.groupby('county')[v].transform('median'); dev = (d[v] - med).abs(); mad = dev.groupby(d.county).transform('median')
    d['atyp_' + v] = (dev / mad.where(mad > 0)).clip(upper=10).fillna(0)
cts = d.groupby(['county', 'ptype']).y.transform('size'); tot = d.groupby('county').y.transform('size'); d['rarity'] = -np.log(cts / tot)
d['lsq2'] = d.lsq ** 2; d['llot2'] = d.llot ** 2; d['age2'] = d.age2020 ** 2
valid_range = d.Low_Value.notna() & d.High_Value.notna() & (d.Low_Value > 0) & (d.High_Value > d.Low_Value)
d['above'] = (valid_range & (P > d.High_Value)).astype(int)
d['below'] = (valid_range & (P < d.Low_Value)).astype(int)
pl = d[d.plaus & d.ConfidenceScore.notna() & valid_range].copy().reset_index(drop=True)
R = {'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(), 'n_sample': len(pl),
     'base_rate_sample': float(pl.y.mean())}
GROUPS = [(0, 'Confidence $<80$'), (1, 'Confidence $\\geq80$')]

# ---------------- Analysis 1: raw alignment ----------------
edges = [-np.inf, -.20, -.10, -.05, 0, .05, .10, .20, np.inf]
labels = ['$<-20$', '$-20$ to $-10$', '$-10$ to $-5$', '$-5$ to 0', '0 to 5', '5 to 10', '10 to 20', '$>20$']
plain = ['<-20', '-20 to -10', '-10 to -5', '-5 to 0', '0 to 5', '5 to 10', '10 to 20', '>20']
pl['gbin'] = pd.cut(pl.gap, edges, labels=False, right=False)
R['alignment'] = []
for h, _ in GROUPS:
    for b in range(len(labels)):
        z = pl[(pl.hiconf == h) & (pl.gbin == b)]
        n = len(z); p = float(z.y.mean()) if n else float('nan')
        R['alignment'].append({'hiconf': h, 'bin': plain[b], 'n': n, 'share': p,
                               'se': float(np.sqrt(p * (1 - p) / n)) if n else float('nan')})
fig, ax = plt.subplots(figsize=(7.2, 3.8))
xs = np.arange(len(labels))
for (h, lab), mk, off in zip(GROUPS, ['o', 's'], [-.08, .08]):
    rows = [r for r in R['alignment'] if r['hiconf'] == h]
    sh = np.array([r['share'] for r in rows]) * 100; se = np.array([r['se'] for r in rows]) * 100
    ax.errorbar(xs + off, sh, yerr=1.96 * se, marker=mk, capsize=3, lw=1.2,
                label=lab.replace('$<80$', '< 80').replace('$\\geq80$', '≥ 80'))
ax.set_xticks(xs); ax.set_xticklabels(plain, fontsize=8)
ax.set_xlabel('Asking price relative to AVM (log points × 100)'); ax.set_ylabel('Pending or contingent (%)')
ax.axvline(3.5, color='grey', lw=.8, ls=':'); ax.legend(frameon=False, fontsize=8); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT.parent / 'alignment_figure.pdf'); plt.close(fig)

# ---------------- Analysis 2: pricing outside the valuation range ----------------
R['outside_shares'] = {str(h): {'n': int((pl.hiconf == h).sum()), 'above': float(pl.above[pl.hiconf == h].mean()),
                                'below': float(pl.below[pl.hiconf == h].mean())} for h, _ in GROUPS}
X = pd.DataFrame({'lp': pl.lp, 'lsq': pl.lsq, 'llot': pl.llot, 'age': pl.age2020, 'lsq_m': pl.lsq_m, 'llot_m': pl.llot_m,
                  'age_m': pl.age_m, 'listing_age': pl.listing_age,
                  'lat': pd.to_numeric(pl.SitusLatitude, errors='coerce'), 'lon': pd.to_numeric(pl.SitusLongitude, errors='coerce')})
for k in ['state', 'ptype']:
    zz = pl[k].fillna('NA'); zz = zz.where(zz.map(zz.value_counts()) >= 100, 'OTHER'); X[k] = zz.astype('category').cat.codes
cm = pd.read_csv(county, dtype={'county': str}).set_index('county')
for v in cm.columns: X[v] = pl.county.map(cm[v])
cat_idx = [list(X).index('state'), list(X).index('ptype')]
learner = lambda: HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, min_samples_leaf=100, learning_rate=.05, l2_regularization=1,
                                                 categorical_features=cat_idx, early_stopping=False, random_state=71)
CLIP = (.02, .98)

def fit_prob(Xtr, ytr, Xte):
    if ytr.min() == ytr.max():
        return np.full(len(Xte), float(ytr.mean()))
    m = learner(); m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1]

def aipw(mask, tcol, with_gap=False):
    """Cross-fitted AIPW difference in P(Y=1) between treated (tcol==1) and inside-range records within mask."""
    idx = np.where(mask & ((pl[tcol] == 1) | ((pl.above == 0) & (pl.below == 0))))[0]
    Xs = X.iloc[idx].reset_index(drop=True)
    if with_gap:
        Xs['gap'] = pl.gap.to_numpy()[idx]
    T = pl[tcol].to_numpy()[idx]; Y = pl.y.to_numpy()[idx]; G = pl.county.to_numpy()[idx]
    e = np.zeros(len(idx)); m1 = np.zeros(len(idx)); m0 = np.zeros(len(idx))
    for tr, te in GroupKFold(5).split(Xs, T, G):
        e[te] = fit_prob(Xs.iloc[tr], T[tr], Xs.iloc[te])
        t1 = tr[T[tr] == 1]; t0 = tr[T[tr] == 0]
        m1[te] = fit_prob(Xs.iloc[t1], Y[t1], Xs.iloc[te]); m0[te] = fit_prob(Xs.iloc[t0], Y[t0], Xs.iloc[te])
    clipped = float(((e < CLIP[0]) | (e > CLIP[1])).mean()); e = e.clip(*CLIP)
    psi = m1 - m0 + T * (Y - m1) / e - (1 - T) * (Y - m0) / (1 - e)
    est = float(psi.mean()); n = len(psi)
    cl = pd.Series((psi - est) / n).groupby(G).sum(); k = len(cl)
    se = float(np.sqrt(k / (k - 1) * (cl ** 2).sum()))
    raw = float(Y[T == 1].mean() - Y[T == 0].mean())
    return {'n': int(n), 'n_treated': int(T.sum()), 'estimate': est, 'se': se, 'raw_difference': raw,
            'share_propensity_clipped': clipped}, pd.Series((psi - est) / n, index=G)

for key, wg in [('doubly_robust', False)]:
    R[key] = {}
    for tcol in ['above', 'below']:
        R[key][tcol] = {}; parts = {}
        for h, _ in GROUPS:
            res, contrib = aipw((pl.hiconf == h).to_numpy(), tcol, with_gap=wg)
            R[key][tcol][str(h)] = res; parts[h] = contrib
        diff = R[key][tcol]['1']['estimate'] - R[key][tcol]['0']['estimate']
        cl = parts[1].groupby(level=0).sum().subtract(parts[0].groupby(level=0).sum(), fill_value=0); k = len(cl)
        R[key][tcol]['difference_high_minus_low'] = {'estimate': float(diff), 'se': float(np.sqrt(k / (k - 1) * (cl ** 2).sum()))}
R['max_share_propensity_clipped'] = max(R[k][t][h]['share_propensity_clipped'] for k in ['doubly_robust']
                                        for t in ['above', 'below'] for h in ['0', '1'])

# Fixed-effects cross-check: same controls and effects as the baseline interaction model.
C = 'lp + lsq + llot + age + lsq_m + llot_m + age_m'
m = pf.feols('y ~ above + below + above:hiconf + below:hiconf + prem + disc + prem:hiconf + disc:hiconf + hiconf + ' + C + ' | city + ptype + month', data=pl,
             vcov={'CRV1': 'city'}, fixef_rm='none', fixef_maxiter=100000)
tt = m.tidy()
R['fixed_effects'] = {'n': int(m._N), **{k.replace(':', '_x_'): {'estimate': float(tt.loc[k, 'Estimate']), 'se': float(tt.loc[k, 'Std. Error'])}
                                          for k in ['above', 'below', 'above:hiconf', 'below:hiconf']}}

# ---------------- Analysis 3: slope horse race, finer geography, alternative screens ----------------
C2 = C.replace('age +', 'age2020 +') + ' + lsq2 + llot2 + age2 + atyp_lsq + atyp_llot + atyp_age2020 + rarity'
TERMS = 'prem + disc + prem:hiconf + disc:hiconf + hiconf'
full_s = d[d.ConfidenceScore.notna()].copy()
for v in cm.columns: full_s[v] = full_s.county.map(cm[v])
def z(x): return (x - x.mean()) / x.std()
full_s['atyp_idx'] = z(z(full_s.atyp_lsq) + z(full_s.atyp_llot) + z(full_s.atyp_age2020))
full_s['rar_z'] = z(full_s.rarity)
full_s['cm_miss'] = full_s[list(cm.columns)].isna().any(axis=1).astype(int)
full_s['dom_z'] = z(full_s['county_median_days_on_market']).fillna(0); full_s['newact_z'] = z(full_s['county_new_active']).fillna(0)
MODS = ['atyp_idx', 'rar_z', 'dom_z', 'newact_z']
def fe_est(data, terms, fe, controls, cluster):
    mm = pf.feols('y ~ ' + terms + ' + ' + controls + ' | ' + fe, data=data, vcov={'CRV1': cluster}, fixef_rm='none', fixef_maxiter=100000)
    q = mm.tidy(); return {'n': int(mm._N), **{k.replace(':', '_x_'): {'estimate': float(q.loc[k, 'Estimate']), 'se': float(q.loc[k, 'Std. Error'])}
                                               for k in q.index if k.startswith('prem') or k.startswith('disc')}}
plx = full_s[full_s.plaus]
hr_terms = TERMS + ''.join(f' + prem:{m} + disc:{m} + {m}' for m in MODS) + ' + cm_miss'
R['slopes'] = {
    'expanded_controls': fe_est(plx, TERMS, 'city + ptype + month', C2, 'city'),
    'horse_race': fe_est(plx, hr_terms, 'city + ptype + month', C2, 'city'),
    'zip_fe': fe_est(plx, TERMS, 'zip + ptype + month', C2, 'city'),
    'zip_quarter_fe': fe_est(plx, TERMS, 'zip^quarter + ptype', C2, 'city'),
    'screen_narrow_0.67_1.5': fe_est(full_s[(full_s.FACurrentAVM / full_s.CurrentListingPrice).between(2 / 3, 1.5)], TERMS, 'city + ptype + month', C2, 'city'),
    'screen_wide_0.4_2.5': fe_est(full_s[(full_s.FACurrentAVM / full_s.CurrentListingPrice).between(.4, 2.5)], TERMS, 'city + ptype + month', C2, 'city')}
R['slopes']['zip_count'] = int(plx.zip.nunique())
(ROOT / 'companion_results.json').write_text(json.dumps(R, indent=2))

# ---------------- Table and macros ----------------
pp = lambda v: f"{100 * v:.2f}"
dr = R['doubly_robust']; fe = R['fixed_effects']; sh = R['outside_shares']
t = r'''\begin{table}[tbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Pricing outside the vendor's valuation range}\label{tab:outside}
\begin{tabular}{lrrrr}\toprule
 & \multicolumn{2}{c}{Priced above the range} & \multicolumn{2}{c}{Priced below the range}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}
 & Conf.\ $<80$ & Conf.\ $\geq80$ & Conf.\ $<80$ & Conf.\ $\geq80$\\\midrule
''' + 'Share of records (\\%) & ' + ' & '.join(pp(sh[str(h)][c]) for c in ['above', 'below'] for h in [0, 1]) + r'\\' + '\n'
t += 'Raw difference from inside (pp) & ' + ' & '.join(pp(dr[c][str(h)]['raw_difference']) for c in ['above', 'below'] for h in [0, 1]) + r'\\' + '\n'
t += 'Doubly robust difference (pp) & ' + ' & '.join(pp(dr[c][str(h)]['estimate']) for c in ['above', 'below'] for h in [0, 1]) + r'\\' + '\n'
t += ' & ' + ' & '.join('(' + pp(dr[c][str(h)]['se']) + ')' for c in ['above', 'below'] for h in [0, 1]) + r'\\' + '\n'
t += 'Records in contrast & ' + ' & '.join(f"{dr[c][str(h)]['n']:,}" for c in ['above', 'below'] for h in [0, 1]) + r'\\' + '\n'
t += r'''\midrule
High minus lower confidence, DR (pp) & \multicolumn{2}{c}{''' + f"{pp(dr['above']['difference_high_minus_low']['estimate'])} ({pp(dr['above']['difference_high_minus_low']['se'])})" + r'''} & \multicolumn{2}{c}{''' + f"{pp(dr['below']['difference_high_minus_low']['estimate'])} ({pp(dr['below']['difference_high_minus_low']['se'])})" + r'''}\\
FE interaction with high, given gap (pp) & \multicolumn{2}{c}{''' + f"{pp(fe['above_x_hiconf']['estimate'])} ({pp(fe['above_x_hiconf']['se'])})" + r'''} & \multicolumn{2}{c}{''' + f"{pp(fe['below_x_hiconf']['estimate'])} ({pp(fe['below_x_hiconf']['se'])})" + r'''}\\
\bottomrule\end{tabular}
\par\smallskip\parbox{.97\textwidth}{\footnotesize Sample: ''' + f"{R['n_sample']:,}" + r''' plausible-AVM records with a confidence score and a valid valuation range; base rate ''' + f"{R['base_rate_sample']:.3f}" + r'''. ``Above'' means the asking price exceeds the vendor's high valuation bound; ``below'' means it is under the low bound. Differences are in percentage points of the probability of pending or contingent status, relative to records priced inside the range. Doubly robust estimates are cross-fitted AIPW average differences with gradient-boosting propensity and outcome models over five county-blocked folds; propensities are clipped to [0.02, 0.98] (at most ''' + f"{100 * R['max_share_propensity_clipped']:.1f}" + r''' percent of records in any cell); county-clustered standard errors in parentheses. The last row reports the interaction of each indicator with high confidence in a linear probability model that also contains the premium, discount and their interactions with high confidence, the baseline controls, and city, property-type and listing-month effects (city-clustered). Covariate-adjusted associations with recorded status, not causal effects.}
\end{table}
'''
(ROOT.parent / 'outside_table.tex').write_text(t)
mac = {'XaboveLo': pp(dr['above']['0']['estimate']), 'XaboveLoSE': pp(dr['above']['0']['se']),
       'XaboveHi': pp(dr['above']['1']['estimate']), 'XaboveHiSE': pp(dr['above']['1']['se']),
       'XbelowLo': pp(dr['below']['0']['estimate']), 'XbelowLoSE': pp(dr['below']['0']['se']),
       'XbelowHi': pp(dr['below']['1']['estimate']), 'XbelowHiSE': pp(dr['below']['1']['se']),
       'XaboveDiff': pp(dr['above']['difference_high_minus_low']['estimate']), 'XaboveDiffSE': pp(dr['above']['difference_high_minus_low']['se']),
       'XbelowDiff': pp(dr['below']['difference_high_minus_low']['estimate']), 'XbelowDiffSE': pp(dr['below']['difference_high_minus_low']['se']),
       'XshareAboveLo': pp(sh['0']['above']), 'XshareAboveHi': pp(sh['1']['above']),
       'XshareBelowLo': pp(sh['0']['below']), 'XshareBelowHi': pp(sh['1']['below']),
       'XsampleBase': f"{100 * R['base_rate_sample']:.1f}"}
sl = R['slopes']
def cell(spec, k):
    v = sl[spec].get(k); return (f"{v['estimate']:.4f}", f"({v['se']:.4f})") if v else ('--', '')
st = r'''\begin{table}[tbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Slope heterogeneity, finer geography, and alternative screens}\label{tab:slopes}
\begin{tabular}{lrrrr}\toprule
\multicolumn{5}{l}{\textit{Panel A. Gap-by-confidence interactions}}\\
Specification & Premium $\times$ high & Discount $\times$ high & $N$ & \\\midrule
'''
for spec, lab in [('expanded_controls', 'Expanded controls'), ('horse_race', 'Horse race (Panel B)'),
                  ('zip_fe', 'ZIP effects'), ('zip_quarter_fe', 'ZIP $\\times$ quarter effects'),
                  ('screen_narrow_0.67_1.5', 'Screen $2/3\\leq A/P\\leq 1.5$'), ('screen_wide_0.4_2.5', 'Screen $0.4\\leq A/P\\leq 2.5$')]:
    a = cell(spec, 'prem_x_hiconf'); b = cell(spec, 'disc_x_hiconf')
    st += f"{lab} & {a[0]} & {b[0]} & {sl[spec]['n']:,} & " + r'\\' + '\n' + f" & {a[1]} & {b[1]} & & " + r'\\' + '\n'
st += r'''\midrule
\multicolumn{5}{l}{\textit{Panel B. Horse race: gap slopes by standardized moderator}}\\
 & Atypicality & Type rarity & County DOM & New/active\\\midrule
'''
for g, lab in [('prem', 'Premium $\\times$ moderator'), ('disc', 'Discount $\\times$ moderator')]:
    vals = [cell('horse_race', f'{g}_x_{m}') for m in MODS]
    st += lab + ' & ' + ' & '.join(v[0] for v in vals) + r'\\' + '\n' + ' & ' + ' & '.join(v[1] for v in vals) + r'\\' + '\n'
st += r'''\bottomrule\end{tabular}
\par\smallskip\parbox{.97\textwidth}{\footnotesize Linear probability models with the expanded physical controls of Table~\ref{tab:robust}, row 2 (squared log sizes and age, county-relative atypicality, type rarity). Unless noted, city, property-type and listing-month effects; all standard errors cluster by city. The horse race adds interactions of premium and discount with each standardized moderator and the moderators' main effects: an index of county-relative size, lot and age atypicality; property-type rarity; 2018 county median days on market; and 2018 county new-to-active listings (zero with a missing indicator where the county is unmatched). ZIP specifications use ''' + f"{sl['zip_count']:,}" + r''' ZIP codes; singleton cells are retained. Screens replace $0.5\leq A/P\leq 2$.}
\end{table}
'''
(ROOT.parent / 'slopes_table.tex').write_text(st)
mac.update({'XhrPremHi': f"{sl['horse_race']['prem_x_hiconf']['estimate']:.4f}", 'XhrPremHiSE': f"{sl['horse_race']['prem_x_hiconf']['se']:.4f}",
            'XzipPremHi': f"{sl['zip_fe']['prem_x_hiconf']['estimate']:.4f}", 'XzipPremHiSE': f"{sl['zip_fe']['prem_x_hiconf']['se']:.4f}",
            'XzqPremHi': f"{sl['zip_quarter_fe']['prem_x_hiconf']['estimate']:.4f}", 'XzqPremHiSE': f"{sl['zip_quarter_fe']['prem_x_hiconf']['se']:.4f}",
            'XnarPremHi': f"{sl['screen_narrow_0.67_1.5']['prem_x_hiconf']['estimate']:.4f}", 'XwidePremHi': f"{sl['screen_wide_0.4_2.5']['prem_x_hiconf']['estimate']:.4f}",
            'XhrAtypPrem': f"{sl['horse_race']['prem_x_atyp_idx']['estimate']:.4f}", 'XhrAtypPremSE': f"{sl['horse_race']['prem_x_atyp_idx']['se']:.4f}",
            'XhrDomPrem': f"{sl['horse_race']['prem_x_dom_z']['estimate']:.4f}", 'XhrDomPremSE': f"{sl['horse_race']['prem_x_dom_z']['se']:.4f}",
            'XfeAboveHi': pp(fe['above_x_hiconf']['estimate']), 'XfeAboveHiSE': pp(fe['above_x_hiconf']['se']),
            'XfeBelowHi': pp(fe['below_x_hiconf']['estimate']), 'XfeBelowHiSE': pp(fe['below_x_hiconf']['se']),
            'XclipMax': f"{100 * R['max_share_propensity_clipped']:.1f}"})
(ROOT.parent / 'companion_macros.tex').write_text('\n'.join(r'\newcommand{\%s}{%s}' % (k, v) for k, v in mac.items()) + '\n')
print(json.dumps(R, indent=1))
