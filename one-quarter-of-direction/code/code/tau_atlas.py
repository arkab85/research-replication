"""The resolution atlas: where tau falls across real financial series and across the first-stage estimators that
different literatures use. Design fixed in advance in code/ATLAS_DESIGN.md; run once.

tau = share of evaluation pairs with |X_i - X_j| <= |g_i - g_j|, computed by the same lines as code/tau_collapse.py.
Descriptive only: no bootstrap, no p-value, no rejection rate.

    python code/tau_atlas.py [part nparts]        |        python code/tau_atlas.py collect
"""
import sys, os, glob
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flows'))
import flows_lib as FL                                          # the O(n^2) centering, checked against the engine
from core_lib import *
from sklearn.ensemble import RandomForestRegressor

HORIZONS = (1, 3, 6)
MIN_OBS = 120
MARKETS = ['CAD', 'EUR', 'JPY', 'CHF', 'GBP', 'MXN', 'SPX', 'NDX', 'UST2', 'UST5', 'UST10', 'USTB']
TIC = ['AUD', 'CAD', 'DKK', 'JPY', 'NZD', 'NOK', 'SEK', 'CHF', 'GBP']
ALT = ['nc_net_chg', 'lvl_net', 'z52_net', 'nc_long_chg', 'nc_short_chg', 'comm_net_chg', 'oi_growth',
       'tff_lev_chg', 'tff_am_chg', 'tff_dealer_chg']
MONTHLY_Y = ['sp500', 'dy10', 'vix', 'ebp', 'spread', 'dollar']


# ---------------------------------------------------------------- first stages
class Stage1Linear(Stage1):
    """Ridge on the standardised regressors with no basis expansion: the linear-projection first stage."""
    def fit(self, d, alpha=1.0):
        self.f = d.attrs['f']; self.b = d.attrs['b']
        self.scf = StandardScaler().fit(d[self.f].values); self.scb = StandardScaler().fit(d[self.b].values)
        self.mf = Ridge(alpha=alpha).fit(self.scf.transform(d[self.f].values), d['Y_th'].values)
        self.mb = Ridge(alpha=alpha).fit(self.scb.transform(d[self.b].values), d['X_t'].values); return self
    def resid(self, d):
        return (d['Y_th'].values - self.mf.predict(self.scf.transform(d[self.f].values)),
                d['X_t'].values - self.mb.predict(self.scb.transform(d[self.b].values)))


class Stage1RF(Stage1):
    """Random-forest first stage, scikit-learn defaults but for the tree count. No tuning."""
    def __init__(self, seed=0): self.seed = seed
    def fit(self, d, **kw):
        self.f = d.attrs['f']; self.b = d.attrs['b']
        kws = dict(n_estimators=200, n_jobs=1)
        self.mf = RandomForestRegressor(random_state=self.seed, **kws).fit(d[self.f].values, d['Y_th'].values)
        self.mb = RandomForestRegressor(random_state=self.seed + 1, **kws).fit(d[self.b].values, d['X_t'].values)
        return self
    def resid(self, d):
        return (d['Y_th'].values - self.mf.predict(d[self.f].values),
                d['X_t'].values - self.mb.predict(d[self.b].values))


STAGES = {'linear': Stage1Linear, 'sieve': Stage1, 'ls': Stage1LS, 'nn': Stage1NN, 'rf': Stage1RF}


def fitted_backward(s1, d, stage_name):
    """The fitted backward conditional mean E[X_t | Z^b], for every first stage.

    tau compares |X_i - X_j| with |g_i - g_j| for g the fitted CONDITIONAL MEAN. For the stages whose
    residual is X - prediction one may recover g as X - e, which is what code/tau_collapse.py does. That
    identity fails for the location-scale stage, whose resid() returns a residual divided by a fitted
    standard deviation; there g must be read off the mean model directly. See ATLAS_DESIGN_ADDENDUM.md.
    """
    Zb = d[d.attrs['b']].values
    if stage_name == 'rf': return s1.mb.predict(Zb)
    if stage_name == 'linear': return s1.mb.predict(s1.scb.transform(Zb))
    if stage_name == 'nn': return s1.mb.predict(s1.scb.transform(Zb))
    return s1.mb.predict(sieve(s1.scb.transform(Zb)))            # 'sieve' and 'ls' share the mean model


# ---------------------------------------------------------------- the design list
def designs():
    """Every (family, label, x, y) the panels define, by rule. Returns a list of (family, label, DataFrame)."""
    R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    D = load_panel()
    for x in ('rr', 'jk', 'oil', 'bs'):                                       # 1. shocks and monthly outcomes
        for y in MONTHLY_Y:
            out.append(('shock-price', f'{x}->{y}', sample(D, x, y)))
    for x in ('rr', 'jk', 'oil', 'bs'):                                       # 2. shocks and exchange rates
        for c in FLOAT9:
            out.append(('shock-fx', f'{x}->{c}', sample(D, x, f'fx_{c}')))
    W = pd.read_csv(os.path.join(R, 'flows', 'data', 'weekly_panel.csv'), index_col=0)
    for m in MARKETS:                                                         # 3./4. weekly flows, both labellings
        d = pd.concat([W[f'f_{m}'].rename('x'), W[f'r_{m}'].rename('y')], axis=1).dropna()
        out.append(('flow-price', f'f_{m}->r_{m}', d))
        out.append(('price-flow', f'r_{m}->f_{m}', d.rename(columns={'x': 'y', 'y': 'x'})[['x', 'y']]))
    M = pd.read_csv(os.path.join(R, 'flows', 'data', 'monthly_panel.csv'), index_col=0)
    for c in TIC:                                                             # 5. TIC flows and exchange rates
        out.append(('tic-fx', f'f_{c}->r_{c}',
                    pd.concat([M[f'f_{c}'].rename('x'), M[f'r_{c}'].rename('y')], axis=1).dropna()))
    A = pd.read_csv(os.path.join(R, 'flows', 'data', 'weekly_alt_panel.csv'), index_col=0)
    for a in ALT:                                                             # 6. alternative causes
        for m in MARKETS:
            for y, fam in (('r', 'alt-return'), ('rv', 'alt-variance')):
                cx, cy = f'{a}_{m}', f'{y}_{m}'
                if cx in A.columns and cy in A.columns:
                    out.append((fam, f'{cx}->{cy}',
                                pd.concat([A[cx].rename('x'), A[cy].rename('y')], axis=1).dropna()))
    return out


# ---------------------------------------------------------------- the diagnostic
def measure(d, h, stage_name):
    """Fit the honest split with the named first stage and return the resolution diagnostics, or None if excluded."""
    if len(d) < MIN_OBS: return None
    T = len(d); n = int(np.floor(T / np.log(T))); tr = T - n - GAP
    if tr < 30 or n < 20: return None
    s1 = STAGES[stage_name]().fit(build(d['x'].values[:tr], d['y'].values[:tr], h, P_LAGS))
    dte = build(d['x'].values[tr + GAP:], d['y'].values[tr + GAP:], h, P_LAGS)
    if len(dte) < 20: return None
    Xv = dte['X_t'].values
    if np.std(Xv) == 0: return None
    ghat = fitted_backward(s1, dte, stage_name)                               # fitted backward conditional mean
    eb = Xv - ghat                                                            # unstandardised first-stage error
    dx = np.abs(Xv[:, None] - Xv[None, :]); dg = np.abs(ghat[:, None] - ghat[None, :])
    iu = np.triu_indices(len(Xv), 1)
    tau = float(np.mean(dx[iu] <= dg[iu])); p2 = float(np.mean(dx[iu] == 0))
    d2 = (eb[:, None] - eb[None, :]) ** 2; med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    err = np.std(ghat - np.mean(ghat)) if np.std(ghat) > 0 else 1.0
    return dict(tau=tau, p2=p2, bw_over_err=float(np.sqrt(med) / err),
                n_hsic=float(len(dte) * hsic2(eb, dte[dte.attrs['b']].values)),
                n=len(dte), T=T, eval_sd=float(np.std(Xv)))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'collect':
        D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, 'tau_atlas_part*.csv'))])
        save(D.round(6), 'tau_atlas.csv'); print('collected', len(D), 'rows ->', os.path.join(RES, 'tau_atlas.csv'))
        sys.exit()
    part, nparts = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    L = designs(); print(f'{len(L)} designs x {len(HORIZONS)} horizons x {len(STAGES)} first stages', flush=True)
    rows, dropped = [], 0
    for i, (fam, lab, d) in enumerate(L):
        if i % nparts != part: continue
        for h in HORIZONS:
            for sname in STAGES:
                try: r = measure(d, h, sname)
                except Exception as e: r = None; print(f'  ! {lab} h={h} {sname}: {type(e).__name__}', flush=True)
                if r is None: dropped += 1; continue
                rows.append(dict(family=fam, design=lab, h=h, stage=sname, **r))
        print(f'[{i + 1}/{len(L)}] {fam:14s} {lab}', flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, f'tau_atlas_part{part}.csv'), index=False)
    print(f'done: {len(rows)} measured, {dropped} excluded', flush=True)
