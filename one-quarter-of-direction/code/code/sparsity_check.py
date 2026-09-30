"""Sparsity-robust check. Identified shock series have a mass point at zero (months without an announcement), and
size_under_sparsity.py shows that the wild and paired tests over-reject in that case even with no channel. The
circular-shift test below keeps the shock series exactly as it is and the first stage fixed, and breaks only the
alignment between shock and outcome by rotating the outcome within the test block; whatever a mass of zeros does to
the index is therefore present in the reference distribution as well. It is a test of independence of the two
series that uses the index as its statistic; it is valid whatever the marginal distribution of the shock.

    python code/sparsity_check.py size [R]     # size of the shift test under sparse no-channel designs
    python code/sparsity_check.py apply        # shift test on the eighteen pairs, cell by cell and pooled"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_lib import *


def split(d, h, p=P_LAGS, gap=GAP):
    X = d['x'].values; Y = d['y'].values; T = len(d); n = int(np.floor(T / np.log(T))); tr = T - n - gap
    return Stage1().fit(build(X[:tr], Y[:tr], h, p)), X[tr + gap:], Y[tr + gap:]
def stat(s1, X, Y, h, p=P_LAGS): e = build(X, Y, h, p); ef, eb = s1.resid(e); return dii(ef, eb, e)
def offsets(n, h, B, rng, p=P_LAGS): lo = h + p + 2; return rng.randint(lo, n - lo, size=B)          # rotations that keep the two series well apart
def shift_cell(d, h, B=499, seed=0):
    s1, X, Y = split(d, h); obs = stat(s1, X, Y, h); rng = np.random.RandomState(seed)
    bo = np.array([stat(s1, X, np.roll(Y, k), h) for k in offsets(len(X), h, B, rng)]); return obs, (np.sum(bo >= obs) + 1) / (B + 1), bo.mean()
def shift_matrix(ds, h, B=499, seed=0, p=P_LAGS, gap=GAP):
    """Observed index per unit and a B x units matrix of rotated values, one rotation fraction per draw shared by all
    units. Any pooled subset is then the row mean over its columns: p = (#{mean_b >= mean_obs} + 1) / (B + 1)."""
    S = [split(d, h, p, gap) for d in ds]; obs = np.array([stat(s1, X, Y, h, p) for s1, X, Y in S]); fr = np.random.RandomState(seed).rand(B); lo = h + p + 2
    M = np.array([[stat(s1, X, np.roll(Y, lo + int(f * (len(X) - 2 * lo))), h, p) for s1, X, Y in S] for f in fr]); return obs, M
def shift_p_from(obs, M, cols): o = obs[cols].mean(); b = M[:, cols].mean(1); return o, (np.sum(b >= o) + 1) / (len(b) + 1), b.mean()
def shift_pooled(ds, h, B=499, seed=0):
    """One rotation fraction per draw, shared by all units, so that cross-unit dependence at a common date is preserved."""
    S = [split(d, h) for d in ds]; obs = np.mean([stat(s1, X, Y, h) for s1, X, Y in S]); rng = np.random.RandomState(seed); lo = h + P_LAGS + 2; bo = np.empty(B)
    for b in range(B):
        f = rng.rand(); bo[b] = np.mean([stat(s1, X, np.roll(Y, lo + int(f * (len(X) - 2 * lo))), h) for s1, X, Y in S])
    return obs, (np.sum(bo >= obs) + 1) / (B + 1), bo.mean()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'apply'; D = load_panel()
    if mode == 'size':
        R = int(sys.argv[2]) if len(sys.argv) > 2 else 200; rr = sample(D, 'rr', 'sp500')['x'].values; rows = []
        def ar1y(T, rng):
            e = rng.normal(size=T); Y = np.zeros(T)
            for t in range(1, T): Y[t] = 0.3 * Y[t - 1] + e[t]
            return Y
        for name, draw in (('dense', lambda rng: rng.normal(size=551)), ('sparse p=0.15', lambda rng: rng.normal(size=551) * (rng.rand(551) < .15)), ('actual narrative shock', lambda rng: rr)):
            rng = np.random.RandomState(0); ps = np.array([shift_cell(pd.DataFrame({'x': draw(rng), 'y': ar1y(551, rng)}), 3, 199, seed=r)[1] for r in range(R)])
            rows.append(dict(design=name, reps=R, shift_05=(ps <= .05).mean(), shift_10=(ps <= .10).mean())); print(rows[-1], flush=True)
        save(pd.DataFrame(rows), 'size_shift_test.csv')
    else:
        rows = []
        for x, y in PAIRS18:
            for h in HS: obs, p, m = shift_cell(sample(D, x, y), h); rows.append(dict(x=x, outcome=y, h=h, DII=obs, mean_under_rotation=m, p_shift=p)); print(rows[-1], flush=True)
        save(pd.DataFrame(rows), 'shift_test_cells.csv'); rows = []
        sets = {'eighteen pairs': PAIRS18, 'nine price and volatility pairs': [q for q in PV if q[0] != 'bs'], 'twelve price and volatility pairs': PV, 'oil pairs only (dense shock)': [q for q in PAIRS18 if q[0] == 'oil']}
        for name, pairs in sets.items():
            for h in HS: obs, p, m = shift_pooled([sample(D, x, y) for x, y in pairs], h, 999); rows.append(dict(set=name, h=h, P_N=obs, mean_under_rotation=m, p_shift=p))
        for name, pairs, kw in (('19 currencies, 1974-2001', [('rr', 'fx_' + c) for c in FLOAT9 + EURO10], dict(end='2001-12')), ('27 currency pairs', [(x, 'fx_' + c) for x in ('rr', 'jk', 'bs') for c in FLOAT9], {})):
            for h in HS: obs, p, m = shift_pooled([sample(D, x, y, **kw) for x, y in pairs], h, 999); rows.append(dict(set=name, h=h, P_N=obs, mean_under_rotation=m, p_shift=p))
        save(pd.DataFrame(rows), 'shift_test_pooled.csv')
