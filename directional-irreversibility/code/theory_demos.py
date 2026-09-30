"""Illustrations of the paper's propositions and Theorem 1.

Everything here is synthetic except Part E, which uses the public-domain
statsmodels `macrodata` quarterly U.S. series (T-bill rate and unemployment,
1959Q1-2009Q3). No identified shock, asset price, or journal input is used.

A. Predictability bound (Proposition prop:innovation): AR(1) drivers.
B. State omission (Proposition prop:stateerasure): pooled vs state-conditioned DII.
C. State errors (Proposition prop:statestability): frozen population values.
D. Immediate incorporation (Proposition prop:incorporation) and Theorem 1:
   fitted joint inference for impact-inclusive vs post-impact outcomes.
E. Public macro data: a predictable level vs its innovation as the driver.
Seeds are fixed; outputs go to theory_demos_results.json and a figure.
"""
from pathlib import Path
import json, sys
import numpy as np

S = Path(__file__).resolve().parent
P = S.parent
sys.path.insert(0, str(S))
from full_study import kernel, center                       # noqa: E402
from dii_joint_nuisance import prepare_direction, joint_nuisance_dii  # noqa: E402

SEED = 20260919


def hsic(u, z):
    """Biased V-statistic of the squared HS norm (the paper's HSIC convention)."""
    K = center(kernel(u)); L = center(kernel(z))
    return float((K * L).sum() / len(u) ** 2)


def poly2(z):
    """Total-degree-two polynomial basis with intercept."""
    z = np.atleast_2d(z)
    if z.shape[0] == 1 and z.shape[1] != 1:
        z = z.T
    cols = [np.ones(len(z))] + [z[:, i] for i in range(z.shape[1])]
    for i in range(z.shape[1]):
        for j in range(i, z.shape[1]):
            cols.append(z[:, i] * z[:, j])
    return np.column_stack(cols)


def ols_resid(y, z):
    Pm = poly2(z)
    b = np.linalg.lstsq(Pm, y, rcond=None)[0]
    return y - Pm @ b


def summarize(v):
    v = np.asarray(v)
    return dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))),
                q05=float(np.quantile(v, .05)), q95=float(np.quantile(v, .95)))


rng = np.random.default_rng(SEED)
out = {}

# ---------------------------------------------------------------- A
A = []
n, reps, gam, sig = 1500, 20, 1.0, 0.5
for phi in [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]:
    hb, hf, dd, ev = [], [], [], []
    for _ in range(reps):
        T = n + 200
        eta = rng.normal(size=T); x = np.empty(T); x[0] = rng.normal()
        for t in range(1, T):
            x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * eta[t]
        x = x[200:]; c = np.r_[np.nan, x[:-1]][1:]; x = x[1:]
        eps = sig * rng.normal(size=len(x))
        y = gam * (x ** 2 - 1) + eps
        ys = (y - y.mean()) / y.std(); cs = (c - c.mean()) / c.std()
        ub = ols_resid(x, np.column_stack([ys, cs]))        # reverse projection residual, X units
        hb.append(hsic(ub, np.column_stack([ys, cs])))
        hf.append(hsic(eps / y.std(), np.column_stack([x, cs])))  # oracle forward residual
        dd.append(hb[-1] - hf[-1]); ev.append(float(np.mean(ub ** 2)))
    A.append(dict(phi=phi, bound=1 - phi ** 2, reverse_hsic=summarize(hb), forward_hsic=summarize(hf),
                  dii=summarize(dd), mean_sq_reverse_residual=summarize(ev)))
out['A_predictability'] = dict(n=n, reps=reps, gamma=gam, sigma=sig, rows=A)

# ---------------------------------------------------------------- B
B = []
sig = 1.0
for n in [100, 200, 400, 800, 1600]:
    reps = 100 if n <= 800 else 40
    dS, d0, c0 = [], [], []
    for _ in range(reps):
        x = rng.choice([-1., 1.], n); s = rng.choice([-1., 1.], n)
        e = sig * rng.normal(size=n); y = s * x + e
        uf, ubS = e, x - s * np.tanh(y / sig ** 2)
        dS.append(hsic(ubS, np.column_stack([y, s])) - hsic(uf, np.column_stack([x, s])))
        c0.append(hsic(x, y))                # pooled: both conditional means are zero, so
        d0.append(c0[-1] - hsic(y, x))       # each pooled component is HSIC(X,Y) -> 0
    B.append(dict(n=n, reps=reps, dii_state_conditioned=summarize(dS), dii_pooled=summarize(d0), pooled_component=summarize(c0)))
out['B_state_omission'] = dict(sigma=sig, oracle_residuals=True, rows=B)

# ---------------------------------------------------------------- C (frozen)
pop = json.load(open(P / 'state_observation_population.json'))
out['C_state_error_population'] = [dict(sigma=r['sigma'], q=r['q'], dii=r['dii']) for r in pop]

# ---------------------------------------------------------------- D
gam, m, gap, n = 0.5, 204, 39, 120
Dpop = []
for h in range(1, 7):
    inc, post = [], []
    for _ in range(10):
        N = 2000 + h
        s = rng.normal(size=N); e = rng.normal(size=N); dp = gam * (s ** 2 - 1) + e
        cs = np.cumsum(dp)
        yin = cs[h:N] - np.r_[0, cs[:N - 1]][:N - h]   # P_{t+h}-P_{t-1}
        yq = cs[h:N] - cs[:N - h]                      # P_{t+h}-P_t
        st = s[:N - h]
        yi = (yin - yin.mean()) / yin.std(); yqq = (yq - yq.mean()) / yq.std()
        uf_i = yi - gam * (st ** 2 - 1) / yin.std()    # oracle forward residual, standardized units
        inc.append(hsic(st, yi[:, None]) - hsic(uf_i, st[:, None]))
        post.append(hsic(st, yqq[:, None]) - hsic(yqq, st[:, None]))
    Dpop.append(dict(h=h, dii_inclusive=summarize(inc), dii_post_impact=summarize(post)))

Drej = []
reps, draws = 200, 399
te = np.arange(m + gap, m + gap + n)
for h in [1, 3, 6]:
    rej = {'inclusive': 0, 'post_impact': 0}
    for r in range(reps):
        N = m + gap + n + h + 1
        s = rng.normal(size=N); e = rng.normal(size=N); dp = gam * (s ** 2 - 1) + e
        cs = np.cumsum(dp)
        idx = np.arange(1, m + gap + n + 1)
        yin = cs[idx + h] - cs[idx - 1]; yq = cs[idx + h] - cs[idx]; st = s[idx]
        comps = {}
        for name, yv in [('inclusive', yin), ('post_impact', yq)]:
            fwd = prepare_direction(yv, st[:, None], m, te, poly2)
            rev = prepare_direction(st, yv[:, None], m, te, poly2)
            comps[name] = [fwd, rev]
        for name in comps:
            res = joint_nuisance_dii({name: comps[name]}, bootstrap_draws=draws, seed=int(rng.integers(1 << 31)))
            rej[name] += res['comparisons'][name]['p_intersection'] <= 0.05
    Drej.append(dict(h=h, reps=reps, reject_inclusive=rej['inclusive'] / reps, reject_post_impact=rej['post_impact'] / reps))
out['D_incorporation'] = dict(gamma=gam, population_by_h=Dpop, fitted_joint_rejection=dict(m=m, gap=gap, n=n, draws=draws, rows=Drej))

# ---------------------------------------------------------------- E (public macro data)
import statsmodels.api as sm
md = sm.datasets.macrodata.load_pandas().data
tb, un = md.tbilrate.to_numpy(), md.unemp.to_numpy()
du = np.r_[np.nan, np.diff(un)]
m, gap, n = 96, 4, 96
t0 = 3
orig = np.arange(t0, t0 + m + gap + n)          # common origins for every horizon
te = np.arange(m + gap, m + gap + n)
# Innovation: tbill_t minus its training-sample AR(2) prediction
Xar = np.column_stack([np.ones(len(tb) - 2), tb[1:-1], tb[:-2]])
tr = orig[:m] - 2
coef = np.linalg.lstsq(Xar[tr], tb[2:][tr], rcond=None)[0]
innov = np.r_[np.nan, np.nan, tb[2:] - Xar @ coef]
E = {}
for label, drv in [('level', tb), ('innovation', innov)]:
    comps, extra = {}, {}
    for h in [1, 2, 4]:
        x = drv[orig]; y = du[orig + h]
        cmat = np.column_stack([drv[orig - 1], du[orig]])
        fwd = prepare_direction(y, np.column_stack([x, cmat]), m, te, poly2)
        rev = prepare_direction(x, np.column_stack([y, cmat]), m, te, poly2)
        comps[f'h{h}'] = [fwd, rev]
        # Exact in-sample check of the bound: ||C_b||^2 <= mean(u_b^2) * 1 (Gaussian kernels, M=1)
        # History-only predictability: standardized driver regressed on the history basis (training fit).
        xs = (x - x[:m].mean()) / x[:m].std(); cs = (cmat - cmat[:m].mean(0)) / cmat[:m].std(0)
        Pc = poly2(cs); b = np.linalg.lstsq(Pc[:m], xs[:m], rcond=None)[0]
        extra[f'h{h}'] = dict(mean_sq_reverse_residual=float(np.mean(rev['u'] ** 2)),
                              history_only_share=float(np.mean((xs - Pc @ b)[te] ** 2)))
    res = joint_nuisance_dii(comps, bootstrap_draws=999, seed=SEED)
    for k, v in res['comparisons'].items():
        v.update(extra[k])
        assert v['reverse_hsic'] <= v['mean_sq_reverse_residual'] + 1e-12
    E[label] = res['comparisons']
out['E_macro'] = dict(source='statsmodels macrodata (FRED; BLS unemployment), public domain',
                      sample='1959Q1-2009Q3; origins from 1959Q4; training 96, gap 4, evaluation 96 quarters',
                      driver='3-month T-bill rate: level, or residual from a training-sample AR(2)',
                      outcome='one-quarter change in the unemployment rate at t+h',
                      history='lagged driver and current unemployment change', basis='total-degree-two polynomial',
                      bootstrap_draws=999, results=E)

json.dump(out, open(P / 'theory_demos_results.json', 'w'), indent=1)
print('theory demos written')
