"""Is the level a function of tau alone? The paper claims the statistic's level under independence is a function of
tau alone, not of the mechanism that produced it. Fit level ~ f(tau), then add design dummies and interactions, and ask
whether they add anything. Also check the monotonicity of the bandwidth ratio and the rotation test's level."""
import numpy as np, pandas as pd, os, sys
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
D = pd.read_csv(os.path.join(R, 'tau_collapse.csv')).sort_values('tau').reset_index(drop=True)
pd.set_option('display.width', 200)
print('=== the axis, sorted by tau ===')
print(D[['design', 'par', 'tau', 'p2', 'n_hsic', 'bw_over_err', 'wild', 'rotation']].to_string(index=False))

def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b; return b, float(r @ r)

for lab, y in (('n*HSIC (level under independence)', D.n_hsic.values), ('wild-bootstrap rejection rate', D.wild.values)):
    t = D.tau.values; n = len(t)
    Xt = np.column_stack([np.ones(n), t, t ** 2])                                  # tau only
    dm = pd.get_dummies(D.design, drop_first=True).values.astype(float)            # design dummies
    Xd = np.column_stack([Xt, dm])
    Xi = np.column_stack([Xd, dm * t[:, None]])                                    # dummies + interactions
    _, s0 = ols(y, Xt); _, s1 = ols(y, Xd); _, s2 = ols(y, Xi)
    tot = float(((y - y.mean()) ** 2).sum())
    f1 = ((s0 - s1) / dm.shape[1]) / (s1 / (n - Xd.shape[1])); f2 = ((s0 - s2) / (2 * dm.shape[1])) / (s2 / (n - Xi.shape[1]))
    print(f'\n=== {lab} ===')
    print(f'  R2 with tau alone (quadratic)      {1 - s0 / tot:.4f}')
    print(f'  R2 adding design dummies           {1 - s1 / tot:.4f}    F({dm.shape[1]},{n - Xd.shape[1]}) = {f1:.2f}')
    print(f'  R2 adding dummies + interactions   {1 - s2 / tot:.4f}    F({2 * dm.shape[1]},{n - Xi.shape[1]}) = {f2:.2f}')
    print('  5% critical F ~ 3.6 and 3.0 respectively; mechanism adds nothing if F is below that.')

print('\n=== matched pairs: same tau, different mechanism ===')
for a, b in ((('atom', 0.5), ('drift', 0.12)), (('atom', 0.6), ('drift', 0.06)), (('atom', 0.65), ('drift', 0.03))):
    ra = D[(D.design == a[0]) & np.isclose(D.par, a[1])].iloc[0]; rb = D[(D.design == b[0]) & np.isclose(D.par, b[1])].iloc[0]
    print(f'  tau {ra.tau:.3f} vs {rb.tau:.3f}   level {ra.n_hsic:.3f} vs {rb.n_hsic:.3f}   p2 {ra.p2:.2f} vs {rb.p2:.2f}   ({a[0]} {a[1]} / {b[0]} {b[1]})')

print('\n=== monotonicity ===')
print('  Spearman tau vs level      ', round(float(pd.Series(D.tau).corr(pd.Series(D.n_hsic), method="spearman")), 4))
print('  Spearman tau vs bandwidth  ', round(float(pd.Series(D.tau).corr(pd.Series(D.bw_over_err), method="spearman")), 4))
print('  Spearman tau vs wild       ', round(float(pd.Series(D.tau).corr(pd.Series(D.wild), method="spearman")), 4))
print('  bandwidth/error range      ', f'{D.bw_over_err.max():.2f} down to {D.bw_over_err.min():.2f}')

print('\n=== the no-atom claim ===')
dr = D[D.design == 'drift']
print('  every drift design has p2 = 0:', bool((dr.p2 == 0).all()))
print('  highest tau reached with no atom:', f'{dr.tau.max():.3f}', ' wild there:', f'{dr.wild.max():.1f}%')
print('\n=== rotation calibration across the whole axis ===')
k = int(round(D.reps.max())); lo, hi = D.rotation.min(), D.rotation.max()
se = 100 * np.sqrt(.05 * .95 / k)
print(f'  range {lo:.1f} to {hi:.1f} percent over {len(D)} designs, {k} reps each (binomial se {se:.1f} points)')
print(f'  designs outside 5% +/- 2se ({5 - 2 * se:.1f} to {5 + 2 * se:.1f}):', int(((D.rotation < 5 - 2 * se) | (D.rotation > 5 + 2 * se)).sum()), 'of', len(D))

rows = []
for lab, y in (('level', D.n_hsic.values), ('wild', D.wild.values)):
    t_ = D.tau.values; n = len(t_); Xt = np.column_stack([np.ones(n), t_, t_ ** 2]); dm = pd.get_dummies(D.design, drop_first=True).values.astype(float)
    Xd = np.column_stack([Xt, dm]); Xi = np.column_stack([Xd, dm * t_[:, None]]); tot = float(((y - y.mean()) ** 2).sum())
    _, s0 = ols(y, Xt); _, s1 = ols(y, Xd); _, s2 = ols(y, Xi)
    rows.append(dict(outcome=lab, r2_tau=1 - s0 / tot, r2_plus_design=1 - s1 / tot, r2_plus_interactions=1 - s2 / tot,
                     F_design=((s0 - s1) / dm.shape[1]) / (s1 / (n - Xd.shape[1])), F_design_df=f'{dm.shape[1]},{n - Xd.shape[1]}',
                     F_inter=((s0 - s2) / (2 * dm.shape[1])) / (s2 / (n - Xi.shape[1])), F_inter_df=f'{2 * dm.shape[1]},{n - Xi.shape[1]}'))
out = pd.DataFrame(rows).round(4); out.to_csv(os.path.join(R, 'collapse_test.csv'), index=False); print('\n=== written to results/collapse_test.csv ==='); print(out.to_string(index=False))
