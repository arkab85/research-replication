"""Summarise the resolution atlas and the published-design study into the three panels the paper reports,
and the macros the prose quotes. Descriptive only.

    python code/atlas_summary.py
writes results/atlas_panelA.csv, atlas_panelB.csv, atlas_panelC.csv, atlas_macros.csv
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
STAGE_LABEL = {'linear': 'Linear projection', 'ols': 'Linear projection', 'sieve': 'Sieve ridge',
               'ls': 'Location--scale', 'nn': 'Neural network', 'rf': 'Random forest'}
FAM_LABEL = {'shock-price': 'Event-dated shocks and prices', 'shock-fx': 'Event-dated shocks and exchange rates',
             'flow-price': 'Weekly flows to prices', 'price-flow': 'Weekly prices to flows',
             'tic-fx': 'Monthly cross-border flows and rates', 'alt-return': 'Alternative weekly causes, returns',
             'alt-variance': 'Alternative weekly causes, variance',
             'P1-equity-AR': 'Autoregression on equity returns', 'P2-fredmd-AR': 'Autoregressions on FRED-MD series',
             'P3-VAR': 'Named VAR systems'}


def agg(D, by):
    g = D.groupby(by, sort=False)
    out = pd.DataFrame({
        'designs': g.size(),
        'median_tau': g.tau.median(),
        'q25_tau': g.tau.quantile(.25),
        'q75_tau': g.tau.quantile(.75),
        'share_above_half': 100 * g.tau.apply(lambda s: float(np.mean(s > 0.5))),
        'median_bw_over_err': g.bw_over_err.median(),
        'median_n': g.n.median(),
    }).reset_index()
    return out.round(4)


def main():
    macros = {}
    A = os.path.join(RES, 'tau_atlas.csv'); P = os.path.join(RES, 'tau_published.csv')

    if os.path.exists(A):
        D = pd.read_csv(A)
        print(f'atlas: {len(D)} measurements over {D.design.nunique()} designs, '
              f'{D.h.nunique()} horizons, {D.stage.nunique()} first stages')
        pa = agg(D, ['stage']); pa['label'] = pa.stage.map(STAGE_LABEL)
        pb = agg(D, ['family']); pb['label'] = pb.family.map(FAM_LABEL)
        pa.to_csv(os.path.join(RES, 'atlas_panelA.csv'), index=False)
        pb.to_csv(os.path.join(RES, 'atlas_panelB.csv'), index=False)
        print('\n--- Panel A: by first stage ---'); print(pa[['label', 'designs', 'median_tau', 'share_above_half', 'median_bw_over_err']].to_string(index=False))
        print('\n--- Panel B: by data family ---'); print(pb[['label', 'designs', 'median_tau', 'share_above_half']].to_string(index=False))
        macros['atlasDesigns'] = f'{D.design.nunique()}'
        macros['atlasMeasurements'] = f'{len(D)}'
        macros['atlasShareAbove'] = f'{100 * np.mean(D.tau > 0.5):.0f}'
        macros['atlasMedianTau'] = f'{D.tau.median():.2f}'
        for s in D.stage.unique():
            sub = D[D.stage == s]
            macros[f'atlasShare{s.capitalize()}'] = f'{100 * np.mean(sub.tau > 0.5):.0f}'
            macros[f'atlasTau{s.capitalize()}'] = f'{sub.tau.median():.2f}'
        # how much of tau is explained by the first stage vs the data
        macros['atlasMaxFamilyShare'] = f'{pb.share_above_half.max():.0f}'
        macros['atlasMinFamilyShare'] = f'{pb.share_above_half.min():.0f}'
    else:
        print('tau_atlas.csv not present yet')

    if os.path.exists(P):
        Q = pd.read_csv(P)
        print(f'\npublished designs: {len(Q)} measurements over {Q.design.nunique()} designs')
        pc = agg(Q, ['family', 'convention']); pc['label'] = pc.family.map(FAM_LABEL)
        pc.to_csv(os.path.join(RES, 'atlas_panelC.csv'), index=False)
        print('\n--- Panel C: published designs ---')
        print(pc[['label', 'convention', 'designs', 'median_tau', 'share_above_half', 'median_bw_over_err']].to_string(index=False))
        macros['pubDesigns'] = f'{Q.design.nunique()}'
        for c in ('insample', 'split'):
            sub = Q[Q.convention == c]
            if len(sub):
                macros[f'pubShare{c.capitalize()}'] = f'{100 * np.mean(sub.tau > 0.5):.0f}'
                macros[f'pubTau{c.capitalize()}'] = f'{sub.tau.median():.2f}'
        ols = Q[(Q.stage == 'ols') & (Q.convention == 'insample')]
        if len(ols):
            macros['pubShareOlsInsample'] = f'{100 * np.mean(ols.tau > 0.5):.0f}'
            macros['pubTauOlsInsample'] = f'{ols.tau.median():.2f}'
        # the FRED-MD autoregressions, by lag length, in-sample and by OLS: the SVAR-like case
        fm = Q[(Q.family == 'P2-fredmd-AR') & (Q.stage == 'ols') & (Q.convention == 'insample')]
        for p in sorted(fm.p.unique()):
            s = fm[fm.p == p]
            macros[f'pubFredOlsP{p}'] = f'{100 * np.mean(s.tau > 0.5):.0f}'
            print(f'   FRED-MD, OLS, in-sample, p={p:2d}: median tau {s.tau.median():.3f}, '
                  f'{100 * np.mean(s.tau > 0.5):.0f}% above one half  (n={len(s)})')
    else:
        print('tau_published.csv not present yet')

    if macros:
        pd.DataFrame(sorted(macros.items()), columns=['macro', 'value']).to_csv(
            os.path.join(RES, 'atlas_macros.csv'), index=False)
        print('\n--- macros ---')
        for k, v in sorted(macros.items()): print(f'  {k:26s} {v}')


if __name__ == '__main__':
    main()
