"""Compares results/*.csv with the values printed in the journal submission (7 Sept 2026) and writes
results/verification_report.md. Point estimates are deterministic and should agree to the printed precision;
p-values are Monte Carlo (the original seeds were not recorded) and are compared with a tolerance."""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(ROOT, 'results')
rd = lambda f: pd.read_csv(os.path.join(RES, f))

# ---- published values -------------------------------------------------------------------------------------
T2 = {  # (x, outcome): [(DII, wild, paired, n*rho_f) at h=1,3,6]
 ('rr','sp500'):[(.0003,.34,.38,.41),(.0013,.13,.30,.34),(.0007,.33,.35,.43)], ('rr','dy10'):[(.0010,.22,.24,.36),(.0009,.25,.26,.29),(.0000,.49,.54,.40)],
 ('rr','vix'):[(-.0021,.88,.90,.52),(.0006,.44,.36,.44),(.0003,.34,.34,.47)], ('jk','sp500'):[(.0013,.15,.40,.55),(.0052,.02,.07,.32),(.0025,.18,.29,.43)],
 ('jk','dy10'):[(.0018,.31,.33,.41),(.0036,.12,.12,.45),(-.0049,.96,.91,.65)], ('jk','vix'):[(.0042,.06,.19,.52),(.0056,.04,.07,.41),(.0041,.05,.17,.42)],
 ('oil','sp500'):[(.0001,.51,.50,.48),(.0021,.06,.12,.29),(-.0017,.92,.82,.52)], ('oil','dy10'):[(-.0022,.92,.77,.67),(.0019,.18,.28,.27),(.0016,.14,.26,.35)],
 ('oil','vix'):[(.0010,.32,.42,.53),(.0002,.31,.32,.47),(-.0010,.61,.49,.51)], ('bs','sp500'):[(.0029,.08,.23,.41),(.0032,.15,.20,.36),(.0018,.32,.32,.39)],
 ('bs','dy10'):[(.0033,.09,.17,.35),(-.0016,.84,.59,.43),(-.0034,.94,.74,.59)], ('bs','vix'):[(.0049,.11,.15,.54),(.0067,.06,.12,.44),(.0026,.20,.32,.43)],
 ('rr','ebp'):[(.0004,.53,.48,.42),(.0002,.58,.56,.57),(.0030,.08,.19,.26)], ('jk','ebp'):[(.0040,.04,.10,.28),(.0038,.10,.21,.28),(.0016,.28,.41,.43)],
 ('bs','ebp'):[(.0003,.38,.41,.31),(-.0003,.55,.61,.36),(.0003,.54,.56,.36)], ('rr','spread'):[(-.0022,.87,.78,.51),(-.0025,.94,.76,.50),(-.0015,.73,.72,.41)],
 ('rr','dollar'):[(.0020,.24,.27,.37),(.0017,.30,.37,.32),(.0026,.10,.14,.27)], ('oil','spread'):[(.0014,.21,.28,.30),(-.0002,.53,.49,.32),(.0018,.19,.29,.23)]}
T3 = {'eighteen pairs':[(15,.0012,.054,.14),(14,.0018,.047,.060),(13,.0006,.24,.33)], 'fifteen pairs excl. EBP':[(None,.0012,.10,.17),(None,.0019,.046,.047),(None,.0004,.33,.35)],
      'twelve pairs excl. Bauer-Swanson and EBP':[(None,.0006,.26,.28),(None,.0017,.044,.033),(None,.0004,.27,.35)], 'nine price and volatility pairs':[(None,.0006,.27,.34),(None,.0024,.023,.026),(None,.0002,.38,.41)]}
T4A = {'Narrative MP':(-.0003,.0009,.0004), 'High-freq. MP':(.0024,.0048,.0006), 'Bauer-Swanson MP':(.0037,.0028,.0003), 'Oil news':(-.0004,.0014,-.0004)}
T4B = {'all four shocks (12 pairs)':(.032,.023,.067,.071), 'drop narrative':(.025,.023,.048,.097), 'drop Jarocinski-Karadi':(.076,.087,.14,.20),
       'drop Bauer-Swanson (the nine pairs)':(.023,.023,.030,.043), 'drop oil':(.040,.037,.094,.14)}
T5 = {('nine price and volatility pairs','P_N(3)'):(.0024,.023,.026), ('nine price and volatility pairs','hump H'):(.0020,.026,.046), ('nine price and volatility pairs','P_N(3)-P_N(1)'):(.0018,.047,.064),
      ('nine price and volatility pairs','P_N(3)-P_N(6)'):(.0022,.039,.068), ('all eighteen rule-based pairs','P_N(3)'):(.0018,.047,.060), ('all eighteen rule-based pairs','hump H'):(.0009,.16,.20),
      ('all eighteen rule-based pairs','P_N(3)-P_N(1)'):(.0006,.28,.26), ('all eighteen rule-based pairs','P_N(3)-P_N(6)'):(.0012,.12,.19),
      ('placebo calendar, 18 pairs, h=3','shift +12 months'):(.0007,.16,None), ('placebo calendar, 18 pairs, h=3','shift +24 months'):(.0010,.10,None),
      ('placebo calendar, 18 pairs, h=3','shift -12 months'):(.0003,.40,None), ('placebo calendar, 18 pairs, h=3','shift -24 months'):(.0016,.07,None)}
T6 = {'rr':[[.0018,-.0001,.0006,-.0008,.0003,.0059,.0048,-.0058,.0002],[.0010,-.0003,.0009,.0001,-.0009,-.0002,.0034,-.0005,.0031],[.0018,.0010,.0018,.0008,.0003,.0045,.0043,-.0025,.0059]],
      'jk':[[.0035,-.0011,.0030,-.0013,-.0007,.0000,.0048,.0017,.0028],[.0023,-.0027,.0056,-.0034,.0005,.0026,.0085,.0004,.0027],[.0020,-.0007,.0054,.0009,.0014,.0037,.0059,.0001,.0020]],
      'bs':[[.0010,-.0003,.0008,.0015,-.0005,.0002,.0004,.0014,-.0009],[.0027,-.0041,.0039,.0004,-.0002,.0004,.0029,.0004,-.0014],[.0008,-.0026,.0013,.0023,.0017,-.0003,-.0002,-.0001,.0006]]}
T6P = {'Narrative MP, 9 floaters':[(6,.0008,.09,.26),(5,.0007,.07,.29),(8,.0020,.008,.07)], 'High-freq. MP, 9 floaters':[(6,.0014,.11,.21),(7,.0018,.030,.18),(8,.0023,.06,.14)],
       'Bauer-Swanson MP, 9 floaters':[(6,.0004,.32,.33),(6,.0006,.14,.37),(5,.0004,.34,.59)], 'All 27 pairs':[(18,.0009,.09,.20),(18,.0010,.054,.15),(21,.0016,.048,.15)],
       '19 currencies, 1974-2001':[(17,.0072,.012,.007),(14,.0014,.20,.30),(17,.0033,.038,.15)], '9 floaters, 1974-2001':[(7,.0049,.024,.013),(7,.0011,.22,.29),(8,.0026,.054,.09)]}
T7 = {'all 19 basket':[(.0015,.10,.30),(.0025,.030,.12),(.0016,.09,.19)], 'European basket':[(.0067,.022,.030),(.0024,.15,.27),(.0036,.12,.30)], 'non-European basket':[(.0014,.13,.20),(.0010,.28,.34),(.0021,.09,.18)]}
T7LP = {0:(.41,.11,15), 1:(.55,.15,15), 3:(.19,.08,14), 6:(-.06,.05,7)}
T9 = {('Domestic price and volatility pairs, h=3',1):(8,.0027), ('Domestic price and volatility pairs, h=3',3):(8,.0017), ('Narrative x 19 currencies, h=1',1):(15,.0041), ('Narrative x 19 currencies, h=1',3):(15,.0032),
      ('Narrative x 19 currencies, h=6',1):(15,.0016), ('Narrative x 19 currencies, h=6',3):(17,.0022), ('All 27 currency pairs, h=6',1):(23,.0033), ('All 27 currency pairs, h=6',3):(18,.0010)}

# ---- comparison -------------------------------------------------------------------------------------------
L = []; tally = {}
def se_p(p, B): return np.sqrt(max(p * (1 - p), 1e-4) * 2 / B)          # MC s.e. of the difference of two independent bootstrap p-values
def chk(table, label, mine, paper, kind, B=None, tol=0.00006):
    if paper is None or (isinstance(mine, float) and np.isnan(mine)): return
    if kind == 'stat': ok = abs(mine - paper) <= tol
    elif kind == 'count': ok = int(mine) == int(paper)
    else: ok = abs(mine - paper) <= max(3 * se_p(paper, B), 0.011)
    k = (table, 'deterministic' if kind != 'p' else 'p-values'); a, b = tally.get(k, (0, 0)); tally[k] = (a + ok, b + 1)
    if not ok: L.append(f'| {table} | {label} | {kind} | {mine:.4f} | {paper:.4f} |')

have = lambda f: os.path.exists(os.path.join(RES, f))
if have('table2_pairs.csv'):
    t = rd('table2_pairs.csv')
    for (x, y), ref in T2.items():
        for h, (d_, w, p, f) in zip((1, 3, 6), ref):
            r = t[(t.x == x) & (t.outcome == y) & (t.h == h)].iloc[0]; lab = f'{x} x {y}, h={h}'
            chk('Table 2', lab + ' DII', r.DII, d_, 'stat'); chk('Table 2', lab + ' n*rho_f', r.n_rho_f, f, 'stat', tol=.0051); chk('Table 2', lab + ' wild', r.p_wild, w, 'p', 499); chk('Table 2', lab + ' paired', r.p_paired, p, 'p', 199)
if have('table3_pooled.csv'):
    t = rd('table3_pooled.csv'); t = t[t.h.astype(str).isin(['1', '3', '6'])]
    for s, ref in T3.items():
        for h, (c, P, w, p) in zip((1, 3, 6), ref):
            r = t[(t.set == s) & (t.h.astype(int) == h)].iloc[0]; Bw, Bp = (4999, 499) if h == 3 else (499, 149)
            chk('Table 3', f'{s}, h={h} count', r.positive, c, 'count'); chk('Table 3', f'{s}, h={h} P_N', r.P_N, P, 'stat'); chk('Table 3', f'{s}, h={h} wild', r.p_wild, w, 'p', Bw); chk('Table 3', f'{s}, h={h} paired', r.p_paired, p, 'p', Bp)
if have('table4A_family_means.csv'):
    t = rd('table4A_family_means.csv')
    for s, ref in T4A.items():
        for h, v in zip((1, 3, 6), ref): chk('Table 4A', f'{s}, h={h}', t[t.shock == s].iloc[0][f'h{h}'], v, 'stat')
if have('table4B_leave_one_out.csv'):
    t = rd('table4B_leave_one_out.csv')
    for s, (w, p, hw, hp) in T4B.items():
        r = t[t['sample'] == s].iloc[0]; Bw, Bp = (4999, 999) if s.startswith('all') else (2999, 299)
        chk('Table 4B', s + ' wild', r.p_wild, w, 'p', Bw); chk('Table 4B', s + ' paired', r.p_paired, p, 'p', Bp); chk('Table 4B', s + ' hump wild', r.hump_p_wild, hw, 'p', Bw); chk('Table 4B', s + ' hump paired', r.hump_p_paired, hp, 'p', Bp)
if have('table5_hump_placebo.csv'):
    t = rd('table5_hump_placebo.csv')
    for (s, st), (v, w, p) in T5.items():
        r = t[(t.set == s) & (t.statistic == st)].iloc[0]; tb = 'Table 5 placebo' if 'placebo' in s else 'Table 5'
        chk(tb, f'{s}: {st}', r.value, v, 'stat'); chk(tb, f'{s}: {st} wild', r.p_wild, w, 'p', 999 if 'placebo' in s else 4999); chk(tb, f'{s}: {st} paired', r.p_paired, p, 'p', 499)
if have('table6_currency_cells.csv'):
    t = rd('table6_currency_cells.csv'); cur = ['AUD', 'CAD', 'DKK', 'JPY', 'NZD', 'NOK', 'SEK', 'CHF', 'GBP']
    for x, g in T6.items():
        for h, row in zip((1, 3, 6), g):
            for c, v in zip(cur, row): chk('Table 6 cells', f'{x} x {c}, h={h}', t[(t.x == x) & (t.currency == c) & (t.h == h)].iloc[0].DII, v, 'stat')
    t = rd('table6_currency_pooled.csv')
    for s, ref in T6P.items():
        for h, (c, P, w, p) in zip((1, 3, 6), ref):
            r = t[(t.panel == s) & (t.h == h)].iloc[0]; chk('Table 6 pooled', f'{s}, h={h} count', r.positive, c, 'count'); chk('Table 6 pooled', f'{s}, h={h} P_N', r.P_N, P, 'stat'); chk('Table 6 pooled', f'{s}, h={h} wild', r.p_wild, w, 'p', 499); chk('Table 6 pooled', f'{s}, h={h} paired', r.p_paired, p, 'p', 149)
if have('table7_baskets.csv'):
    t = rd('table7_baskets.csv'); t = t[t.version == 'as published']
    for s, ref in T7.items():
        for h, (v, w, p) in zip((1, 3, 6), ref): r = t[(t.basket == s) & (t.h == h)].iloc[0]; chk('Table 7 baskets', f'{s}, h={h} DII', r.DII, v, 'stat'); chk('Table 7 baskets', f'{s}, h={h} wild', r.p_wild, w, 'p', 499); chk('Table 7 baskets', f'{s}, h={h} paired', r.p_paired, p, 'p', 199)
    t = rd('table7_local_projections.csv'); t = t[t.version == 'as published']
    for h, (b, dsp, c) in T7LP.items():
        r = t[t.h == h].iloc[0]; chk('Table 7 LP', f'mean beta, h={h}', r.mean_beta, b, 'stat', tol=.0051); chk('Table 7 LP', f'dispersion, h={h}', r.dispersion, dsp, 'stat', tol=.0101); chk('Table 7 LP', f'positive, h={h}', r.positive, c, 'count')
if have('table9_lag_robustness.csv'):
    t = rd('table9_lag_robustness.csv')
    for (s, p), (c, P) in T9.items():
        r = t[(t.result == s) & (t.lags == p)].iloc[0]; chk('Table 9', f'{s}, {p} lag(s) count', r.positive, c, 'count'); chk('Table 9', f'{s}, {p} lag(s) P_N', r.P_N, P, 'stat')

out = ['# Verification against the published tables', '', 'Generated by `code/verify.py`. "Deterministic" = point estimates, counts and n*rho_f, which depend only on the data and should match to the',
       'printed precision. "p-values" are bootstrap Monte Carlo quantities; the original seeds were not recorded, so a match means agreement within', 'three Monte Carlo standard errors of the difference (or 0.011 for rounding).', '',
       '| Table | Quantity type | Matched | Compared |', '|---|---|---|---|'] + [f'| {k[0]} | {k[1]} | {a} | {b} |' for k, (a, b) in tally.items()]
out += ['', '## Every quantity that does not match', '', '| Table | Quantity | Type | This package | Paper |', '|---|---|---|---|---|'] + (L or ['| (none) | | | | |'])
open(os.path.join(RES, 'verification_report.md'), 'w', encoding='utf8').write('\n'.join(out) + '\n'); print('\n'.join(out))
