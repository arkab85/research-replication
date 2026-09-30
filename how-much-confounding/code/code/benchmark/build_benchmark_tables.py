"""Build the application tables of the manuscript and supplement from stored outputs.

Writes, in 3_LaTeX_Source/results/:
  fomc_table.tex        Table (monetary-policy surprises)
  revised_vix_table.tex Table (returns and volatility)
  bandwidth_table.tex   Supplement table (bandwidth-set certificates)
Inputs (benchmark/results): fomc.json, fomc_adjusted.json, nu_floors_fomc.json, coskew_fomc.json,
equity_benchmark.json, equity_adjusted_b10.json, equity_adjusted_b33.json, nu_floors_equity.json.
Run: python benchmark/build_benchmark_tables.py
"""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
TEX = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), '3_LaTeX_Source', 'results')


def J(name):
    return json.load(open(os.path.join(RES, name)))


def f3(x):
    return '0' if x <= 0 else f'{x:.3f}'


def f2(x):
    return f'{x:.2f}'


def pv(p):
    return '$<.001$' if p < .001 else f'.{int(round(p * 100)):02d}'


def certified_set(low_sqrt, step):
    """Contiguous arcs of retained cells on the circle of period 90 degrees, modulo signed
    permutations; an arc that wraps through 0 is reported with negative start."""
    keep = np.array(low_sqrt) == 0; m = len(keep)
    if keep.all():
        return 'all'
    runs, seen = [], set()
    for i in range(m):
        if keep[i] and not keep[(i - 1) % m] and i not in seen:
            run = [i]; j = (i + 1) % m
            while keep[j] and j != i:
                run.append(j); j = (j + 1) % m
            runs.append(run); seen.update(run)
    out = []
    for r in runs:
        a = r[0] * step; b = (r[-1] + 1) * step
        if r[0] > r[-1]:
            a -= 90
        out.append(f'[{a:.0f},{b:.0f}]')
    return '$' + r'\cup'.join(out) + '$'


def floor_range(pf, cell_low):
    """Range of the point embedding floor over grid points bounding retained cells."""
    pf = np.array(pf); keep = np.array(cell_low) == 0
    idx = sorted(set(np.where(keep)[0]) | set(np.where(keep)[0] + 1))
    return f"{pf[idx].min():.2f}--{pf[idx].max():.2f}"


def fomc():
    r = J('fomc.json'); a = J('fomc_adjusted.json'); nf = J('nu_floors_fomc.json'); cs = J('coskew_fomc.json')
    B = ('block1', 'block19')
    rec = lambda j: ' & '.join(f"{r['raw'][b]['recursive']['norms'][j]:.3f}; {f3(r['raw'][b]['recursive']['lower_budget'][j])}" for b in B)
    rs = lambda j: ' & '.join(f3(r['rescaled'][b]['recursive']['lower_budget'][j]) for b in B)
    fl = lambda lab, k: ' & '.join(f"{f3(r[lab]['floors'][f'k{k}_' + b]['rho'])} ({f3(r[lab]['floors'][f'k{k}_' + b]['lower'])})" for b in B)
    nu = lambda k: ' & '.join(floor_range(nf[b][f'k{k}']['pair_floor_point'], r['raw'][b]['cell_lower_sqrt']) for b in B)
    iso = ' & '.join(', '.join(pv(r['raw']['isotropy'][f'k{k}_' + b]['p']) for k in (2, 3, 5)) for b in B)
    flp = lambda lab: ' & '.join(', '.join(f2(r[lab]['floors'][f'k{k}_' + b]['rho']) for k in (2, 3, 5)) for b in B)
    fll = lambda lab: ' & '.join(', '.join(f2(r[lab]['floors'][f'k{k}_' + b]['lower']) if r[lab]['floors'][f'k{k}_' + b]['lower'] > 0 else '0' for k in (2, 3, 5)) for b in B)
    def nr(b):
        lo = min(float(x.split('--')[0]) for x in [floor_range(nf[b][f'k{k}']['pair_floor_point'], r['raw'][b]['cell_lower_sqrt']) for k in (2, 3, 5)])
        hi = max(float(x.split('--')[1]) for x in [floor_range(nf[b][f'k{k}']['pair_floor_point'], r['raw'][b]['cell_lower_sqrt']) for k in (2, 3, 5)])
        return f'{lo:.2f}--{hi:.2f}'
    nurange = ' & '.join(nr(b) for b in B)
    nusf = ', '.join(f2(nf['block1'][f'k{k}']['orders_pair_floor_point'][1]) for k in (2, 3, 5))
    pre_s = ' & '.join(', '.join(pv(r['rescaled']['floors'][f'k{k}_' + b]['equal_means_p']) for k in (2, 3, 5)) for b in B)
    adjrow = ' & '.join(', '.join(f3(a[b]['adjusted'][f'k{k}']['breakdown'][1]) for k in (2, 3, 5)) for b in B)
    adjset = ' & '.join(certified_set(a[b]['adjusted_grid']['k3']['cell_lower_sqrt'], 1.0) for b in B)
    rawset = ' & '.join(certified_set(r['raw'][b]['cell_lower_sqrt'], 1.0) for b in B)
    deg = np.array(cs['deg']); w = np.array(cs['wald'])
    ang = r['raw']['block1']['recursive']['angles_deg'][1]
    w0 = w[np.argmin(abs(deg - 0))]; wsf = w[np.argmin(abs(deg - ang))]
    body = rf"""\begin{{table}}[tbp]
\centering\setstretch{{1.5}}\setlength{{\tabcolsep}}{{5pt}}
\caption{{Two monetary-policy surprises: certificate, measured confounding, adjusted certificate}}\label{{tab:fomc}}
\begin{{tabular}}{{lrr}}
\toprule
 & iid & blocks of 19\\
\midrule
\multicolumn{{3}}{{l}}{{\emph{{A. Certificate: $\|\widehat C\|$; breakdown $\widehat\rho^{{\,*}}$ (joint .95 over the two orderings)}}}}\\
Rates first ($0^\circ$) & {rec(0)}\\
Stocks first (${ang:.1f}^\circ$) & {rec(1)}\\
Certified set at zero exposure (degrees) & {rawset}\\
\multicolumn{{3}}{{l}}{{\emph{{B. Measured confounding, pre-announcement VIX regimes $k=2,3,5$}}}}\\
Volatility floor $\widehat\rho_{{W_k}}$, $k=2,3,5$ & {flp('raw')}\\
\hspace{{1em}}95\% lower bounds & {fll('raw')}\\
Embedding floor at stocks first, $k=2,3,5$ & \multicolumn{{2}}{{c}}{{{nusf}}}\\
Isotropy $p$-value, $k=2,3,5$ & {iso}\\
\multicolumn{{3}}{{l}}{{\emph{{C. Removing it: stocks-first breakdown}}}}\\
Conditioning on $k=2,3,5$ regimes & {adjrow}\\
\hspace{{1em}}Adjusted certified set, $k=3$ & {adjset}\\
Dividing by VIX & {rs(1)}\\
\hspace{{1em}}Pretest of zero rescaled floor, $p$-value, $k=2,3,5$ & {pre_s}\\
\multicolumn{{3}}{{l}}{{\emph{{D. Specification: co-skewness Wald statistic}}}}\\
Rates first; stocks first; minimum over rotations & \multicolumn{{2}}{{c}}{{{w0:.1f}; {wsf:.1f}; {w.min():.1f} (at ${cs['argmin']:.0f}^\circ$)}}\\
\bottomrule\end{{tabular}}
\tablenotes{{325 announcements, February 1990 to March 2026; the surprises are treated as innovations. Computations as in Table~\ref{{tab:vix}}, with 999 draws, the stocks-first angle recomputed in every draw and one-degree cells; floor lower bounds are zero when the boundary pretest does not reject. Embedding floors are point estimates of $f_{{12}}$ from U-statistics at the rejected ordering; lower bounds from a 97.5\% band joint over rotations are at most 0.05. Adjusted breakdowns use the regime-conditioned operator with regimes recomputed in every draw; the adjusted set uses three regimes. Rescaled breakdowns divide both surprises by pre-announcement VIX. Co-skewness statistics use an iid bootstrap covariance; the $\chi^2_2$ .95 value is 5.99, which over-rejects in Table~\ref{{tab:recsize}}.}}
\end{{table}}
"""
    open(os.path.join(TEX, 'fomc_table.tex'), 'w').write(body)


def equity():
    r = J('equity_benchmark.json')
    A = {b: J(f'equity_adjusted_{b}.json') for b in ('b10', 'b33')}
    nf = J('nu_floors_equity.json') if os.path.exists(os.path.join(RES, 'nu_floors_equity.json')) else None
    B = ('block10', 'block33'); AB = ('b10', 'b33')
    rec = lambda j: ' & '.join(f"{r['raw'][b]['norms'][j]:.3f}; {f3(r['raw'][b]['lower_budget'][j])}" for b in B)
    fl = lambda k: ' & '.join(f"{f3(r['raw']['floors'][f'k{k}_' + b]['rho'])} ({f3(r['raw']['floors'][f'k{k}_' + b]['lower'])})" for b in B)
    nu = (lambda k: ' & '.join(floor_range(nf[b][f'k{k}']['pair_floor_point'], r['raw'][b]['cell_lower_sqrt']) for b in B)) if nf else (lambda k: '. & .')
    adj = lambda k: ' & '.join(', '.join(f3(x) for x in A[b]['adjusted'][f'k{k}']['breakdown']) for b in AB)
    G = {b: J(f'equity_grid1_{b}.json') for b in ('b10', 'b33')} if all(os.path.exists(os.path.join(RES, f'equity_grid1_{b}.json')) for b in ('b10', 'b33')) else None
    if not G:
        adjset = ' & '.join(certified_set(A[b]['adjusted_grid']['cell_lower_sqrt'], 3.0) for b in AB)
    if G:
        rawset = ' & '.join(certified_set(G[b]['raw']['cell_lower_sqrt'], 1.0) for b in AB)
        adjset = ' & '.join(certified_set(G[b]['cond']['cell_lower_sqrt'], 1.0) for b in AB)
        arrow = ' & '.join('; '.join(f"[{min(G[b]['ar'][f'k{k}']['set']):.0f},{max(G[b]['ar'][f'k{k}']['set']):.0f}]" if G[b]['ar'][f'k{k}']['set'] else 'empty' for k in (2, 3, 5)) for b in AB)
    else:
        rawset = ' & '.join(certified_set(r['raw'][b]['cell_lower_sqrt'], 3.0) for b in B)
        arrow = '. & .'
    rs = ' & '.join(', '.join(f3(x) for x in r['rescaled'][b]['lower_budget']) for b in B)
    condresc = ' & '.join(', '.join(f3(x) for x in A[ab]['adjusted']['k3']['breakdown']) + '; ' + ', '.join(f3(x) for x in r['rescaled'][b]['lower_budget']) for ab, b in zip(AB, B))
    flp = ' & '.join(', '.join(f2(r['raw']['floors'][f'k{k}_' + b]['rho']) for k in (2, 3, 5)) for b in B)
    fll = ' & '.join(', '.join(f2(r['raw']['floors'][f'k{k}_' + b]['lower']) for k in (2, 3, 5)) for b in B)
    hi = int(round(r['raw']['het']['k2']['angle'] / 3.0))
    flc = ' & '.join(', '.join(f"{r['raw']['floors'][f'k{k}_' + b]['rho']:.2f} ({r['raw']['floors'][f'k{k}_' + b]['lower']:.2f})" for k in (2, 3, 5)) for b in B)
    nuo = [', '.join(f2(nf['block10'][f'k{k}']['orders_pair_floor_point'][j]) for k in (2, 3, 5)) for j in range(2)]
    het33 = (lambda h: f"{h['angle']:.1f} [{h['ci95'][0]:.1f}, {h['ci95'][1]:.1f}]")(J('equity_het33.json')['k2']) if os.path.exists(os.path.join(RES, 'equity_het33.json')) else '.'
    hx = J('equity_hetexposure.json')
    het = lambda k: f"{r['raw']['het'][f'k{k}']['angle']:.1f} [{r['raw']['het'][f'k{k}']['ci95'][0]:.1f}, {r['raw']['het'][f'k{k}']['ci95'][1]:.1f}]"
    sf = r['raw']['floors']['k2_shock_floors']
    iso = '$<.001$' if max(r['raw']['isotropy'][k]['p'] for k in r['raw']['isotropy']) < .001 else 'see text'
    csw = ' & '.join(f"{r['raw'][b]['coskew_wald'][0]:.1f}, {r['raw'][b]['coskew_wald'][1]:.1f}" for b in B)
    csh = ' & '.join(f"{A[b]['coskew_het']['wald']:.1f}" for b in AB)
    ang2 = r['raw']['block10']['angles_deg'][1]
    a2 = ', '.join(f3(x) for x in A['b10']['adjusted']['k2']['breakdown']); a5 = ', '.join(f3(x) for x in A['b10']['adjusted']['k5']['breakdown'])
    body = rf"""\begin{{table}}[tbp]
\centering\setstretch{{1.4}}\setlength{{\tabcolsep}}{{4pt}}
\caption{{Returns and volatility: certificate, measured confounding, adjusted certificate}}\label{{tab:vix}}
\begin{{tabular}}{{lrr}}
\toprule
 & blocks of 10 & blocks of 33\\
\midrule
\multicolumn{{3}}{{l}}{{\emph{{A. Certificate: $\|\widehat C\|$; breakdown $\widehat\rho^{{\,*}}$ (joint .95 over the two orderings)}}}}\\
Returns first ($0^\circ$) & {rec(0)}\\
VIX first (${ang2:.1f}^\circ$) & {rec(1)}\\
Certified set at zero exposure (degrees) & {rawset}\\
\multicolumn{{3}}{{l}}{{\emph{{B. Measured confounding, lagged VIX regimes $k=2,3,5$}}}}\\
Volatility floor $\widehat\rho_{{W_k}}$ & {flp}\\
\hspace{{1em}}95\% lower bounds & {fll}\\
Embedding floor, returns first; VIX first & \multicolumn{{2}}{{c}}{{{nuo[0]}; {nuo[1]}}}\\
\multicolumn{{3}}{{l}}{{\emph{{C. Removing it: breakdowns at the two orderings}}}}\\
Conditioning on three regimes & {adj(3)}\\
Dividing by lagged VIX & {rs}\\
Heteroskedasticity rotation, $k=2$ (95\% interval) & {het(2)} & {het33}\\
\hspace{{1em}}Robust set, $k=2$; $3$; $5$ & {arrow}\\
\hspace{{1em}}Exposure at it (lower bound), raw; cond. & {hx['block10']['exposure_lower']:.3f}; {hx['block10']['residual_exposure_lower']:.3f} & {hx['block33']['exposure_lower']:.3f}; {hx['block33']['residual_exposure_lower']:.3f}\\
\multicolumn{{3}}{{l}}{{\emph{{D. Specification: co-skewness Wald statistic}}}}\\
Two orderings; heteroskedasticity rotation & {csw.split(' & ')[0]}; {csh.split(' & ')[0]} & {csw.split(' & ')[1]}; {csh.split(' & ')[1]}\\
\bottomrule\end{{tabular}}
\tablenotes{{1,039 weekly innovations, 1999--2018, VAR(4), 399 refitted draws. Breakdowns as in Definition~\ref{{def:breakdown}} with the joint .95 critical value at the two orderings, the VIX-first angle recomputed in every draw; certified sets use one-degree cells, the .975 radius and the curvature allowance. Volatility floors use \eqref{{eq:rhohat}} with 999 draws; the boundary pretest and the isotropy test reject at every $k$ ($p<.001$). Embedding floors are point estimates of $f_{{12}}$ at the two orderings; a 97.5\% band joint over rotations runs from at most 0.04 to between 0.14 and 0.21. Conditioned breakdowns recompute the regimes in every draw; with two and five regimes they are {a2} and {a5} (blocks of ten); the conditioned certified sets are {adjset.replace(' & ', ' and ')}. Heteroskedasticity rotations use 499 refitted draws; with three and five regimes they are {het(3)} and {het(5)} degrees. Robust sets invert the Wald test of zero within-regime covariance of the candidate shocks over one-degree angles ($\chi^2_{{k-1}}$ critical values); an empty set means the overidentifying restrictions are rejected at every angle (smallest $p$-values .03 and .002 for $k=3$ and $5$). The rotation is re-estimated in every draw for the exposure bounds and the co-skewness statistic ($\chi^2_2$ .95 value 5.99).}}
\end{{table}}
"""
    open(os.path.join(TEX, 'revised_vix_table.tex'), 'w').write(body)


def bandwidth():
    a = J('fomc_adjusted.json'); e = {b: J(f'equity_adjusted_{b}.json') for b in ('b10', 'b33')}
    fmt = lambda x: ', '.join(f3(v) for v in x)
    rows = []
    for lab, src, keys in (('Monetary-policy surprises', a, ('block1', 'block19')), ('Returns and volatility', e, ('b10', 'b33'))):
        get = (lambda k: src[k]) if lab.startswith('Mon') else (lambda k: src[k])
        rows.append(rf"\multicolumn{{3}}{{l}}{{\emph{{{lab}}}}}\\")
        for kind, name in (('bandwidth_raw', 'Raw'), ('bandwidth_adjusted', 'Conditioned on three regimes')):
            for s in ('0.5', '1.0', '2.0'):
                rows.append(f"{name}, bandwidth {s} alone & " + ' & '.join(fmt(get(k)[kind]['breakdown_single'][s]) for k in keys) + r'\\')
            rows.append(f"{name}, all three jointly & " + ' & '.join(fmt(get(k)[kind]['breakdown_joint']) for k in keys) + r'\\')
    body = r"""\begin{table}[tbp]
\centering\setstretch{1.5}\setlength{\tabcolsep}{5pt}
\caption{Bandwidth robustness of the recursive-ordering breakdowns}\label{tab:bandwidth}
\begin{tabular}{lrr}
\toprule
 & short blocks & long blocks\\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule\end{tabular}
\tablenotes{Entries are breakdown budgets at the two recursive orderings (rates first, stocks first; returns first, VIX first). Short blocks are iid and ten weeks, long blocks 19 announcements and 33 weeks. Since $\|\mathcal C^{s}\|\le\nu^{(s)}_i\nu^{(s)}_j\le\rho_i\rho_j/s^2$ for bandwidth $s$, the joint version uses the .95 quantile of the maximum over bandwidths and orderings of $s^2\|\widehat{\mathcal C}^{s*}-\widehat{\mathcal C}^{s}\|$ and reports $\max_s\{[s^2\|\widehat{\mathcal C}^{s}\|-q]_+\}^{1/2}$, a breakdown for the budgets $\rho_i$ valid for all three bandwidths at once; single-bandwidth rows use their own .95 critical values. The draws differ from those of Tables~\ref{tab:vix} and~\ref{tab:fomc}, so unit-bandwidth entries can differ slightly from them.}
\end{table}
"""
    open(os.path.join(TEX, 'bandwidth_table.tex'), 'w').write(body)


if __name__ == '__main__':
    fomc(); equity(); bandwidth(); print('tables written')


def calibration():
    """Supplement table: post hoc calibration of the conditioned recursive test."""
    rows = []
    specs = [('k5', 'Two-state, proxy noise 0.05, $k=2$'), ('k50', 'Two-state, proxy noise 0.5, $k=2$'),
             ('k150_weakproxy', 'Two-state, proxy noise 1.5, $k=2$'), ('k50_lognormal', 'Log-normal SV, proxy noise 0.5, $k=3$'),
             ('k50_fomclike', '$n=325$, Student-$t_5$, blocks 19, $k=3$')]
    for tag, lab in specs:
        f = os.path.join(RES, f'adjusted_calibration_{tag}.json')
        if not os.path.exists(f):
            continue
        c = J(f'adjusted_calibration_{tag}.json'); m = c['summary']
        rows.append(f"{lab} & {c['rho']:.3f} & {c['rho_residual']:.3f} & {100 * m['raw_true_reject']:.1f} & {100 * m['raw_true_exceeds_rho']:.1f} & "
                    f"{100 * m['adj_true_reject']:.1f} & {100 * m['adj_true_exceeds_rho_residual']:.1f} & {100 * m['raw_false_reject']:.0f}, {100 * m['adj_false_reject']:.0f}\\\\")
    body = r"""\begin{table}[tbp]
\centering\setstretch{1.5}\setlength{\tabcolsep}{3pt}
\caption{Calibration of the conditioned recursive-ordering test (\%, designed after the applications)}\label{tab:adjcal}
\begin{tabular}{lrrrrrrr}
\toprule
 & & & \multicolumn{2}{c}{Raw test} & \multicolumn{2}{c}{Conditioned test} & False order\\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Design & $\rho$ & $\rho^{\mid W_k}$ & reject & $>\rho$ & reject & $>\rho^{\mid W_k}$ & raw, cond.\\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule\end{tabular}
\tablenotes{True first-ordered structure $A=\bigl(\begin{smallmatrix}1&0\\.5&1\end{smallmatrix}\bigr)$, common volatility with $\rho=0.371$ (two-state chain with stay probability .95, or log-normal AR(1) log-volatility with persistence .95), demeaned exponential innovations unless stated, proxy $W_t$ equal to the volatility state plus Gaussian noise, $n=600$ and blocks of ten unless stated, 199 bootstrap draws and 200 replications per design. ``Reject'' is the frequency with which the breakdown at the true ordering is positive; ``$>\rho$'' and ``$>\rho^{\mid W_k}$'' the frequencies with which it exceeds the true budget or the true residual budget of Theorem~\ref{thm:adjusted}; the last column is the rejection frequency at the false ordering. These designs are not part of the protocols fixed in advance.}
\end{table}
"""
    open(os.path.join(TEX, 'adjusted_calibration_table.tex'), 'w').write(body)


if __name__ == '__main__':
    calibration()


def fomc4():
    r = J('fomc4.json'); bw = J('fomc4_bw.json') if os.path.exists(os.path.join(RES, 'fomc4_bw.json')) else None
    B = ('block1', 'block18'); names = {'0': 'MP1', '1': '2y', '2': '10y', '3': 'S\\&P'}

    def lab(k):
        return '--'.join(names[c] for c in k)

    def summ(b, key):
        o = r[b]['orderings']; rej = [v[key] for v in o.values() if v[key] > 0]
        return len(rej), (min(rej), max(rej)) if rej else (0, 0)
    rows = []
    for key, name in (('breakdown', 'Raw'), ('breakdown_cond', 'Conditioned, three regimes')):
        c = [summ(b, key) for b in B]
        rows.append(f"{name}: rejected (of 24) & {c[0][0]} & {c[1][0]}\\\\")
        rows.append(f"\\hspace{{1em}}breakdowns when rejected & {c[0][1][0]:.2f}--{c[0][1][1]:.2f} & {c[1][1][0]:.2f}--{c[1][1][1]:.2f}\\\\")
    surv = [sorted(k for k, v in r[b]['orderings'].items() if v['breakdown'] == 0) for b in B]
    mp1 = [sum(k[0] == '0' for k in s) for s in surv]
    rows.append(f"Not rejected, raw: MP1 first; other & {mp1[0]}; {len(surv[0]) - mp1[0]} & {mp1[1]}; {len(surv[1]) - mp1[1]}\\\\")
    if bw:
        rows.append("Bandwidths $0.5,1,2$ jointly: rejected, raw; cond. & " + ' & '.join(f"{bw[b]['raw']['rejected']}; {bw[b]['cond']['rejected']}" for b in B) + "\\\\")
    fl = ' & '.join(', '.join(f"{r['floors'][f'k{k}_' + b]['rho']:.2f}" for k in (2, 3, 5)) for b in B)
    fll = ' & '.join(', '.join(f"{r['floors'][f'k{k}_' + b]['lower']:.2f}" for k in (2, 3, 5)) for b in B)
    iso = ' & '.join('$<.001$ each' if max(r['isotropy'][f'k{k}_' + b]['p'] for k in (2, 3, 5)) < .001 else ', '.join(pv(r['isotropy'][f'k{k}_' + b]['p']) for k in (2, 3, 5)) for b in B)
    h = [r[b]['het'] for b in B]
    ang = r['min_dependence']['angles_to_het']
    body = rf"""\begin{{table}}[tbp]
\centering\setstretch{{1.5}}\setlength{{\tabcolsep}}{{4pt}}
\caption{{Four monetary-policy surprises: orderings, measured confounding, identified rotation}}\label{{tab:fomc4}}
\begin{{tabular}}{{lrr}}
\toprule
 & iid & blocks of 18\\
\midrule
\multicolumn{{3}}{{l}}{{\emph{{A. The 24 recursive orderings (joint .95 over orderings and pairs)}}}}\\
{rows[0]}
{rows[1]}
{rows[4]}
\multicolumn{{3}}{{l}}{{\emph{{B. Measured confounding, pre-announcement VIX regimes $k=2,3,5$}}}}\\
Volatility floor & {fl}\\
\hspace{{1em}}95\% lower bounds & {fll}\\
Isotropy $p$-value (df $9,18,36$) & {iso}\\
\multicolumn{{3}}{{l}}{{\emph{{C. Removing it}}}}\\
{rows[2]}
{rows[3]}
{rows[5] if bw else ''}
\multicolumn{{3}}{{l}}{{\emph{{D. Heteroskedasticity-identified rotation, three regimes}}}}\\
Bootstrap column-angle radius, degrees & {min(h[0]['column_angle_q95']):.0f}--{max(h[0]['column_angle_q95']):.0f} & {min(h[1]['column_angle_q95']):.0f}--{max(h[1]['column_angle_q95']):.0f}\\
Exposure at it (lower bound), raw; conditioned & {f3(h[0]['exposure_lower'])}; {f3(h[0]['residual_exposure_lower'])} & {f3(h[1]['exposure_lower'])}; {f3(h[1]['residual_exposure_lower'])}\\
Co-skewness Wald, 12 df ($p$-value) & {h[0]['coskew_wald']:.1f} ({pv(h[0]['coskew_p'])}) & {h[1]['coskew_wald']:.1f} ({pv(h[1]['coskew_p'])})\\
Angles to independence-minimizing rotation & \multicolumn{{2}}{{c}}{{{', '.join(f'{a:.0f}' for a in sorted(ang))}}}\\
\bottomrule\end{{tabular}}
\tablenotes{{{r['n']} announcements, January 1991 to March 2026, with surprises in the near-term fed funds futures rate (MP1), 2- and 10-year Treasury futures and the S\&P 500, the variables of \citet{{Jarocinski2024}} with futures in place of on-the-run yields; 499 draws. Orderings are recursive (Cholesky) orderings of these four variables, recomputed in every draw; an ordering is rejected when its breakdown, the square root of the largest pairwise operator norm minus the joint critical value, is positive. The bandwidth row uses the joint critical value of Table~\ref{{tab:bandwidth}}. Volatility floors use 999 draws and the boundary pretest. The identified rotation jointly diagonalizes the three regime covariances and is re-estimated in every draw; the column-angle radius is the range over shocks of the .95 quantile of the angle between bootstrap and original columns, after signed-permutation matching; the exposure bounds are those of Corollary~\ref{{cor:hetexposure}}.}}
\end{{table}}
"""
    open(os.path.join(TEX, 'fomc4_table.tex'), 'w').write(body)


if __name__ == '__main__':
    fomc4()
