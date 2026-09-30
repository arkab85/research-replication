"""Aggregate the pre-specified addendum and rebuild the three calibration tables so that they
show the main confirmatory study and the addendum together (post hoc reporting code)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from confirm_addendum import RB_DESIGNS, JW_DESIGNS, WB_DESIGNS, RES
from summarize_confirm import LABEL, pct, se, TEX
import summarize_confirm


def load(part, name):
    path = os.path.join(RES, f'addendum_{part}_{name}.jsonl')
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def main():
    summarize_confirm.main()
    main_sum = json.load(open(os.path.join(RES, 'confirm_summary.json')))
    out = {'RB': {}, 'JW': {}, 'WB': {}}
    for name in RB_DESIGNS:
        rows = load('RB', name)
        if rows:
            d = dict(reps=len(rows), rho=rows[0]['rho'], cpop=rows[0]['cpop'])
            for k in ('op_cover', 'op_reject_true', 'op_reject_false', 'budget_violation'):
                x = [r[k] for r in rows]; d[k] = pct(x); d[k + '_se'] = se(x)
            for lab, cv in (('95', 5.991464547107979), ('975', 7.377758908227871)):
                d['coskew_true_rej_' + lab] = pct([r['W_true'] > cv for r in rows])
                d['coskew_false_rej_' + lab] = pct([r['W_false'] > cv for r in rows])
            out['RB'][name] = d
    for name in JW_DESIGNS:
        rows = load('JW', name)
        if rows:
            d = dict(reps=len(rows))
            for k in ('rot_cover', 'L_old', 'L_new', 'joint_old', 'joint_new'):
                x = [r[k] for r in rows]; d[k] = pct(x); d[k + '_se'] = se(x)
            d['ind_share'] = pct([r['ind_share'] for r in rows])
            d['median_c_new'] = float(np.median([r['c_new'] for r in rows]))
            out['JW'][name] = d
    for name in WB_DESIGNS:
        rows = load('WB', name)
        if rows:
            c = [r['cover'] for r in rows]
            out['WB'][name] = dict(reps=len(rows), block=rows[0]['ell'], coverage=pct(c), mc_se=se(c),
                                   median_critical=float(np.median([r['c'] for r in rows])))
    json.dump(out, open(os.path.join(RES, 'addendum_summary.json'), 'w'), indent=2)

    # ---- Table 1 with addendum rows
    J = main_sum['J']
    a = r"""\begin{table}[tbp]
\centering\setstretch{1.5}\setlength{\tabcolsep}{4pt}
\caption{Pre-specified calibration on fresh seeds: rotation, whitening and joint coverage (\%)}\label{tab:svarsimfs}
\begin{tabular}{lrrrrrr}
\toprule
 & Rotation & \multicolumn{2}{c}{Whitening region} & \multicolumn{2}{c}{Joint} & Retained\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Common volatility & cells & Percentile & Studentized & Percentile & Studentized & cells\\
\midrule
"""
    lab = {'J1_iid_d0_n300': 'None', 'J2_iid_d4_n300': r'iid, $\rho=0.371$', 'J3_mk95_d4_n300': r'Persistent, $\rho=0.371$'}
    for name, d in J.items():
        a += f"{lab[name]} & {d['budget_cover']:.1f} & {d['L_old']:.1f} & {d['L_new']:.1f} & {d['joint_old']:.1f} & {d['joint_new']:.1f} & {d['budget_share']:.1f} \\\\\n"
    ses = [d[k + '_se'] for d in J.values() for k in ('L_old', 'L_new', 'joint_old', 'joint_new')]
    if out['JW']:
        a += "\\midrule\n"
        lab2 = {'J4_gauss_n300': r'None; Gaussian shocks', 'J5_chi40_n300': r'None; $\chi^2_{40}$ shocks'}
        for name, d in out['JW'].items():
            a += f"{lab2[name]} & {d['rot_cover']:.1f} & {d['L_old']:.1f} & {d['L_new']:.1f} & {d['joint_old']:.1f} & {d['joint_new']:.1f} & {d['ind_share']:.1f} \\\\\n"
            ses += [d[k + '_se'] for k in ('L_old', 'L_new', 'joint_old', 'joint_new')]
    a += r"""\bottomrule\end{tabular}
\par\smallskip\noindent 500 replications per row, $n=300$, VAR(1), six-degree grid without the true angle, same samples for all columns. Skewed exponential innovations in the first three rows; the last two rows (addendum) have Gaussian or standardized $\chi^2_{40}$ shocks. Operator and percentile region: 199 refitted draws; studentized region: 1,999. Nominal: .975 per region, .95 jointly. Persistent design: blocks of ten. Largest Monte Carlo standard error: """ + f"{max(ses):.1f}" + r""" points.
\end{table}
"""
    open(os.path.join(TEX, 'confirm_joint_table.tex'), 'w').write(a)

    # ---- Table 2 with the long-block column
    W = main_sum['W']
    key = {n.split('_')[0][:-1] if n.split('_')[0].endswith('L') else n.split('_')[0]: n for n in WB_DESIGNS}
    a = r"""\begin{table}[tbp]
\centering\setstretch{1.5}\setlength{\tabcolsep}{4pt}
\caption{Pre-specified whitening-region calibration (nominal 97.5\%)}\label{tab:whitecal}
\begin{tabular}{lrrrrr}
\toprule
 & & \multicolumn{3}{c}{Blocks as listed} & Blocks $\lceil\sqrt n\,\rceil$\\
\cmidrule(lr){3-5}\cmidrule(lr){6-6}
Common volatility & $n$ & Block & Coverage & Median $c$ & Coverage\\
\midrule
"""
    for name, d in W.items():
        l, n, ell = LABEL[name]
        tag = name.split('_')[0]
        wb = out['WB'].get(key.get(tag, ''), None) if tag in key else None
        col = f"{wb['coverage']:.1f} ({wb['mc_se']:.1f})" if wb else '--'
        a += f"{l} & {n} & {ell} & {d['coverage']:.1f} ({d['mc_se']:.1f}) & {d['median_critical']:.1f} & {col} \\\\\n"
    a += r"""\bottomrule\end{tabular}
\par\smallskip\noindent 1,000 replications per cell, 1,999 draws, Monte Carlo standard errors in parentheses. $\rho=0.371$ with common volatility; exponential innovations unless stated; block 1 is the iid bootstrap. Last column: addendum with moving blocks of $\lceil\sqrt n\,\rceil$ (18 or 25). The $t_5$ design violates the moment condition of Proposition~\ref{prop:studentized}. The $\chi^2_3$ .975 quantile is 9.35.
\end{table}
"""
    open(os.path.join(TEX, 'confirm_whitening_table.tex'), 'w').write(a)

    # ---- Table 3 with addendum rows
    R = main_sum['R']
    a = r"""\begin{table}[tbp]
\centering\setstretch{1.5}\setlength{\tabcolsep}{4pt}
\caption{Pre-specified calibration of the recursive-ordering tests (\%)}\label{tab:recsize}
\begin{tabular}{lrrrrrr}
\toprule
 & \multicolumn{3}{c}{Operator test} & \multicolumn{3}{c}{Co-skewness Wald}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
 & Coverage & Reject & Reject & Reject true & Reject true & Reject\\
Common volatility & true & true & false & at .05 & at .025 & false\\
\midrule
"""
    lab = {'R1_iid_d0_n600': 'None', 'R2_iid_d4_n600': r'iid two-point', 'R3_mk95_d4_n600': r'Two-state, stay .95',
           'R4_logsv_n600': r'Log-normal SV, $\phi=.95$', 'R5_mk98_mix_n600': r'Two-state, stay .98$^\dagger$'}
    for name, d in list(R.items()) + ([('__mid__', None)] if out['RB'] else []) + list(out['RB'].items()):
        if name == '__mid__':
            a += "\\midrule\n"; continue
        a += f"{lab[name]} & {d['op_cover']:.1f} & {d['op_reject_true']:.1f} & {d['op_reject_false']:.1f} & {d['coskew_true_rej_95']:.1f} & {d['coskew_true_rej_975']:.1f} & {d['coskew_false_rej_975']:.1f} \\\\\n"
    a += r"""\bottomrule\end{tabular}
\par\smallskip\noindent 500 replications per row, $n=600$, true first-ordered recursive structure, 199 moving-block draws, blocks of ten. $\rho=0.371$ with common volatility; the last two rows (addendum) are more persistent; $^\dagger$contaminated innovations. Coverage: the operator lower bound at the true ordering does not exceed the population norm (0, 0.0255, 0.0255, 0.0206, 0.0163 by row). Reject: positive lower bound; a size without common volatility. Co-skewness: rejection at the $\chi^2_2$ .95 and .975 critical values; last column .975 at the false ordering.
\end{table}
"""
    open(os.path.join(TEX, 'confirm_recursive_table.tex'), 'w').write(a)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
