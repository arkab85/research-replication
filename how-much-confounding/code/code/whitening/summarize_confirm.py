"""Aggregate the confirmatory study and write summary JSON and LaTeX tables."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from confirm_study import W_DESIGNS, J_DESIGNS, R_DESIGNS, REPS, RES

TEX = os.path.abspath(os.path.join(HERE, '..', '..', '..', '3_LaTeX_Source', 'results'))


def load(part, name):
    path = os.path.join(RES, f'confirm_{part}_{name}.jsonl')
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path)]


def pct(x):
    return 100 * float(np.mean(x))


def se(x):
    p = float(np.mean(x)); return 100 * np.sqrt(p * (1 - p) / len(x))


LABEL = {
    'V1_iid_d0_n300': (r'None', 300, 1),
    'V2_iid_d4_n300': (r'iid two-point', 300, 1),
    'V3_mk95_d4_n300': (r'Two-state, stay .95', 300, 10),
    'V3b_mk95_d4_n300_l20': (r'Two-state, stay .95', 300, 20),
    'V4_iid_d4_n600': (r'iid two-point', 600, 1),
    'V5_mk95_d4_n600': (r'Two-state, stay .95', 600, 10),
    'V5b_mk95_d4_n600_l20': (r'Two-state, stay .95', 600, 20),
    'V6_logsv_n600': (r'Log-normal SV, $\phi=.95$', 600, 10),
    'V7_mk98_mix_n600': (r'Two-state, stay .98; contaminated $\chi^2_2$', 600, 10),
    'V8_iid_t5_n600': (r'iid two-point; Student-$t_5$ (stress)', 600, 1),
}


def main():
    out = {'W': {}, 'J': {}, 'R': {}}
    for name in W_DESIGNS:
        rows = load('W', name)
        if rows:
            c = [r['cover'] for r in rows]
            out['W'][name] = dict(reps=len(rows), coverage=pct(c), mc_se=se(c),
                                  median_critical=float(np.median([r['c'] for r in rows])))
    for name in J_DESIGNS:
        rows = load('J', name)
        if rows:
            d = dict(reps=len(rows), rho=rows[0]['rho'])
            for k in ('ind_cover', 'budget_cover', 'L_old', 'L_new', 'joint_old', 'joint_new'):
                x = [r[k] for r in rows]; d[k] = pct(x); d[k + '_se'] = se(x)
            d['budget_share'] = pct([r['budget_share'] for r in rows]); d['ind_share'] = pct([r['ind_share'] for r in rows])
            d['median_c_new'] = float(np.median([r['c_new'] for r in rows])); d['median_r_old'] = float(np.median([r['r_old'] for r in rows]))
            out['J'][name] = d
    for name in R_DESIGNS:
        rows = load('R', name)
        if rows:
            d = dict(reps=len(rows), rho=rows[0]['rho'], cpop=rows[0]['cpop'])
            for k in ('op_cover', 'op_reject_true', 'op_reject_false', 'budget_violation'):
                x = [r[k] for r in rows]; d[k] = pct(x); d[k + '_se'] = se(x)
            for lab, cv in (('95', 5.991464547107979), ('975', 7.377758908227871)):
                x = [r['W_true'] > cv for r in rows]; d['coskew_true_rej_' + lab] = pct(x); d['coskew_true_rej_' + lab + '_se'] = se(x)
                x = [r['W_false'] > cv for r in rows]; d['coskew_false_rej_' + lab] = pct(x)
            d['median_false_angle_deg'] = float(np.median([r['angle_false'] for r in rows]))
            out['R'][name] = d
    json.dump(out, open(os.path.join(RES, 'confirm_summary.json'), 'w'), indent=2)
    os.makedirs(TEX, exist_ok=True)
    # ---------------- main-text joint table
    if out['J']:
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
        for name, d in out['J'].items():
            a += f"{lab[name]} & {d['budget_cover']:.1f} & {d['L_old']:.1f} & {d['L_new']:.1f} & {d['joint_old']:.1f} & {d['joint_new']:.1f} & {d['budget_share']:.1f} \\\\\n"
        reps = max(d['reps'] for d in out['J'].values())
        a += r"""\bottomrule\end{tabular}
\par\smallskip\noindent """ + f"{reps}" + r""" replications per row, $n=300$, VAR(1), skewed innovations, fixed six-degree grid without the true angle. Both whitening regions and the rotation cells are evaluated on the same data. Operator and percentile region: 199 fully refitted bootstrap draws; studentized region: 1,999 draws. Nominal levels: .975 for each region, .95 jointly. The persistent design uses moving blocks of ten for both bootstraps. Largest Monte Carlo standard error: """ + f"{max(max(d[k + '_se'] for k in ('L_old', 'L_new', 'joint_old', 'joint_new')) for d in out['J'].values()):.1f}" + r""" points.
\end{table}
"""
        open(os.path.join(TEX, 'confirm_joint_table.tex'), 'w').write(a)
    # ---------------- main-text recursive table
    if out['R']:
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
        lab = {'R1_iid_d0_n600': 'None', 'R2_iid_d4_n600': r'iid, $\rho=0.371$', 'R3_mk95_d4_n600': r'Persistent, $\rho=0.371$'}
        for name, d in out['R'].items():
            a += f"{lab[name]} & {d['op_cover']:.1f} & {d['op_reject_true']:.1f} & {d['op_reject_false']:.1f} & {d['coskew_true_rej_95']:.1f} & {d['coskew_true_rej_975']:.1f} & {d['coskew_false_rej_975']:.1f} \\\\\n"
        reps = max(d['reps'] for d in out['R'].values())
        a += r"""\bottomrule\end{tabular}
\par\smallskip\noindent """ + f"{reps}" + r""" replications per row, $n=600$, VAR(1) with a true first-ordered recursive structure, 199 moving-block draws with blocks of ten (the equity application uses blocks of ten and 399 draws). Coverage: the operator lower bound at the true ordering does not exceed the population operator norm (0 without common volatility; 0.0255 by quadrature at $\rho=0.371$). Reject: the operator lower bound is positive, i.e.\ independence is rejected at that ordering; it is a size under no common volatility and a power-type frequency when the shocks are dependent. Co-skewness: rejection at the $\chi^2_2$ .95 and .975 critical values; the last column uses .975 at the false ordering.
\end{table}
"""
        open(os.path.join(TEX, 'confirm_recursive_table.tex'), 'w').write(a)
    # ---------------- supplement whitening table
    if out['W']:
        a = r"""\begin{table}[tbp]
\centering\setstretch{1.5}\setlength{\tabcolsep}{5pt}
\caption{Pre-specified whitening-region calibration (nominal 97.5\%)}\label{tab:whitecal}
\begin{tabular}{lrrrr}
\toprule
Common volatility & $n$ & Block & Coverage (\%) & Median $c$\\
\midrule
"""
        for name, d in out['W'].items():
            l, n, ell = LABEL[name]
            a += f"{l} & {n} & {ell} & {d['coverage']:.1f} ({d['mc_se']:.1f}) & {d['median_critical']:.1f} \\\\\n"
        reps = max(d['reps'] for d in out['W'].values())
        a += r"""\bottomrule\end{tabular}
\par\smallskip\noindent """ + f"{reps:,}" + r""" replications per design, 1,999 bootstrap draws, Monte Carlo standard errors in parentheses. Designs with common volatility have $\rho=0.371$. Innovations are demeaned exponential unless stated. Block 1 is the iid residual bootstrap. The Student-$t_5$ design violates the eighth-moment condition of Proposition~\ref{prop:studentized}. The $\chi^2_3$ .975 quantile is 9.35.
\end{table}
"""
        open(os.path.join(TEX, 'confirm_whitening_table.tex'), 'w').write(a)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
