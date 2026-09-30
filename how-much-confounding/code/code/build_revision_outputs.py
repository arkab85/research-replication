from pathlib import Path
import json,numpy as np
r=Path(__file__).resolve().parents[2];out=r/'4_Replication/code/revision_results';tex=r/'3_LaTeX_Source/results'
def get(n):return json.loads((out/(n+'.json')).read_text())
e=get('equity');o=get('oil')
a=r'''\begin{table}[tbp]
\centering\setstretch{1.5}
\caption{Equity recursive-ordering tests with fully refitted inference}\label{tab:vix}
\begin{tabular}{lrrrr}
\toprule
Ordering & $\|\widehat C\|$ & Lower $\rho$ & Wald & $p$-value\\
\midrule
'''
for j,name in enumerate(['Returns first','VIX first']):a+=f"{name} & {e['recursive_norms'][j]:.4f} & {e['recursive_budget_lower'][j]:.3f} & {e['recursive_wald'][j]:.2f} & {e['recursive_p'][j]:.5f} \\\\\n"
a+=r'''\bottomrule\end{tabular}
\par\smallskip\noindent The joint recursive-operator radius is 0.02435. Wald $p$-values use the asymptotic $\chi^2_2$ distribution, which over-rejects in the calibration of Table~\ref{tab:recsize}; the two-test asymptotic familywise critical value is 7.378. Lower $\rho$ is the square root of the positive operator lower bound, with the projection-defined exposure units of Assumption~\ref{ass:svar}.
\end{table}
''';(tex/'revised_vix_table.tex').write_text(a)
ow=get('oil_whitening')
a=r'''\begin{table}[tbp]
\centering\setstretch{1.5}
\caption{Oil: certified outer projection with studentized whitening uncertainty}\label{tab:oilsvar}
\begin{tabular}{rrc}
\toprule
Exposure budget $\rho$ & Retained cells (\%) & Elasticity outer range\\
\midrule
'''
for rho in ('0','0.025','0.05','0.1','0.15'):
 x=ow['sets'][rho];assert x['elasticity_min']==0.0 and x['elasticity_max']=='unbounded'
 a+=f"{float(rho):.3f} & {100*x['cell_share']:.1f} & $[0,\\infty)$ \\\\\n"
assert all(v==0.0 for v in ow['breakdown_lower'].values())
a+=r'''\bottomrule\end{tabular}
\par\smallskip\noindent Joint asymptotic confidence is at least .95 under Assumption~\ref{ass:svarboot} and asymptotic coverage of the studentized region. The outer range includes all retained cells and the whitening region. Infinite upper endpoints are reported rather than capped. The certificate breakdown lower bound is zero for elasticity thresholds 0.0258, 0.05, 0.10 and 0.20.
\end{table}
''';(tex/'revised_oil_table.tex').write_text(a)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7,3.8));rho=np.linspace(0,.2,401)
for name,label,color in [('oil','Oil','#9b4d15'),('equity','Equity','#174a7e')]:
 low=np.load(out/(name+'.npz'))['cell_lower'];share=np.mean(low[:,None]<=rho[None,:]**2,axis=0);ax.plot(rho,100*share,label=label,color=color,lw=2)
ax.set(xlabel='Common-state exposure budget',ylabel='Rotation cells retained (%)',xlim=(0,.2),ylim=(0,103));ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(r/'3_LaTeX_Source/figures/revised_frontier.pdf');plt.close(fig)

o=out;t=r/'3_LaTeX_Source'
a=r'''\begin{table}[tbp]
\centering\setstretch{1.5}\setlength{\tabcolsep}{5pt}
\caption{Fully recomputed conditional influence diagnostics}\label{tab:oilrobust}
\begin{tabular}{lrrrrr}
\toprule
 & $n$ & \multicolumn{2}{c}{Lower budget} & \multicolumn{2}{c}{Co-skewness Wald}\\
Sample & & First & Second & First & Second\\
\midrule
'''
for name,label in [('equity','Equity: full'),('equity_top1','Largest week'),('equity_top5','Largest five weeks'),('equity_crisis','Crisis interval'),('oil','Oil: full'),('oil_may','May 2020 impulse')]:
 d=json.loads((o/(name+'.json')).read_text());x=d['recursive_budget_lower'];w=d['recursive_wald'];a+=f"{label} & {d['n']} & {x[0]:.3f} & {x[1]:.3f} & {w[0]:.2f} & {w[1]:.2f} \\\\\n"
a+=r'''\bottomrule\end{tabular}
\par\smallskip\noindent First/second denote returns-first/VIX-first for equity and production-first/price-first for oil. The recursive-operator event covers the two orderings jointly within each row; co-skewness uses the separate asymptotic familywise critical value 7.378, which over-rejects in the calibration of Table~\ref{tab:recsize}. No multiplicity adjustment covers searches across influence specifications. All influence rows retain the original calendar with impulse dummies and exclude their saturated residuals from the statistic.
\end{table}
''';(t/'results/revised_influence_table.tex').write_text(a)

# The earlier 200-replication calibration table (revised_mc_table.tex) is superseded by
# whitening/summarize_confirm.py, which writes the confirmatory tables.
