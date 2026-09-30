"""Build all journal revision displays from stored numerical results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];R=P/'results'

def table(name,caption,label,align,head,rows,note):
    s='\\begin{table}[tbp]\n\\centering\\small\n'
    s+=f'\\caption{{{caption}}}\\label{{{label}}}\n'
    s+=f'\\begin{{tabular}}{{{align}}}\n\\toprule\n{head}\\\\\n\\midrule\n'
    s+='\n'.join(' & '.join(row)+r' \\' for row in rows)
    s+='\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n'
    s+='\\begin{minipage}{\\linewidth}\\footnotesize '+note+'\\end{minipage}\n\\end{table}\n'
    (R/name).write_text(s)

def build():
    mc=pd.read_csv(R/'revision_mc_summary.csv')
    assert (mc.cover_aware==1).all() and (mc.median_rv_aware==0).all()
    q=mc[mc.model=='quadratic']
    rows=[[f'{r.phi:.1f}',str(r.n),'Linear' if r.degree==1 else 'Quadratic',
           f'{100*r.cover_zero:.1f}',f'{100*r.cover_aware:.1f}',f'{r.median_r_f:.3f}',
           f'{r.median_r_b:.3f}',f'{r.median_rv_zero:.4f}'] for r in q.itertuples()]
    table('revision_learning_table.tex','Learning and coverage under quadratic transmission','tab:learning',
          'rrlrrrrr',r'$\phi$ & $n$ & Fit & Cover 0 & Cover $r$ & Med. $r_f$ & Med. $r_b$ & Med. RV 0',rows,
          r'Coverage is in percent for the population contrast $D=0.016427715$. Each row uses 200 replications. '
          r'Cover 0 ignores regression error; Cover $r$ uses exact model-known RMS bounds on L1 error. '
          r'Training and evaluation each contain $n$ observations from independent stationary paths. '
          r'All error-aware median robustness values and exclusion frequencies at $\gamma=0.01$ are zero. '
          r'The final column is the zero-budget calculation, not an error-robust claim.')
    tr=pd.read_csv(R/'revision_training_summary.csv');rows=[]
    for phi in [0.,.6]:
        base=q[(q.phi==phi)&(q.n==600)&(q.degree==2)].iloc[0]
        rows.append([f'{phi:.1f}','600',f'{base.median_r_f:.4f}',f'{base.median_r_b:.4f}',
                     f'{base.median_rv_aware:.4f}',f'{100*base.power_001_aware:.1f}'])
        for r in tr[tr.phi==phi].itertuples():
            rows.append([f'{phi:.1f}',f'{r.n_train:,}',f'{r.median_r_f:.4f}',f'{r.median_r_b:.4f}',
                         f'{r.median_rv_aware:.4f}',f'{100*r.power_aware:.1f}'])
    table('revision_training_table.tex','Training information and error-aware exclusion','tab:training',
          'rrrrrr',r'$\phi$ & Training $n$ & Med. $r_f$ & Med. $r_b$ & Med. RV & Exclude 0.01 (\%)',rows,
          r'Quadratic transmission, quadratic fits, 600 evaluation observations, 200 replications per row. '
          r'The 600-observation rows are from the baseline study; the larger-training rows are a disclosed follow-up. '
          r'Error-aware diagnostic coverage is 100 percent in every row. The reference has $K_b=28.4444$. '
          r'Model-known error bounds are used; the table does not claim a data-only error estimator.')
    er=pd.read_csv(R/'revision_error_budgets.csv');rows=[]
    for r in er.itertuples():
        rows.append(['Equity' if r.design.startswith('Equity') else 'Bond',f'{r.r:.3f}',
                     f'{1000*r.lower:+.3f}',f'{1000*r.upper:+.3f}',f'{r.rv:.4f}',f'{r.ceiling:.4f}'])
    table('revision_error_table.tex','Pooled evidence under explicit regression-error allowances','tab:errors',
          'lrrrrr',r'Outcome & $r_f=r_b$ & $10^3L$ & $10^3U$ & Lower RV & Upper certificate',rows,
          r'Operator averaging, 2,000 equity events or 2,378 bond events, nine strata. The point contrasts are '
          r'$0.00001053$ and $0.00103063$; constants are $41.6089$ and $43.2948$. '
          r'The original multiplier uses four-event blocks within strata and 1,999 draws. '
          r'Each outcome has its own joint forward/backward confidence event. Budgets are L1 conditional-mean errors '
          r'in training-standard-deviation units. The zero budget is an optimistic scenario, not a verified fit property.')
    ca=pd.read_csv(R/'revision_calendar.csv');assert (ca.rv==0).all();rows=[]
    for r in ca[(ca.calendar_months==3)&(ca.r==0)].itertuples():
        rows.append(['Equity' if r.outcome.startswith('Equity') else 'Bond',
                    'Average' if r.pooling=='average' else 'Direct sum',f'{1000*r.D:+.3f}',
                    f'{1000*r.lower:+.3f}',f'{1000*r.upper:+.3f}',f'{r.ceiling:.4f}'])
    table('revision_pooling_table.tex','Calendar dependence and the choice of pooling target','tab:poolingrevision',
          'llrrrr',r'Outcome & Pooling & $10^3\widehat D$ & $10^3L$ & $10^3U$ & Upper certificate',rows,
          r'Three-calendar-month blocks share multipliers across banks and event types; 1,999 draws. '
          r'The zero regression-error scenario is displayed. Every lower robustness value is zero. '
          r'Direct-sum pooling prevents backward-operator cancellation under transmission and has its own envelope constant. '
          r'These additional sensitivity analyses do not verify the required dependence approximation.')
    rows=[]
    for r in mc.itertuples():
        rows.append([{'confounding':'Common state','quadratic':'Quadratic','linear':'Linear'}[r.model],
                     f'{r.phi:.1f}',str(r.n),str(r.degree),f'{100*r.cover_zero:.1f}',
                     f'{100*r.cover_aware:.1f}',f'{r.median_r_f:.3f}',f'{r.median_r_b:.3f}'])
    table('revision_full_mc_table.tex','All 24 baseline learned-regression cells','tab:fullmc',
          'lrrrrrrr',r'Model & $\phi$ & $n$ & Degree & Cover 0 & Cover $r$ & Med. $r_f$ & Med. $r_b$',rows,
          r'Coverage in percent; 200 replications per cell. Each training path and independent evaluation path has length $n$. '
          r'The actual no-channel loading is 0.3. No false exclusion of that budget occurs. '
          r'The nonlinear transmission model has a quadratic conditional mean; the linear model is a Gaussian transmission control. '
          r'Every error-aware lower robustness value is zero in these baseline cells. Raw replication outcomes and Monte Carlo standard errors are retained.')
    sh=pd.read_csv(R/'revision_sharpness.csv')
    rows=[[f'{r.delta:.3f}',f'{r.Hf:.8f}',f'{r.Hb:.8f}',f'{r.D_over_delta2:.8f}',f'{r.quadrature_difference:.1e}'] for r in sh.itertuples()]
    table('revision_sharpness_table.tex','Numerical check of the explicit sharpness coefficient','tab:sharpnum',
          'rrrrr',r'$\delta$ & $H_f$ & $H_b$ & $D/\delta^2$ & Quadrature gap',rows,
          r'$a=0.8$, unit noise and kernel bandwidths. The limit is $0.007585185$. '
          r'The gap is the largest absolute difference between component or contrast values using 100 and 180 quadrature nodes. '
          r'The analytical proof, not the numerical slope, establishes rate sharpness.')
    rows=[]
    for r in ca[ca.r.isin([0.,.05])].itertuples():
        rows.append(['Equity' if r.outcome.startswith('Equity') else 'Bond',
                     'Avg.' if r.pooling=='average' else 'Direct',str(r.calendar_months),
                     f'{r.r:.2f}',f'{1000*r.lower:+.2f}',f'{1000*r.upper:+.2f}',f'{r.ceiling:.4f}'])
    table('revision_full_calendar_table.tex','Calendar-block sensitivity: all endpoint scenarios','tab:calendarall',
          'llrrrrr',r'Outcome & Pooling & Months & $r$ & $10^3L$ & $10^3U$ & Upper certificate',rows,
          r'1,999 multipliers; every lower robustness value is zero. Intermediate error budgets 0.01 and 0.025 are in the CSV file. '
          r'Rows are scenario-specific intervals, not a simultaneous family over all choices. '
          r'The calendar grouping is a dependence sensitivity check and does not establish independence between groups.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.7),constrained_layout=True)
    axs[0].plot(sh.delta,sh.D_over_delta2,'o-',color='#154f7b',label=r'$D(\delta)/\delta^2$')
    axs[0].axhline(sh.limit.iloc[0],ls='--',color='#a34d18',label='Proved limiting coefficient')
    axs[0].set(xscale='log',xlabel=r'Nonlinear loading $\delta$',ylabel='Scaled directional contrast',title='A. Explicit sharpness construction')
    axs[0].legend(loc='lower left',fontsize=8);axs[0].grid(alpha=.2)
    e=er[er.design=='Equity index']
    axs[1].plot(e.r,e.ceiling,'o-',label='Average, within-stratum blocks',color='#154f7b')
    for p,col,mark in [('average','#b67920','s'),('direct_sum','#39815b','^')]:
        d=ca[(ca.outcome=='Equity index')&(ca.pooling==p)&(ca.calendar_months==3)]
        axs[1].plot(d.r,d.ceiling,marker=mark,ls='--',label=('Average' if p=='average' else 'Direct sum')+', calendar blocks',color=col)
    axs[1].set(xlabel=r'Common regression-error allowance $r$',ylabel='Upper certificate bound',title='B. Equity event-data sensitivity')
    axs[1].legend(fontsize=7.5);axs[1].grid(alpha=.2)
    fig.savefig(P/'figures/revision_sharpness_error.pdf')
    fig.savefig(P/'figures/revision_sharpness_error.png',dpi=200)
    plt.close(fig)
    # Make the optimistic regression assumption visible in retained legacy tables.
    for name in ['event_table.tex','pooled_sensitivity_table.tex','pooled_table.tex']:
        path=R/name;s=path.read_text()
        addition=r' The displayed intervals use the optimistic zero regression-error budget; positive budgets widen them.'
        if addition not in s:s=s.replace('\\end{table}',addition+'\n\\end{table}')
        s=s.replace('robustness ceiling','upper certificate bound')
        s=s.replace('robustness value and ceiling','robustness value and upper certificate bound')
        if name=='pooled_sensitivity_table.tex':s=s.replace('\\begin{table}[tbp]','\\begin{table}[H]')
        path.write_text(s)
    print('Revision displays built from verified stored results.')

if __name__=='__main__':build()
