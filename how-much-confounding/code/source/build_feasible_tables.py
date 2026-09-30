from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];R=P/'results'
def table(file,caption,label,cols,head,rows,note):
 s='\\begin{table}[tbp]\n\\centering\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+head+r' \\'+'\n\\midrule\n'
 s+='\n'.join(' & '.join(r)+r' \\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\\noindent '+note+'\n\\end{table}\n'
 (R/file).write_text(s)
s=pd.read_csv(R/'feasible_mc_summary.csv');q=s[(s.model=='quadratic')&(s.degree==2)]
rows=[]
for r in q.itertuples():
 rows.append([f'{r.phi:.1f}',str(r.n),f'{100*r.cover_joint:.1f}',f'{100*r.power_ignore:.1f}',f'{100*r.power_feasible_r:.1f}',f'{100*r.power_joint:.1f}',f'{r.rv_joint:.4f}'])
table('feasible_power_table.tex','Feasible inference at moderate training and evaluation sizes','tab:feasiblepower','rrrrrrr',r'$\phi$ & $m=n$ & Cover & Ignore & Bound & Joint & Med. RV',rows,
 r'Quadratic transmission; 200 replications per row and 399 multiplier draws. Coverage and exclusion frequencies are percentages. Ignore omits coefficient uncertainty; Bound uses the feasible RMS allowance; Joint uses Theorem~\ref{thm:joint}. Exclusion is at $\gamma=0.01$ with $K_b=28.4444$; $\eta_f=\eta_b=0$ is correct in this design. Med. RV is the joint procedure. Each row has $n$ training and $n$ evaluation observations.')
rows=[]
for r in s[s.degree==2].itertuples():
 rows.append([{'confounding':'Common state','quadratic':'Transmission','linear':'Gaussian'}[r.model],f'{r.phi:.1f}',str(r.n),f'{100*r.cover_ignore:.1f}',f'{100*r.cover_joint:.1f}',f'{100*r.power_joint:.1f}',f'{r.width_joint:.3f}'])
table('feasible_full_table.tex','All correctly specified feasible-inference cells','tab:feasiblefull','lrrrrrr',r'Model & $\phi$ & $n$ & Cover I & Cover J & Power J & Width J',rows,
 r'200 replications per row. I ignores first-stage uncertainty; J is joint inference. Power is exclusion at $\gamma=0.01$. All feasible-RMS-comparator exclusion frequencies are zero. Gaussian denotes the linear transmission control with zero population diagnostic. Coverage of 100 percent is an observed frequency, not a theorem of exact finite-sample coverage.')
rows=[]
for r in s[s.degree==1].itertuples():
 rows.append([{'confounding':'Common state','quadratic':'Transmission'}[r.model],f'{r.phi:.1f}',f'{100*r.cover_ignore:.1f}',f'{100*r.cover_joint:.1f}',f'{100*r.cover_misspec:.1f}',f'{r.width_joint:.3f}'])
table('feasible_misspec_table.tex','Deliberately misspecified linear regressions','tab:feasiblemisspec','lrrrrr',r'Model & $\phi$ & Cover I & Cover J & Cover J+$\eta$ & Width J',rows,
 r'$m=n=600$, 200 replications per row. The zero-allowance joint procedure has no conditional-mean coverage guarantee here. J+$\eta$ adds the model-known RMS misspecification allowance (0.192 or 1 forward, zero backward); this is a mechanism check, not an empirical estimator of specification error.')
a=pd.read_csv(R/'feasible_application.csv');rows=[]
for r in a[(a.calendar_months==3)&(a.eta==0)].itertuples():
 rows.append(['Equity' if r.outcome.startswith('Equity') else 'Bond',r.basis.capitalize(),f'{r.q_ignore:.3f}',f'{r.q:.3f}',f'$[{r.lower:.3f},{r.upper:.3f}]$',f'{r.upper_certificate:.3f}'])
table('feasible_application_table.tex','First-stage uncertainty in the fixed-calendar event analysis','tab:feasibleapp','llrrcr',r'Outcome & Basis & $q_0$ & $q_J$ & Joint interval & Upper RV',rows,
 r'Direct-sum pooling, eight strata, three-month calendar multipliers, 1,999 draws. $q_0$ holds regressions fixed; $q_J$ includes their estimation. Preprocessing uses dates through 2015, regressions use 2016--2020, and evaluation starts in 2021. $\eta=0$ is an optimistic specification scenario. Every lower certificate is zero. These are sensitivity calculations under a maintained calendar sampling approximation.')
# Compact positive approximation allowances for the bounded empirical dictionary.
rows=[]
for r in a[(a.calendar_months==3)&(a.basis=='bounded')].itertuples():
 rows.append(['Equity' if r.outcome.startswith('Equity') else 'Bond',f'{r.eta:.3f}',f'{r.lower:.3f}',f'{r.upper:.3f}',f'{r.upper_certificate:.3f}'])
table('feasible_application_eta_table.tex','Specification allowances with the bounded dictionary','tab:feasibleeta','lrrrr',r'Outcome & $\eta_f=\eta_b$ & $L$ & $U$ & Upper RV',rows,
 r'Three-month calendar blocks. Allowances measure L1 projection misspecification in fixed preprocessing units; coefficient uncertainty is already in the joint radius. The bounded basis was added after the quadratic fit proved unstable, and both specifications are retained. All 48 combinations of outcome, basis, calendar block and allowance are supplied in the replication output.')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(6.5,3.3),constrained_layout=True)
for phi,c,m in [(0.,'#174f78','o'),(.6,'#ad5b20','s')]:
 d=q[q.phi==phi];ax.plot(d.n,100*d.power_joint,marker=m,color=c,label=f'Joint, persistence {phi:.1f}')
ax.plot([250,600,1200],[0,0,0],'k--',label='Feasible RMS bound, both persistence values')
ax.set(xlabel='Training observations = evaluation observations',ylabel='Exclusion frequency (%)',ylim=(-3,104),xticks=[250,600,1200])
ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.2)
fig.savefig(P/'figures/feasible_power.pdf');fig.savefig(P/'figures/feasible_power.png',dpi=180);plt.close(fig)
print('Feasible displays built.')

# Reproducible sample-count display and consistent submission font sizes.
c=pd.read_csv(R/'feasible_application_counts.csv')
rows=[]
for r in c.itertuples():
 name=r.stratum.replace('press conferences','press conf.').replace('MPR press conf.','press conf.')
 rows.append(['Equity' if r.outcome.startswith('Equity') else 'Bond',name,str(r.n_pre),str(r.n_train),str(r.n_eval),'Yes' if r.included else 'No'])
table('feasible_counts_table.tex','Counts in the fixed-calendar analysis','tab:feasiblecounts','llrrrr',r'Outcome & Stratum & Preprocess & Train & Evaluate & Used',rows,
 'Preprocessing ends in 2015; training is 2016--2020; evaluation starts in 2021. The declared rule requires at least 20 observations in each period. Counts precede the inclusion rule.')
for f in R.glob('*.tex'):
 f.write_text(f.read_text().replace(r'\centering\small',r'\centering').replace(r'\footnotesize',r'\normalsize'))

for f in R.glob('*.tex'):
 s=f.read_text()
 if r'\setstretch{1.5}\centering' not in s:s=s.replace(r'\centering',r'\setstretch{1.5}\centering')
 f.write_text(s)
