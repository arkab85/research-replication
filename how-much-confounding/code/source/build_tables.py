from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import benchmark,envelope_constant,amplitude
P=Path(__file__).resolve().parents[1];R=P/'results'
def table(path,caption,label,cols,head,rows,note):
 text='\\begin{table}[t]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+head+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(row)+' \\\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n\\par\\medskip\\noindent '+note+'\n\\end{table}\n'
 (R/path).write_text(text)
pop=pd.read_csv(R/'population.csv');s=pop.query('delta==0.3 or delta==0.6')
rows=[[f'{r.phi:.1f}',f'{r.rx:.1f}',f'{r.ry:.1f}',f'{r.delta:.1f}',f'{1e3*r.dii:+.3f}',f'{1e3*r.mcse:.3f}'] for r in s.itertuples()]
table('population_table.tex','Directional evidence under nonlinear common states','tab:population','rrrrrr',r'$\phi$ & $r_X$ & $r_Y$ & $\delta$ & $10^3D_h$ & MC s.e.',rows,'Monte Carlo standard errors use the same $10^3$ units as the index. Each cell averages ten independent batches of 2,000 stationary tuples, with fixed Gaussian kernels. Full results, including zero and 0.1 loading, are in the replication archive. The shares refer to the Gaussian reference.')
inf=pd.read_csv(R/'inference.csv')
rows=[[str(int(r.n)),f'{r.rx:.1f}',f'{r.ry:.1f}',f'{r.delta:.2f}',f'{100*r.false_exclusion:.1f}',f'{r.mean_radius:.3f}'] for r in inf.itertuples()]
table('inference_table.tex','False exclusion of the true confounding budget','tab:inference','rrrrrr',r'$n$ & $r_X$ & $r_Y$ & $\delta$ & Rejected (\%) & Radius',rows,'Each cell uses 120 independent replications, 399 multiplier draws, persistence 0.8, and oracle conditional means. Radius is the mean simultaneous covariance-norm radius. The nonlinear envelope is conservative; these designs do not establish performance at a least-favorable boundary.')
power=pd.read_csv(R/'power.csv');praw=pd.read_csv(R/'power_raw.csv')
praw['rv']=np.sqrt(np.maximum(praw.lower,0)/praw.K);rv=praw.query('gamma==0').groupby('n').rv.agg(['mean','median'])
rows=[[str(int(r.n)),f'{r.gamma:.3f}',f'{100*r.rejection_rate:.1f}',f'{1e3*r.mean_dii:.2f}',f'{1e3*r.mean_lower:.2f}',f'{rv.loc[r.n,"median"]:.4f}'] for r in power.itertuples()]
table('power_table.tex','Certifying transmission against a specified confounding class','tab:power','rrrrrr',r'$n$ & Budget $\gamma$ & Rejected (\%) & $10^3\widehat D_h$ & $10^3L_h$ & Median $\mathrm{RV}_{.05}$',rows,r'Each cell uses 120 replications of the quadratic transmission design. $\widehat D_h$ and $L_h$ are replication means; the last column is the median robustness value $\mathrm{RV}_{.05}=\{\pos{L_h}/K_b\}^{1/2}$ across replications, which does not depend on $\gamma$. The Gaussian no-channel reference is fixed at $\phi=0.8,r_X=0.3,r_Y=0.7$, for which $K_b=51.26$. Budgets are in fixed reference-standard-deviation units; larger budgets enlarge the null class.')
tr=pd.read_csv(R/'transmission.csv');Kref=envelope_constant(benchmark(.8,.3,.7))
rows=[[f'{r.phi:.1f}',r.loading,f'{1e3*r.Hf:+.3f} ({1e3*r.Hf_se:.3f})',f'{1e3*r.Hb:.3f} ({1e3*r.Hb_se:.3f})',f'{np.sqrt(max(r.dii,0)/Kref):.4f}'] for r in tr.itertuples()]
table('transmission_table.tex','Population components under additive-noise transmission','tab:transmission','rlrrr',r'Cause $\phi$ & Loading $f$ & $10^3H_f$ (s.e.) & $10^3H_b$ (s.e.) & Population RV',rows,r'$Y_{t+1}=f(X_t)+e_{t+1}$ with $e\sim N(0,0.3^2)$, a stationary Gaussian AR(1) cause with unit variance, and $C_t=X_{t-1}$. Quadratic: $f(x)=(x^2-1)/\sqrt2$; cubic: $f(x)=(x+x^3)/\sqrt{22}$. Each cell averages ten batches of 2,000 tuples using unbiased U-statistics and oracle conditional means; standard errors in parentheses. Population RV is $(D_h/K_b)^{1/2}$ at the reference of Table~\ref{tab:power}.')
app=pd.read_csv(R/'application.csv').query('ell==12 and r==0')
rows=[[r.asset,str(int(r.h)),f'{1000*r.dii:+.3f}',f'{1000*r.lower:+.2f}',f'{1000*(r.dii-r.lower):.2f}',f'{r.K:.2f}',f'{r.gamma_lower:.3f}',f'{np.sqrt(max(r.upper,0)/r.K):.4f}'] for r in app.itertuples()]
table('application_table.tex','Robustness values for monetary-shock directional evidence','tab:application','lrrrrrrr',r'Outcome & $h$ & $10^3\widehat D_h$ & $10^3 L_h$ & $10^3(\widehat D_h-L_h)$ & $K_b$ & $\mathrm{RV}_{.05}$ & $\overline{\mathrm{RV}}_{.05}$',rows,r'All twelve comparisons enter a simultaneous 95-percent operator confidence event. There are 204 training and 120 evaluation origins; block length is twelve and there are 1,999 multiplier draws. The column $\widehat D_h-L_h$ is the statistical margin; holding it fixed, a point index at least this large would be needed for a positive robustness value. The table uses the optimistic zero-approximation-error budget and the stated reference scenario. Every robustness value is zero; no confounding class is excluded.')
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':12})
fig,axs=plt.subplots(2,1,figsize=(6.5,5.8))
L=np.linspace(0,.02,201)
for rx,ry,style in [(.2,.8,'-'),(.5,.5,'--'),(.8,.2,':')]:
 q=benchmark(.8,rx,ry);K=envelope_constant(q)
 axs[0].plot(L,np.sqrt(L/K),style,label=f'({rx:.1f}, {ry:.1f}); K={K:.1f}',color='black')
 z=pop.query('phi==0.8 and rx==@rx and ry==@ry')
 axs[1].errorbar(z.delta,1000*z.dii,yerr=1.96*1000*z.mcse,linestyle=style,marker='o',color='black',capsize=3,label=f'({rx:.1f}, {ry:.1f})')
axs[0].set(xlabel='Positive lower DII bound L',ylabel='Necessary total RMS loading',title='A. Inverting the class-wide bound')
axs[0].set_xticks([0,.005,.010,.015,.020],['0.000','0.005','0.010','0.015','0.020'])
axs[1].set(xlabel='Outcome RMS loading',ylabel='Population DII × 1,000',title='B. A specified nonlinear shape')
for ax in axs:
 ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=12);ax.grid(axis='y',alpha=.18)
axs[1].axhline(0,color='gray',lw=.6);fig.tight_layout();fig.savefig(P/'figures/envelope.pdf');fig.savefig(P/'figures/envelope.png',dpi=160)
ev=pd.read_csv(R/'event_application.csv').query('ell==4 and r==0')
rows=[]
for r in ev.itertuples():
    qs=(np.sqrt(r.backward_hsic)-np.sqrt(r.forward_hsic))/2
    nstar=r.test_n*(r.radius/qs)**2
    rows.append([('S\\&P 500, window' if r.design.startswith('E1') else 'VIX, post-event'),str(int(r.h)),f'{1000*r.dii:+.2f}',f'{1000*r.lower:+.2f}',f'{1000*(r.dii-r.lower):.2f}',f'{r.K:.1f}',f'{r.rv:.3f}',f'{np.sqrt(max(r.upper,0)/r.K):.4f}',f'{100*round(nstar/100):,.0f}'])
table('event_table.tex','Robustness values at announcement frequency (pre-specified)','tab:event','lrrrrrrrr',r'Outcome & $h$ & $10^3\widehat D_h$ & $10^3L_h$ & $10^3(\widehat D_h-L_h)$ & $K_b$ & $\mathrm{RV}_{.05}$ & $\overline{\mathrm{RV}}_{.05}$ & $n^*$',rows,r'Units are FOMC announcements; training announcements precede 2008 and evaluation announcements (142 or 143 per cell) run from 2009 to March 2026. The within-window design relates the policy surprise to the S\&P 500 surprise in the same 30-minute window; the VIX design relates the monetary-policy shock to the log VIX change over $h$ trading days after the announcement day. All eight covariance operators enter one simultaneous 95-percent event with block length four and 1,999 draws; block length one gives the same conclusions. $K_b$ is the maximum over the pre-specified correlation-matched reference set; $\overline{\mathrm{RV}}_{.05}$ is the robustness ceiling of Corollary~\ref{cor:ceiling}. $n^*$ is a plug-in heuristic: the evaluation sample at which the confidence radius would equal $(\widehat H_b^{1/2}-\widehat H_f^{1/2})/2$, holding the point components fixed and scaling the radius as $n^{-1/2}$; it is a design calculation, not an estimate.')
pl=pd.read_csv(R/'pooled_application.csv').query('ell==4 and r==0')
rows=[[r.design,f'{int(r.n):,}',str(int(r.strata)),f'{1000*r.dii:+.2f}',f'{1000*r.lower:+.2f}',f'{1000*r.upper:+.2f}',f'{r.Kbar:.1f}',f'{r.rv:.3f}',f'{np.sqrt(max(r.upper,0)/r.Kbar):.4f}'] for r in pl.itertuples()]
table('pooled_table.tex','Pooled central-bank communication events (pre-specified)','tab:pooled','lrrrrrrrr',r'Outcome & $n$ & Strata & $10^3\overline D_h$ & $10^3L_h$ & $10^3U_h$ & $\overline K_b$ & $\mathrm{RV}_{.05}$ & $\overline{\mathrm{RV}}_{.05}$',rows,r'Stratified pooling (Proposition~\ref{prop:pool}) over Fed statements, press conferences, and minutes (USMPD), ECB press releases and press conferences (EA-MPD), speeches by the ECB President and other Executive Board members (EA-EMPD), and Bank of England MPC announcements and MPR press conferences (UKMPD). $X$ is the first principal component of in-window money-market rate changes up to one year; $Y$ is the in-window equity-index return or ten-year government yield change. The first half of each stratum trains; $n$ counts evaluation events. One simultaneous 95-percent event per outcome, blocks of four events within stratum, 1,999 draws; blocks of one give the same conclusions. $\overline{\mathrm{RV}}_{.05}=\{\pos{U_h}/\overline K_b\}^{1/2}$ is the robustness ceiling of Corollary~\ref{cor:ceiling}.')
ps=pd.read_csv(R/'pooled_strata.csv')  # placed with [H] in the appendix
rows=[]
for r in ps.itertuples():
    ab={'Fed statements':'Fed stat.','Fed press conferences':'Fed press conf.','Fed minutes':'Fed minutes','ECB press releases':'ECB press rel.','ECB press conferences':'ECB press conf.','ECB Board-member speeches':'ECB Board sp.','ECB President speeches':'ECB Pres. sp.','BoE MPC announcements':'BoE MPC','BoE MPR press conferences':'BoE MPR conf.'}
    rows.append([('Equity' if r.design.startswith('Equity') else 'Bond'),ab[r.stratum],f'{r.start[:4]}--{r.end[:4]}',str(r.n_eval),f'{r.train_corr:+.2f}',f'{1000*r.Hf:.2f}',f'{1000*r.Hb:.2f}',f'{1000*r.dii:+.2f}'])
table('pooled_strata_table.tex','Stratum-level components of the pooled design','tab:strata','llrrrrrr',r'Outcome & Stratum & Years & $n_s$ & $\rho_s$ & $10^3H_f$ & $10^3H_b$ & $10^3D_h$',rows,r'Point components by stratum on the evaluation half; $\rho_s$ is the training correlation of $X$ and $Y$. No stratum-level inference is reported, as pre-specified; the pooled operator averages the stratum operators with weights $n_s/n$.')

(R/'pooled_strata_table.tex').write_text((R/'pooled_strata_table.tex').read_text().replace('\\begin{table}[t]','\\begin{table}[H]',1))
rs=pd.read_csv(R/'rate_study.csv');rows=[]
def Mconst(q,side):return (2*np.sqrt(2*q[side]['v'])+3*np.sqrt(2))**2
for (phi,rx,ry),g in rs.groupby(['phi','rx','ry'],sort=False):
    q=benchmark(phi,rx,ry);side='b' if phi<.5 else 'f';col='Hb' if side=='b' else 'Hf'
    kl=lambda d:.5*(d/q['se'])**2  # Lemma 2: outcome-only loading, KL <= S(d)/2
    prev=None
    for r in g.itertuples():
        h,se=getattr(r,col),getattr(r,col+'_se')
        sl=''
        if prev is not None and prev[1]>2*prev[2] and h>2*se: sl=f'{np.log(h/prev[1])/np.log(r.delta/prev[0]):.2f}'
        prev=(r.delta,h,se)
        rows.append([f'$\\phi={phi},r_X={rx}$',f'$H_{side}$',f'{r.delta:.1f}',f'{1000*h:.3f} ({1000*se:.3f})',f'{envelope_constant(q,side)*r.delta**2:.2f}',f'{Mconst(q,side)*kl(r.delta):.2f}',sl])
table('rate_table.tex','The envelope has the right order; its constant is conservative','tab:rate','llrrrrr',r'Design & Component & $\delta$ & $10^3H$ (s.e.) & $K_r\delta^2$ & $M_r\kappa(\delta)$ & Local slope',rows,r'Outcome-only loading departures of size $\delta$ with the shape of Section~\ref{sec:numerical}; each entry averages twenty batches of 2,000 tuples with unbiased U-statistics. The local slope is $\log(H/H_{\mathrm{prev}})/\log(\delta/\delta_{\mathrm{prev}})$ relative to the previous row, shown when both estimates exceed two standard errors. $K_r\delta^2$ is the loading envelope of Corollary~\ref{cor:budget}; $M_r\kappa(\delta)$ is the entropy envelope of Proposition~\ref{prop:entropy} at the Lemma~\ref{lem:entropy} bound $\kappa(\delta)=\delta^2/(2\sigma_e^2)$. Bounds are in raw units.')
sv=pd.read_csv(R/'pooled_sensitivity.csv');rows=[]
for r in sv.itertuples():
    grid=r.variant.startswith('Reference')
    lab=('Baseline' if r.variant.startswith('Baseline') else 'Reference grid' if grid else r.variant.replace('History (previous event), ','History, ').replace('Bandwidth','Bandwidth'))
    rows.append([('Equity' if r.outcome.startswith('Equity') else 'Bond'),lab,f'{1000*r.dii:+.2f}',f'{1000*r.lower:+.2f}',f'{1000*r.upper:+.2f}',
        (f'{r.Kmin:.0f}--{r.Kmax:.0f}' if grid else f'{r.Kbar:.1f}'),f'{r.rv:.4f}' if not grid else '0',
        (f'{r.ceiling_min:.4f}--{r.ceiling:.4f}' if grid else f'{r.ceiling:.4f}')])
table('pooled_sensitivity_table.tex','Sensitivity of the pooled robustness value and ceiling','tab:sens','llrrrrrr',r'Outcome & Variant & $10^3\overline D_h$ & $10^3L_h$ & $10^3U_h$ & $\overline K_b$ & $\mathrm{RV}_{.05}$ & $\overline{\mathrm{RV}}_{.05}$',rows,r'Every variant of the pooled design that we computed is reported. The reference-grid row replaces the correlation-matched reference by each of sixteen fixed signal-share pairs $r_X,r_Y\in\{0.2,0.4,0.6,0.8\}$ and reports ranges; it changes $\overline K_b$ only. Bandwidth rows scale all kernel bandwidths. History rows add the previous event in the same stratum to both regressions, with quadratic conditional means, and set the reference persistence across consecutive events to $\phi$. Each row uses its own simultaneous 95-percent event. The one positive robustness value (bond, bandwidth $0.5$) does not survive a Bonferroni adjustment for the seven variants per outcome: at level $0.05/7$ (4,999 draws) its lower bound is $-0.00013$. The baseline is the pre-specified design.')
(R/'pooled_sensitivity_table.tex').write_text((R/'pooled_sensitivity_table.tex').read_text().replace('\\begin{table}[t]','\\begin{table}[tbp]',1))
