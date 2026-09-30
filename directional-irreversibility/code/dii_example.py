"""Executable synthetic illustration with KNOWN conditional-mean residuals.
Not a new empirical application or a finite-sample validation of size/power.
"""
from pathlib import Path
import numpy as np,json
from dii_core import dii_profile
P=Path(__file__).resolve().parent.parent
rng=np.random.default_rng(20260912);n=216
x=rng.normal(size=n);v=rng.normal(size=n+1);noise=(v[1:]+v[:-1])/np.sqrt(2)
a=.6;g=.6;lam=.6
linear=a*x+noise;curved=g*(x*x-1)+noise;scale=np.sqrt(1+lam*x*x)*noise
cases={'linear_gaussian':(noise,x,x-a/(1+a*a)*linear,linear),
       'nonlinear_additive':(noise,x,x,curved),
       'conditional_scale':(scale,x,x,scale)}
r=dii_profile(cases,6,399,20260912)
# With oracle zero conditional means, matched kernels imply exact cancellation.
assert abs(r['comparisons']['conditional_scale']['dii'])<1e-12
assert r['comparisons']['conditional_scale']['p_intersection']==1
# Swapping the two representations reverses the observed contrast exactly.
swap={'swapped':(x,curved,noise,x)}
z=dii_profile(swap,6,399,20260912)
assert abs(z['comparisons']['swapped']['dii']+r['comparisons']['nonlinear_additive']['dii'])<1e-12
r['purpose']='Synthetic software example only. Residuals are oracle-known, not estimated. Exact cancellation is outside the nonconstant-limit theorem; its p-value is computational output only.'
(P/'dii_example_results.json').write_text(json.dumps(r,indent=2))
print('Reference implementation: cancellation and representation-swap checks passed.')
# Analytic figure: no simulation estimates or institutional data.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=np.linspace(-2.5,2.5,300)
fig,axs=plt.subplots(1,3,figsize=(10.2,3.3),sharex=True,sharey=True)
for ax,title,mean,sd,label in zip(axs,['Linear Gaussian exposure','Nonlinear additive exposure','Shock-dependent risk'],[a*s,g*(s*s-1),0*s],[0*s+1,0*s+1,np.sqrt(1+lam*s*s)],['DII = 0; linear slope nonzero','DII > 0; linear slope zero','DII = 0; variance depends on shock']):
 ax.fill_between(s,mean-sd,mean+sd,color='#cadeed',alpha=.8,label='Mean +/- one conditional SD')
 ax.plot(s,mean,color='#163c60',lw=2,label='Conditional mean')
 ax.axhline(0,color='#999999',lw=.6);ax.set_title(title,fontsize=10);ax.set_xlabel('Standardized shock S')
 ax.text(.5,.97,label,transform=ax.transAxes,ha='center',va='top',fontsize=8)
 ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('Standardized outcome Y');axs[0].set_ylim(-4,5)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=9)
fig.tight_layout(rect=[0,.10,1,1])
fig.savefig(P/'dii_population_examples.pdf');fig.savefig(P/'dii_population_examples.png',dpi=150);plt.close(fig)
