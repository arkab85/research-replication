"""Reference DII inference for fixed, aligned residual/regressor comparisons.

This is a research implementation of the paper's specified block algorithms.
Inputs are evaluation residuals after a separately trained conditional-mean model.
The caller must justify nuisance rates, common history, fixed feature scales,
finite-memory dependence, and the covered null configuration. Computing a p-value
cannot establish these conditions. No automatic causal interpretation is supplied.
"""
import numpy as np
from full_study import kernel,component

def dii_profile(comparisons, block_length, bootstrap_draws=399, seed=20260910, contrasts=None):
    """comparisons: ordered dict name -> (forward_residual, forward_regressors,
    reverse_residual, reverse_regressors). All rows share one evaluation calendar.
    Inputs are in prespecified units; kernel bandwidth is one in each coordinate.
    contrasts: dict label -> fixed weights in comparison order (e.g. horizon hump).
    Returns DII components, upper-tail component/intersection p-values, and
    conjunction p-value for positivity at every supplied comparison.
    No adjustment is needed for this prespecified conjunction. Searching for any
    significant horizon is a different question and requires multiplicity control.
    """
    if not comparisons: raise ValueError('At least one comparison is required')
    names=list(comparisons); n=len(comparisons[names[0]][0]);ell=int(block_length);B=int(bootstrap_draws)
    if ell<1 or n%ell or B<1: raise ValueError('Require n divisible by positive block length and B >= 1')
    for name,arrs in comparisons.items():
        if len(arrs)!=4 or any(len(a)!=n or not np.isfinite(a).all() for a in map(np.asarray,arrs)):
            raise ValueError('All arrays must be finite and calendar-aligned')
    rng=np.random.default_rng(seed);J=n//ell
    xi=rng.normal(size=(B,J));xi-=xi.mean(axis=1)[:,None];W=np.repeat(xi,ell,axis=1)
    starts=rng.integers(0,n,size=(B,J));idx=(starts[:,:,None]+np.arange(ell)[None,None,:])%n
    counts=np.zeros((B,n));np.add.at(counts,(np.repeat(np.arange(B),n),idx.ravel()),1)
    obs=[];ws=[];ps=[];components={}
    for name in names:
        rf,zf,rb,zb=map(np.asarray,comparisons[name])
        hf,wf,pf=component(kernel(rf),kernel(zf),W,counts)
        hb,wb,pb=component(kernel(rb),kernel(zb),W,counts)
        obs.append(hb-hf);ws.append(wb-wf);ps.append(pb-pf-(hb-hf))
        components[name]={'forward_hsic':float(hf),'reverse_hsic':float(hb)}
    obs=np.asarray(obs);ws=np.asarray(ws);ps=np.asarray(ps)
    def report(weights):
        weights=np.asarray(weights,dtype=float)
        if weights.shape!=(len(names),) or not np.isfinite(weights).all():raise ValueError('Invalid contrast weights')
        d=float(weights@obs);dw=weights@ws;db=weights@ps
        pw=(1+np.count_nonzero(dw>=d-1e-12))/(B+1);pb=(1+np.count_nonzero(db>=d-1e-12))/(B+1)
        return {'dii':d,'p_wild':float(pw),'p_paired':float(pb),'p_intersection':float(max(pw,pb))}
    for j,name in enumerate(names):components[name].update(report(np.eye(len(names))[j]))
    return {'n':n,'block_length':ell,'bootstrap_draws':B,'seed':seed,'comparisons':components,
            'contrasts':{name:report(w) for name,w in (contrasts or {}).items()},
            'p_all_positive':max(r['p_intersection'] for r in components.values()),
            'interpretation':'Pointwise inference requires the assumptions in the manuscript; no causal or economic-effect certification.'}
