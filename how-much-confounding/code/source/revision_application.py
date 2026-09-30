"""Post-review empirical additions: error budgets, calendar blocks, direct-sum pooling."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from core import gram, strat_center, center, interval
from pooled_application import load, build, radius_starts

P=Path(__file__).resolve().parents[1];R=P/'results'

def cluster_radius(As, labels, B=1999, seed=20260922):
    n=len(labels);_,g=np.unique(labels,return_inverse=True);m=g.max()+1
    # Sum within arbitrary calendar clusters using a sparse membership matrix.
    from scipy.sparse import csr_matrix
    membership=csr_matrix((np.ones(n),(g,np.arange(n))),shape=(m,n))
    weights=np.random.default_rng(seed).normal(size=(B,m));mx=np.zeros(B)
    for A in As:
        cov=np.asarray((membership@center(A))@membership.T)
        mx=np.maximum(mx,np.sqrt(np.maximum(0,np.einsum('bi,ij,bj->b',weights,cov,weights,optimize=True)))/n)
    return float(np.quantile(mx,.95,method='higher')),int(m)

def main():
    threadpool_limits(limits=1);S=load();out=[];points=[]
    baseline=pd.read_csv(R/'pooled_application.csv')
    for idx,label in [(0,'Equity index'),(1,'10-year bond yield')]:
        rows=build(S,idx);n=sum(r['n_eval'] for r in rows)
        st=np.concatenate([np.repeat(i,r['n_eval']) for i,r in enumerate(rows)])
        x,y,rf,rb=[np.concatenate([r[k] for r in rows]) for k in ['x','y','rf','rb']]
        dates=pd.DatetimeIndex(np.concatenate([r['dates'] for r in rows]))
        af=strat_center(gram(rf),st)*strat_center(gram(x),st)
        ab=strat_center(gram(rb),st)*strat_center(gram(y),st)
        pi=np.array([r['n_eval']/n for r in rows]);ks=np.array([r['K'] for r in rows])
        kb=float((pi@np.sqrt(ks))**2)
        hf,hb=float(af.mean()),float(ab.mean())
        old=baseline[(baseline.design==label)&(baseline.ell==4)&(baseline.r==0)].iloc[0]
        assert abs((hb-hf)-old.dii)<1e-12
        for row in rows:
            from core import component
            f=component(row['rf'],row['x'])[0];b=component(row['rb'],row['y'])[0]
            points.append(dict(outcome=label,stratum=row['name'],n=row['n_eval'],Hf=f,Hb=b,D=b-f))
        for kind in ['average','direct_sum']:
            if kind=='direct_sum':
                mask=(st[:,None]==st[None,:])/pi[st,None]
                As=[af*mask,ab*mask];K=float(pi@ks)
            else:As=[af,ab];K=kb
            Hf,Hb=[float(a.mean()) for a in As]
            for months in [1,3,6]:
                # Same calendar multiplier for all banks and communication types.
                labels=(dates.year*12+dates.month-1)//months
                rad,groups=cluster_radius(As,np.asarray(labels))
                for r in [0.,.01,.025,.05]:
                    lo,hi=interval(Hf,Hb,rad,r,r)
                    out.append(dict(outcome=label,pooling=kind,calendar_months=months,
                        groups=groups,n=n,r=r,Hf=Hf,Hb=Hb,D=Hb-Hf,radius=rad,
                        lower=lo,upper=hi,K=K,rv=np.sqrt(max(lo,0)/K),
                        ceiling=np.sqrt(max(hi,0)/K)))
            print(label,kind,'complete',flush=True)
    pd.DataFrame(out).to_csv(R/'revision_calendar.csv',index=False)
    pd.DataFrame(points).to_csv(R/'revision_strata_components.csv',index=False)
    baseline['ceiling']=np.sqrt(np.maximum(baseline.upper,0)/baseline.Kbar)
    baseline.query('ell==4').to_csv(R/'revision_error_budgets.csv',index=False)
    (R/'revision_application_metadata.json').write_text(json.dumps(dict(
        design_status='post-review sensitivity analysis, not preregistered',
        multiplier_draws=1999,seed=20260922,
        calendar_blocks_months=[1,3,6],
        interpretation='Conditional on fixed trained functions; calendar blocks do not prove sampling validity.'),indent=2))
    print(pd.DataFrame(out).query('calendar_months==3 and r==0').to_string(index=False))

if __name__=='__main__':main()
