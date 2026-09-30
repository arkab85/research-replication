"""Additional dependence check; does not correct estimated VAR preprocessing."""
from pathlib import Path
import numpy as np,json
from jointproc import analyse_multi
P=Path(__file__).resolve().parent
y=np.loadtxt(P/'oil_extended.txt');T=len(y);p=24
X=np.column_stack([np.ones(T-p)]+[y[p-j:T-j] for j in range(1,p+1)]);Y=y[p:]
rows=[]
for end in [408,312]:
 tr=np.arange(p,T)<end;coef=np.linalg.lstsq(X[tr],Y[tr],rcond=None)[0];U=Y-X@coef
 U=U/U[tr].std(0,ddof=1);a=U[tr];b=U[~tr];lag=int(4*(len(a)/100)**(2/9))
 pairs=[(0,2),(0,1)]
 r,q0,qJ=analyse_multi([(a[:,i],a[:,j]) for i,j in pairs],[(b[:,i],b[:,j]) for i,j in pairs],seed=7,block=3,hac_lags=lag)
 rows.append(dict(training_end=end,training_hac_lags=lag,q0=q0,qJ=qJ,results=r))
 assert all(v['joint'][0]<0 for v in r)
(P/'oil_hac_check.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
