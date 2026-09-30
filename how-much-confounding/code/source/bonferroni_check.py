"""Bonferroni check for the single positive sensitivity variant (bond, bandwidth 0.5)."""
import json
from pooled_application import *
S=load();rows=build(S,1);s=.5
st=np.concatenate([[i]*r['n_eval'] for i,r in enumerate(rows)])
g=lambda v:gram(np.asarray(v)/s)
zf=np.concatenate([r['x'] for r in rows])[:,None];zb=np.concatenate([r['y'] for r in rows])[:,None]
rf=np.concatenate([r['rf'] for r in rows]);rb=np.concatenate([r['rb'] for r in rows])
Af=strat_center(g(rf),st)*strat_center(g(zf),st);Ab=strat_center(g(rb),st)*strat_center(g(zb),st)
starts=[];off=0
for r in rows:starts+=list(range(off,off+r['n_eval'],4));off+=r['n_eval']
out={}
for a in [.05,.05/7]:
    rad=radius_starts([Af,Ab],np.array(starts),B=4999,alpha=a);lo,hi=interval(max(0,Af.mean()),max(0,Ab.mean()),rad);out[str(a)]={'radius':rad,'lower':lo,'upper':hi}
(P/'results/bonferroni_check.json').write_text(json.dumps(out,indent=2));print(out)
