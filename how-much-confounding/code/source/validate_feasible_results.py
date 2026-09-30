"""Check full-cell aggregation and reproduce independent representative draws."""
from pathlib import Path
import json, ast
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from revision_study import draw
from feasible_inference import joint_analyze
P=Path(__file__).resolve().parents[1];R=P/'results';threadpool_limits(limits=1)
d=pd.read_csv(R/'feasible_mc_raw.csv');s=pd.read_csv(R/'feasible_mc_summary.csv')
assert len(d)==4400 and len(s)==22
assert d.groupby(['model','phi','n','degree']).size().eq(200).all()
for row in s.itertuples():
 g=d[(d.model==row.model)&(d.phi==row.phi)&(d.n==row.n)&(d.degree==row.degree)]
 for tag in ['joint','ignore','feasible_r','misspec']:
  for kind in ['cover','power','width','rv']:
   col=f'{kind}_{tag}';v=g[col].median() if kind=='rv' else g[col].mean()
   assert np.isclose(v,getattr(row,col),rtol=1e-12,atol=1e-12),(col,row)
for model in ['confounding','quadratic','linear']:
 r=d[(d.model==model)&(d.phi==.6)&(d.n==250)&(d.degree==2)&(d.rep==0)].iloc[0]
 rng=np.random.default_rng(int(r.seed));xt,yt=draw(250,.6,model,rng);x,y=draw(250,.6,model,rng)
 actual,_=joint_analyze(xt,yt,x,y,2,True,399,int(r.seed)+1)
 for k,v in actual.items():assert np.isclose(v,r[k],rtol=1e-10,atol=1e-12),(model,k,v,r[k])
a=pd.read_csv(R/'feasible_application.csv');assert len(a)==48
assert (a.lower<=a.upper).all() and (a.lower>=-1).all() and (a.upper<=1).all()
assert a.rv.eq(0).all()
for f in (P/'source').glob('*.py'):ast.parse(f.read_text())
report={'replications':4400,'cells':22,'replications_per_cell':200,'aggregations_verified':True,'independently_reproduced_first_replications':3,'application_scenarios':48,'all_application_lower_certificates_zero':True,'source_syntax_valid':True}
(P/'audit/feasible_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
