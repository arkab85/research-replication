"""Full grid fixed in notes/feasible_study_design.md; independent of pilot seeds."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd,json,time
from threadpoolctl import threadpool_limits
from revision_study import draw,truth
from feasible_inference import joint_analyze
from core import interval
P=Path(__file__).resolve().parents[1];R=P/'results'
def run_cell(task):
 model,phi,n,degree=task;threadpool_limits(limits=1)
 mid={'confounding':1,'quadratic':2,'linear':3}[model]
 seed=77119000+mid*100000+int(phi*10)*10000+n*10+degree
 rng=np.random.default_rng(seed);dtrue=truth(model)[-1];out=[];K=28.444444444444443
 for rep in range(200):
  xt,yt=draw(n,phi,model,rng);x,y=draw(n,phi,model,rng)
  row,bs=joint_analyze(xt,yt,x,y,degree,phi>0,399,seed+rep+1)
  eta=(.3*.8**2 if model=='confounding' else 1.) if degree==1 and model!='linear' else 0.
  lm,um=interval(row['Hf'],row['Hb'],row['q'],eta,0.)
  row.update(model=model,phi=phi,n=n,n_train=n,degree=degree,rep=rep,seed=seed,D_true=dtrue,K=K,
             eta_model_known=eta,lower_misspec_allowance=lm,upper_misspec_allowance=um)
  for tag,lo,hi in [('joint',row['lower'],row['upper']),('ignore',row['lower_ignore'],row['upper_ignore']),('feasible_r',row['lower_feasible_r'],row['upper_feasible_r']),('misspec',lm,um)]:
   row['cover_'+tag]=lo<=dtrue<=hi;row['power_'+tag]=lo>K*.01**2
   row['rv_'+tag]=np.sqrt(max(lo,0)/K);row['width_'+tag]=hi-lo
  out.append(row)
 return out
if __name__=='__main__':
 tasks=[(m,p,n,2) for m in ['confounding','quadratic','linear'] for p in [0.,.6] for n in [250,600,1200]]
 tasks += [(m,p,600,1) for m in ['confounding','quadratic'] for p in [0.,.6]]
 allrows=[];start=time.time()
 with ProcessPoolExecutor(max_workers=6) as pool:
  fs={pool.submit(run_cell,t):t for t in tasks}
  for f in as_completed(fs):
   allrows.extend(f.result());print('Complete',fs[f],round(time.time()-start,1),flush=True)
 d=pd.DataFrame(allrows).sort_values(['model','phi','n','degree','rep']);d.to_csv(R/'feasible_mc_raw.csv',index=False)
 metrics={}
 for tag in ['joint','ignore','feasible_r','misspec']:
  for kind in ['cover','power','width']:metrics[f'{kind}_{tag}']=(f'{kind}_{tag}','mean')
  metrics[f'rv_{tag}']=(f'rv_{tag}','median')
 for f in ['q','q_ignore','feasible_r_f','feasible_r_b']:metrics[f]= (f,'median')
 s=d.groupby(['model','phi','n','degree']).agg(reps=('rep','size'),D_true=('D_true','first'),**metrics).reset_index()
 for col in [c for c in s if c.startswith('cover_') or c.startswith('power_')]:s[col+'_mcse']=np.sqrt(s[col]*(1-s[col])/s.reps)
 s.to_csv(R/'feasible_mc_summary.csv',index=False)
 (R/'feasible_mc_metadata.json').write_text(json.dumps({'replications':len(d),'cells':len(tasks),'bootstrap_draws':399,'design':'disclosed post-pilot method-development study','seconds':time.time()-start},indent=2))
 print(s[['model','phi','n','degree','cover_joint','power_joint','power_feasible_r']].to_string(index=False),flush=True)
