"""Exact Gaussian-kernel full-procedure Monte Carlo. No external data.
Run: OPENBLAS_NUM_THREADS=1 python source/full_study.py --reps 300 --boot 399
"""
from pathlib import Path
import numpy as np
import json,csv,argparse,time
P=Path(__file__).resolve().parent.parent

def kernel(x):
 x=np.asarray(x);x=x[:,None] if x.ndim==1 else x
 dist=np.sum(x*x,axis=1)[:,None]+np.sum(x*x,axis=1)[None,:]-2*x@x.T
 return np.exp(-.5*np.maximum(dist,0))
def center(K):return K-K.mean(0)[None,:]-K.mean(1)[:,None]+K.mean()
def basis(z):
 z=np.atleast_2d(z)
 return np.column_stack([np.ones(len(z)),z,z*z])
def draw(rng,N,design,n):
 e=rng.normal(size=N+1);v=rng.normal(size=N+1);c=rng.choice([-1.,1.],size=N)
 if design in ['double','cancel']:
  x=e[1:];y=e[:-1];f=np.zeros(N);g=np.zeros(N)
 elif design in ['regular','near']:
  a=.65 if design=='regular' else n**(-.25)
  s=1+a*c;x=s*e[1:];y=s*e[:-1];f=np.zeros(N);g=np.zeros(N)
 else:
  theta=.5 if design=='weak' else 1.
  x=e[1:];noise=(v[1:]+v[:-1])/np.sqrt(2);f=theta*(x*x-1);y=f+noise;g=np.zeros(N)
 # Cancellation omits C, so oracle HSIC(Y,X)=HSIC(X,Y) exactly.
 zf=x[:,None] if design=='cancel' else np.column_stack([x,c])
 zb=y[:,None] if design=='cancel' else np.column_stack([y,c])
 return x,y,zf,zb,y-f,x-g

def component(K,L,W,counts):
 n=len(K);Kc=center(K);Lc=center(L);H=np.sum(Kc*Lc)/n**2
 wild=np.einsum('bi,bi->b',W@(Kc*Lc),W)/n**2
 q=counts/n;qK=q@K;qL=q@L
 paired=np.einsum('bi,bi->b',q@(K*L),q)-2*np.sum(q*qK*qL,axis=1)+np.sum(q*qK,axis=1)*np.sum(q*qL,axis=1)
 return H,wild,paired

def one(rng,n,ell,m,design,B):
 x,y,zf,zb,uf,ub=draw(rng,m+32+n,design,n)
 sl=slice(m+32,m+32+n)
 bf=np.linalg.lstsq(basis(zf[:m]),y[:m],rcond=None)[0]
 bb=np.linalg.lstsq(basis(zb[:m]),x[:m],rcond=None)[0]
 uhatf=y[sl]-basis(zf[sl])@bf;uhatb=x[sl]-basis(zb[sl])@bb
 Lf=kernel(zf[sl]);Lb=kernel(zb[sl]);J=n//ell
 xi=rng.normal(size=(B,J));xi-=xi.mean(axis=1)[:,None];W=np.repeat(xi,ell,axis=1)
 starts=rng.integers(0,n,size=(B,J));idx=(starts[:,:,None]+np.arange(ell)[None,None,:])%n
 counts=np.zeros((B,n));np.add.at(counts,(np.repeat(np.arange(B),n),idx.reshape(-1)),1)
 result={}
 for label,rf,rb in [('oracle',uf[sl],ub[sl]),('feasible',uhatf,uhatb)]:
  hf,wf,pf=component(kernel(rf),Lf,W,counts);hb,wb,pb=component(kernel(rb),Lb,W,counts)
  D=hb-hf;Dw=wb-wf;Db=pb-pf-D
  # Cancellation can be exactly zero up to floating arithmetic; tie tolerance fixed.
  tol=1e-12
  pw=(1+np.count_nonzero(Dw>=D-tol))/(B+1);pp=(1+np.count_nonzero(Db>=D-tol))/(B+1)
  result[label]=(D,pw,pp,max(pw,pp))
 return result

def wilson(k,N):
 z=1.95996398454;p=k/N;den=1+z*z/N;mid=(p+z*z/(2*N))/den
 half=z*np.sqrt(p*(1-p)/N+z*z/(4*N*N))/den
 return [float(mid-half),float(mid+half)]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=300);ap.add_argument('--boot',type=int,default=399);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 rng=np.random.default_rng(20260911);rows=[];summary=[];t=time.time()
 cells=[(d,'quadratic') for d in ['double','regular','near','weak','strong','cancel']]+[('double','equal'),('regular','equal')]
 for n,ell in [(125,5),(216,6)]:
  for design,allocation in cells:
   m=n*n if allocation=='quadratic' else n
   start=len(rows)
   for rep in range(a.reps):
    r=one(rng,n,ell,m,design,a.boot)
    for label,(D,pw,pb,pc) in r.items():rows.append([design,allocation,n,ell,m,rep,label,D,pw,pb,pc])
   for label in ['oracle','feasible']:
    group=[r for r in rows[start:] if r[6]==label];rates={}
    for name,col in [('wild',8),('paired',9),('intersection',10)]:
     k=int(sum(r[col]<=.05 for r in group));rates[name]={'rejections':k,'rate':k/a.reps,'wilson95':wilson(k,a.reps)}
    summary.append({'design':design,'allocation':allocation,'n':n,'ell':ell,'m':m,'version':label,'reps':a.reps,'rates':rates,'median_D':float(np.median([r[7] for r in group]))})
   print(json.dumps({'n':n,'design':design,'allocation':allocation,'oracle_intersection':summary[-2]['rates']['intersection']['rate'],'feasible_intersection':summary[-1]['rates']['intersection']['rate'],'elapsed_seconds':round(time.time()-t,1)}),flush=True)
 prefix='smoke' if a.smoke else 'full_study'
 with (P/f'{prefix}_raw.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['design','allocation','n','block_length','m','rep','version','D','p_wild','p_paired','p_intersection']);w.writerows(rows)
 out={'seed':20260911,'bootstrap_draws':a.boot,'replications_per_cell':a.reps,'kernel':'fixed Gaussian bandwidth one in every coordinate','alpha':.05,'tie_tolerance':1e-12,'designs':cells,'summary':summary,'elapsed_seconds':time.time()-t}
 (P/f'{prefix}_results.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
