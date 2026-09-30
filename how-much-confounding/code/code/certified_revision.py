"""Revision: exact Gaussian kernels, certified rotation cells and joint projection.
Run with OPENBLAS_NUM_THREADS=1 python certified_revision.py --workers 6.
All bootstrap draws and seeds are retained. Influence exercises are conditional,
post-selection diagnostics, not simultaneous tests across selected deletions.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import chi2
from svarcore import var_regenerate,mbb_indices,whiten,rot,gram,_center,coskew
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'revision_results'; OUT.mkdir(exist_ok=True)
TH=np.linspace(0,np.pi/2,31)
CURV=4*np.sqrt(3)+8

def op_inner(E,F):
    return np.sum(_center(gram(E[:,0],F[:,0]))*_center(gram(E[:,1],F[:,1])))/(len(E)*len(F))
def stats(Z,angles):
    return np.array([op_inner(Z@rot(t),Z@rot(t)) for t in angles])
def distances(Zb,Z,ab,ao,orig):
    out=[]
    for t,s,v in zip(ab,ao,orig):
        Eb=Zb@rot(t); E=Z@rot(s)
        out.append(np.sqrt(max(0,op_inner(Eb,Eb)+v-2*op_inner(Eb,E))))
    return np.array(out)
def rec(L):return np.array([0.,np.arctan2(-L[1,0],L[1,1])%(np.pi/2)])
def fit(y,p,drop):
    n=len(y)-p
    X=np.column_stack([np.ones(n)]+[y[p-j:len(y)-j] for j in range(1,p+1)])
    D=np.eye(n)[:,drop] if len(drop) else np.empty((n,0))
    B=np.linalg.lstsq(np.column_stack([X,D]),y[p:],rcond=None)[0]
    U=y[p:]-np.column_stack([X,D])@B
    keep=np.ones(n,bool);keep[drop]=False
    return B[:X.shape[1]], D@B[X.shape[1]:], U[keep],keep

def spec(name):
    if name.startswith('oil'):
        y=np.loadtxt(ROOT/'oil_extended.txt');p=24;cols=[0,2];block=12
        dates=pd.date_range('1974-01-01',periods=len(y),freq='MS')[p:]
        drop=np.where(dates==pd.Timestamp('2020-05-01'))[0] if name=='oil_may' else np.array([],int)
    else:
        data=pd.read_csv(ROOT/'weekly.csv',index_col=0,parse_dates=True);y=data.values;p=4;cols=[0,1];block=10;dates=data.index[p:]
        drop=np.array([],int)
        if name=='equity_crisis':drop=np.where((dates>='2008-09-01')&(dates<='2009-03-31'))[0]
        if name in ('equity_top1','equity_top5'):
            _,_,u,_=fit(y,p,[]);z,l=whiten(u);g=[]
            for t in rec(l):
                e=z@rot(t);e=(e-e.mean(0))/e.std(0);v=np.abs(e[:,0]**2*e[:,1])+np.abs(e[:,0]*e[:,1]**2);g.append(v/v.sum())
            drop=np.argsort(-np.sum(g,axis=0))[:1 if name.endswith('1') else 5]
    B,F,U,keep=fit(y,p,drop);Z,L=whiten(U[:,cols]);aa=rec(L)
    return dict(name=name,y=y,p=p,cols=cols,block=block,drop=drop,dates=dates,B=B,F=F,U=U-U.mean(0),keep=keep,Z=Z,L=L,aa=aa,orig=stats(Z,TH),origr=stats(Z,aa))

STATE=None
def init(state):
    global STATE;STATE=state

def draw(seed):
    s=STATE;rng=np.random.default_rng(seed)
    # Fixed impulse design, original calendar and original sample length.
    idx=mbb_indices(len(s['U']),s['block'],rng)
    pool=s['U'][idx]
    # Resample enough innovations for every original date; impulse dates are
    # conditioned on and their innovations set to zero (saturated nuisance).
    innovations=np.zeros((len(s['keep']),s['y'].shape[1]));innovations[s['keep']]=pool
    ys=var_regenerate(s['B'],s['y'][:s['p']],innovations+s['F'])
    _,_,u,_=fit(ys,s['p'],s['drop']);z,l=whiten(u[:,s['cols']]);a=rec(l)
    d=distances(z,s['Z'],TH,TH,s['orig']);dr=distances(z,s['Z'],a,s['aa'],s['origr'])
    return d,dr,np.linalg.norm(np.linalg.solve(s['L'],l-s['L']),'fro'),coskew(z,a),l

def cell_lower(Z,orig,q):
    lower=[]; raw=[]
    for j in range(len(TH)-1):
        a,b=orig[j:j+2];c=op_inner(Z@rot(TH[j]),Z@rot(TH[j+1]));v=max(a+b-2*c,0)
        t=np.clip((a-c)/v,0,1) if v>1e-16 else 0
        m=np.sqrt(max(a+2*t*(c-a)+t*t*v,0));err=CURV*(TH[j+1]-TH[j])**2/8
        lower.append(max(0,m-q-err));raw.append(m)
    return np.array(lower),np.array(raw)

def elasticity_cells(L,r):
    bounds=[]
    for a,b in zip(TH[:-1],TH[1:]):
        A=L@rot((a+b)/2);err=np.linalg.norm(L,axis=1)*(r+2*np.sin((b-a)/4)); intervals=[]
        for k in range(2):
            xlo,xhi=A[0,k]-err[0],A[0,k]+err[0];ylo,yhi=A[1,k]-err[1],A[1,k]+err[1]
            # Sign-normalize each potentially co-moving column, with conservative
            # rectangular entry bounds; denominators near zero imply infinity.
            for sign in (1,-1):
                xl,xh=(xlo,xhi) if sign==1 else (-xhi,-xlo)
                yl,yh=(ylo,yhi) if sign==1 else (-yhi,-ylo)
                if xh>=0 and yh>0:
                    intervals.append((max(0,xl)/yh, float('inf') if yl<=0 else max(0,xh)/yl))
        bounds.append([min(v[0] for v in intervals),max(v[1] for v in intervals)] if intervals else [float('nan')]*2)
    return np.array(bounds)

def report(s,res,seeds):
    D,DR,R,CS,LB=[np.array(v) for v in zip(*res)]
    q=float(np.quantile(D.max(1),.975,method='higher'));r=float(np.quantile(R,.975,method='higher'))
    qr=float(np.quantile(DR.max(1),.95,method='higher'))
    lower,raw=cell_lower(s['Z'],s['orig'],q);bounds=elasticity_cells(s['L'],r)
    cs=coskew(s['Z'],s['aa']);W=[float(cs[j]@np.linalg.solve(np.cov(CS[:,j,:].T),cs[j])) for j in range(2)]
    sets={}
    for rho in (0,.025,.05,.1,.15,.2):
        keep=lower<=rho*rho
        sets[str(rho)]={'cell_share':float(keep.mean()),'elasticity_min':float(np.nanmin(bounds[keep,0])) if keep.any() else None,'elasticity_max':float(np.nanmax(bounds[keep,1])) if keep.any() else None}
    breakdown={str(c):float(np.sqrt(np.min(lower[bounds[:,1]>=c]))) if np.any(bounds[:,1]>=c) else None for c in (.0258,.05,.1,.2)}
    ans={'name':s['name'],'n':len(s['Z']),'B':len(res),'block':s['block'],'grid_degrees':3,'q_operator_975':q,'r_whitening_975':r,'curvature_allowance':float(CURV*(TH[1]-TH[0])**2/8),'joint_confidence':.95,'recursive_q95':qr,'recursive_angles':np.rad2deg(s['aa']).tolist(),'recursive_norms':np.sqrt(s['origr']).tolist(),'recursive_budget_lower':np.sqrt(np.maximum(np.sqrt(s['origr'])-qr,0)).tolist(),'recursive_wald':W,'recursive_p':chi2.sf(W,2).tolist(),'recursive_familywise_critical':float(chi2.ppf(.975,2)),'sets':sets,'breakdown_lower':breakdown,'dropped_dates':[str(s['dates'][j].date()) for j in s['drop']]}
    np.savez_compressed(OUT/(s['name']+'.npz'),D=D,DR=DR,R=R,CS=CS,Lboot=LB,seeds=seeds,theta=TH,L=s['L'],norms=np.sqrt(s['orig']),cell_lower=lower,cell_segment_min=raw,elasticity_bounds=bounds)
    if s['name'].startswith('equity'):
        ans['sets']={k:{'cell_share':v['cell_share']} for k,v in ans['sets'].items()}
        ans.pop('breakdown_lower')
    def clean(v):
        if isinstance(v,float) and np.isinf(v): return 'unbounded'
        if isinstance(v,float) and np.isnan(v): return None
        if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list): return [clean(x) for x in v]
        return v
    ans=clean(ans)
    (OUT/(s['name']+'.json')).write_text(json.dumps(ans,indent=2,allow_nan=False))
    return ans

def run(name,B,workers,seed):
    s=spec(name);seeds=np.random.SeedSequence(seed).generate_state(B);t=time.time()
    with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(s,)) as ex:
        res=[]
        for j,row in enumerate(ex.map(draw,seeds,chunksize=1)):
            res.append(row)
            if (j+1)%25==0:print(name,j+1,'/',B,'seconds',round(time.time()-t),flush=True)
    ans=report(s,res,seeds);print(json.dumps(ans),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--names',nargs='+',default=['oil','equity','oil_may','equity_top1','equity_top5','equity_crisis']);ap.add_argument('--B',type=int,default=399);args=ap.parse_args()
    for i,name in enumerate(args.names):run(name,args.B if name in ('oil','equity') else min(args.B,199),args.workers,260926+['oil','equity','oil_may','equity_top1','equity_top5','equity_crisis'].index(name)*100)
