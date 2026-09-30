"""Sensitivity of the pooled robustness value and ceiling to the reference, bandwidth and history.
Reported in full; the baseline is the pre-specified design."""
from pooled_application import *
from core import amplitude
import itertools
S=load()
def Kq(cov,noise,labels,sides):
    q={'labels':labels,'noise':noise,'cov':cov}
    for side,(j,z) in sides.items():
        z=np.array(z);beta=np.linalg.solve(cov[np.ix_(z,z)],cov[z,j]);q[side]={'j':j,'z':z,'beta':beta,'v':float(cov[j,j]-cov[j,z]@beta)}
    return 4*max(amplitude(q,1,0,'b'),amplitude(q,0,1,'b'))**2
def K_nohist(rx,ry,sign):
    a,b=np.sqrt(rx),sign*np.sqrt(ry);cov=np.array([[1,a*b],[a*b,1]])
    return Kq(cov,np.array([np.sqrt(1-rx),np.sqrt(1-ry)]),[('x',0),('y',0)],{'b':(0,[1]),'f':(1,[0])})
def K_hist(rx,ry,sign,phi):
    a,b=np.sqrt(rx),sign*np.sqrt(ry);lab=[('x',0),('y',0),('x',-1),('y',-1)]
    load=np.array([a,b,a,b]);t=np.array([0,0,-1,-1]);noise=np.array([np.sqrt(1-rx),np.sqrt(1-ry)]*2)
    cov=np.outer(load,load)*phi**np.abs(t[:,None]-t[None,:])+np.diag(noise**2)
    return Kq(cov,noise,lab,{'b':(0,[1,2,3]),'f':(1,[0,2,3])})
def matched(rho,fn):
    return max(fn(rx,rho**2/rx,np.sign(rho)) for rx in [.4,.6,.8] if rho**2/rx<=.95)
def pooled(rows,s=1.,hist=False,phi=0.,ell=4):
    st=np.concatenate([[i]*r['n_eval'] for i,r in enumerate(rows)]);n=len(st)
    g=lambda v:gram(np.asarray(v)/s)
    if hist:
        zf=np.concatenate([np.column_stack([r['x'],r['xl'],r['yl']]) for r in rows]);zb=np.concatenate([np.column_stack([r['y'],r['xl'],r['yl']]) for r in rows])
    else:
        zf=np.concatenate([r['x'] for r in rows])[:,None];zb=np.concatenate([r['y'] for r in rows])[:,None]
    rf=np.concatenate([r['rf'] for r in rows]);rb=np.concatenate([r['rb'] for r in rows])
    Af=strat_center(g(rf),st)*strat_center(g(zf),st);Ab=strat_center(g(rb),st)*strat_center(g(zb),st)
    Hf,Hb=max(0,Af.mean()),max(0,Ab.mean())
    starts=[];off=0
    for r in rows:starts+=list(range(off,off+r['n_eval'],ell));off+=r['n_eval']
    rad=radius_starts([Af,Ab],np.array(starts))
    lo,hi=interval(Hf,Hb,rad)
    pi=np.array([r['n_eval'] for r in rows])/n
    K=[(matched(r['rho'],(lambda rx,ry,sg,phi=phi:K_hist(rx,ry,sg,phi)) if hist else K_nohist)) for r in rows]
    Kbar=float((pi@np.sqrt(K))**2)/s**2
    return dict(n=n,dii=Hb-Hf,lower=lo,upper=hi,Kbar=Kbar,rv=np.sqrt(max(lo,0)/Kbar),ceiling=np.sqrt(max(hi,0)/Kbar),Hf=Hf,Hb=Hb,rows=rows)
import itertools as it
def poly3(Z):
    cols=[np.ones(len(Z))]+[Z[:,i] for i in range(Z.shape[1])]+[Z[:,i]*Z[:,j] for i,j in it.combinations_with_replacement(range(Z.shape[1]),2)]
    return np.column_stack(cols)
def build_hist(S,idx):
    """Same rule as build(), adding the previous event's standardized (X,Y) in the stratum as history."""
    out=[]
    for name,(d,cand,longest,eq,bond) in S.items():
        y=eq if idx==0 else bond
        d=d[d[y].notna()];cand=[c for c in cand if c in d.columns and d[c].notna().mean()>=.95]
        dd=d.dropna(subset=cand+[y]).sort_values('t').reset_index(drop=True)
        if len(dd)<41:continue
        ntr=len(dd)//2;M=dd[cand].to_numpy(float);mu=M[:ntr].mean(0);sd=M[:ntr].std(0);keep=sd>0
        Z=(M[:,keep]-mu[keep])/sd[keep];cols=[c for c,k in zip(cand,keep) if k]
        w=np.linalg.svd(Z[:ntr],full_matrices=False)[2][0];ref=next(c for c in longest if c in cols)
        if w[cols.index(ref)]<0:w=-w
        X=np.column_stack([Z@w,dd[y].to_numpy(float)]);m=X[:ntr].mean(0);sdv=X[:ntr].std(0);X=(X-m)/sdv
        L=np.vstack([[np.nan,np.nan],X[:-1]]);X=X[1:];L=L[1:];ntr-=1
        F=np.column_stack([X[:,0],L]);B=np.column_stack([X[:,1],L])
        bf=np.linalg.lstsq(poly3(F[:ntr]),X[:ntr,1],rcond=None)[0];bb=np.linalg.lstsq(poly3(B[:ntr]),X[:ntr,0],rcond=None)[0]
        te=slice(ntr,None);rho=float(np.corrcoef(X[:ntr].T)[0,1])
        out.append(dict(name=name,n_eval=len(X)-ntr,rho=rho,x=X[te,0],y=X[te,1],xl=L[te,0],yl=L[te,1],
            rf=X[te,1]-poly3(F[te])@bf,rb=X[te,0]-poly3(B[te])@bb))
    return out
res=[]
for idx,label in [(0,'Equity index'),(1,'10-year bond yield')]:
    rows=build(S,idx);base=pooled(rows)
    res.append({'outcome':label,'variant':'Baseline (pre-specified)',**{k:v for k,v in base.items() if k!='rows'}})
    # reference shares grid, U fixed
    ks=[]
    for rx,ry in itertools.product([.2,.4,.6,.8],[.2,.4,.6,.8]):
        pi=np.array([r['n_eval'] for r in rows])/base['n']
        Kbar=float((pi@np.sqrt([K_nohist(rx,ry,np.sign(r['rho'])) for r in rows]))**2)
        ks.append((np.sqrt(max(base['lower'],0)/Kbar),np.sqrt(max(base['upper'],0)/Kbar),Kbar))
    ks=np.array(ks)
    res.append({'outcome':label,'variant':'Reference shares $r_X,r_Y\\in\\{.2,.4,.6,.8\\}$','n':base['n'],'dii':base['dii'],'lower':base['lower'],'upper':base['upper'],'Kbar':np.nan,'Kmin':ks[:,2].min(),'Kmax':ks[:,2].max(),'rv':ks[:,0].max(),'ceiling':ks[:,1].max(),'ceiling_min':ks[:,1].min()})
    for s in [.5,2.]:
        o=pooled(rows,s=s);res.append({'outcome':label,'variant':f'Bandwidth {s:g}',**{k:v for k,v in o.items() if k!='rows'}})
    hr=build_hist(S,idx)
    for phi in [0.,.3,.6]:
        o=pooled(hr,hist=True,phi=phi);res.append({'outcome':label,'variant':f'History (previous event), $\\phi={phi:g}$',**{k:v for k,v in o.items() if k!='rows'}})
pd.DataFrame(res).to_csv(R/'pooled_sensitivity.csv',index=False)
print(pd.DataFrame(res)[['outcome','variant','n','dii','lower','upper','Kbar','rv','ceiling','ceiling_min']].to_string(index=False))
