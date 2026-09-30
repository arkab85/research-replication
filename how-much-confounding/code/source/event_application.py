"""Original event design (audit/original_submission/event_design_prespecification.md)."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from core import pair,radius,interval,amplitude
P=Path(__file__).resolve().parents[1];D=P/'data'/'event';R=P/'results'
ev=pd.read_csv(D/'shocks_fed_jk_t.csv');ev['date']=pd.to_datetime(ev.start).dt.normalize()
vix=pd.read_csv(D/'vix-daily.csv',parse_dates=['DATE']).set_index('DATE').CLOSE
def ref_K(rho):
    """max K_b over the pre-specified correlation-matched Gaussian reference set, no history."""
    Ks=[]
    for rx in [.4,.6,.8]:
        ry=rho**2/rx
        if ry>.95: continue
        a=np.sqrt(rx);b=np.sign(rho)*np.sqrt(ry)
        cov=np.array([[1,a*b],[a*b,1]]);noise=np.array([np.sqrt(1-rx),np.sqrt(1-ry)])
        q={'labels':[('x',0),('y',0)],'noise':noise,'cov':cov}
        for side,j in [('b',0),('f',1)]:
            z=np.array([1-j]);beta=np.linalg.solve(cov[np.ix_(z,z)],cov[z,j]);v=cov[j,j]-cov[j,z]@beta
            q[side]={'j':j,'z':z,'beta':beta,'v':float(v)}
        Ks.append(4*max(amplitude(q,1,0,'b'),amplitude(q,0,1,'b'))**2)
    return max(Ks)
def poly(x):return np.column_stack([np.ones(len(x)),x,x*x])
cells={}
cells[('E1: S&P 500 surprise',0)]=ev.assign(x=ev.pc1,y=ev.SP500)
days=vix.index
for h in [1,5,20]:
    rows=[]
    for _,r in ev.iterrows():
        i=days.searchsorted(r.date)
        if i>=len(days) or days[i]!=r.date or i+h>=len(days):rows.append(np.nan);continue
        rows.append(100*np.log(vix.iloc[i+h]/vix.iloc[i]))
    cells[('E2: VIX change',h)]=ev.assign(x=ev.MP_median,y=np.array(rows))
comps={};As=[];meta=[]
for key,df in cells.items():
    df=df[['date','x','y']].dropna()
    tr=df[df.date<'2008-01-01'];te=df[df.date>='2009-01-01']
    mu=tr[['x','y']].mean().to_numpy();sd=tr[['x','y']].std(ddof=0).to_numpy()
    ztr=(tr[['x','y']].to_numpy()-mu)/sd;zte=(te[['x','y']].to_numpy()-mu)/sd
    bf=np.linalg.lstsq(poly(ztr[:,0]),ztr[:,1],rcond=None)[0];bb=np.linalg.lstsq(poly(ztr[:,1]),ztr[:,0],rcond=None)[0]
    rf=zte[:,1]-poly(zte[:,0])@bf;rb=zte[:,0]-poly(zte[:,1])@bb
    f,b=pair((rf,zte[:,[0]],rb,zte[:,[1]]));As+= [f[1],b[1]];comps[key]=(f,b)
    rho=float(np.corrcoef(ztr.T)[0,1])
    meta.append({'design':key[0],'h':key[1],'train_n':len(tr),'test_n':len(te),'train_corr':rho,'K':ref_K(rho)})
out=[]
for ell in [4,1]:
    rad=radius(As,ell=ell,B=1999,seed=20260919)
    for (key,(f,b)),m in zip(comps.items(),meta):
        for r in [0.,.01,.025,.05]:
            lo,hi=interval(f[0],b[0],rad,r,r)
            out.append({**m,'ell':ell,'r':r,'forward_hsic':f[0],'backward_hsic':b[0],'dii':b[0]-f[0],'radius':rad,'lower':lo,'upper':hi,'rv':np.sqrt(max(lo,0)/m['K']),'rv_point':np.sqrt(max(b[0]-f[0],0)/m['K'])})
pd.DataFrame(out).to_csv(R/'event_application.csv',index=False)
(R/'event_metadata.json').write_text(json.dumps({'files':{f:hashlib.sha256((D/f).read_bytes()).hexdigest() for f in ['shocks_fed_jk_t.csv','vix-daily.csv']},'components':len(As),'B':1999,'seed':20260919},indent=2))
print(pd.DataFrame(out).query('r==0')[['design','h','ell','train_n','test_n','train_corr','forward_hsic','backward_hsic','dii','lower','K','rv','rv_point']].to_string(index=False))
