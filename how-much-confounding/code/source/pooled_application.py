"""Original pooled design (audit/original_submission/pooled_design_prespecification.md)."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from core import gram,strat_center,center,amplitude,interval,component
P=Path(__file__).resolve().parents[1];D=P/'data'/'pooled';R=P/'results'
US=['MP1','MP2','FF1','FF2','FF3','FF4','FF5','FF6','ED1','ED2','ED3','ED4','OIS1Y','UST3M','UST6M']
EA=['OIS_SW','OIS_1M','OIS_3M','OIS_6M','OIS_1Y'];EE=['OIS_1W','OIS_1M','OIS_2M','OIS_3M','OIS_6M','OIS_1Y']
UK=['cm1','cm2','cm3','cm4','GBP1MOIS=','GBP2MOIS=','GBP3MOIS=','GBP1YOIS=','GB1YT=RR']
def load():
    S={}
    for sh,name in [('Statements','Fed statements'),('Press Conferences','Fed press conferences'),('Minutes','Fed minutes')]:
        d=pd.read_excel(D/'USMPD.xlsx',sheet_name=sh);d['t']=pd.to_datetime(d.date_time)
        S[name]=(d,US,['OIS1Y','ED4'],'SP500','UST10Y')
    for sh,name in [('Press Release Window','ECB press releases'),('Press Conference Window','ECB press conferences')]:
        d=pd.read_excel(D/'Dataset_EA-MPD.xlsx',sheet_name=sh);d['t']=pd.to_datetime(d.date)
        S[name]=(d,EA,['OIS_1Y'],'STOXX50','DE10Y')
    e=pd.read_excel(D/'EA-EMPD.en.xlsx',sheet_name='EA-EMPD');e['t']=pd.to_datetime(e.Date_time)
    for et,name in [('EB','ECB Board-member speeches'),('P','ECB President speeches')]:
        S[name]=(e[e.Event_type==et].copy(),EE,['OIS_1Y'],'STOXX50E','DE10Y')
    u=pd.read_excel(D/'UKMPD.xlsx',sheet_name='surprises');u['t']=pd.to_datetime(u.Datetime)
    for k in range(1,5):u[f'cm{k}']=u[f'FSScm{k}'].fillna(u[f'SON3c{k}'])
    for flag,name in [(True,'BoE MPC announcements'),(False,'BoE MPR press conferences')]:
        S[name]=(u[u.isMPC==flag].copy(),UK,['cm4'],'.FTSE','GB10YT=RR')
    return S
def ref_K(rho):
    Ks=[]
    for rx in [.4,.6,.8]:
        ry=rho**2/rx
        if ry>.95:continue
        a=np.sqrt(rx);b=np.sign(rho)*np.sqrt(ry);cov=np.array([[1,a*b],[a*b,1]])
        q={'labels':[('x',0),('y',0)],'noise':np.array([np.sqrt(1-rx),np.sqrt(1-ry)]),'cov':cov}
        for side,j in [('b',0),('f',1)]:
            z=np.array([1-j]);beta=np.linalg.solve(cov[np.ix_(z,z)],cov[z,j]);q[side]={'j':j,'z':z,'beta':beta,'v':float(cov[j,j]-cov[j,z]@beta)}
        Ks.append(4*max(amplitude(q,1,0,'b'),amplitude(q,0,1,'b'))**2)
    return max(Ks)
poly=lambda x:np.column_stack([np.ones(len(x)),x,x*x])
def build(S,ycol_idx,devol=False):
    rows=[]
    for name,(d,cand,longest,eq,bond) in S.items():
        y=eq if ycol_idx==0 else bond
        d=d[d[y].notna()];cand=[c for c in cand if c in d.columns and d[c].notna().mean()>=.95]
        dd=d.dropna(subset=cand+[y]).sort_values('t').reset_index(drop=True)
        if len(dd)<40:print('drop stratum',name,len(dd));continue
        ntr=len(dd)//2;M=dd[cand].to_numpy(float);mu=M[:ntr].mean(0);sd=M[:ntr].std(0);keep=sd>0
        Z=(M[:,keep]-mu[keep])/sd[keep];cols=[c for c,k in zip(cand,keep) if k]
        w=np.linalg.svd(Z[:ntr],full_matrices=False)[2][0]
        ref=next(c for c in longest if c in cols)
        if w[cols.index(ref)]<0:w=-w
        x=Z@w;yy=dd[y].to_numpy(float)
        if devol:
            # exploratory: divide by trailing root-mean-square over the previous 20 events (past only)
            sx_=pd.Series(x).pow(2).shift(1).rolling(20,min_periods=10).mean().pow(.5).to_numpy()
            sy_=pd.Series(yy).pow(2).shift(1).rolling(20,min_periods=10).mean().pow(.5).to_numpy()
            ok=np.isfinite(sx_)&np.isfinite(sy_)&(sx_>0)&(sy_>0)
            x=x[ok]/sx_[ok];yy=yy[ok]/sy_[ok];dd=dd[ok].reset_index(drop=True);ntr=len(dd)//2
        X=np.column_stack([x,yy]);m=X[:ntr].mean(0);s=X[:ntr].std(0);X=(X-m)/s
        bf=np.linalg.lstsq(poly(X[:ntr,0]),X[:ntr,1],rcond=None)[0];bb=np.linalg.lstsq(poly(X[:ntr,1]),X[:ntr,0],rcond=None)[0]
        te=X[ntr:];rho=float(np.corrcoef(X[:ntr].T)[0,1])
        rows.append(dict(name=name,instruments=cols,n_train=ntr,n_eval=len(te),rho=rho,K=ref_K(rho),
             x=te[:,0],y=te[:,1],rf=te[:,1]-poly(te[:,0])@bf,rb=te[:,0]-poly(te[:,1])@bb,
             dates=dd.t.iloc[ntr:].to_numpy(),
             start=str(dd.t.iloc[0].date()),split=str(dd.t.iloc[ntr].date()),end=str(dd.t.iloc[-1].date())))
    return rows
def radius_starts(As,starts,B=1999,alpha=.05,seed=20260920):
    n=len(As[0]);w=np.random.default_rng(seed).normal(size=(B,len(starts)));mx=np.zeros(B)
    for A in As:
        Sm=np.add.reduceat(np.add.reduceat(center(A),starts,axis=0),starts,axis=1)
        mx=np.maximum(mx,np.sqrt(np.maximum(0,np.einsum('bi,ij,bj->b',w,Sm,w,optimize=True)))/n)
    return float(np.quantile(mx,1-alpha,method='higher'))
def run(rows,label):
    st=np.concatenate([[i]*r['n_eval'] for i,r in enumerate(rows)]);n=len(st)
    x=np.concatenate([r['x'] for r in rows]);y=np.concatenate([r['y'] for r in rows])
    rf=np.concatenate([r['rf'] for r in rows]);rb=np.concatenate([r['rb'] for r in rows])
    Af=strat_center(gram(rf),st)*strat_center(gram(x[:,None]),st);Ab=strat_center(gram(rb),st)*strat_center(gram(y[:,None]),st)
    Hf,Hb=max(0,Af.mean()),max(0,Ab.mean())
    pi=np.array([r['n_eval'] for r in rows])/n;Kbar=float((pi@np.sqrt([r['K'] for r in rows]))**2)
    out=[];per=[]
    for r in rows:
        f=component(r['rf'],r['x'][:,None])[0];b=component(r['rb'],r['y'][:,None])[0]
        per.append({'design':label,'stratum':r['name'],'n_train':r['n_train'],'n_eval':r['n_eval'],'train_corr':r['rho'],'K':r['K'],'Hf':f,'Hb':b,'dii':b-f,'start':r['start'],'split':r['split'],'end':r['end'],'instruments':' '.join(r['instruments'])})
    for ell in [4,1]:
        starts=[];off=0
        for r in rows:starts+=list(range(off,off+r['n_eval'],ell));off+=r['n_eval']
        rad=radius_starts([Af,Ab],np.array(starts))
        for rr in [0.,.01,.025,.05]:
            lo,hi=interval(Hf,Hb,rad,rr,rr)
            out.append({'design':label,'ell':ell,'r':rr,'n':n,'strata':len(rows),'Hf':Hf,'Hb':Hb,'dii':Hb-Hf,'radius':rad,'lower':lo,'upper':hi,'Kbar':Kbar,'rv':np.sqrt(max(lo,0)/Kbar),'rv_point':np.sqrt(max(Hb-Hf,0)/Kbar)})
    return out,per
if __name__=='__main__':
    S=load();res=[];per=[]
    for idx,label in [(0,'Equity index'),(1,'10-year bond yield')]:
        o,p=run(build(S,idx),label);res+=o;per+=p
    pd.DataFrame(res).to_csv(R/'pooled_application.csv',index=False);pd.DataFrame(per).to_csv(R/'pooled_strata.csv',index=False)
    (R/'pooled_metadata.json').write_text(json.dumps({'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(D.iterdir())},'B':1999,'seed':20260920},indent=2))
    print(pd.DataFrame(per)[['design','stratum','n_train','n_eval','train_corr','K','Hf','Hb','dii']].to_string(index=False))
    print(pd.DataFrame(res).query('r==0').drop(columns=['r']).to_string(index=False))
