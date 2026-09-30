"""Recompute a monetary-shock sensitivity illustration from public data.
This is retrospective and includes every preselected asset/horizon cell.
"""
from pathlib import Path
import itertools,json,hashlib,shutil
import numpy as np,pandas as pd
from core import *
P=Path(__file__).resolve().parents[1];D=P/'data';R=P/'results'
panel_path=D/'analysis_panel.csv'
panel_bytes=panel_path.read_bytes()
expected=(D/'panel_sha256.txt').read_text().strip()
assert hashlib.sha256(panel_bytes).hexdigest()==expected, 'Frozen analysis panel changed; do not combine new data with frozen results.'
base=pd.read_csv(panel_path,index_col='month',float_precision='round_trip');base.index=pd.PeriodIndex(base.index,freq='M')
panel=panel_bytes.decode()
tr=pd.period_range('1991-01','2007-12',freq='M');te=pd.period_range('2009-01','2018-12',freq='M')
def poly(z):
 return np.column_stack([np.ones(len(z)),z]+[z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(z.shape[1]),2)])
comps={};meta=[];As=[]
for asset in ['FX','Treasury','Credit','VIX']:
 for h in [1,3,6]:
  df=pd.DataFrame({'x':base.MP,'y':base[asset].shift(-h),'c1':base.MP.shift(1),'c2':base[asset].shift(1)}).reindex(tr.append(te))
  ar=df.to_numpy();assert np.isfinite(ar).all();mu=ar[:len(tr)].mean(0);sd=ar[:len(tr)].std(0);z=(ar-mu)/sd
  zf=np.column_stack([z[:,0],z[:,2:]]);zb=np.column_stack([z[:,1],z[:,2:]])
  F=poly(zf);B=poly(zb);rf=z[len(tr):,1]-F[len(tr):]@np.linalg.lstsq(F[:len(tr)],z[:len(tr),1],rcond=None)[0]
  rb=z[len(tr):,0]-B[len(tr):]@np.linalg.lstsq(B[:len(tr)],z[:len(tr),0],rcond=None)[0]
  cf,cb=pair((rf,zf[len(tr):],rb,zb[len(tr):]));As.extend([cf[1],cb[1]])
  comps[(asset,h)]=(cf,cb);meta.append({'asset':asset,'h':h,'train_n':len(tr),'test_n':len(te),'x_mean':mu[0],'x_sd':sd[0],'y_mean':mu[1],'y_sd':sd[1],'xlag_sd':sd[2],'ylag_sd':sd[3]})
out=[]
for ell in [6,12]:
 rad=radius(As,ell=ell,B=1999,seed=20260913)
 for (asset,h),(f,b) in comps.items():
  for r in [0.,.01,.025,.05]:
   lo,hi=interval(f[0],b[0],rad,r,r)
   q=benchmark(.8,.3,.7,p=1,h=h,k=1,both=True)
   m=next(v for v in meta if v['asset']==asset and v['h']==h)
   # Reference X and Y units are their response/current scales; histories have
   # their own fixed training scales, represented as bandwidth ratios.
   q['kernel_scales']=np.array([1.,1.,m['xlag_sd']/m['x_sd'],m['ylag_sd']/m['y_sd']])
   K=envelope_constant(q)
   out.append({'asset':asset,'h':h,'ell':ell,'r':r,'forward_hsic':f[0],'backward_hsic':b[0],'dii':b[0]-f[0],'radius':rad,'lower':lo,'upper':hi,'K':K,'gamma_point':np.sqrt(max(0,b[0]-f[0])/K),'gamma_lower':np.sqrt(max(0,lo)/K)})
pd.DataFrame(out).to_csv(R/'application.csv',index=False)
(R/'application_scaling.json').write_text(json.dumps(meta,indent=2))
(R/'application_metadata.json').write_text(json.dumps({'panel_sha256':hashlib.sha256(panel.encode()).hexdigest(),'simultaneous_components':24,'B':1999,'seed':20260913,'comment':'Retrospective extension of existing related-paper design; all cells retained. Conditional fitted-target inference; oracle transfer requires stated L1 approximation tolerance. Gaussian benchmark is a scenario, not estimated or validated.'},indent=2))
print(pd.DataFrame(out).query('ell==12 and r==0')[['asset','h','dii','lower','upper','K','gamma_lower']].to_string(index=False))
