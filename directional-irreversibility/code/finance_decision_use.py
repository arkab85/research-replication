from pathlib import Path
import numpy as np,pandas as pd,json,hashlib
from scipy.optimize import nnls
P=Path(__file__).resolve().parent.parent/'finance';f=P/'analysis_panel_local.csv';assert hashlib.sha256(f.read_bytes()).hexdigest()==(P/'analysis_panel_sha256.txt').read_text().strip()
base=pd.read_csv(f,index_col='month');base.index=pd.PeriodIndex(base.index,freq='M');tr=pd.period_range('1991-01','2007-12',freq='M');te=pd.period_range('2009-01','2018-12',freq='M');m=len(tr);n=len(te);B=1999;ell=12
rng=np.random.default_rng(20260919);starts=rng.integers(0,n,size=(B,n//ell));idx=((starts[:,:,None]+np.arange(ell))%n).reshape(B,n);rows=[];detail=[]
for asset in ['FX','Treasury','Credit','VIX']:
 for h in [1,3,6]:
  a=pd.DataFrame({'x':base.MP,'y':base[asset].shift(-h),'c1':base.MP.shift(1),'c2':base[asset].shift(1)}).reindex(tr.append(te)).to_numpy();assert np.isfinite(a).all()
  z=(a-a[:m].mean(0))/a[:m].std(0);x=z[:,0];y=z[:,1];c=z[:,2:]
  M=np.column_stack([np.ones(len(y)),c,x]);beta=np.linalg.lstsq(M[:m],y[:m],rcond=None)[0];u=y-M@beta
  V0=np.column_stack([np.ones(len(y)),c*c]);V1=np.column_stack([V0,x*x]);b0,_=nnls(V0[:m],u[:m]**2);b1,_=nnls(V1[:m],u[:m]**2)
  floor=.01*np.mean(u[:m]**2);r0=V0[m:]@b0;r1=V1[m:]@b1;v0=np.maximum(r0,floor);v1=np.maximum(r1,floor);actual=u[m:]**2
  l0=np.log(v0)+actual/v0;l1=np.log(v1)+actual/v1;gain=l0-l1;est=float(gain.mean());boot=gain[idx].mean(1);ci=np.quantile(boot,[.025,.975]);p=(1+np.count_nonzero(boot-est>=est-1e-12))/(B+1)
  rows.append(dict(asset=asset,h=h,score_gain=est,gain_ci_low=float(ci[0]),gain_ci_high=float(ci[1]),p_positive=float(p),baseline_score=float(l0.mean()),augmented_score=float(l1.mean()),floor=float(floor),baseline_floor_count=int((r0<floor).sum()),augmented_floor_count=int((r1<floor).sum()),trained_shock_square_coefficient=float(b1[-1])))
  for j,date in enumerate(te):detail.append(dict(asset=asset,h=h,month=str(date),baseline_score=float(l0[j]),augmented_score=float(l1[j]),score_gain=float(gain[j])))
order=sorted(range(len(rows)),key=lambda i:rows[i]['p_positive']);prev=0
for rank,i in enumerate(order):prev=max(prev,(12-rank)*rows[i]['p_positive']);rows[i]['p_holm']=min(1.,prev)
out=dict(seed=20260919,training_n=m,evaluation_n=n,draws=B,block_length=ell,protocol_sha256=hashlib.sha256((P.parent/'decision_use_protocol.md').read_bytes()).hexdigest(),status='Secondary post-results audit. Squared errors of the frozen fitted affine forecast, not structural variance. Conditional evaluation uncertainty under working stability/dependence assumptions; no real-time or investment claim.',comparisons=rows)
(P/'decision_use_results.json').write_text(json.dumps(out,indent=2));pd.DataFrame(rows).to_csv(P/'decision_use_results.csv',index=False);pd.DataFrame(detail).to_csv(P/'decision_use_scores.csv',index=False)
print(pd.DataFrame(rows)[['asset','h','score_gain','gain_ci_low','gain_ci_high','p_positive','p_holm']].to_string(index=False))
