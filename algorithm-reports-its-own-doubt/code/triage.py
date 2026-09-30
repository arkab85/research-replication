"""Ground-truth triage evaluation, calibrated to the data: with a review budget of the top K percent,
how many TRULY overpriced units (markup > log 1.10) does each ranking rule catch?
Rules: raw gap g; posterior probability pi = P(m > log1.10 | g, s)."""
import numpy as np, json
from scipy.stats import norm
from prep import load
d=load(); pl=d[d.plaus&d.ConfidenceScore.notna()&(d.s>0)]
S=pl.s.values; N=len(S); C2=0.195; SM2=0.0208; MU=0.02; TH=np.log(1.10)
rng=np.random.default_rng(7); R=200
res={k:{"raw":[],"post":[],"oracle":[]} for k in [5,10,20]}
share=[]
for r in range(R):
    idx=rng.integers(0,N,N); s=S[idx]
    m=MU+np.sqrt(SM2)*rng.standard_normal(N); e=np.sqrt(C2)*s*rng.standard_normal(N); g=m+e
    lam=SM2/(SM2+C2*s**2); mp=MU+lam*(g-MU); tau=np.sqrt(lam*C2*s**2)
    pi=1-norm.cdf((TH-mp)/tau)
    truth=m>TH; share.append(truth.mean())
    for K,st in res.items():
        k=int(N*K/100)
        for name,score in [("raw",g),("post",pi),("oracle",m)]:
            sel=np.argpartition(-score,k)[:k]
            st[name].append(truth[sel].mean())
out={"true_overpriced_share":float(np.mean(share))}
for K,st in res.items():
    out[f"top{K}"]={n:[float(np.mean(v)),float(np.std(v))] for n,v in st.items()}
    out[f"top{K}"]["lift_pct"]=round(100*(np.mean(st["post"])/np.mean(st["raw"])-1),1)
json.dump(out,open("triage.json","w"),indent=1); print(json.dumps(out,indent=1))
