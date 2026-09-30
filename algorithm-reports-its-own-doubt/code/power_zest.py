"""Web Appendix G power analysis: detecting the confidence interaction (delta_P=-.07) at varying N,
binary (5.4% base) vs continuous outcome. Calibrated to empirical s_i, c2=.195, sm2=.0208. HC0 t-tests."""
import numpy as np
from scipy.stats import norm as nd
from prep import load
d=load(); pl=d[d.plaus&d.ConfidenceScore.notna()&(d.s>0)]
S=pl.s.values; HI=(pl.ConfidenceScore>=80).values.astype(float)
c2,sm2,bP,bD,dP=0.195,0.0208,-.08,.07,-.07
rng=np.random.default_rng(3); REPS=300
def run(N,binary,null=False):
    rej=0
    for _ in range(REPS):
        i=rng.integers(0,len(S),N); s=S[i]; hi=HI[i]
        m=.02+np.sqrt(sm2)*rng.standard_normal(N); g=m+np.sqrt(c2)*s*rng.standard_normal(N)
        lam=sm2/(sm2+c2*s*s); mp=.02+lam*(g-.02); tau=np.sqrt(lam*c2*s*s)
        a=(0-mp)/tau; P=mp*(1-nd.cdf(a))+tau*nd.pdf(a); D=mp-P
        xb=.054+bP*np.maximum(m,0)+bD*np.minimum(m,0)+(0 if null else dP)*np.maximum(m,0)*hi
        y=(rng.random(N)<np.clip(xb,0,1)).astype(float) if binary else xb+.10*rng.standard_normal(N)
        X=np.column_stack([np.ones(N),P,D,P*hi,D*hi,hi])
        b,_,_,_=np.linalg.lstsq(X,y,rcond=None); e=y-X@b
        XtXi=np.linalg.inv(X.T@X); V=XtXi@(X.T*(e*e))@X@XtXi
        rej+=abs(b[3]/np.sqrt(V[3,3]))>1.96
    return rej/REPS
for N in [4000,10000,25000,43000]:
    print(N,"binary power",run(N,True),"size",run(N,True,null=True),"continuous power",run(N,False))
