"""Implied median absolute AVM error from the calibrated interval scale, for comparison
with Zillow's published Zestimate accuracy (1.9% on-market, 7.0% off-market)."""
import numpy as np
from prep import load
d=load(); pl=d[d.plaus&d.ConfidenceScore.notna()&(d.s>0)]; s=pl.s.values; c=np.sqrt(0.1949)
rng=np.random.default_rng(1)
e=np.abs(c*s[rng.integers(0,len(s),2_000_000)]*rng.standard_normal(2_000_000))
print("implied median abs AVM error: %.1f%%"%(100*(np.exp(np.median(e))-1)))
for name,mask in [("hi-conf",pl.ConfidenceScore>=80),("lo-conf",pl.ConfidenceScore<80)]:
    ss=s[mask.values]; ee=np.abs(c*ss[rng.integers(0,len(ss),500_000)]*rng.standard_normal(500_000))
    print(name,": %.1f%%"%(100*(np.exp(np.median(ee))-1)))
