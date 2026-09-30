import numpy as np
from localconst import directions, shape_fun
# random-direction check of the local constant at the simulation reference
a,sx,b,sy=0.8,0.6,0.0,1.0; kb=0.0083967; back,fwd=directions(a,sx,b,sy); rng=np.random.default_rng(3)
mx={0.2:0,0.5:0}
for k in range(24):
    wx=rng.uniform(); cX=rng.standard_normal(5); cX/=np.linalg.norm(cX); cY=rng.standard_normal(5); cY/=np.linalg.norm(cY)
    fX=shape_fun(cX); fY=shape_fun(cY)
    FX=lambda u,w=wx,f=fX: w*f(u); FY=lambda u,w=wx,f=fY: (1-w)*f(u)
    for g in (0.2,0.5):
        D=back.exact_H(FX,FY,g)-fwd.exact_H(FY,FX,g); mx[g]=max(mx[g],D/g**2)
print("random directions (24, degree<=6): max D/g^2 at g=0.2:",round(mx[0.2],6)," at g=0.5:",round(mx[0.5],6)," kappa_b:",kb)
