import numpy as np, json
from localconst import *
a,sx,b,sy=0.8,0.6,0.0,1.0
res={}
for N in (4,6,8,10,12):
    k=kappa(a,sx,b,sy,N=N)
    print(f"N={N:2d}  kb_X={k['X']['kb']:.3e}  kb_Y={k['Y']['kb']:.7f}  kD_Y={k['Y']['kD']:.7f}  Hf@shape={k['Y']['Hf_at_shape']:.2e}")
k=kappa(a,sx,b,sy,N=10)
c=k['Y']['shape']; c=c*np.sign(c[np.argmax(abs(c))])
print("least favourable Y-shape coefficients (He_2..):", np.round(c,4))
Kb=K_env(a,sx,b,sy); kb=k['kappa_b']
print(f"K_b={Kb:.4f} (256/9={256/9:.4f})  kappa_b={kb:.6f}  ratio={Kb/kb:.1f}  sqrt ratio={np.sqrt(Kb/kb):.2f}")
# exact contrast along least-favourable direction
back,fwd=directions(a,sx,b,sy); Gs=shape_fun(c); z=lambda u: np.zeros_like(u)
rows=[]
for g in (0.01,0.05,0.1,0.2,0.3,0.4):
    Hb=back.exact_H(z,Gs,g); Hf=fwd.exact_H(Gs,z,g)
    rows.append((g,Hb,Hf,(Hb-Hf)/g**2)); print(f"gamma={g:.2f}  Hb={Hb:.3e}  Hf={Hf:.3e}  D/g^2={(Hb-Hf)/g**2:.6f}  Hb/g^2={Hb/g**2:.6f}")
D_tr=0.0164277150
print("population RV quadratic transmission: envelope",np.sqrt(D_tr/Kb)," local",np.sqrt(D_tr/kb))
for med in (0.0115,0.0170,0.0084,0.0150):
    print("median RV env",med,"-> local",med*np.sqrt(Kb/kb))
json.dump(dict(Kb=Kb,kb=kb,shape=list(c),rows=rows,kbX=k['X']['kb'],kDY=k['Y']['kD'],HfY=k['Y']['Hf_at_shape']),open('simref.json','w'))
