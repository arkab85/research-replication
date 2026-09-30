import numpy as np, json
from svarcore import demand_elasticity
o=np.load('oil_orig.npz'); th=o['thetas']; nm=o['norms']; cs=o['cs']; L=o['L']; n=int(o['n'])
D=np.concatenate([np.load(f'oil_boot_{c}.npz')['diff'] for c in range(3)]); CS=np.concatenate([np.load(f'oil_boot_{c}.npz')['cs'] for c in range(3)])
B=len(D); q=float(np.quantile(D.max(1),0.95)); el=np.array([demand_elasticity(L,t) for t in th])
exc=np.maximum(nm-q,0)
print(f"bootstrap reps {B}; first-stage-aware uniform 95% band q={q:.4f} (fixed-first-stage multiplier band was 0.0265)")
print(f"||C|| range {nm.min():.4f}-{nm.max():.4f}; recursive rotation {nm[0]:.4f}")
rows=[]
for rho in (0.0,0.05,0.10,0.15,0.20):
    a=exc<=rho**2; e=el[a]; e=e[np.isfinite(e)]
    rows.append((rho,float(a.mean()),float(e.min()) if len(e) else np.nan,float(e.max()) if len(e) else np.nan))
    print(f"rho={rho:.2f}: rotations admitted {a.mean():.2f}; elasticity [{rows[-1][2]:.3f}, {rows[-1][3]:.3f}]")
bd={}
for c in (0.05,0.10):
    m=np.isfinite(el)&(el>=c); bd[c]=float(np.sqrt(exc[m].min())) if m.any() else np.nan
    print(f"lower confidence bound on breakdown budget for 'elasticity < {c}': {bd[c]:.3f}")
print("rho that admits every rotation:",float(np.sqrt(exc.max())))
# co-skewness set (robust to latent common-state confounding)
wald=[]
for i in range(len(th)):
    V=np.cov(CS[:,i,:].T); m=cs[i]; wald.append(float(m@np.linalg.solve(V,m)))
wald=np.array(wald); a=wald<=5.991; e=el[a]; e=e[np.isfinite(e)]
print(f"co-skewness set: rotations admitted {a.mean():.2f}; elasticity [{e.min() if len(e) else np.nan:.3f}, {e.max() if len(e) else np.nan:.3f}]; recursive admitted: {bool(a[0])}; min Wald {wald.min():.2f} at {np.degrees(th[wald.argmin()]):.0f} deg")
print("co-skewness admitted angles (deg):",np.degrees(th[a]).astype(int).tolist())
print("elasticity by angle (deg:el) for admitted:",[(int(np.degrees(t)),round(float(x),3)) for t,x in zip(th[a],el[a])])
json.dump(dict(B=B,q=q,rows=rows,breakdown={str(k):v for k,v in bd.items()},rho_all=float(np.sqrt(exc.max())),
   cs_share=float(a.mean()),cs_el=[float(e.min()),float(e.max())] if len(e) else None,cs_recursive=bool(a[0]),
   norms=nm.tolist(),wald=wald.tolist(),el=el.tolist(),theta_deg=np.degrees(th).tolist()),open('oil_svar_results.json','w'))
