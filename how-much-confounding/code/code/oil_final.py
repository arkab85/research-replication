import numpy as np, json, time
from pathlib import Path
P=Path(__file__).resolve().parent
from jointproc import analyse_multi
from localconst import Setup, K_env
from scipy.stats import skew, kurtosis
t0=time.time()
y=np.loadtxt(P/'oil_extended.txt'); T=len(y); p=24
def innov(end):
    X=np.column_stack([np.ones(T-p)]+[y[p-j:T-j] for j in range(1,p+1)]); Y=y[p:]
    tr=np.arange(p,T)<end; B=np.linalg.lstsq(X[tr],Y[tr],rcond=None)[0]; U=Y-X@B
    sd=U[tr].std(0,ddof=1); return U[tr]/sd, U[~tr]/sd
def kappa_b(a,sx,b,sy,N=6):
    S=Setup(a,sx,b,sy,nq=50,nf=40)
    return max(np.linalg.eigvalsh(S.local_matrix('resp',N))[-1],np.linalg.eigvalsh(S.local_matrix('reg',N))[-1])
names=["production","activity","price"]; pairs=[(0,2),(0,1)]
out={}
for lab,end in (("A",(2008-1974)*12),("B",(2000-1974)*12)):
    Utr,Uev=innov(end)
    res,q0,qJ=analyse_multi([(Utr[:,i],Utr[:,j]) for i,j in pairs],[(Uev[:,i],Uev[:,j]) for i,j in pairs],seed=7)
    res3,q03,qJ3=analyse_multi([(Utr[:,i],Utr[:,j]) for i,j in pairs],[(Uev[:,i],Uev[:,j]) for i,j in pairs],block=3,seed=7)
    print(f"design {lab}: m={len(Utr)} n={len(Uev)} q0={q0:.4f} qJ={qJ:.4f} | blocks of 3: q0={q03:.4f} qJ={qJ3:.4f}  skew/kurt prod eval {skew(Uev[:,0]):.2f}/{kurtosis(Uev[:,0]):.2f} price {skew(Uev[:,2]):.2f}/{kurtosis(Uev[:,2]):.2f}")
    rows=[]
    for (i,j),r,r3 in zip(pairs,res,res3):
        rho=float(np.corrcoef(Utr[:,i],Utr[:,j])[0,1]); Kb=kb=0
        for rX in (0.4,0.6,0.8):
            a=np.sqrt(rX); b=rho/a
            if b*b>0.95: continue
            Kb=max(Kb,K_env(a,np.sqrt(1-rX),b,np.sqrt(1-b*b))); kb=max(kb,kappa_b(a,np.sqrt(1-rX),b,np.sqrt(1-b*b)))
        hb,hf=np.sqrt(r['Hb']),np.sqrt(r['Hf'])
        ratio=(qJ/((hb-hf)/2))**2 if hb>hf else float('inf')
        row=dict(pair=f"{names[i]}->{names[j]}",rho=rho,Hf=r['Hf'],Hb=r['Hb'],D=r['D'],Lfix=r['fixed'][0],Ufix=r['fixed'][1],L=r['joint'][0],U=r['joint'][1],
                 L3=r3['joint'][0],Kb=Kb,kb=kb,upenv=np.sqrt(max(r['joint'][1],0)/Kb),uploc=np.sqrt(max(r['joint'][1],0)/kb),
                 upenvfix=np.sqrt(max(r['fixed'][1],0)/Kb),uplocfix=np.sqrt(max(r['fixed'][1],0)/kb),ptloc=np.sqrt(max(r['D'],0)/kb),ratio=ratio)
        rows.append(row)
        print("  {pair:22s} rho={rho:+.3f} 1e3Hf={a:.2f} 1e3Hb={b:.2f} 1e3D={c:+.2f} fixed[{Lfix:+.4f},{Ufix:+.4f}] joint[{L:+.4f},{U:+.4f}] L(blk3)={L3:+.4f} Kb={Kb:.1f} kb={kb:.4f} up env/loc fixed {upenvfix:.4f}/{uplocfix:.3f} joint {upenv:.4f}/{uploc:.3f} ptloc={ptloc:.3f} ratio={ratio:.3g}".format(
            a=1e3*row['Hf'],b=1e3*row['Hb'],c=1e3*row['D'],**row))
    out[lab]=dict(m=len(Utr),n=len(Uev),q0=q0,qJ=qJ,q03=q03,qJ3=qJ3,skew=float(skew(Uev[:,0])),kurt=float(kurtosis(Uev[:,0])),rows=rows)
json.dump(out,open(P/'oil_final.json','w'),indent=1,default=float)
print("done",round(time.time()-t0),"s")
