"""journal revision: exact population calculations and learned-regression Monte Carlo.

Added after the first review. No design is described as preregistered.
The error-aware simulation uses a model-known RMS bound on the L1 regression
error. It is a calibration experiment, not a data-only error estimator.
"""
from pathlib import Path
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.special import roots_hermitenorm
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits
from core import component, radius, interval, benchmark, envelope_constant

P = Path(__file__).resolve().parents[1]
R = P / 'results'

def population(a, delta, sigma=1., nodes=120):
    """Exact characteristic functions integrated by Gaussian quadrature.
    X=a U+sqrt(1-a^2) xi, Y=delta*(U^2-1)/sqrt(2)+sigma e.
    Unit kernels, Var(X)=1, population conditional-mean residuals.
    a=1 is the quadratic transmission alternative.
    """
    z, w = roots_hermitenorm(nodes)
    w = w / np.sqrt(2*np.pi)
    t, s = z[:, None], z[None, :]
    den = 1 - 1j*np.sqrt(2)*s*delta
    fy = np.exp(-1j*s*delta/np.sqrt(2)-.5*sigma*sigma*s*s)/np.sqrt(den)
    joint = fy*np.exp(-.5*t*t*((1-a*a)+a*a/den))
    diff = joint - np.exp(-.5*t*t)*fy
    hb = float(w @ (abs(diff)**2) @ w)
    if a == 1. or delta == 0.:
        hf = 0.
    else:
        # Symmetric square root avoids branch ambiguities for quadratic forms.
        cov = np.array([[1., a], [a, 1.]])
        ev, evec = np.linalg.eigh(cov)
        root = (evec*np.sqrt(ev))@evec.T
        vals, vecs = np.linalg.eigh(root@np.diag([1., -a*a])@root*delta/np.sqrt(2))
        d = vecs.T@root[:, 1]
        ds = 1-2j*s[..., None]*vals
        base = np.exp(-1j*s*delta*(1-a*a)/np.sqrt(2)-.5*sigma*sigma*s*s)
        base = base / np.prod(np.sqrt(ds), axis=-1)
        jointf = base*np.exp(-.5*t*t*np.sum(d*d/ds, axis=-1))
        difff = jointf-base*np.exp(-.5*t*t)
        hf = float(w @ (abs(difff)**2) @ w)
    return hf, hb, hb-hf

def sharpness_coefficient(a=.8, sigma=1., sx=1., sy=1.):
    return 1.5*a**4/sx**4/sy**2/(1+2/sx**2)**2.5/(1+2*sigma**2/sy**2)**1.5

def ar1(n, phi, rng):
    innovations = rng.normal(size=n)*np.sqrt(1-phi*phi)
    innovations[0] = rng.normal()
    return lfilter([1.], [1., -phi], innovations)

def draw(n, phi, model, rng):
    u = ar1(n, phi, rng)
    if model == 'confounding':
        a, lam, sigma = .8, .3, 1.
        x = a*u+np.sqrt(1-a*a)*rng.normal(size=n)
        y = lam*(u*u-1)/np.sqrt(2)+sigma*rng.normal(size=n)
    elif model == 'quadratic':
        a, lam, sigma = 1., 1., .5
        x = u
        y = lam*(u*u-1)/np.sqrt(2)+sigma*rng.normal(size=n)
    else:
        a, lam, sigma = 1., .8, .5
        x = u
        y = lam*x+sigma*rng.normal(size=n)
    return x, y

def truth(model):
    mx = np.array([[1.,0,1.],[0,1.,0],[1.,0,3.]])
    if model == 'linear':
        vy = .8**2+.5**2
        my = np.array([[1.,0,vy],[0,vy,0],[vy,0,3*vy*vy]])
        return np.array([0.,.8,0.]),np.array([0.,.8/vy,0.]),mx,my,0.
    a, lam, sigma = (.8,.3,1.) if model == 'confounding' else (1.,1.,.5)
    vy = lam*lam+sigma*sigma
    m3 = 2*np.sqrt(2)*lam**3
    m4 = 15*lam**4+6*lam*lam*sigma*sigma+3*sigma**4
    my = np.array([[1.,0,vy],[0,vy,m3],[vy,m3,m4]])
    cf = lam*a*a/np.sqrt(2)
    return np.array([-cf,0.,cf]),np.zeros(3),mx,my,population(a,lam,sigma,180)[2]

def features(x):
    return np.column_stack([np.ones(len(x)),x,x*x])

def cell(task):
    model, phi, n, degree, reps, boot = task[:6]
    train_multiple = task[6] if len(task)>6 else 1
    n_train = n*train_multiple
    threadpool_limits(limits=1)
    model_id={'confounding':1,'quadratic':2,'linear':3}[model]
    seed0 = 20260922+100000*model_id+10000*int(phi*10)+10*n+degree+10000000*(train_multiple-1)
    rng = np.random.default_rng(seed0)
    bf0,bb0,mx,my,dtrue=truth(model)
    K = envelope_constant(benchmark(phi=.6,rx=.64,ry=0.,p=0,h=0,k=0))
    ell = max(1,int(round(n**(1/3))))
    records=[]
    for rep in range(reps):
        xt,yt=draw(n_train,phi,model,rng)
        x,y=draw(n,phi,model,rng) # independent training and evaluation paths
        bf=np.zeros(3);bb=np.zeros(3)
        bf[:degree+1]=np.linalg.lstsq(features(xt)[:,:degree+1],yt,rcond=None)[0]
        bb[:degree+1]=np.linalg.lstsq(features(yt)[:,:degree+1],xt,rcond=None)[0]
        ef=y-features(x)@bf;eb=x-features(y)@bb
        hf,af,*_=component(ef,x)
        hb,ab,*_=component(eb,y)
        qr=radius([af,ab],ell=ell,B=boot,seed=seed0+rep+1)
        rf=float(np.sqrt(max(0.,(bf-bf0)@mx@(bf-bf0))))
        rb=float(np.sqrt(max(0.,(bb-bb0)@my@(bb-bb0))))
        l0,u0=interval(hf,hb,qr)
        l1,u1=interval(hf,hb,qr,rf,rb)
        records.append(dict(model=model,phi=phi,n=n,n_train=n_train,degree=degree,rep=rep,
            seed=seed0,block=ell,bootstrap=boot,D_true=dtrue,D_hat=hb-hf,
            Hf=hf,Hb=hb,radius=qr,r_f=rf,r_b=rb,L_zero=l0,U_zero=u0,
            L_aware=l1,U_aware=u1,cover_zero=l0<=dtrue<=u0,
            cover_aware=l1<=dtrue<=u1,rv_zero=np.sqrt(max(l0,0)/K),
            rv_aware=np.sqrt(max(l1,0)/K),K=K,
            exclude_001_zero=l0>K*.01**2,exclude_001_aware=l1>K*.01**2,
            false_exclude_zero=(l0>K*.3**2) if model=='confounding' else np.nan,
            false_exclude_aware=(l1>K*.3**2) if model=='confounding' else np.nan))
    return records

def population_audit():
    rows=[]
    c=sharpness_coefficient()
    for delta in [.005,.01,.02,.05,.1,.2,.4]:
        h1=population(.8,delta,1.,100)
        h2=population(.8,delta,1.,180)
        rows.append(dict(delta=delta,Hf=h2[0],Hb=h2[1],D=h2[2],
            D_over_delta2=h2[2]/delta**2,limit=c,
            quadrature_difference=max(abs(np.array(h1)-h2))))
    pd.DataFrame(rows).to_csv(R/'revision_sharpness.csv',index=False)
    assert rows[0]['D']>0
    assert abs(rows[0]['D_over_delta2']/c-1)<.001
    assert max(r['quadrature_difference'] for r in rows)<1e-9
    print('Sharpness coefficient',c,'small-delta ratio',rows[0]['D_over_delta2'],flush=True)

def summarize(df):
    keys=['model','phi','n','degree']
    out=df.groupby(keys,sort=False).agg(reps=('rep','size'),D_true=('D_true','first'),
       cover_zero=('cover_zero','mean'),cover_aware=('cover_aware','mean'),
       median_r_f=('r_f','median'),median_r_b=('r_b','median'),
       median_rv_zero=('rv_zero','median'),median_rv_aware=('rv_aware','median'),
       power_001_zero=('exclude_001_zero','mean'),power_001_aware=('exclude_001_aware','mean'),
       false_zero=('false_exclude_zero','mean'),false_aware=('false_exclude_aware','mean')).reset_index()
    for col in ['cover_zero','cover_aware','power_001_zero','power_001_aware']:
        out[col+'_mcse']=np.sqrt(out[col]*(1-out[col])/out.reps)
    out.to_csv(R/'revision_mc_summary.csv',index=False)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=200)
    ap.add_argument('--bootstrap',type=int,default=399);ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
    R.mkdir(exist_ok=True);population_audit()
    tasks=[(m,p,n,d,a.reps,a.bootstrap) for m in ['confounding','quadratic','linear']
           for p in [0.,.6] for n in [250,600] for d in [1,2]]
    if a.pilot:tasks=tasks[:1]
    all_rows=[];t0=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs={pool.submit(cell,t):t for t in tasks}
        for future in as_completed(futs):
            rows=future.result();all_rows.extend(rows)
            print('Finished',futs[future][:4],'elapsed',round(time.time()-t0,1),flush=True)
    df=pd.DataFrame(all_rows).sort_values(['model','phi','n','degree','rep'])
    suffix='_pilot' if a.pilot else ''
    df.to_csv(R/f'revision_mc_raw{suffix}.csv',index=False)
    if not a.pilot:
        out=summarize(df);print(out.to_string(index=False),flush=True)
        (R/'revision_mc_metadata.json').write_text(json.dumps(dict(
            seed=20260922,replications_per_cell=a.reps,cells=len(tasks),
            bootstrap_draws=a.bootstrap,training='independent stationary path of length n',
            error_budget='exact model-known RMS upper bound on conditional L1 error',
            population='characteristic functions and 180-point Gauss-Hermite quadrature',
            design_status='post-review revision; not preregistered',seconds=time.time()-t0),indent=2))

if __name__=='__main__': main()
