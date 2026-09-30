# Exact check of the common-volatility envelope ||C_12|| <= rho_1 rho_2 / (s_1 s_2) and its sharp rate.
# eps_i = sigma(V) eta_i, eta_i iid unit variance (Gaussian or Laplace), sigma two-point, E sigma^2 = 1.
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
x,w=hermegauss(120); w=w/np.sqrt(2*np.pi)          # spectral measure N(0,1) for unit bandwidth
T,S=np.meshgrid(x,x,indexing='ij'); WW=np.outer(w,w)
def cf(eta, t):                                      # cf of unit-variance eta at t
    return np.exp(-t**2/2) if eta=='gauss' else 1.0/(1.0+t**2/2)
def normC(delta, eta):
    sig=np.array([1+delta,1-delta])/np.sqrt(1+delta**2); p=np.array([.5,.5])
    joint=sum(pk*cf(eta,sk*T)*cf(eta,sk*S) for pk,sk in zip(p,sig))
    m1=sum(pk*cf(eta,sk*T) for pk,sk in zip(p,sig)); m2=sum(pk*cf(eta,sk*S) for pk,sk in zip(p,sig))
    hs=np.sum(WW*np.abs(joint-m1*m2)**2); rho2=float(np.sum(p*(sig-np.sum(p*sig))**2))
    return np.sqrt(hs), rho2
print("eta     delta   ||C||        rho^2 (bound)  ratio ||C||/rho^2")
for eta in ('gauss','laplace'):
    for d in (0.4,0.2,0.1,0.05,0.02,0.01):
        c,r2=normC(d,eta); print(f"{eta:7s} {d:5.2f}  {c:.3e}   {r2:.3e}      {c/r2:.4f}")
print("Gaussian local constant 3^(-3/2) =",3**-1.5)
