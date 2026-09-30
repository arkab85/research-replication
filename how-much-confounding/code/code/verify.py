import numpy as np, time
from localconst import *
t0=time.time()
# 1. sharpness coefficient: backward, Y (regressor) loaded with psi_2
for nq,nf in ((60,48),):
    back=Setup(0.8,0.6,0.0,1.0,nq=nq,nf=nf)
    M=back.local_matrix('reg',N=3)
    print(f"[1] nq={nq} nf={nf}: M[0,0]={M[0,0]:.9f}  target a^4/54={0.8**4/54:.9f}")
# 2. exact H at the sharpness construction vs the paper's table (delta=0.1: Hb=7.438e-5, Hf=1.9e-7; D/d^2=0.00741847)
back=Setup(0.8,0.6,0.0,1.0); fwd=Setup(0.0,1.0,0.8,0.6)
p2=basis(2)[0]; z=lambda u: np.zeros_like(u)
for d in (0.005,0.1,0.2,0.4):
    Hb=back.exact_H(z,p2,d); Hf=fwd.exact_H(p2,z,d)
    print(f"[2] delta={d}: Hb={Hb:.8f} Hf={Hf:.8f} D/d2={(Hb-Hf)/d**2:.8f}")
# 3. first-order conditional mean with b != 0
S=Setup(0.7,0.7,0.5,0.8); F=basis(3)[0]; G=basis(2)[0]
zz=np.linspace(-2,2,5); d=1e-4
xp=np.linspace(-12,12,4801); wp=np.exp(-0.5*xp**2)*(xp[1]-xp[0])/np.sqrt(2*np.pi)
def mP(z,d):
    lik=np.exp(-0.5*((z[:,None]-S.c2*xp[None,:]-d*G(xp)[None,:])/S.s2)**2)*wp[None,:]
    return (lik*(S.c1*xp+d*F(xp))[None,:]).sum(1)/lik.sum(1)
fd=(mP(zz,d)-mP(zz,-d))/(2*d)
print("[3] m1 formula :", np.round(S.m1(F,G,zz),6)); print("[3] finite diff:", np.round(fd,6))
# 4. local coefficient vs exact H/delta^2 in a generic case (b != 0), mixed loading on response and regressor
S=Setup(0.7,0.7,0.5,0.8)
cF=np.array([0.3,0.8,0.2]); cG=np.array([0.6,-0.5,0.3])
Fm=shape_fun(cF); Gm=shape_fun(cG)
# local via delta_prime of the combined loading
Dp=S.delta_prime(Fm,Gm); loc=float(np.sum(S.WTS*np.abs(Dp)**2))
for d in (0.02,0.01,0.005):
    print(f"[4] delta={d}: exact H/d^2={S.exact_H(Fm,Gm,d)/d**2:.8f}  local={loc:.8f}")
print("time",round(time.time()-t0,1))
