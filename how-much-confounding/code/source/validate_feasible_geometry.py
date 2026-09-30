from pathlib import Path
import json,numpy as np
from core import gram,center,component
from feasible_inference import derivative_geometry
rng=np.random.default_rng(91037);n=37;e=rng.normal(size=n);z=rng.normal(size=n)
v=np.array([.3,-.2]);P=np.column_stack([z,z*z]);Lc=center(gram(z));G,B=derivative_geometry(e,z)
t=2e-4;en=e-t*(P@v);cross=np.exp(-(e[:,None]-en[None,:])**2/2)
h0=component(e,z)[0];hn=component(en,z)[0];cp=(cross*Lc).mean()
fd=(hn+h0-2*cp)/t**2;analytic=v@G@v
old=gram(e)
raw=((cross-cross.mean(axis=0))*Lc).mean(axis=1)-((old-old.mean(axis=0))*Lc).mean(axis=1)
bfd=(raw-raw.mean())/t
err=float(np.max(abs(bfd-B@v)))
assert abs(fd-analytic)<2e-6,(fd,analytic)
assert err<2e-6,err
out={'operator_derivative_norm_numeric':float(fd),'operator_derivative_norm_analytic':float(analytic),'cross_gram_derivative_max_error':err,'passed':True}
p=Path(__file__).resolve().parents[1];(p/'audit/feasible_geometry_validation.json').write_text(json.dumps(out,indent=2));print(out)
