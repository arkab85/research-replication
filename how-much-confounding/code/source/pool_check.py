"""Naive versus stratified pooling under independence with stratum-specific scales."""
from pathlib import Path
import json,numpy as np
from core import component,strat_component
P=Path(__file__).resolve().parents[1];rng=np.random.default_rng(0);n=300
rng.normal(size=n);rng.normal(size=n)  # same draws as the documented check
st=np.repeat([0,1],n//2);scale=np.where(st==0,.3,3.)
x=rng.normal(size=n)*scale;y=rng.normal(size=n)*scale
out={'naive':component(y,x[:,None])[0],'stratified':strat_component(y,x[:,None],st)[0]}
(P/'results/pool_check.json').write_text(json.dumps(out,indent=2));print(out)
