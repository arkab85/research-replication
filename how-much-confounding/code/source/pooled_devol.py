"""Exploratory (NOT pre-specified): pooled design after removing slow-moving volatility
regimes by trailing 20-event scale normalization within stratum (past events only)."""
from pooled_application import *
S=load();res=[]
for idx,label in [(0,'Equity index'),(1,'10-year bond yield')]:
    o,_=run(build(S,idx,devol=True),label);res+=o
pd.DataFrame(res).to_csv(R/'pooled_devol.csv',index=False)
print(pd.DataFrame(res).query('r==0 and ell==4')[['design','n','Hf','Hb','dii','radius','lower','upper','Kbar','rv']].to_string(index=False))
