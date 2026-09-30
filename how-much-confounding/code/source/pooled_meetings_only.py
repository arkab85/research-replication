"""Exploratory (NOT pre-specified; added after the pre-specified results were seen):
pooled design restricted to policy-meeting strata, excluding EA-EMPD speeches and P events."""
from pooled_application import *
S=load();S={k:v for k,v in S.items() if not k.endswith('speeches')}
res=[]
for idx,label in [(0,'Equity index'),(1,'10-year bond yield')]:
    o,_=run(build(S,idx),label);res+=o
pd.DataFrame(res).to_csv(R/'pooled_meetings_only.csv',index=False)
print(pd.DataFrame(res).query('r==0 and ell==4')[['design','n','Hf','Hb','dii','radius','lower','upper','Kbar','rv','rv_point']].to_string(index=False))
