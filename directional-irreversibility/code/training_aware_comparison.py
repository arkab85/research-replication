from pathlib import Path
import pandas as pd,numpy as np,json
P=Path(__file__).resolve().parent.parent;d=pd.read_csv(P/'training_aware_study_raw.csv');out=[]
for (rho,design),g in d.groupby(['rho','design']):
 p=g.pivot(index='rep',columns='method',values='reject').astype(int);v=p.training_aware-p.original_nbb;se=v.std(ddof=1)/np.sqrt(len(v))
 out.append({'rho':float(rho),'design':design,'difference':float(v.mean()),'mc95':[float(v.mean()-1.96*se),float(v.mean()+1.96*se)]})
(P/'training_aware_paired_comparisons.json').write_text(json.dumps(out,indent=2))
