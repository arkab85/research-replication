"""Post-diagnostic trend sensitivity; labeled as such, never substituted for primary estimates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from estimate import fit_cluster, infer, design, ROOT, OUT

d = pd.read_csv(ROOT / "data/estimation_sample_with_package_flags.csv.gz", parse_dates=["date"])
rows = []
for sector in ["Retail", "Office", "Multifamily", "Industrial"]:
    s = d[(d.sector == sector) & ~d.package_flag & (d.year != 2020)].reset_index(drop=True)
    f = fit_cluster(s.log_ppsf, design(s,trends=True), s.neigh, s.neigh)
    rows.append({"sector":sector,"spec":"Exposure-specific linear trends (diagnostic)", "n":f["n"], "clusters":f["groups"], "parcels":s.bbl.nunique(),**infer(f,{"R_post":.1,"O_post":-.1})})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT/"trend_sensitivity.csv",index=False)
# Familywise adjustment applies to the four prespecified sectoral evenness tests.
e = pd.read_csv(OUT / "estimates.csv")
m = e[e.spec.eq("Three-use evenness")].sort_values("p").copy()
m["holm_p"] = np.minimum(1,np.maximum.accumulate(m.p.to_numpy()*np.arange(len(m),0,-1)))
m.to_csv(OUT/"evenness_multiplicity.csv",index=False)
# Distribution of exposures among properties actually traded, without causal interpretation.
s = d[~d.package_flag].copy()
s["period"] = np.where(s.year<=2019,"2017–2019",np.where(s.year>=2021,"2021–2025","2020"))
s.groupby(["sector","period"]).agg(n=("bbl","size"),parcels=("bbl","nunique"),median_ppsf=("ppsf","median"),mean_res=("residential_share_500","mean"),mean_office=("office_share_500","mean"),mean_retail=("retail_share_500","mean"),median_area=("area","median")).reset_index().to_csv(OUT/"sample_description.csv",index=False)
