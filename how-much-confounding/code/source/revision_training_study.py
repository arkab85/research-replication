"""Declared follow-up: training-size mechanism behind conservative certificates.
Run after the baseline revision study; report all four added cells.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
from revision_study import cell
P=Path(__file__).resolve().parents[1];R=P/'results'
if __name__=='__main__':
    tasks=[('quadratic',p,600,2,200,399,m) for p in [0.,.6] for m in [10,100]]
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(cell,t):t for t in tasks}
        for f in as_completed(fs):
            rows.extend(f.result());print('Finished',fs[f],flush=True)
    d=pd.DataFrame(rows).sort_values(['phi','n_train','rep'])
    d.to_csv(R/'revision_training_raw.csv',index=False)
    s=d.groupby(['phi','n','n_train']).agg(reps=('rep','size'),
        cover_zero=('cover_zero','mean'),cover_aware=('cover_aware','mean'),
        median_r_f=('r_f','median'),median_r_b=('r_b','median'),
        median_rv_zero=('rv_zero','median'),median_rv_aware=('rv_aware','median'),
        power_zero=('exclude_001_zero','mean'),power_aware=('exclude_001_aware','mean')).reset_index()
    s.to_csv(R/'revision_training_summary.csv',index=False)
    print(s.to_string(index=False),flush=True)
