from pathlib import Path
import csv,json
P=Path(__file__).resolve().parent.parent
s=json.loads((P/'full_study_results.json').read_text())['summary']
lookup={(r['design'],r['allocation'],r['n'],r['version']):r for r in s}
out=[]
for n in [125,216]:
 null=lookup['regular','quadratic',n,'feasible']['rates']
 for alt in ['weak','strong']:
  a=lookup[alt,'quadratic',n,'feasible']['rates'];fpw=null['wild']['rate'];fpi=null['intersection']['rate'];tw=a['wild']['rate'];ti=a['intersection']['rate']
  out.append({'n':n,'alternative':alt,'false_positive_reduction':fpw-fpi,'power_reduction':tw-ti,'critical_cost_ratio_at_equal_prior':(tw-ti)/(fpw-fpi),'wild_equal_prior_equal_cost_loss':.5*fpw+.5*(1-tw),'intersection_equal_prior_equal_cost_loss':.5*fpi+.5*(1-ti)})
rows=list(csv.DictReader((P/'full_study_raw.csv').open()))
feas=[r for r in rows if r['version']=='feasible'];disagree=sum((float(r['p_paired'])<=.05)!=(float(r['p_intersection'])<=.05) for r in feas)
(P/'decision_results.json').write_text(json.dumps({'analysis':'post-simulation illustrative loss analysis, not a prespecified or empirically calibrated managerial loss model','null':'equal positive dependence baseline','rows':out,'feasible_datasets':len(feas),'paired_intersection_decision_disagreements':disagree},indent=2))
print(json.dumps(out,indent=2));print('Paired/intersection disagreements:',disagree,'of',len(feas))
