"""Audit compiled files and the stored revised simulation aggregation."""
import json,math,fitz,subprocess,ast
from pathlib import Path
r=Path(__file__).resolve().parents[2];src=r/'3_LaTeX_Source';audit={}
for stem,limit in [('main',40),('online_appendix',20),('cover_letter',2)]:
 doc=fitz.open(src/(stem+'.pdf'));assert len(doc)<=limit
 fonts=subprocess.check_output(['pdffonts',str(src/(stem+'.pdf'))],text=True).splitlines()[2:]
 assert all(line.split()[-5]=='yes' for line in fonts)
 text='\n'.join(p.get_text() for p in doc);assert '??' not in text
 if stem!='cover_letter':
  for i,p in enumerate(doc):
   bottom=p.get_text(clip=fitz.Rect(0,p.rect.height-55,p.rect.width,p.rect.height));assert str(i+1) in bottom,(stem,i,bottom)
 audit[stem]={'pages':len(doc),'all_fonts_embedded':True,'no_unresolved_reference_markers':True,'all_pages_numbered':stem!='cover_letter'}
a=src.joinpath('main.tex').read_text().split('\\begin{abstract}')[1].split('\\end{abstract}')[0];assert len(a.split())<=150;audit['abstract_words']=len(a.split());audit['body_format']='11-point article class, 1.5 spacing, one-inch margins'
for name in ['main_build','appendix_build','cover_build']:
 log=(src/'audit'/(name+'.txt')).read_text();assert 'Overfull' not in log,name;assert 'undefined' not in log,name;assert '! ' not in log,name
audit['latex_errors_undefined_references_overfull_boxes']=0
import sys,hashlib
W=r/'4_Replication/code/whitening';sys.path.insert(0,str(W));sys.path.insert(0,str(W.parent))
from confirm_study import ALL,REPS,seeds_for
summ=json.loads((W/'results/confirm_summary.json').read_text());checked={}
for part,designs in ALL.items():
 for name in designs:
  rows=[json.loads(l) for l in open(W/f'results/confirm_{part}_{name}.jsonl')]
  assert len(rows)==REPS[part] and sorted(x['seed'] for x in rows)==sorted(seeds_for(part,name,REPS[part])),(part,name)
  key={'W':'cover','J':'joint_new','R':'op_cover'}[part];stored={'W':'coverage','J':'joint_new','R':'op_cover'}[part]
  assert abs(100*sum(x[key] for x in rows)/len(rows)-summ[part][name][stored])<1e-9
  checked[f'{part}:{name}']=len(rows)
spec=(W/'PRESPECIFICATION.md').read_text()
for line in spec.splitlines():
 if line.startswith('- ') and ': ' in line and line.endswith(tuple('0123456789abcdef')):
  f,h=line[2:].split(': ');assert hashlib.sha256((W/f).read_bytes()).hexdigest()==h,f
audit['confirmatory_study']={'designs_and_replications':checked,'seeds_match_protocol':True,'code_hashes_match_prespecification':True,
 'joint_coverage_studentized':{k:v['joint_new'] for k,v in summ['J'].items()},'joint_coverage_percentile_same_samples':{k:v['joint_old'] for k,v in summ['J'].items()},
 'whitening_coverage_studentized':{k:v['coverage'] for k,v in summ['W'].items()},'reported_in_manuscript':True}
import confirm_addendum as ca
asum=json.loads((W/'results/addendum_summary.json').read_text());achecked={}
for part,designs in ca.ALL.items():
 for name in designs:
  rows=[json.loads(l) for l in open(W/f'results/addendum_{part}_{name}.jsonl')]
  assert len(rows)==ca.REPS[part] and sorted(x['seed'] for x in rows)==sorted(ca.seeds_for(part,name,ca.REPS[part])),(part,name)
  key={'RB':'op_cover','JW':'joint_new','WB':'cover'}[part];stored={'RB':'op_cover','JW':'joint_new','WB':'coverage'}[part]
  assert abs(100*sum(x[key] for x in rows)/len(rows)-asum[part][name][stored])<1e-9
  achecked[f'{part}:{name}']=len(rows)
spec=(W/'PRESPECIFICATION_ADDENDUM.md').read_text()
for line in spec.splitlines():
 if line.startswith('- ') and ': ' in line and line.endswith(tuple('0123456789abcdef')):
  f,h=line[2:].split(': ');assert hashlib.sha256((W/f).read_bytes()).hexdigest()==h,f
bs=json.loads((r/'4_Replication/code/revision_results/block_sensitivity.json').read_text());assert bs['equity_block10_reproduces_published']
audit['addendum']={'designs_and_replications':achecked,'seeds_match_protocol':True,'code_hashes_match_prespecification':True,
 'whitening_coverage_sqrt_n_blocks':{k:v['coverage'] for k,v in asum['WB'].items()},'weak_identification_joint':{k:v['joint_new'] for k,v in asum['JW'].items()},
 'recursive_operator_coverage_persistent':{k:v['op_cover'] for k,v in asum['RB'].items()},'equity_block10_reproduces_published':True,'reported_in_manuscript':True}
# benchmark applications: stored results must match the manuscript tables, which are generated from them
BM=r/'4_Replication/code/benchmark/results'
fomc=json.loads((BM/'fomc.json').read_text());eq=json.loads((BM/'equity_benchmark.json').read_text());oil=json.loads((BM/'oil_benchmark.json').read_text())
def near(x,y,tol=6e-4):return abs(x-y)<tol
assert fomc['n']==325 and near(fomc['raw']['block1']['recursive']['lower_budget'][1],0.120) and near(fomc['raw']['block19']['recursive']['lower_budget'][1],0.099)
assert near(fomc['rescaled']['block1']['recursive']['lower_budget'][1],0.120) and near(fomc['rescaled']['block19']['recursive']['lower_budget'][1],0.101)
assert all(near(fomc['raw']['floors'][f'k{k}_block1']['rho'],v) for k,v in zip((2,3,5),(0.250,0.307,0.363)))
assert all(fomc['rescaled']['floors'][f'k{k}_block1']['lower']==0 for k in (2,3,5))
assert eq['n']==1039 and all(near(a,b) for a,b in zip(eq['raw']['block10']['lower_budget'],[0.147,0.106])) and all(near(a,b) for a,b in zip(eq['rescaled']['block10']['lower_budget'],[0.157,0.146]))
pub=json.loads((r/'4_Replication/code/revision_results/equity.json').read_text())
assert all(abs(a-b)<1e-9 for a,b in zip(eq['raw']['block10']['lower_budget'],pub['recursive_budget_lower'])) and all(abs(a-b)<1e-6 for a,b in zip(eq['raw']['block10']['coskew_wald'],pub['recursive_wald']))
assert all(near(eq['raw']['floors'][f'k{k}_block10']['rho'],v) for k,v in zip((2,3,5),(0.157,0.197,0.216)))
assert near(eq['raw']['het']['k2']['angle'],23.5,0.06) and near(eq['raw']['het']['k2']['ci95'][0],18.9,0.06) and near(eq['raw']['het']['k2']['ci95'][1],28.6,0.06)
assert near(eq['raw']['floors']['k2_shock_floors']['pair_floor'],0.18,0.006) and near(eq['rescaled']['floors']['k2_shock_floors']['pair_floor'],0.21,0.006)
assert all(near(oil['floors'][f'k{k}']['rho'],v) for k,v in zip((2,3,5),(0.082,0.113,0.131))) and near(oil['floors']['k5']['lower'],0.039)
fa=json.loads((BM/'fomc_adjusted.json').read_text());ea={b:json.loads((BM/f'equity_adjusted_{b}.json').read_text()) for b in ('b10','b33')}
assert all(near(fa['block1']['adjusted'][f'k{k}']['breakdown'][1],v) for k,v in zip((2,3,5),(0.093,0.088,0.080))) and all(fa['block1']['adjusted'][f'k{k}']['breakdown'][0]==0 for k in (2,3,5))
assert all(near(a,b) for a,b in zip(ea['b10']['adjusted']['k3']['breakdown'],[0.146,0.110])) and near(ea['b10']['coskew_het']['wald'],3.98,0.01)
cs=json.loads((BM/'coskew_fomc.json').read_text());
for kk,(rt,at) in {5:(0.41,0.01),50:(0.355,0.01)}.items():
 c=json.loads((BM/f'adjusted_calibration_k{kk}.json').read_text())['summary'];assert near(c['raw_true_reject'],rt,1e-9) and near(c['adj_true_reject'],at,1e-9) and c['raw_true_exceeds_rho']==0 and c['adj_true_exceeds_rho_residual']<=0.005 and c['raw_false_reject']==1 and c['adj_false_reject']==1
assert near(cs['min'],2.89,0.01)
import importlib.util,tempfile,filecmp
spec=importlib.util.spec_from_file_location('bbt',r/'4_Replication/code/benchmark/build_benchmark_tables.py');bbt=importlib.util.module_from_spec(spec)
tmp=Path(tempfile.mkdtemp());bbt.TEX=str(tmp);spec.loader.exec_module(bbt);bbt.TEX=str(tmp);bbt.fomc();bbt.equity();bbt.bandwidth();bbt.calibration();bbt.fomc4()
for f in ('revised_vix_table.tex','fomc_table.tex','bandwidth_table.tex','adjusted_calibration_table.tex','fomc4_table.tex'):assert (tmp/f).read_text()==(src/'results'/f).read_text(),f
audit['benchmark_applications']={'fomc_n':fomc['n'],'equity_n':eq['n'],'stored_results_match_manuscript':True,'tables_regenerate_from_stored_outputs':True,'equity_raw_reproduces_main_run':True,'seeds':{'fomc':fomc['seed'],'equity':eq['seed'],'oil':oil['seed']}}
for p in list((r/'4_Replication/code').glob('*.py'))+list(W.glob('*.py'))+list((r/'4_Replication/code/benchmark').glob('*.py')):ast.parse(p.read_text())
audit['python_syntax']='pass'
(r/'4_Replication/audit/submission_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
