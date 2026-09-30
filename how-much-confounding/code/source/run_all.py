"""Run the complete frozen-data replication from any working directory."""
from pathlib import Path
import os,subprocess,sys
P=Path(__file__).resolve().parents[1]
env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
for script in ['study.py','power.py','transmission.py','application.py','event_application.py','pool_check.py','pooled_application.py','pooled_meetings_only.py','pooled_devol.py','rate_study.py','pooled_sensitivity.py','bonferroni_check.py','verify.py','build_tables.py','revision_study.py','revision_training_study.py','revision_application.py','build_revision_tables.py','validate_revision.py','feasible_study.py','feasible_application.py','build_feasible_tables.py','validate_feasible_geometry.py','validate_feasible_results.py']:
 print('Running',script,flush=True)
 subprocess.run([sys.executable,str(P/'source'/script)],cwd=P,env=env,check=True)
subprocess.run(['sh','build_paper.sh'],cwd=P,env=env,check=True)
print('Complete:',P/'main.pdf',flush=True)
