from pathlib import Path
import subprocess,sys,os,shutil
S=Path(__file__).resolve().parent;P=S.parent;B=P/'build';B.mkdir(exist_ok=True)
env={**os.environ,'OPENBLAS_NUM_THREADS':'1'}
if '--rerun-study' in sys.argv:
 subprocess.run([sys.executable,str(S/'full_study.py'),'--reps','300','--boot','399'],check=True,env=env)
subprocess.run([sys.executable,str(S/'decision_analysis.py')],check=True,env=env)
subprocess.run([sys.executable,str(S/'reposition.py')],check=True)
subprocess.run([sys.executable,str(S/'second_pass.py')],check=True)
subprocess.run([sys.executable,str(S/'dii_example.py')],check=True,env=env)
subprocess.run([sys.executable,str(S/'dii_reframe.py')],check=True)
if '--rerun-finance' in sys.argv:
 subprocess.run([sys.executable,str(S/'finance_download.py')],check=True)
 subprocess.run([sys.executable,str(S/'finance_study.py')],check=True,env=env)
 subprocess.run([sys.executable,str(S/'finance_bounds.py')],check=True,env=env)
 subprocess.run([sys.executable,str(S/'orthogonal_finance.py')],check=True,env=env)
 subprocess.run([sys.executable,str(S/'training_aware_finance.py')],check=True,env=env)
 subprocess.run([sys.executable,str(S/'joint_nuisance_finance.py')],check=True,env=env)
 subprocess.run([sys.executable,str(S/'finance_decision_use.py')],check=True,env=env)
subprocess.run([sys.executable,str(S/'finance_integrate.py')],check=True)
if '--rerun-diagnostic' in sys.argv:
 subprocess.run([sys.executable,str(S/'persistent_diagnostic.py')],check=True,env=env)
subprocess.run([sys.executable,str(S/'weak_integrate.py')],check=True)
if '--rerun-orthogonal' in sys.argv:
 subprocess.run([sys.executable,str(S/'orthogonal_diagnostic.py')],check=True,env=env)
subprocess.run([sys.executable,str(S/'orthogonal_integrate.py')],check=True)
if '--rerun-training-aware' in sys.argv:
 subprocess.run([sys.executable,str(S/'oracle_derivative_audit.py')],check=True,env=env)
 subprocess.run([sys.executable,str(S/'training_aware_study.py')],check=True,env=env)
subprocess.run([sys.executable,str(S/'training_aware_comparison.py')],check=True,env=env)
subprocess.run([sys.executable,str(S/'training_aware_integrate.py')],check=True)
if '--rerun-joint' in sys.argv:
 subprocess.run([sys.executable,str(S/'joint_nuisance_study.py')],check=True,env=env)
subprocess.run([sys.executable,str(S/'joint_nuisance_integrate.py')],check=True)
if '--rerun-decision-use' in sys.argv:
 subprocess.run([sys.executable,str(S/'decision_use_study.py')],check=True,env=env)
subprocess.run([sys.executable,str(S/'decision_use_integrate.py')],check=True)
subprocess.run([sys.executable,str(S/'focus_revision.py')],check=True)
subprocess.run([sys.executable,str(S/'scenario_revision.py')],check=True)
subprocess.run([sys.executable,str(S/'original_recovery.py')],check=True)
subprocess.run([sys.executable,str(S/'decision_regret.py')],check=True)
subprocess.run([sys.executable,str(S/'decision_regret_revision.py')],check=True)
if '--rerun-state-observation' in sys.argv:
 subprocess.run([sys.executable,str(S/'state_observation.py')],check=True,env=env)
subprocess.run([sys.executable,str(S/'state_observation_revision.py')],check=True)
subprocess.run([sys.executable,str(S/'finance_revision.py')],check=True)
subprocess.run([sys.executable,str(S/'submission_revision.py')],check=True)
subprocess.run([sys.executable,str(S/'figures_main.py')],check=True)
subprocess.run([sys.executable,str(S/'revision_r2.py')],check=True)
if '--rerun-theory-demos' in sys.argv:
 subprocess.run([sys.executable,str(S/'theory_demos.py')],check=True,env=env)
subprocess.run([sys.executable,str(S/'theory_demos_figure.py')],check=True)
subprocess.run([sys.executable,str(S/'revision_r3.py')],check=True)
if '--rerun-oilvix' in sys.argv:
 subprocess.run([sys.executable,str(S/'oilvix_application.py')],check=True,env=env)
subprocess.run([sys.executable,str(S/'revision_r4.py')],check=True)
if '--rerun-fan-pelger' in sys.argv:
 subprocess.run([sys.executable,str(S/'fan_pelger_experiments.py'),'200'],check=True,env=env)
subprocess.run([sys.executable,str(S/'revision_r5.py')],check=True)
if '--rerun-fd' in sys.argv:
 subprocess.run([sys.executable,str(S/'verify_fd_nuisance.py')],check=True,env=env)
 subprocess.run([sys.executable,str(S/'fd_experiments.py'),'loading'],check=True,env=env)
 subprocess.run([sys.executable,str(S/'fd_experiments.py'),'bandwidth'],check=True,env=env)
subprocess.run([sys.executable,str(S/'revision_r6.py')],check=True)
subprocess.run([sys.executable,str(S/'revision_r7_singular.py')],check=True)
subprocess.run([sys.executable,str(S/'revision_r8_check.py')],check=True)
# A clean auxiliary state avoids carrying interrupted builds into a revision.
for ext in ['aux','out','toc']:
 (B/f'main.{ext}').unlink(missing_ok=True)
for i in range(3):
 r=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={B}','main.tex'],cwd=S,capture_output=True,text=True)
 (B/f'compile_{i}.txt').write_text(r.stdout)
 if r.returncode:raise RuntimeError(r.stdout[-3000:])
subprocess.run([sys.executable,str(S/'split_companion.py')],check=True)
r=subprocess.run(['pandoc',str(S/'editor_strategy.md'),'-o',str(P/'DII_Management_Science_Editor_Strategy.pdf'),'--pdf-engine=pdflatex','-V','geometry:margin=1in','-V','fontsize=11pt','-V','colorlinks=true','-V','urlcolor=blue'],capture_output=True)
if r.returncode: print('reviewer strategy PDF not rebuilt (pandoc template); use source/editor_strategy.md.')
if '--v' in sys.argv:
 subprocess.run([sys.executable,str(S/'summary_statistics.py')],check=True)
 subprocess.run([sys.executable,str(S/'paper_version.py')],check=True)
print('Built manuscript and reviewer strategy.')
