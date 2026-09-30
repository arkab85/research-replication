"""One-command replication. Run from the code folder:  python run_all.py [--skip-streams]
Rebuilds every table, figure and number in the paper from the raw extracts on F:\\ and recompiles both manuscripts if tectonic is on PATH.
--skip-streams reuses the cached outputs of the three passes over the 20M-row note log (about 10 minutes each)."""
import subprocess, sys, os, time
H = os.path.dirname(os.path.abspath(__file__)); skip = "--skip-streams" in sys.argv
STEPS = [("loanframe.py", False), ("main_analysis.py", False), ("cares.py", False), ("round2_a.py", False), ("round2_notes.py", True), ("rd.py", False), ("round2_c.py", False),
         ("round3_notes.py", True), ("round3_b.py", False), ("round3_c.py", False), ("benchmark.py", False), ("mk_tex_assets.py", False), ("mk_tex_assets2.py", False),
         ("rd_assets.py", False), ("extra_analysis_b.py", False), ("build_version_a.py", False)]
for s, stream in STEPS:
    if stream and skip: print("skip (cached):", s); continue
    t = time.time(); print(">>", s, flush=True); r = subprocess.run([sys.executable, os.path.join(H, s)], capture_output=True, text=True)
    if r.returncode: print(r.stdout[-2000:], r.stderr[-3000:]); sys.exit(f"FAILED at {s}")
    print(f"   ok ({time.time()-t:.0f}s)")
print("All numbers regenerated. Compare tex/*/numbers*.tex with the versions in the submission before compiling.")
