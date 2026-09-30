import hashlib, glob, os, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__)); pat = [os.path.join(H, "tex", "v", "numbers*.tex"), os.path.join(H, "tex", "v", "t*.tex")]
def snap(): return {os.path.basename(f): hashlib.md5(open(f, "rb").read()).hexdigest() for p in pat for f in glob.glob(p)}
before = snap(); print("files tracked:", len(before), flush=True)
r = subprocess.run([sys.executable, os.path.join(H, "run_all.py"), "--skip-streams"], capture_output=True, text=True); print(r.stdout[-1500:]);
if r.returncode: print(r.stderr[-2000:]); sys.exit("pipeline failed")
after = snap(); diff = [k for k in before if before[k] != after.get(k)]
print("REPRODUCED IDENTICALLY" if not diff else "DIFFERENCES IN: " + ", ".join(diff))
