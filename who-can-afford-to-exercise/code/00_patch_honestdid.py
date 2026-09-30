"""Idempotent upstream fix for honestdid, required before 118_honest_avg.py.

honestdid/core.py calls float() on a 1x1 array in the conditional test. NumPy >= 1.25
raises TypeError on that, so any sensitivity analysis with a non-unit l_vec fails --
which is why a single-period target works and an aggregate does not. The fix takes the
sole element instead of relying on implicit scalar conversion. No numerical content
changes. Run once; running again is a no-op.
"""
import re, os, shutil, sys
import honestdid.core as core

src_path = core.__file__
TAG = "# patched for numpy>=1.25 scalar conversion"
if os.path.exists(src_path + ".orig"):          # always patch from pristine source
    shutil.copy(src_path + ".orig", src_path)
s = open(src_path, encoding="utf-8").read()
assert TAG not in s

# any float(<matrix product>) with two or three operands
V = r"[A-Za-z_][\w.]*(?:\.T)?"
pat = re.compile(rf"float\((\s*{V}\s*@\s*{V}(?:\s*@\s*{V})?\s*)\)")
hits = pat.findall(s)
print(f"{src_path}\n  quadratic-form float() calls found: {len(hits)}")
for h in hits:
    print(f"    float({h.strip()})")
assert hits, "nothing to patch -- check the library version"

shutil.copy(src_path, src_path + ".orig")
s2 = pat.sub(lambda m: f"float(np.asarray({m.group(1).strip()}).reshape(-1)[0])", s)
s2 = s2.replace("import numpy as np", "import numpy as np  " + TAG, 1)
open(src_path, "w", encoding="utf-8").write(s2)
print(f"  patched {len(hits)} call(s); original saved to {src_path}.orig")
