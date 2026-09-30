"""Assemble and compile the full-length journal manuscript; report length."""
import os, re, glob, shutil, subprocess
H = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(H, "tex", "v"); J = os.path.join(H, "tex", "juefull"); TT = os.path.join(H, "tex", "tectonic.exe")
for f in glob.glob(os.path.join(S, "*")):
    b = os.path.basename(f)
    if b.startswith(("t", "f", "numbers")) and b.endswith((".tex", ".pdf")) and not os.path.exists(os.path.join(J, b)): shutil.copy2(f, J)
for b in ("numbers_strm.tex", "numbers_v.tex", "tJ_geo.tex", "highlights.txt"): shutil.copy2(os.path.join(S, b), J)
p = os.path.join(J, "main.tex"); s = open(p, encoding="utf-8").read()
s = s.replace("None of sixteen interaction terms is significant at five percent.", "None of the twenty interaction terms is significant at five percent.").replace("(\\,$p=0.08$)", "($p=0.08$)").replace("With a dozen subgroups one such rejection is expected by chance,", "With some twenty subgroup estimates of this outcome one such rejection is expected by chance,")
open(p, "w", encoding="utf-8").write(s)
for f in ("main.tex", "cover_letter.tex"):
    if os.path.exists(os.path.join(J, f)):
        r = subprocess.run([TT, f], cwd=J, capture_output=True, text=True); print(f, "compiled" if r.returncode == 0 else r.stderr[-700:]); print("\n".join(l for l in r.stderr.split("\n") if "Overfull" in l or "Undefined" in l or "undefined" in l)[:1500])
b = s[s.index("\\section{Introduction}"):s.index("\\section*{Declaration of competing interest}")]; b = b.replace("\\$", " "); b = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", " ", b, flags=re.S); b = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", b, flags=re.S); b = re.sub(r"\$[^$]*\$", " X ", b); b = re.sub(r"\\(ref|label|eqref)\{[^}]*\}", " 1 ", b); b = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " 1 ", b)
print("body words:", len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-.,]*", re.sub(r"[{}~]", " ", b)))); m = s[:s.index("\\appendix")]; print("main tables:", len(re.findall(r"\\begin\{table\}", m)), "| main figures:", len(re.findall(r"\\begin\{figure\}", m)))
i = s.index("\\begin{abstract}"); a = re.sub(r"\\[a-zA-Z]+(\{\})?", " 1 ", s[i + 16:s.index("\\end{abstract}")]); print("abstract words:", len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", a)))
try:
    from pypdf import PdfReader; print("pages:", len(PdfReader(os.path.join(J, "main.pdf")).pages))
except Exception as e: print(e)
