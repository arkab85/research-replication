"""Put the tables in first-citation order and give the two appendix tables A-numbers.

A reviewer reads sequentially. Before this, the first table cited in the text was
numbered 18 and the eighth was numbered 1. Nothing about the estimates changes.
"""
import re, os
from config import PAPER as PAP

p = os.path.join(PAP, "jmp.tex")
t = open(p, encoding="utf-8").read()

head, _, tabsec = t.partition(r"\section*{Tables}")
assert tabsec, "no Tables section"

# pull every table environment out of the Tables section
envs = re.findall(r"\\begin\{table\}.*?\\end\{table\}", tabsec, re.S)
by_lab = {re.search(r"\\label\{tab:([a-z0-9]+)\}", e).group(1): e for e in envs}
assert len(by_lab) == len(envs), "duplicate labels"

# first citation of each label in the body
body = head + tabsec.split(envs[0])[0]
order, seen = [], set()
for m in re.finditer(r"\\ref\{tab:([a-z0-9]+)\}", body):
    k = m.group(1)
    if k in by_lab and k not in seen:
        seen.add(k); order.append(k)
missing = [k for k in by_lab if k not in seen]
assert not missing, f"never cited: {missing}"

APPX = ["sec", "issuers"]               # balance sheets, issuer list
main = [k for k in order if k not in APPX]
appx = [k for k in order if k in APPX]
print("main :", " ".join(main))
print("appx :", " ".join(appx))

blocks = "\n\n".join(by_lab[k] for k in main)
blocks += ("\n\n\\clearpage\n\\setcounter{table}{0}\n"
           "\\renewcommand{\\thetable}{A.\\arabic{table}}\n"
           "\\section*{Appendix Tables}\n"
           "\\addcontentsline{toc}{section}{Appendix Tables}\n\n")
blocks += "\n\n".join(by_lab[k] for k in appx)

pre = tabsec.split(envs[0])[0]
post = tabsec.rsplit(envs[-1], 1)[1]
open(p, "w", encoding="utf-8").write(
    head + r"\section*{Tables}" + pre + blocks + post)
print(f"reordered {len(envs)} tables")
