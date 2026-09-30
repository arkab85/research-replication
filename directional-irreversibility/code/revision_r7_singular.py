"""Single-author voice: first-person plural -> singular throughout (runs last)."""
from pathlib import Path
import re
T = Path(__file__).resolve().parent / 'main.tex'
tex = T.read_text()
for a, b in [(r'\bWe\b', 'I'), (r'\bwe\b', 'I'), (r'\bOur\b', 'My'), (r'\bour\b', 'my')]:
    tex = re.sub(a, b, tex)
assert not re.search(r'\b([Ww]e|[Oo]urs?|us)\b', tex)
T.write_text(tex)
cl = T.parent / 'cover_letter.md'
c = cl.read_text()
c = re.sub(r'\bWe\b', 'I', c); c = re.sub(r'\bwe\b', 'I', c); c = re.sub(r'\bOur\b', 'My', c); c = re.sub(r'\bour\b', 'my', c)
cl.write_text(c)
print('revision_r7_singular: applied')
