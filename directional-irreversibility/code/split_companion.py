"""Split the fully cross-referenced TeX build into main paper and companion."""
from pathlib import Path
import fitz
P=Path(__file__).resolve().parent.parent
full=fitz.open(P/'build/main.pdf')
hits=[i for i,p in enumerate(full) if 'Electronic Companion' in p.get_text()]
assert len(hits)==1,hits
split=hits[0]
for name,start,end in [('Manuscript',0,split-1),('Electronic_Companion',split,len(full)-1)]:
 d=fitz.open();d.insert_pdf(full,from_page=start,to_page=end);d.set_metadata({'title':'Directional Irreversibility in Economic Dynamics'+(' - Electronic Companion' if start else ''),'author':''});d.save(P/f'DII_Management_Science_{name}.pdf');print(name,len(d))
