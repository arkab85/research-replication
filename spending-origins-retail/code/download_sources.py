"""Verify bundled official sources or recover missing files from their exact URLs."""
from pathlib import Path
import argparse,hashlib,json,urllib.request
R=Path(__file__).resolve().parent

def verify(path,item):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 if path.stat().st_size!=item['bytes'] or h.hexdigest()!=item['sha256']:
  raise ValueError('Source differs from the analyzed snapshot: '+str(path))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
 sources=json.loads((R/'sources.json').read_text())['sources']
 for item in sources:
  p=R/item['file']
  if not p.exists():
   if item.get('note','').startswith('Pretty-printed'):raise FileNotFoundError('Restore the bundled formatted metadata snapshot: '+str(p))
   if args.verify_only:raise FileNotFoundError(p)
   p.parent.mkdir(parents=True,exist_ok=True)
   part=p.with_suffix(p.suffix+'.partial')
   req=urllib.request.Request(item['url'],headers={'User-Agent':'Research replication source retrieval'})
   with urllib.request.urlopen(req,timeout=120) as response,part.open('wb') as output:
    while chunk:=response.read(1024*1024):output.write(chunk)
   verify(part,item);part.replace(p)
  verify(p,item)
 print(f'Verified {len(sources)} official source files against the analyzed snapshot.')
