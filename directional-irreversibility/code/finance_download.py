from pathlib import Path
import urllib.request,json,hashlib,datetime,concurrent.futures
P=Path(__file__).resolve().parent.parent/'finance';D=P/'data';D.mkdir(parents=True,exist_ok=True)
urls={'jk_monthly.csv':'https://raw.githubusercontent.com/marekjarocinski/jkshocks_update_fed/3b48967babad46f819163ad1c66fa3595dac9f18/shocks_fed_jk_m.csv','jk_readme.md':'https://raw.githubusercontent.com/marekjarocinski/jkshocks_update_fed/3b48967babad46f819163ad1c66fa3595dac9f18/README.md'}
for name in ['DEXJPUS','GS10','BAA','AAA','VIXCLS']:urls[name+'.csv']='https://fred.stlouisfed.org/graph/fredgraph.csv?id='+name
# Independent read-only public data downloads.
def get(item):
 name,url=item
 if (D/name).exists():
  b=(D/name).read_bytes()
 else:
  with urllib.request.urlopen(url,timeout=45) as r:b=r.read()
 (D/name).write_bytes(b)
 return {'file':name,'url':url,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:r=list(ex.map(get,urls.items()))
if not (P/'sources.json').exists():(P/'sources.json').write_text(json.dumps(r,indent=2))
print([(x['file'],x['bytes']) for x in r])
