import sys,json,time; from icrc import *
seed=int(sys.argv[1]); B=int(sys.argv[2])
d,pl=sample(); rng=np.random.default_rng(seed)
cities=pl.city.unique(); ci=pl.city.map({c:i for i,c in enumerate(cities)}).values
out=[]; t=time.time()
for b in range(B):
    draw=rng.integers(0,len(cities),len(cities)); cnt=np.bincount(draw,minlength=len(cities))
    w=cnt[ci].astype(float)
    row={}
    bi,_,c2,sm2,_=icrc(pl,"inter",w=w); row["icrc_inter"]=bi.tolist(); row["c2"]=c2; row["sm2"]=sm2
    bb,_,_,_,_=icrc(pl,"base",w=w); row["icrc_base"]=bb.tolist()
    dat=pl.assign(w=w); dat=dat[dat.w>0]
    row["naive_inter"]=stage2(dat,"inter",w=True)[0].tolist(); row["naive_base"]=stage2(dat,"base",w=True)[0].tolist()
    out.append(row)
json.dump(out,open(f"boot_{seed}.json","w")); print(len(out),time.time()-t)
