"""Bootstrap intervals for the heteroskedasticity-identified rotation in the FOMC data
(raw and VIX-rescaled surprises), appended to benchmark/results/fomc.json."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bcore import np, rot, het_angle, boot_index
from fomc_application import load, OUT

B = 999; SEED = 2026092704


def het_boot(u, W, k, block, rng):
    n = len(u); a, L, Dv = het_angle(u, W, k)
    ab = np.empty(B)
    for b in range(B):
        idx = boot_index(n, block, rng); ub = u[idx] - u[idx].mean(0)
        ab[b] = het_angle(ub, W[idx], k)[0]
    ab = (ab - a + 45) % 90 - 45 + a
    return dict(k=k, angle=a, ci95=np.percentile(ab, [2.5, 97.5]).tolist(), ci90=np.percentile(ab, [5, 95]).tolist(),
                bin_variances=Dv.tolist(), impact=(L @ rot(np.deg2rad(a))).tolist())


def main():
    d = load(); u = d[['pc1', 'SP500']].values; W = d.vix_prev.values
    res = json.load(open(os.path.join(OUT, 'fomc.json')))
    rng = np.random.default_rng(SEED); nblk = int(np.ceil(np.sqrt(len(u))))
    for lab, uu in (('raw', u), ('rescaled', u / W[:, None])):
        uu = uu - uu.mean(0)
        res[lab]['het'] = {f'k{k}_block{blk}': het_boot(uu, W, k, blk, rng) for k in (2, 3, 5) for blk in (1, nblk)}
        print(lab, {k: (round(v['angle'], 1), np.round(v['ci95'], 1).tolist()) for k, v in res[lab]['het'].items()}, flush=True)
    json.dump(res, open(os.path.join(OUT, 'fomc.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
