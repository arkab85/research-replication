"""Gate for Part C: size of the circular-shift test at daily-level sparsity (shock nonzero on 3 percent of days, no
channel, T = 3000, h = 21, two lags, gap 126), next to the wild and paired rejection rates from
results/size_under_sparsity.csv for the same design (both 100 percent).   python extension/size_shift_daily.py [R]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'code'))
import run_extension  # noqa: F401  (installs the O(n^2) Gram routine; it runs nothing on import)
import sparsity_check as SC; from core_lib import np, pd
R = int(sys.argv[1]) if len(sys.argv) > 1 else 100; rng = np.random.RandomState(0); ps = []
for r in range(R):
    T = 3000; e = rng.normal(size=T); Y = np.zeros(T)
    for t in range(1, T): Y[t] = 0.3 * Y[t - 1] + e[t]
    d = pd.DataFrame({'x': rng.normal(size=T) * (rng.rand(T) < .03), 'y': Y}); obs, M = SC.shift_matrix([d], 21, 99, r, 2, 126); ps.append(SC.shift_p_from(obs, M, [0])[1])
    if (r + 1) % 20 == 0: print(r + 1, 'reps: size at 5% =', np.mean(np.array(ps) <= .05), ' at 10% =', np.mean(np.array(ps) <= .10), flush=True)
pd.DataFrame([dict(design='daily-like p=0.03, T=3000, h=21', reps=R, shift_05=np.mean(np.array(ps) <= .05), shift_10=np.mean(np.array(ps) <= .10))]).to_csv(os.path.join(HERE, 'results', 'size_shift_daily.csv'), index=False)
