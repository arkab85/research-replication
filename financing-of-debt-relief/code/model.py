"""Numerical examples and verification for "The Financing of Debt Relief".

Every input below is an illustrative parameter chosen in the text; nothing is
estimated. The script reproduces Figures 1-3, the rows of Tables 1-2, and a
verification report (data/model_verification.json) that checks each
proposition's numerical claims, including an independent linear-programming
solution of the allocation problem.

Run from any directory:  python code/model.py
Requires NumPy, SciPy and Matplotlib.
"""
from pathlib import Path
import csv
import json

import numpy as np
from scipy.optimize import brentq, linprog
from scipy.stats import beta as beta_dist
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
for folder in ("figures", "tables", "data"):
    (ROOT / folder).mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})
BLACK, GRAY, LIGHT = "#111111", "#6b6b6b", "#a8a8a8"
FIGSIZE = (6.6, 2.7)
report = {"status": "passed", "illustrative_not_estimated": True}


def trapz(y, x):
    y, x = np.asarray(y), np.asarray(x)
    return float(((y[1:] + y[:-1]) * np.diff(x)).sum() / 2)


def fmt(x, d=3):
    return f"{x:.{d}f}"


# ---------------------------------------------------------------------------
# 1. Propositions 1-2: rationing by value per dollar (uniform posterior example)
#    g(q) = R q - c,  l(q) = k - w q,  q ~ U[0,1]
# ---------------------------------------------------------------------------
R, c, k, w = 1.2, 0.2, 0.3, 0.1
q0 = c / R


def cash_used(u):
    """Cash needed to fund every opportunity with posterior above u."""
    return k * (1 - u) - w * (1 - u * u) / 2


Kbar = cash_used(q0)


def solve_uniform(K):
    if K <= 0:
        return 1.0, (R - c) / (k - w), 0.0, 0.0, np.nan
    u = q0 if K >= Kbar else brentq(lambda x: cash_used(x) - K, q0, 1.0)
    lam = max(0.0, (R * u - c) / (k - w * u))
    share = 1 - u
    value = R * (1 - u * u) / 2 - c * share
    success = (1 + u) / 2
    return u, lam, share, value, success


Ks = np.linspace(0, Kbar, 301)
sol = np.array([solve_uniform(K) for K in Ks])
with (ROOT / "data" / "uniform_model.csv").open("w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["liquidity_K", "cutoff_q", "shadow_price", "funded_share",
                 "private_value", "success_among_funded"])
    wr.writerows(np.column_stack([Ks, sol]))

# Independent check: discretised LP with 4,000 posterior types.
qgrid = (np.arange(4000) + 0.5) / 4000
g_lp, l_lp = R * qgrid - c, k - w * qgrid
max_gap = 0.0
for K in np.linspace(0.005, Kbar - 0.005, 17):
    lp = linprog(-g_lp / len(g_lp), A_ub=[l_lp / len(l_lp)], b_ub=[K],
                 bounds=(0, 1), method="highs")
    assert lp.success
    max_gap = max(max_gap, abs(-lp.fun - solve_uniform(K)[3]))
assert max_gap < 1e-6
assert np.all(np.diff(sol[:, 2]) >= -1e-12)          # access rises with K
assert np.all(np.diff(sol[1:, 4]) <= 1e-12)          # selected success falls with K
assert np.all(np.diff(sol[:, 1]) <= 1e-12)           # shadow price falls with K
report["prop1_2"] = {"lp_comparisons": 17, "lp_types": 4000,
                     "max_abs_lp_gap": max_gap, "Kbar": Kbar,
                     "params": {"R": R, "c": c, "k": k, "omega": w}}

rows = []
for frac in (0.1, 0.3, 0.6, 1.0):
    u, lam, share, val, succ = solve_uniform(frac * Kbar)
    rows.append((frac, frac * Kbar, u, lam, share, val, succ))
with (ROOT / "tables" / "uniform_rows.tex").open("w") as f:
    f.write("\\newcommand{\\UniformRows}{%\n")
    for r in rows:
        f.write(" & ".join(fmt(v) for v in r) + r" \\" + "\n")
    f.write("}\n")

fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)
x = Ks / Kbar
ax[0].plot(x, sol[:, 2], color=BLACK, lw=1.4, label="Funded share")
ax[0].plot(x, sol[:, 4], color=GRAY, lw=1.4, ls="--", label="Success rate among funded")
ax[0].set(xlabel=r"Liquidity, $K/\bar K$", ylabel="Share / probability", ylim=(0, 1.04))
ax[0].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.40, 0.02))
ax[0].set_title("A. Access and selected performance", loc="left", fontsize=9.5)
ax[1].plot(x, sol[:, 1], color=BLACK, lw=1.4, label=r"$\lambda$ (left axis)")
ax2 = ax[1].twinx()
ax2.plot(x, sol[:, 0], color=GRAY, lw=1.4, ls="--", label=r"$q^*$ (right axis)")
ax2.set_ylim(0, 1.04)
ax2.spines["right"].set_visible(True)
ax2.set_ylabel(r"Cutoff $q^*$")
ax[1].set(xlabel=r"Liquidity, $K/\bar K$", ylabel=r"$\lambda$")
h1, l1 = ax[1].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left", bbox_to_anchor=(0.52, 0.52))
ax[1].set_title("B. Shadow value and cutoff", loc="left", fontsize=9.5)
fig.tight_layout(w_pad=2.2)
fig.savefig(ROOT / "figures" / "fig1_rationing.pdf")
plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Proposition 4: the value of information is hump-shaped in liquidity
#    Continuous example: Q ~ Beta(1.5, 1), p = 0.6, R = 1, c = 0.35
# ---------------------------------------------------------------------------
RB, cB = 1.0, 0.35
aB, bB = 1.5, 1.0
dist = beta_dist(aB, bB)
pB = dist.mean()
N = 200_000
u = (np.arange(N) + 0.5) / N
Qq = dist.ppf(u)                                   # quantile function on a grid
ms = np.linspace(0, 1, 1001)


def top_integral(vals, m):
    """Integral over the top-m quantiles of vals (vals sorted ascending in u)."""
    n = int(round(m * N))
    return vals[N - n:].sum() / N if n > 0 else 0.0


VG = np.array([top_integral(np.maximum(RB * Qq - cB, 0), m) for m in ms])
V0 = ms * max(RB * pB - cB, 0)
dV = VG - V0
Arank = np.array([top_integral(RB * (Qq - pB), m) for m in ms])
Sscr = np.array([top_integral(np.maximum(cB - RB * Qq, 0), m) for m in ms])
assert np.max(np.abs(dV - (Arank + Sscr))) < 1e-9                  # decomposition
assert np.all(dV >= -1e-12)
assert np.all(np.diff(dV, 2) <= 1e-8)                               # concavity
mstar = 1 - dist.cdf(pB)
peak_theory = RB * dist.expect(lambda q: max(q - pB, 0))
mad_half = 0.5 * RB * dist.expect(lambda q: abs(q - pB))
end_theory = dist.expect(lambda q: max(cB - RB * q, 0))
i_peak = int(np.argmax(dV))
assert abs(ms[i_peak] - mstar) < 2e-3
assert abs(dV.max() - peak_theory) < 1e-4
assert abs(peak_theory - mad_half) < 1e-6
assert abs(dV[-1] - end_theory) < 1e-5
mbar = 1 - dist.cdf(cB / RB)                                        # profitable share
report["theorem1"] = {
    "posterior": f"Beta({aB},{bB})", "p": pB, "R": RB, "c": cB,
    "m_star_theory": mstar, "m_star_grid": float(ms[i_peak]),
    "peak_theory_R_E(Q-p)+": peak_theory, "peak_grid": float(dV.max()),
    "half_MAD": mad_half, "value_at_full_funding_theory": end_theory,
    "value_at_full_funding_grid": float(dV[-1]),
    "profitable_share": mbar}

# marginal value of liquidity with and without information
dVG = np.gradient(VG, ms)
fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)
ax[0].plot(ms, dV, color=BLACK, lw=1.5, label=r"Total value $\Delta V(m)$")
ax[0].plot(ms, Arank, color=GRAY, lw=1.3, ls="--", label=r"Ranking value $\mathcal{A}(m)$")
ax[0].plot(ms, Sscr, color=GRAY, lw=1.3, ls=":", label=r"Screening value $\mathcal{S}(m)$")
ax[0].axvline(mstar, color=LIGHT, lw=0.8)
ax[0].text(mstar + 0.012, 0.003, r"$m^*$", fontsize=8.5, color=GRAY)
ax[0].set(xlabel=r"Fundable share, $m=K/k$", ylabel="Value of information")
ax[0].legend(fontsize=7.8, loc="upper left")
ax[0].set_ylim(0, 0.2)
ax[0].set_title("A. Ranking and screening value", loc="left", fontsize=9.5)
ax[1].plot(ms[1:-1], dVG[1:-1], color=BLACK, lw=1.5, label="With the signal")
ax[1].plot(ms, np.full_like(ms, RB * pB - cB), color=GRAY, lw=1.3, ls="--",
           label="Hard information only")
ax[1].axvline(mstar, color=LIGHT, lw=0.8)
ax[1].set(xlabel=r"Fundable share, $m=K/k$", ylabel=r"Marginal value, $k\,V'(K)$")
ax[1].set_ylim(bottom=0)
ax[1].legend(fontsize=7.8, loc="upper right")
ax[1].set_title("B. Marginal value of liquidity", loc="left", fontsize=9.5)
fig.tight_layout(w_pad=2.2)
fig.savefig(ROOT / "figures" / "fig2_information_value.pdf")
plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Proposition 6: which score is worth more depends on the balance sheet
#    Two calibrated scores with the same mean p = 0.5 (R = 1, c = 0.1).
# ---------------------------------------------------------------------------
R2, c2, p2 = 1.0, 0.1, 0.5
scores = {
    "A": [(1.0, 0.2), (0.375, 0.8)],     # precise at the top
    "B": [(0.8, 0.5), (0.2, 0.5)],       # broad
}


def T_discrete(atoms, m):
    """Expected successes among the top-m share (atoms = [(q, mass)])."""
    tot, left = 0.0, m
    for q, mass in sorted(atoms, key=lambda t: -t[0]):
        take = min(mass, left)
        tot += q * take
        left -= take
        if left <= 1e-15:
            break
    return tot


def auc_discrete(atoms, p):
    pos = [(q, q * mass / p) for q, mass in atoms]
    neg = [(q, (1 - q) * mass / (1 - p)) for q, mass in atoms]
    a = 0.0
    for q1, w1 in pos:
        for q0_, w0 in neg:
            a += w1 * w0 * (1.0 if q1 > q0_ else 0.5 if q1 == q0_ else 0.0)
    return a


for name, atoms in scores.items():
    assert abs(sum(m for _, m in atoms) - 1) < 1e-12
    assert abs(sum(q * m for q, m in atoms) - p2) < 1e-12
    assert all(R2 * q > c2 for q, _ in atoms)          # all actions profitable

grid = np.linspace(0, 1, 100_001)
rank_val = {n: np.array([R2 * (T_discrete(a, m) - m * p2) for m in grid])
            for n, a in scores.items()}
aucs = {n: auc_discrete(a, p2) for n, a in scores.items()}
for n in scores:
    integral = trapz(rank_val[n], grid)
    assert abs(integral - R2 * p2 * (1 - p2) * (aucs[n] - 0.5)) < 1e-8   # AUC identity
m_nb, m_dep = 0.072, 0.862     # June 2020 exercise rates in this paper's sample, nonbank and depository (Section 6.2)
val = {n: {m: R2 * (T_discrete(a, m) - m * p2) for m in (m_nb, 0.25, 0.5, 0.75, m_dep)}
       for n, a in scores.items()}
assert val["A"][m_nb] > val["B"][m_nb] and val["B"][m_dep] > val["A"][m_dep]
report["theorem2"] = {
    "scores": {n: a for n, a in scores.items()}, "AUC": aucs,
    "value_at_nonbank_share_0.072": {n: val[n][m_nb] for n in scores},
    "value_at_depository_share_0.862": {n: val[n][m_dep] for n in scores},
    "ratio_A_over_B_at_0.072": val["A"][m_nb] / val["B"][m_nb],
    "ratio_B_over_A_at_0.862": val["B"][m_dep] / val["A"][m_dep],
    "auc_identity_checked": True}

# General check of the AUC identity on random discrete score distributions.
rng = np.random.default_rng(20260926)
max_auc_err = 0.0
for _ in range(200):
    nat = rng.integers(2, 7)
    qs = np.sort(rng.uniform(0.02, 0.98, nat))
    mass = rng.dirichlet(np.ones(nat))
    pp = float(qs @ mass)
    atoms = list(zip(qs, mass))
    gg = np.linspace(0, 1, 20_001)
    vals = np.array([T_discrete(atoms, m) - m * pp for m in gg])
    lhs = trapz(vals, gg)
    rhs = pp * (1 - pp) * (auc_discrete(atoms, pp) - 0.5)
    max_auc_err = max(max_auc_err, abs(lhs - rhs))
assert max_auc_err < 1e-6
report["theorem2"]["auc_identity_random_cases"] = 200
report["theorem2"]["auc_identity_max_abs_error"] = max_auc_err

# Convex order implies dominance at every capacity: B vs a garbling of B.
garbled = [(0.65, 0.5), (0.35, 0.5)]     # mean-preserving contraction of B
tb = np.array([T_discrete(scores["B"], m) for m in grid])
tg = np.array([T_discrete(garbled, m) for m in grid])
assert np.all(tb >= tg - 1e-12)

with (ROOT / "tables" / "score_rows.tex").open("w") as f:
    f.write("\\newcommand{\\ScoreRows}{%\n")
    for m in (m_nb, 0.25, 0.5, 0.75, m_dep):
        a, b = val["A"][m], val["B"][m]
        f.write(f"{m:.3f} & {a + 1e-12:.4f} & {b + 1e-12:.4f} & {a / b:.2f}" + r" \\" + "\n")
    f.write("}\n")
    f.write("\\newcommand{\\AUCA}{%.2f}\n\\newcommand{\\AUCB}{%.2f}\n" % (aucs["A"], aucs["B"]))

fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)
ax[0].plot(grid, rank_val["A"], color=BLACK, lw=1.5, label=f"Score A (AUC = {aucs['A']:.2f})")
ax[0].plot(grid, rank_val["B"], color=GRAY, lw=1.5, ls="--", label=f"Score B (AUC = {aucs['B']:.2f})")
for m, lab, ha, dx in ((m_nb, "Nonbank", "left", 0.012), (m_dep, "Depository", "right", -0.012)):
    ax[0].axvline(m, color=LIGHT, lw=0.8)
    ax[0].text(m + dx, 0.004, lab, fontsize=7.2, color=GRAY, va="bottom", ha=ha)
ax[0].set(xlabel=r"Fundable share, $m$", ylabel="Value of the score")
ax[0].set_ylim(0, 0.2)
ax[0].legend(fontsize=7.8, loc="upper right")
ax[0].set_title("A. Value by liquidity state", loc="left", fontsize=9.5)
cap = {n: np.array([T_discrete(a, m) / p2 for m in grid]) for n, a in scores.items()}
ax[1].plot(grid, cap["A"], color=BLACK, lw=1.5, label="Score A")
ax[1].plot(grid, cap["B"], color=GRAY, lw=1.5, ls="--", label="Score B")
ax[1].plot(grid, grid, color=LIGHT, lw=0.9, ls=":", label="Uninformative")
ax[1].set(xlabel=r"Share of loans funded, top-ranked first", ylabel="Share of successes captured")
ax[1].legend(fontsize=7.8, loc="lower right")
ax[1].set_title("B. Cumulative accuracy profiles cross", loc="left", fontsize=9.5)
fig.tight_layout(w_pad=2.2)
fig.savefig(ROOT / "figures" / "fig3_score_value.pdf")
plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Binary benchmark (Corollary 1) and adoption cost
# ---------------------------------------------------------------------------
p, eta = 0.5, 0.8
piH = p * eta + (1 - p) * (1 - eta)
qH, qL = p * eta / piH, p * (1 - eta) / (1 - piH)
tent = np.array([R2 * (T_discrete([(qH, piH), (qL, 1 - piH)], m) - m * p) for m in ms])
assert abs(tent.max() - 0.15) < 1e-12 and abs(ms[np.argmax(tent)] - 0.5) < 1e-12
assert abs(tent[0]) < 1e-12 and abs(tent[-1]) < 1e-12
adopt = ms[tent > 0.06 + 1e-12]
report["corollary1_binary"] = {"qH": qH, "qL": qL, "piH": piH, "peak": float(tent.max()),
                               "adoption_interval_cost_0.06": [float(adopt.min()), float(adopt.max())]}


# ---------------------------------------------------------------------------
# 5. Proposition 5: reliance on information (level and relative responses)
# ---------------------------------------------------------------------------
def sensitivity(m, ms_):
    return np.where(m <= ms_, m / ms_, (1 - m) / (1 - ms_))


Dm = sensitivity(ms, piH)
assert abs(Dm.max() - 1) < 1e-12
# Continuous case (Beta example of Proposition 4, where RQ >= c fails only for
# screening; use c = 0 so every action is profitable): the level slope of a
# linear probability regression of x on Q is A(m)/(R Var Q), and the slope
# relative to the base rate is nonincreasing in m.
varQ = dist.var()
level = np.array([top_integral(Qq - pB, m) for m in ms]) / varQ
rel = level[1:] / ms[1:]
assert abs(ms[int(np.argmax(level))] - mstar) < 2e-3
assert np.all(np.diff(rel) <= 1e-9)
# direct regression check at three funded shares
for mm in (0.1, 0.4, 0.8):
    xx = (u >= 1 - mm).astype(float)
    slope = np.cov(xx, Qq, bias=True)[0, 1] / Qq.var()
    assert abs(slope - top_integral(Qq - pB, mm) / varQ) < 1e-3
report["prop_reliance"] = {"binary_peak_at": float(ms[np.argmax(Dm)]),
                           "beta_level_slope_peak_at": float(ms[int(np.argmax(level))]),
                           "relative_slope_nonincreasing": True}


# ---------------------------------------------------------------------------
# 6. Proposition 7: financing value of recognised information (score B)
# ---------------------------------------------------------------------------
kF, LF, K0 = 1.0, 0.6, 0.2
atomsB = scores["B"]
m_hard = min(1.0, K0 / (kF - LF * p2))
m_rec = brentq(lambda m: kF * m - LF * T_discrete(atomsB, m) - K0, 1e-9, 1.0)
assert m_rec > m_hard
def VGB(m):
    """Private value of funding the top-m share under score B."""
    return (R2 * 0.8 - c2) * min(m, 0.5) + (R2 * 0.2 - c2) * max(m - 0.5, 0.0)
value_rec = VGB(m_rec)
value_hard = VGB(m_hard) - LF * (T_discrete(atomsB, m_hard) - m_hard * p2)
value_none = m_hard * (R2 * p2 - c2)
assert value_rec > value_hard > value_none
report["prop_financing"] = {"k": kF, "L": LF, "K0": K0,
                            "fundable_share_hard_pricing": m_hard,
                            "fundable_share_recognised": m_rec,
                            "private_value_no_signal": value_none,
                            "private_value_signal_hard_pricing": value_hard,
                            "private_value_signal_recognised": value_rec}


# ---------------------------------------------------------------------------
# 7. Proposition 8: private selection and borrower surplus (self-cure ordering)
# ---------------------------------------------------------------------------
qs7, g7, b7 = np.array([0.8, 0.2]), np.array([0.7, 0.1]), np.array([0.1, 0.9])
m7 = 0.5
priv_rand, priv_sel = m7 * g7.mean(), 0.5 * g7[0]
bor_rand, bor_sel = m7 * b7.mean(), 0.5 * b7[0]
assert np.isclose(priv_sel - priv_rand, 0.15) and np.isclose(bor_sel - bor_rand, -0.20)
assert np.isclose((priv_sel + bor_sel) - (priv_rand + bor_rand), -0.05)
planner = 0.5 * max(g7 + b7)
assert planner >= priv_rand + bor_rand
report["prop_welfare"] = {"private_gain": 0.15, "borrower_gain": -0.20, "total_gain": -0.05,
                          "planner_total_with_signal": planner,
                          "total_without_signal": priv_rand + bor_rand}


# ---------------------------------------------------------------------------
# 8. Proposition 9: liquidity versus ex post compensation
# ---------------------------------------------------------------------------
n8 = 3000
g8 = rng.uniform(0.0, 1.0, n8)
l8 = rng.uniform(0.2, 1.0, n8)
K8 = 0.25 * l8.sum() / n8


def allocate(gv, lv, K):
    lp = linprog(-gv / n8, A_ub=[lv / n8], b_ub=[K], bounds=(0, 1), method="highs")
    assert lp.success
    return lp.x


x_base = allocate(g8, l8, K8)
x_s = allocate(g8 + 0.3, l8, K8)
x_prop = allocate(g8 + 0.3 * l8, l8, K8)
x_liq = allocate(g8, l8, 1.2 * K8)
cash = lambda xx: float(l8 @ xx / n8)
count = lambda xx: float(xx.sum() / n8)
assert abs(cash(x_base) - K8) < 1e-9 and abs(cash(x_s) - K8) < 1e-9
assert count(x_s) >= count(x_base) - 1e-9
assert np.max(np.abs(x_prop - x_base)) < 1e-6
assert count(x_liq) > count(x_base)
# homogeneous cash need: an ex post payment leaves the number of actions unchanged
x_h = allocate(g8, np.full(n8, 0.5), 0.1)
x_hs = allocate(g8 + 0.3, np.full(n8, 0.5), 0.1)
assert abs(count(x_h) - count(x_hs)) < 1e-9
report["prop_policy"] = {
    "heterogeneous": {"cash_base": cash(x_base), "cash_flat_payment": cash(x_s),
                      "count_base": count(x_base), "count_flat_payment": count(x_s),
                      "proportional_payment_changes_selection": False,
                      "count_after_20pct_liquidity": count(x_liq)},
    "homogeneous": {"count_base": count(x_h), "count_flat_payment": count(x_hs)}}

# elastic funding: quadratic external finance cost Phi(B) = phi B^2 / 2
phi, Kint, kk = 4.0, 0.05, 0.3
qU = (np.arange(20000) + 0.5) / 20000


def elastic(s=0.0, dK=0.0):
    def foc(mm):
        qm = 1 - mm
        B = max(mm * kk - Kint - dK, 0.0)
        return (1.2 * qm - 0.2 + s) - kk * phi * B
    mm = brentq(foc, 1e-9, 1 - 1e-9)
    return mm, max(mm * kk - Kint - dK, 0.0)


m_e0, B_e0 = elastic()
m_es, B_es = elastic(s=0.05)
target = m_es
dK_equiv = brentq(lambda d: elastic(dK=d)[0] - target, 0.0, 0.2)
m_el, B_el = elastic(dK=dK_equiv)
assert abs(m_el - m_es) < 1e-8 and B_es > B_e0 > B_el
report["prop_policy"]["elastic"] = {"m_base": m_e0, "B_base": B_e0,
                                    "m_payment": m_es, "B_payment": B_es,
                                    "m_injection": m_el, "B_injection": B_el,
                                    "equivalent_injection": dK_equiv}

(ROOT / "data" / "model_verification.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
print(json.dumps(report, indent=2, default=float))
