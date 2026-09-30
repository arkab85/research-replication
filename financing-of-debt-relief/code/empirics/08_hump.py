"""Figure 6 and the macros for Section 6.3: the level response traced against the funded share.

Proposition 5(i) says the level response of exercise to the ranking is concave in the funded
share, vanishes at both extremes, and peaks where the funded share equals the share of loans
ranked above the cell mean. This script traces that profile directly from the decision cells
and tests the theory's one-parameter shape against a flat alternative, bootstrapping over
issuers. Inputs are the bundled cell aggregates in data/empirics/; no loan-level data are used.

    python code/empirics/08_hump.py
"""
import csv, collections, json, os, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data", "empirics")
MIN_N = 25          # cells below this are too noisy to carry a within-cell covariance
B_DRAWS = 2000
SEED = 11


def load(fname, typ):
    """Cells with interior funded shares and positive index variance."""
    out, drop0, drop1 = [], 0, 0
    with open(os.path.join(DATA, fname)) as fh:
        for r in csv.DictReader(fh):
            n, nx, v = float(r["n"]), float(r["nx"]), float(r["var"])
            if n < MIN_N or v <= 0:
                continue
            if nx == 0:
                drop0 += 1
                continue
            if nx == n:
                drop1 += 1
                continue
            out.append(dict(iss=r["issuer"], ym=r["ym"], per=r["period"], typ=typ,
                            m=nx / n, cov=float(r["cov"]), var=v, n=n))
    return out, drop0, drop1


def fit(cells):
    """Theory shape tau(m) = g*m(1-m) against the flat alternative tau(m) = t.

    Under the model the within-cell covariance of exercise with the ranking is
    cov_c = var_c * tau(m_c), so both are least squares through the origin on cov_c.
    """
    xa = np.array([c["var"] * c["m"] * (1.0 - c["m"]) for c in cells])
    xb = np.array([c["var"] for c in cells])
    y = np.array([c["cov"] for c in cells])
    g = float(xa @ y / (xa @ xa))
    t = float(xb @ y / (xb @ xb))
    sy = float(y @ y)
    r2a = 1.0 - float(((y - g * xa) ** 2).sum()) / sy
    r2b = 1.0 - float(((y - t * xb) ** 2).sum()) / sy
    return g, t, r2a, r2b


def profile(cells, edges):
    """Level response within bins of the funded share: sum(cov)/sum(var)."""
    num = collections.defaultdict(float)
    den = collections.defaultdict(float)
    cnt = collections.Counter()
    msum = collections.defaultdict(float)
    for c in cells:
        i = min(int(c["m"] * (len(edges) - 1)), len(edges) - 2)
        num[i] += c["cov"]; den[i] += c["var"]; cnt[i] += 1; msum[i] += c["m"]
    rows = []
    for i in sorted(cnt):
        rows.append((msum[i] / cnt[i], num[i] / den[i], cnt[i]))
    return rows


def main():
    dep, d0, d1 = load("cells_depository.csv", "dep")
    nb, n0, n1 = load("cells_nonbank.csv", "nb")
    cells = dep + nb
    edges = np.linspace(0, 1, 11)

    g, t, r2a, r2b = fit(cells)

    by = collections.defaultdict(list)
    for c in cells:
        by[c["iss"]].append(c)
    keys = list(by)
    rng = random.Random(SEED)
    gs, dr2 = [], []
    for _ in range(B_DRAWS):
        samp = []
        for _ in keys:
            samp += by[rng.choice(keys)]
        gg, tt, ra, rb = fit(samp)
        gs.append(gg); dr2.append(ra - rb)
    gs.sort(); dr2.sort()
    g_se = float(np.std(gs, ddof=1))
    q = lambda a, p: a[int(p * len(a))]

    g_nb, t_nb, r2a_nb, r2b_nb = fit(nb)
    g_dep, t_dep, r2a_dep, r2b_dep = fit(dep)
    gs_nb = []
    by_nb = collections.defaultdict(list)
    for c in nb:
        by_nb[c["iss"]].append(c)
    knb = list(by_nb)
    rng2 = random.Random(SEED + 1)
    for _ in range(B_DRAWS):
        samp = []
        for _ in knb:
            samp += by_nb[rng2.choice(knb)]
        gs_nb.append(fit(samp)[0])
    gs_nb.sort()

    res = dict(
        gamma_nb=g_nb, gamma_nb_se=float(np.std(gs_nb, ddof=1)),
        r2_theory_nb=r2a_nb, r2_flat_nb=r2b_nb,
        gamma_dep=g_dep, r2_theory_dep=r2a_dep, r2_flat_dep=r2b_dep,
        cells=len(cells), cells_dep=len(dep), cells_nb=len(nb), issuers=len(keys),
        dropped_m0=d0 + n0, dropped_m1=d1 + n1, min_n=MIN_N, draws=B_DRAWS,
        gamma=g, gamma_se=g_se, gamma_lo=q(gs, .025), gamma_hi=q(gs, .975),
        peak_tau=g / 4.0, flat_tau=t, r2_theory=r2a, r2_flat=r2b,
        dr2=r2a - r2b, dr2_lo=q(dr2, .025), dr2_hi=q(dr2, .975),
        dr2_share_pos=sum(1 for x in dr2 if x > 0) / len(dr2),
        profile_pooled=profile(cells, edges),
        profile_dep=profile(dep, edges), profile_nb=profile(nb, edges),
    )
    with open(os.path.join(DATA, "hump.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    # ---- macros -------------------------------------------------------------
    def f2(x): return ("%.2f" % x)
    def f3(x): return ("%.3f" % x)
    lines = [
        "%% Section 6.3, generated by code/empirics/08_hump.py; do not edit.",
        "\\newcommand{\\eHumpCells}{%s}" % format(len(cells), ","),
        "\\newcommand{\\eHumpIssuers}{%d}" % len(keys),
        "\\newcommand{\\eHumpDropZero}{%s}" % format(d0 + n0, ","),
        "\\newcommand{\\eHumpMinN}{%d}" % MIN_N,
        "\\newcommand{\\eHumpGamma}{%s}" % f2(g),
        "\\newcommand{\\eHumpGammaSE}{%s}" % f2(g_se),
        "\\newcommand{\\eHumpGammaLo}{%s}" % f2(q(gs, .025)),
        "\\newcommand{\\eHumpGammaHi}{%s}" % f2(q(gs, .975)),
        "\\newcommand{\\eHumpPeak}{%s}" % f3(g / 4.0),
        "\\newcommand{\\eHumpRtwoTheory}{%s}" % f2(r2a),
        "\\newcommand{\\eHumpRtwoFlat}{%s}" % f2(r2b),
        "\\newcommand{\\eHumpDrTwoLo}{%s}" % f2(q(dr2, .025)),
        "\\newcommand{\\eHumpDrTwoHi}{%s}" % f2(q(dr2, .975)),
        "\\newcommand{\\eHumpDraws}{%s}" % format(B_DRAWS, ","),
        "\\newcommand{\\eHumpGammaNb}{%s}" % f2(g_nb),
        "\\newcommand{\\eHumpGammaNbSE}{%s}" % f2(float(np.std(gs_nb, ddof=1))),
        "\\newcommand{\\eHumpPeakNb}{%s}" % f3(g_nb / 4.0),
        "\\newcommand{\\eHumpRtwoNb}{%s}" % f2(r2a_nb),
        "\\newcommand{\\eHumpRtwoNbFlat}{%s}" % f2(r2b_nb),
        "\\newcommand{\\eHumpGammaDep}{%s}" % f2(g_dep),
        "\\newcommand{\\eHumpCellsNb}{%s}" % format(len(nb), ","),
        "\\newcommand{\\eHumpCellsDep}{%s}" % format(len(dep), ","),
    ]
    with open(os.path.join(ROOT, "tables", "hump_macros.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
    grid = np.linspace(0, 1, 200)

    ax[0].plot(grid, g * grid * (1 - grid), color="0.25", lw=1.6,
               label=r"theory: $\tau(m)=\gamma\,m(1-m)$")
    for a in ax:
        a.axvline(0.5, color="0.7", lw=.8, ls=":")
        a.set_xlabel("funded share of the decision cell, $m$")
        a.set_xlim(0, 1)
        a.spines[["top", "right"]].set_visible(False)

    pm, pt, pc = zip(*res["profile_pooled"])
    ax[0].scatter(pm, pt, s=[max(14, min(150, c / 2.5)) for c in pc],
                  facecolor="#1f6f6a", edgecolor="white", zorder=3, label="all cells")
    ax[0].set_ylabel("level response of exercise to the ranking")
    ax[0].set_title("A. Pooled", loc="left", fontsize=11)
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")

    for cells_t, lab, col, mk, gt in [(res["profile_nb"], "nonbanks", "#1f6f6a", "o", g_nb),
                                      (res["profile_dep"], "depositories", "#8c3a4f", "s", g_dep)]:
        m_, t_, c_ = zip(*cells_t)
        ax[1].scatter(m_, t_, s=[max(14, min(150, c / 2.5)) for c in c_], marker=mk,
                      facecolor=col, edgecolor="white", zorder=3, label=lab)
        ax[1].plot(grid, gt * grid * (1 - grid), color=col, lw=1.3, alpha=.8)
    ax[1].set_title("B. By issuer type", loc="left", fontsize=11)
    ax[1].legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "fig6_hump.pdf"))

    print("pooled gamma = %.3f (s.e. %.3f), peak tau = %.3f" % (g, g_se, g / 4))
    print("nonbanks gamma = %.3f, R2 %.2f vs flat %.2f" % (g_nb, r2a_nb, r2b_nb))
    print("depositories gamma = %.3f, R2 %.2f vs flat %.2f" % (g_dep, r2a_dep, r2b_dep))
    print("R2 theory %.3f vs flat %.3f; difference positive in %.1f%% of draws"
          % (r2a, r2b, 100 * res["dr2_share_pos"]))
    print("wrote figures/fig6_hump.pdf, tables/hump_macros.tex, data/empirics/hump.json")


if __name__ == "__main__":
    main()
