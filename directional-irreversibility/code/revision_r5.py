"""the journal submission stage, round 5.

Adds a corollary for balanced-panel averages (Theorem 1(iii) with fixed N), a
simulation of cross-sectional pooling with and without conditioning on an
observed common factor, and a heavy-tail size study. Reads
fan_pelger_results.json (fan_pelger_experiments.py, seed 20260921).
No identified shock, asset-price data, or journal input.
"""
from pathlib import Path
import json

S = Path(__file__).resolve().parent
P = S.parent
T = S / 'main.tex'
tex = T.read_text()
d = json.load(open(P / 'fan_pelger_results.json'))


def rep(old, new, count=1):
    global tex
    n = tex.count(old)
    assert n == count, (n, old[:100])
    tex = tex.replace(old, new)


pan = {(r['theta'], r['N'], r['mode']): r for r in d['panel']}
ht = {(r['design'], str(r['df'])): r['reject'] for r in d['heavy_tails']}
reps = d['panel'][0]['reps']
P1u, P16u = 100 * pan[(0.35, 1, 'unadjusted')]['reject'], 100 * pan[(0.35, 16, 'unadjusted')]['reject']
P1a, P16a = 100 * pan[(0.35, 1, 'factor_adjusted')]['reject'], 100 * pan[(0.35, 16, 'factor_adjusted')]['reject']
nullmax = 100 * max(pan[(0.0, N, md)]['reject'] for N in [1, 4, 16] for md in ['unadjusted', 'factor_adjusted', 'defactored_known'])
t3 = 100 * ht[('regular', '3')]
K1, K16 = 100 * pan[(0.35, 1, 'defactored_known')]['reject'], 100 * pan[(0.35, 16, 'defactored_known')]['reject']
E16 = 100 * pan[(0.35, 16, 'defactored_estimated')]['reject']
E4null = 100 * max(pan[(0.0, N, 'defactored_estimated')]['reject'] for N in [1, 4, 16])
modes = [('unadjusted', 'Unconditioned'), ('factor_adjusted', 'Factor in regressor kernels'),
         ('defactored_known', 'Defactored, known loading'), ('defactored_estimated', 'Defactored, estimated loading')]

rows = ''
for key, lab in modes:
    rows += lab + ''.join(f" & {100*pan[(0.0,N,key)]['reject']:.1f}" for N in [1, 4, 16]) + ''.join(f" & {100*pan[(0.35,N,key)]['reject']:.1f}" for N in [1, 4, 16]) + "\\\\\n"

hrows = ''
for lab, key in [('Gaussian', 'Gaussian'), ('Student $t_5$', '5'), ('Student $t_3$', '3')]:
    hrows += f"{lab} & {100*ht[('regular', key)]:.1f} & {100*ht[('double', key)]:.1f}\\\\\n"

SUB = rf"""
\subsection{{Balanced panels, factor conditioning, and heavy tails}}\label{{sec:panel}}
Financial applications often observe many outcomes on one calendar. Theorem~\ref{{thm:main}} already covers any fixed signed contrast of squared operator norms, which gives the following.

\begin{{corollary}}[Balanced-panel averages]\label{{cor:panel}}
Let $N$ units share one calendar, one driver, and one training and evaluation split, and let each unit $i$ have its own fitted forward and reverse representations. Under the conditions of Theorem~\ref{{thm:main}} for the stacked vector of $2N$ operators, the weighted average $\bar D=\sum_{{i=1}}^N w_i D_i$ with fixed weights is tested by the maximum of its quadratic and linear bootstrap p-values, computed from common evaluation and training paths for all units. The test has pointwise asymptotic level at most $\alpha$ on the covered null strata for fixed $N$.
\end{{corollary}}

The corollary is Theorem~\ref{{thm:main}}(iii) with $a_d=\pm w_i$; common paths retain the dependence created by the shared driver and by common shocks (Appendix~\ref{{app:training}}). It does not cover $N$ growing with the sample.

Table~\ref{{tab:panel}} studies pooling in a design with a common driver $X_t$ and $N$ outcomes $Y_{{i,t+1}}=\theta(X_t^2-1)+f_{{t+1}}+e_{{i,t+1}}$. Here $f$ is an observed common factor independent of $X$, and every unit has positive DII when $\theta>0$. The table compares four ways of treating the factor, with {reps} samples per cell.

The unconditioned equal-weight average gains only moderately from pooling: power at $\theta=0.35$ goes from {P1u:.1f} percent for one unit to {P16u:.1f} percent for 16. The shared factor makes the unit contrasts strongly dependent. Conditioning on the factor inside the regressor kernels is worse still ({P1a:.1f} and {P16a:.1f} percent), because the kernel must detect dependence on a higher-dimensional regressor.

The effective treatment follows the logic of factor-adjusted testing (Fan et al.\ 2019): remove the factor from the outcome and test the defactored outcome. With a prespecified loading, as in market-adjusted returns, the defactored outcome is a fixed transformation. Theorem~\ref{{thm:main}} and Corollary~\ref{{cor:panel}} then apply unchanged. Power rises from {K1:.1f} percent for one unit to {K16:.1f} percent for 16, and pooling becomes highly effective.

With a loading estimated on the training sample, the forward direction treats the factor as an additional coefficient of the fitted mean, and its training influence is propagated like any other coefficient. In the reverse direction the estimated loading enters the regressor kernel and its uncertainty is not propagated, so this version is supported by simulation only. Its power is {E16:.1f} percent at $N=16$. Its largest null rejection, {E4null:.1f} percent, is above the known-loading version at the double-independence null, where the procedure is otherwise conservative. Extending Theorem~\ref{{thm:main}} to estimated loadings in both directions, and to $N$ growing with the sample, are the natural next steps. With factors removed, the cross-section can substitute for time-series length; without removal, it largely cannot.

\begin{{table}}[htbp]\centering\small
\caption{{Balanced-panel averages: rejection percentages at nominal 5\%}}\label{{tab:panel}}
\begin{{tabular}}{{lcccccc}}\toprule
& \multicolumn{{3}}{{c}}{{Null, $\theta=0$}} & \multicolumn{{3}}{{c}}{{Positive DII, $\theta=0.35$}}\\\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}
Treatment of the factor & $N=1$ & $N=4$ & $N=16$ & $N=1$ & $N=4$ & $N=16$\\\midrule
{rows}\bottomrule\end{{tabular}}
\par\vspace{{4pt}}\raggedright\noindent {reps} samples per cell; 204 training, 39-row gap, 120 evaluation rows; total-degree-two means; joint procedure with 399 draws and common paths across units. The null is a double-independence law, at which the procedure is conservative.
\end{{table}}

Heavy tails are the rule in financial data, and Theorem~\ref{{thm:main}} assumes eighth moments. Table~\ref{{tab:heavy}} reruns the equal-positive and double-independence null designs of Section~\ref{{sec:sim}} with unit-variance Student-$t$ innovations. Rejection stays at or below 5 percent, including $t_3$ ({t3:.1f} percent at the equal-positive null), which violates the moment condition. The bounded Gaussian feature map is the likely reason. This is design-specific evidence outside the theorem, not a guarantee.

\begin{{table}}[htbp]\centering\small
\caption{{Heavy-tailed innovations: null rejection percentages at nominal 5\%}}\label{{tab:heavy}}
\begin{{tabular}}{{lcc}}\toprule
Innovations & Equal-positive null & Double-independence null\\\midrule
{hrows}\bottomrule\end{{tabular}}
\par\vspace{{4pt}}\raggedright\noindent 300 samples per cell; same sizes and procedure as Table~\ref{{tab:panel}}; means quadratic in the continuous coordinate, linear in the state, with their interaction.
\end{{table}}
"""
anchor = "\\section{A reproducible financial application}\\label{sec:application}"
rep("\\newtheorem{theorem}{Theorem}\n", "\\newtheorem{theorem}{Theorem}\n\\newtheorem{corollary}{Corollary}\n")
rep(anchor, SUB.strip() + "\n\n" + anchor)

rep("Toy examples with known answers show each proposition at work (Figure~\\ref{fig:toys}).",
    f"Toy examples with known answers show each proposition at work (Figure~\\ref{{fig:toys}}). In a balanced panel with a common factor, removing the factor and averaging 16 units raises power from {P1u:.0f} to {K16:.0f} percent, and size is maintained with Student-$t_3$ innovations (Section~\\ref{{sec:panel}}).")

rep("Three design conclusions follow.",
    "Three design conclusions follow, together with a fourth for panels: remove common factors from outcomes rather than conditioning on them inside the kernel, after which pooling units raises power sharply.")

rep("\\bibitem{fanyao}",
    "\\bibitem{farm} Fan J, Ke Y, Sun Q, Zhou WX (2019) FarmTest: Factor-adjusted robust multiple testing with approximate false discovery control. \\emph{J. Amer. Statist. Assoc.} 114(528):1880--1893.\n\\bibitem{fanyao}")

rep("Component bounds also limit the error from reusing fitted residuals in scenario-based decisions.",
    "In panels, removing common factors before pooling raises power sharply.")
a_ = tex[tex.index('\\begin{abstract}') + 16:tex.index('\\end{abstract}')]
assert len(a_.split()) <= 250, len(a_.split())
T.write_text(tex)
cl = (S / 'cover_letter.md').read_text()
old = "The paper's joint procedure (Theorem 1) restores size while carrying the uncertainty from fitted models and their units."
assert old in cl
cl = cl.replace(old, old + f" A corollary extends the test to balanced-panel averages. Following the logic of factor-adjusted testing, removing a common factor before pooling 16 units raises power from {P1u:.0f} to {K16:.0f} percent, whereas conditioning on the factor inside the kernel lowers it. Size is maintained with Student-t(3) innovations.")
st = cl.index('Abstract:\n') + len('Abstract:\n'); en = cl.index('\n\nSuggested Associate Editors')
cl = cl[:st] + a_.strip().replace('--', '–') + cl[en:]
(S / 'cover_letter.md').write_text(cl)
print('revision_r5: applied')
