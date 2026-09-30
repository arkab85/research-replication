"""the journal submission stage, round 7.

Adds (i) a proposition extending Theorem 1 to training-estimated factor loadings in
both directions, with a proof sketch and exact numerical verification
(verify_fd_nuisance.py), and simulation evidence (fd_experiments.py); and (ii) a
bandwidth-multiplier sensitivity study. Reads fd_experiments_results.json and
fd_nuisance_verification.json. Simulation only; no journal inputs.
"""
from pathlib import Path
import json

S = Path(__file__).resolve().parent
P = S.parent
T = S / 'main.tex'
tex = T.read_text()
d = json.load(open(P / 'fd_experiments_results.json'))
v = json.load(open(P / 'fd_nuisance_verification.json'))


def rep(old, new, count=1):
    global tex
    n = tex.count(old)
    assert n == count, (n, old[:100])
    tex = tex.replace(old, new)


L = {(r['theta'], r['N']): r for r in d['loading']}
W = {(r['c'], r['design']): r for r in d['bandwidth']}
Lnull = 100 * max(L[(0.0, N)]['reject'] for N in [1, 4, 16])
L16 = 100 * L[(0.35, 16)]['reject']; L4 = 100 * L[(0.35, 4)]['reject']; L1 = 100 * L[(0.35, 1)]['reject']
maxerr = max(v['defactored_forward_refit_rel_error'], v['defactored_reverse_refit_rel_error'])

# ---------------------------------------------------------------- replace the estimated-loading paragraph
a = tex.index("With a loading estimated on the training sample, the forward direction treats the factor")
b = tex.index("\\begin{table}[htbp]\\centering\\small\n\\caption{Balanced-panel averages")
newpar = (
    "With a loading estimated on the training sample, its uncertainty enters both directions: as a coefficient of the forward mean, "
    "and through the defactored regressor $Y-\\hat\\lambda f$ in the reverse direction. Proposition~\\ref{prop:loading} extends Theorem~\\ref{thm:main} "
    "to this case by appending the loading's training influence to the nuisance vector. Table~\\ref{tab:panel} reports two versions. "
    "The row labeled ``estimated loading'' holds the reverse-direction loading fixed. The row labeled ``estimated, propagated'' implements "
    f"Proposition~\\ref{{prop:loading}}. Propagation lowers the largest null rejection to {Lnull:.1f} percent, with power of {L1:.1f}, {L4:.1f}, "
    f"and {L16:.1f} percent for 1, 4, and 16 units. With factors removed, the cross-section can substitute for time-series length; without removal, it largely cannot.\n\n"
    "\\begin{proposition}[Estimated factor loadings]\\label{prop:loading}\n"
    "Suppose the conditions of Theorem~\\ref{thm:main} hold with an observed factor $f$ that has finite eighth moments. Let the loading "
    "$\\hat\\lambda$ be the training least-squares coefficient on $f$ in the forward mean, and let the reverse representation use the "
    "standardized regressor $Y-\\hat\\lambda f$ with its own training standardization and fit. Then the conclusions of Theorem~\\ref{thm:main} "
    "hold with $Z_\\eta$ augmented by the training influence of $\\hat\\lambda$, and with $\\mathcal J$ augmented by the derivative of the "
    "reverse covariance operator with respect to the loading, the reverse standardization and fit being re-solved at each loading.\n"
    "\\end{proposition}\n"
    "Appendix~\\ref{app:loading} gives the proof sketch and the implementation. The derivative is computed by finite differences of the "
    f"feature map and agrees with a full weighted refit to a relative error of {maxerr:.0e}.\n\n")
tex = tex[:a] + newpar + tex[b:]

rep("The table compares four ways of treating the factor", "The table compares five ways of treating the factor")
# add propagated row to Table 5
prow = "Defactored, estimated, propagated" + ''.join(f" & {100*L[(0.0,N)]['reject']:.1f}" for N in [1, 4, 16]) + ''.join(f" & {100*L[(0.35,N)]['reject']:.1f}" for N in [1, 4, 16]) + "\\\\\n"
i = tex.index("Defactored, estimated loading")
j = tex.index("\\\\\n", i) + 3
tex = tex[:j] + prow + tex[j:]
rep("200 samples per cell; 204 training, 39-row gap, 120 evaluation rows;",
    "200 samples per cell (100 for the propagated version at $N=16$); 204 training, 39-row gap, 120 evaluation rows;")

# ---------------------------------------------------------------- bandwidth sensitivity
brows = ''.join(f"{c:g} & {100*W[(c,'regular')]['reject']:.1f} & {100*W[(c,'double')]['reject']:.1f} & {100*W[(c,'quadratic')]['reject']:.1f}\\\\\n" for c in [0.5, 1.0, 2.0])
BW = rf"""
The bandwidth is the natural next question for a kernel method. Throughout, both kernels use bandwidth one after training-sample standardization. The bandwidth is therefore data-driven through the estimated scales, and Theorem~\ref{{thm:main}} propagates their uncertainty. A different multiple $c$ of those scales defines a different fixed-bandwidth target, to which the theorem applies unchanged. Table~\ref{{tab:bandwidth}} reports the procedure at $c\in\{{0.5,1,2\}}$, using the same finite-difference implementation, which is verified against full weighted refits (Appendix~\ref{{app:loading}}). Size is controlled at each multiple. Power against the quadratic exposure changes with $c$ and is highest at the default $c=1$ among the three multiples. This is why the multiple should be fixed before estimation rather than chosen after seeing p-values. Bandwidths that shrink with the sample, or that are chosen to maximize a test statistic, are not covered.

\begin{{table}}[htbp]\centering\small
\caption{{Bandwidth multiples: rejection percentages at nominal 5\%}}\label{{tab:bandwidth}}
\begin{{tabular}}{{cccc}}\toprule
Multiple $c$ & Equal-positive null & Double-independence null & Quadratic exposure\\\midrule
{brows}\bottomrule\end{{tabular}}
\par\vspace{{4pt}}\raggedright\noindent 300 samples per null cell and 200 per power cell; 204 training, 39-row gap, 120 evaluation rows; joint procedure with 399 draws. The quadratic design is $Y=0.5(X^2-1)+\varepsilon$ with regressor $X$.
\end{{table}}
"""
anchor = "Heavy tails are the rule in financial data"
rep(anchor, BW.strip() + "\n\n" + anchor)

# ---------------------------------------------------------------- companion appendix
EC = rf"""
\section{{Estimated loadings and bandwidth multiples: proof sketch and implementation}}\label{{app:loading}}
\paragraph{{Proof sketch for Proposition~\ref{{prop:loading}}.}} The proof of Theorem~\ref{{thm:joint}} uses the nuisance parameters in two ways. First, the training estimators are asymptotically linear with finite-memory influence functions. Second, the evaluation covariance operator has a second-order expansion in the nuisance, whose first-order term and mixed second derivative converge in probability. The loading $\hat\lambda$ is a least-squares coefficient, so it is asymptotically linear with influence $s_Y\,\psi_f$, where $\psi_f$ is the forward coefficient influence for $f$ in standardized units. The scale effects of $s_Y$ cancel in the raw-unit loading. In the reverse representation, the standardized regressor, its training center and scale, and the reverse coefficients are smooth functions of the loading, because the design matrix is nonsingular. The Gaussian residual and regressor kernels are smooth in these arguments, and the eighth-moment conditions give the required domination. The expansion therefore holds with the loading appended to the nuisance vector. The chain rule assigns the loading's total effect, including the re-solved standardization and fit, to the new derivative; the direct influences of the other nuisance parameters are unchanged. The remaining steps of the proof of Theorem~\ref{{thm:joint}} apply without change. This is a sketch: the domination arguments follow Appendix~\ref{{app:preprocessing}}.

\paragraph{{Implementation and verification.}} \texttt{{dii\_fd\_nuisance.py}} computes the operator matrices by central finite differences of a feature map. For the loading, the feature map re-solves the reverse standardization and fit at each perturbed loading. \texttt{{verify\_fd\_nuisance.py}} checks three things. On the standard estimand it reproduces the analytic implementation, with maximum errors {v['standard_max_error_M']:.0e} and {v['standard_max_error_J']:.0e}. For the defactored pair, the quadratic form along a random training-weight direction matches a full weighted refit of the forward loading and the reverse standardization and fit; the relative errors are {v['defactored_forward_refit_rel_error']:.0e} (forward) and {v['defactored_reverse_refit_rel_error']:.0e} (reverse). For bandwidth multiples 0.5 and 2, the same refit check gives relative errors {v['bandwidth_0.5_refit_rel_error']:.0e} and {v['bandwidth_2.0_refit_rel_error']:.0e}. The experiments are in \texttt{{fd\_experiments.py}} (seed 20260922).

\paragraph{{What remains open.}} Three extensions are not covered: numbers of units growing with the sample, bandwidths that shrink with the sample or are tuned on the evaluation data, and uniformity near the null boundary.
"""
rep("\\end{document}", EC.strip() + "\n\\end{document}")

rep("Extending Theorem~\\ref{thm:main} to estimated loadings in both directions, and to $N$ growing with the sample, are the natural next steps.", "") if "Extending Theorem~\\ref{thm:main} to estimated loadings in both directions" in tex else None

T.write_text(tex)
cl = (S / 'cover_letter.md').read_text()
old = "whereas conditioning on the factor inside the kernel lowers it."
if old in cl:
    cl = cl.replace(old, old + " A proposition extends the joint inference to training-estimated factor loadings, verified numerically against full refits, and a bandwidth study shows size control at several fixed bandwidth multiples.")
(S / 'cover_letter.md').write_text(cl)
print('revision_r6: applied')
