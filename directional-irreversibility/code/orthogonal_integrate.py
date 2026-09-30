from pathlib import Path
import json,re
S=Path(__file__).resolve().parent;P=S.parent;s=(S/'main.tex').read_text()
marker=r'\section{An optional decision-use illustration}'
assert marker in s;s=s.replace(marker,(S/'orthogonal_theory.tex').read_text()+'\n'+marker)
abstract="""A shock can move an outcome, change its risk, or generate an asymmetric relationship between opposing dynamic models. This paper develops the Directional Irreversibility Index (DII) to separate directional structure from mean responses and general dependence. Financial benchmarks show why positive DII can coexist with a zero linear response and why a persistent footprint need not imply delayed information absorption. Inference combines estimated residuals, dependent horizon comparisons, and null configurations with different convergence rates. We develop joint block approximations under explicit persistent-dependence conditions and simultaneous component bounds that distinguish asymmetry from approximate model validity. An orthogonal operator correction preserves the original DII target while replacing a first-order regression-error requirement with a second-order product-rate condition. This permits comparable training and evaluation sizes under correctly specified parametric mean models and consistent derivative learning. Reproducible experiments document calibration failures and substantial finite-sample costs: neither the component bounds nor the implemented correction demonstrates a power advantage. A fixed four-market application has no primary directional rejection or demonstrated positive nonlinear prediction gain. The framework supplies scoped inference and interpretation for directional economic dynamics, while keeping statistical footprints, model adequacy, and financial mechanisms distinct."""
s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',s,flags=re.S)
main=r'''
\subsection{Reducing sensitivity to fitted conditional means}\label{sec:ortho}
The uncorrected estimator's sufficient condition, $\sqrt n r\to0$, can require an unrealistically large training sample. The paper therefore develops an alternative estimator of the same DII target. Let $u=Y-m(Z)$ and $g(Z)=\E[\phi'(u)\mid Z]$. Replace the residual feature by
\[
T=\phi(u)-g(Z)u.
\]
Because the conditional-mean residual satisfies $\E[u\mid Z]=0$, this correction leaves its covariance with the regressor feature unchanged. The estimand remains the original HSIC component, rather than independence of a newly transformed residual. Estimated versions of both nuisances are learned using training data only.

Appendix~\ref{app:orthogonal} proves that the population covariance error is bounded by a constant times $r^2+rs$, where $r$ is the conditional-mean estimation error and $s$ is the derivative-learner error. Under the stated moment and joint-dependence conditions, $\sqrt n(r^2+rs)\to0$ is sufficient for oracle-equivalent joint block inference using the corrected influence process. When a correctly specified parametric mean model has $r=O_p(m^{-1/2})$, a training size proportional to evaluation size and a consistent derivative learner suffice. Ordinary plug-in inference has the stronger first-order requirement. This is a theoretical relaxation with substantive assumptions, not certification of an arbitrary fitted regression.

The new score follows established orthogonal-estimation principles; the contribution is its construction for the DII operator and joint persistent-process inference. The implementation computes exact Gaussian-RKHS Gram matrices. Estimating the additional nuisance can nevertheless be costly in small samples. The matched experiment below reports lower power for the implemented correction. We therefore distinguish a relaxed asymptotic learning requirement from a practical superiority claim.

'''
s=s.replace(r'\section{Simulation evidence:',main+r'\section{Simulation evidence:')
s=s.replace('The paper has three connected contributions: a DII interpretation that separates directional structure from mean and risk effects; feasible joint inference under explicit persistent-dependence conditions; and simultaneous component bounds that distinguish direction, approximate model validity, and horizon persistence.','The paper has three connected contributions: a DII interpretation separating directional structure from mean and risk effects; joint inference under explicit persistent-dependence conditions, including an orthogonal correction that relaxes the regression-error requirement; and simultaneous component bounds distinguishing direction, approximate model validity, and horizon persistence.')
s=s.replace('The component bounds cover certain cancellation configurations excluded by the contrast test, at a potentially substantial cost in power.','The component bounds cover certain cancellation configurations excluded by the contrast test. Both the bounds and the implemented orthogonal correction have substantial small-sample costs, which are measured rather than assumed away.')
s=s.replace('Appendices give the full residual and bootstrap arguments, including the persistent-dependence extension and simultaneous bounds, followed by an optional decision-loss application.','The electronic companion gives the full residual and bootstrap arguments, including persistent dependence, simultaneous bounds, and orthogonal estimation, followed by an optional decision-loss application.')
s=s.replace('The main methodological contribution is inference for this contrast after fitting the two models.','The main methodological contribution is inference for this contrast after fitting the two models, including a correction that reduces its sensitivity to regression estimation error.')
s=s.replace('Nor do the 204 training observations establish the sufficient asymptotic first-stage rate.','Nor do the 204 training observations establish the sufficient asymptotic first-stage rate. The orthogonal extension relaxes that requirement for its own corrected estimator, but does not verify correct specification or derivative-learning quality in this application.')
s=s.replace('The added nonoverlapping-block route permits persistent dynamics under explicit absolute-regularity conditions.','The added nonoverlapping-block route permits persistent dynamics under explicit absolute-regularity conditions. Orthogonal estimation further replaces first-order sensitivity to fitted means with a second-order product-rate condition, under the stated moment and learning assumptions.')
results=json.loads((P/'orthogonal_diagnostic_results.json').read_text());assert results['datasets']==600
rows=[]
for rho in (0.,.6):
 for theta in (0.,.5,1.):
  r={x['method']:x for x in results['summary'] if x['rho']==rho and x['theta']==theta}
  rows.append(f"{rho:.1f} & {theta:.1f} & {100*r['raw']['rate']:.0f} & {100*r['orthogonal']['rate']:.0f}"+r' \\')
audit=r'''
\subsection{A matched audit of orthogonal estimation}
A second post-results diagnostic isolates the correction using the same persistent-exposure family. Both estimators use identical fitted polynomial means, training observations, evaluation observations, four-row nonoverlapping blocks, and bootstrap draws. The fixed population scales are $1$ for the shock and $\sqrt{1+2\theta^2}$ for the outcome; no sample-derived scaling is used. Training and evaluation sizes are again 204 and 120. The gap is $\lceil8\log(n+1)\rceil=39$, following a rule that diverges with sample size. There are 100 replications per cell and 199 bootstrap draws. All six cells were fixed before running this diagnostic.

\begin{table}[htbp]\centering
\caption{Matched nonoverlapping-block audit: intersection rejection percentages}
\begin{tabular}{rrrr}\toprule
$\rho$ & $\theta$ & Uncorrected & Orthogonal\\\midrule
% ROWS
\bottomrule\end{tabular}
\par\vspace{5pt}\raggedright\noindent These are Monte Carlo frequencies, not an empirical power estimate. Cellwise Wilson intervals, seeds, full output, and exact implementation are supplied. Each procedure uses its own score covariance and the same nonoverlapping algorithm.
\end{table}

The correction is not a finite-sample power improvement in this implementation. At the strong exposure the uncorrected procedure rejects 82\% of datasets in both dependence cells; the corrected procedure rejects 14\% and 12\%. At the weaker exposure the corresponding frequencies are 17\% and 19\% versus 2\% and 3\%. These outcomes show that adding an estimated derivative can be expensive when only 204 training observations are available. They do not contradict the product-rate theorem, but they prevent a recommendation based on asymptotic rates alone.

A separate population calculation verifies the intended cancellation using smooth finite-dimensional features: halving a small regression perturbation approximately halves the uncorrected operator error and quarters the corrected error. The exact Gaussian Gram implementation agrees with an independent spectral-quadrature calculation to within $10^{-14}$. These checks validate the algebra and implementation; the power experiment evaluates a different question, namely finite-sample usefulness. Further derivative-learning improvements require their own evaluation rather than selecting a favorable variant from this experiment.

'''.replace('% ROWS','\n'.join(rows))
s=s.replace(r'\section{A reproducible financial application}',audit+r'\section{A reproducible financial application}')
f=json.loads((P/'finance/orthogonal_results.json').read_text())
assert all(sum(x['p_intersection']<=.05 for x in f[k]['comparisons'].values())==0 for k in ('raw','orthogonal'))
extra=r'''
\paragraph{Secondary correction audit.} The corrected and uncorrected nonoverlapping procedures were also applied to the same fixed 12 financial cells, using identical fitted mean models, four-row blocks, and 999 common draws. Neither has an unadjusted intersection rejection. This post-results comparison preserves the primary protocol and reports every selected cell. It provides no basis for promoting the correction as an empirically superior estimator or strengthening the financial transmission claim. Population-target preservation does not imply identical finite-sample estimates. The empirical nuisance, scaling, and dependence assumptions remain unverified.

'''
s=s.replace(r'\section{Financial transmission and the use of DII}',extra+r'\section{Financial transmission and the use of DII}')
bib=r'''
\bibitem{dml} Chernozhukov, V., D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, and J. Robins (2016a). Double/Debiased Machine Learning for Treatment and Causal Parameters. arXiv:1608.00060. \url{https://arxiv.org/abs/1608.00060}.
\bibitem{lr} Chernozhukov, V., J. C. Escanciano, H. Ichimura, W. K. Newey, and J. M. Robins (2016b). Locally Robust Semiparametric Estimation. arXiv:1608.00033. \url{https://arxiv.org/abs/1608.00033}.
'''
s=s.replace(r'\begin{thebibliography}{99}',r'\begin{thebibliography}{99}'+bib)
# Keep the main paper succinct; full proofs travel in a separate companion.
m=re.search(r'(\\begin\{thebibliography\}\{99\})(.*?)(\\end\{thebibliography\})',s,re.S)
entries=re.findall(r'\\bibitem\{[^}]+\}.*?(?=\\bibitem|$)',m.group(2),re.S);entries.sort(key=lambda x:x.split('}',1)[1].strip())
bib=m.group(1)+'\n'+'\n'.join(x.strip() for x in entries)+'\n'+m.group(3)
s=s[:m.start()]+s[m.end():]
marker=r'\appendix';assert marker in s
s=s.replace(marker,'\\clearpage\n'+bib+'\n\\clearpage\n'+marker+'\n\\setcounter{page}{1}\n\\renewcommand{\\thepage}{EC-\\arabic{page}}\n\\begin{center}\\Large Electronic Companion\\end{center}\n\\noindent Directional Irreversibility in Economic Dynamics: Inference and Shock Transmission\\par\n')
(S/'main.tex').write_text(s);(P/'abstract.txt').write_text(abstract)
print('Orthogonal theory and matched audits integrated; abstract words',len(abstract.split()))
