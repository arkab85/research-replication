from pathlib import Path
import json,re
S=Path(__file__).resolve().parent;P=S.parent;s=(S/'main.tex').read_text()
marker='\\clearpage\n\\section{An optional decision-use illustration}'
assert marker in s
s=s.replace(marker,(S/'weak_dependence.tex').read_text()+'\n'+marker)
abstract="""A shock can move an outcome, change its risk, or generate an asymmetric relationship between opposing dynamic models. This paper develops the Directional Irreversibility Index (DII) to separate that last object from mean responses and general dependence. Financial exposure and outcome-timing benchmarks show why positive DII can coexist with a zero linear response, and why a persistent footprint need not imply delayed information absorption. The inferential problem combines estimated residuals, dependent horizon comparisons, and null configurations with different convergence rates. We establish fitted-operator transfer bounds and, using an existing Hilbert-space bootstrap theorem, a joint nonoverlapping-block approximation under explicit absolute-regularity conditions that permit persistent dynamics. Simultaneous component confidence bounds distinguish directional asymmetry from approximate forward-model validity and remain applicable to some exact-cancellation configurations excluded by contrast tests. A 4,800-dataset calibration study and a separate 600-dataset diagnostic quantify distortion, power, and conservatism. A fixed four-market application has no primary directional rejection or demonstrated positive nonlinear prediction gain. The framework provides a common language and reproducible inference for directional economic dynamics, while keeping statistical footprints, model adequacy, and financial mechanisms distinct."""
s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _: '\\begin{abstract}\n'+abstract+'\n\\end{abstract}',s,flags=re.S)
s=s.replace('The paper establishes operator perturbation bounds and a joint block approximation for a fixed family of contrasts under finite-memory dynamics.','The paper establishes operator perturbation bounds and two explicit joint block routes: a self-contained finite-memory argument and a nonoverlapping-block extension under absolute regularity that permits persistent dynamics. A second inferential output bounds both components and DII simultaneously, allowing researchers to distinguish directional evidence from approximate forward-model validity without selecting a contrast regime.')
s=s.replace('The paper has three connected contributions: a DII interpretation that separates directional structure from mean and risk effects; feasible joint inference for fitted dynamic contrasts with different null limits; and a reusable framework for distinguishing one-horizon evidence, average asymmetry, and persistence.','The paper has three connected contributions: a DII interpretation that separates directional structure from mean and risk effects; feasible joint inference under explicit persistent-dependence conditions; and simultaneous component bounds that distinguish direction, approximate model validity, and horizon persistence. The component bounds cover certain cancellation configurations excluded by the contrast test, at a potentially substantial cost in power.')
s=s.replace('The present argument combines generated-residual perturbation with an explicit joint block approximation on a finite-memory class.','The present argument combines generated-residual perturbation with explicit joint block approximations. The persistent-dependence extension specializes the Hilbert-space results of Dehling, Sharipov, and Wendler (2014), then derives fitted-operator and simultaneous-component consequences. Neither the underlying weak-dependence theorem nor confidence-set projection is claimed as a new general statistical principle.')
s=s.replace('Use a gap that separates training and evaluation under the maintained finite-memory model.','Use a gap justified by the maintained dependence model, accounting for the full lead--lag window. Under absolute regularity, a separating gap gives an asymptotic coupling argument rather than literal independence.')
main=r'''
\subsection{What the component bounds add}\label{sec:boundsuse}
A directional contrast can be positive because the reverse model is much worse even when the forward model also violates residual independence. Report a simultaneous interval for each component, not only a p-value for their difference. Appendix~\ref{app:weak} gives a nonoverlapping-block construction under explicit absolute-regularity conditions, including geometric decay. It yields $[L_{f,h},U_{f,h}]$, $[L_{b,h},U_{b,h}]$, and $[L_h^D,U_h^D]$ on one confidence event.

If $L_h^D>0$, the evidence supports positive asymmetry. If, in addition, $U_{f,h}$ is below a prespecified tolerance, it also supports approximate forward-model validity in those kernel units. Neither conclusion proves an intervention effect. For a risk manager this distinguishes a directional feature of a nonlinear exposure from confidence in the proposed representation. For a researcher it prevents a contrast from being promoted into a stronger structural conclusion than its components support.

The same confidence event supports prespecified averages, hump contrasts, and persistence comparisons across a fixed common-calendar family. Exact cancellation of the DII contrast need not invalidate these intervals if the joint operator limit satisfies its quantile condition. This broader interpretive scope carries a cost: projection bounds can be substantially less powerful than a contrast test. The original circular/wild intersection and the new nonoverlapping confidence procedure have separate algorithms and guarantees; empirical output from one is not attributed to the other.

'''
s=s.replace('\\section{Simulation evidence:',main+'\\section{Simulation evidence:')
s=s.replace('The theoretical conditions also remain consequential. Exact finite dependence is not established for these financial series.','The theoretical conditions also remain consequential. The persistent-dependence extension supplies a broader sampling route, but does not establish its assumptions for these financial series or retroactively change the circular/wild primary algorithm.')
s=s.replace('Such diagnostics neither prove nor disprove a particular finite-memory model.','Such diagnostics do not establish stationarity, the joint absolute-regularity condition, or a first-stage rate.')
s=s.replace('Within the stated finite-memory class, the resulting framework supports one-horizon comparisons, prespecified signed aggregates, and conjunction statements about persistence.','The added nonoverlapping-block route permits persistent dynamics under explicit absolute-regularity conditions. Simultaneous component bounds support one-horizon comparisons, prespecified signed aggregates, and persistence statements while distinguishing asymmetry from approximate model validity. Their regime coverage and conservative power are separate from the original intersection procedure.')
s=s.replace('Appendices give the full residual and bootstrap arguments, followed by an optional decision-loss application.','Appendices give the full residual and bootstrap arguments, including the persistent-dependence extension and simultaneous bounds, followed by an optional decision-loss application.')
s=s.replace('\\begin{thebibliography}{99}','\\begin{thebibliography}{99}\n\\bibitem{dsw} Dehling, H., O. Sh. Sharipov, and M. Wendler (2014). Bootstrap for dependent Hilbert space-valued random variables with application to von Mises statistics. arXiv:1312.3870v3, Theorems 1.1--1.2. \\url{https://arxiv.org/abs/1312.3870v3}.')
r=json.loads((P/'persistent_diagnostic_results.json').read_text());assert r['total_datasets']==600
rows=[]
for x in r['summary']:
 if x['residuals']=='fitted':
  rows.append(f"{x['rho']:.1f} & {x['theta']:.1f} & "+' & '.join(f"{100*x[k]['rate']:.0f}" for k in ['wild_reject','paired_reject','intersection_reject','bounds_reject'])+r' \\')
diag=r'''
\subsection{A persistent-process diagnostic at the financial sample allocation}
A separate post-results experiment holds the small-sample allocation at $m=204$ training and $n=120$ evaluation observations, with a 32-row gap. It is a synthetic power and implementation diagnostic, not a revision of the financial protocol. For independent iid standard-normal $S_t$ and innovations $\zeta_t\sim N(0,1-\rho^2)$, set
\[
e_t=\rho e_{t-1}+\zeta_t,\qquad Y_t=\theta(S_t^2-1)+e_t,
\quad C_t=(S_{t-1},Y_{t-1}),
\]
starting $e_0$ in its stationary distribution. The forward conditional mean is $\theta(S_t^2-1)+\rho\{Y_{t-1}-\theta(S_{t-1}^2-1)\}$ and its residual is $\zeta_t$. Symmetry gives a zero reverse conditional mean. Thus both means belong to the fitted total-degree-two polynomial class; $\theta=0$ gives double independence and $\theta>0$ gives positive DII. The design has infinite memory when $\rho=0.6$.

There are 100 replications per cell, six cells, and 199 draws per procedure. The original intersection uses 12-row blocks; the new simultaneous bounds use the theorem's four-row dyadic rule. Both use a common resampling path within each procedure. Because both the block rule and the inferential output differ, this is not an isolated comparison of p-values with intervals. Training-based standardization is frozen as in the financial exercise; its randomness and the modest training allocation are additional reasons to treat this as a finite-sample diagnostic, rather than a demonstration that every fixed-kernel nuisance premise holds.

\begin{table}[htbp]\centering
\caption{Persistent-process diagnostic: fitted-residual rejection percentages}
\begin{tabular}{rrrrrr}\toprule
$\rho$ & $\theta$ & Wild & Paired & Intersection & Bounds\\\midrule
% ROWS
\bottomrule\end{tabular}
\par\vspace{5pt}\raggedright\noindent Bounds reject positivity's null only when the simultaneous lower DII bound exceeds zero. With 100 replications, zero rejections has a Wilson 95\% upper bound of about 3.7\%. All cellwise Wilson intervals and oracle results are included in the reproducibility files.
\end{table}

At $\theta=0.5$, the fitted intersection rejects 30\% of iid-noise datasets and 20\% with persistent noise. The corresponding oracle frequencies are 32\% and 30\%. This provides a concrete design in which a meaningful nonlinear exposure is often missed at the financial sample allocation. It does not estimate power against an unknown empirical effect. At $\theta=1$, fitted intersection frequencies are 75\% and 80\%, compared with oracle frequencies of 84\% and 87\%. The diagnostic cannot convert the financial nonrejections into positive findings.

The simultaneous lower bounds reject in none of the 600 fitted evaluations, including the positive alternatives. This is evidence of severe finite-sample conservatism in these designs, not an advantage to conceal. The bounds add joint interpretation and coverage of some contrast-degenerate laws; the completed experiment does not establish a power advantage over paired or intersection testing. Larger samples, sharper confidence regions, or better-justified nuisance learning are research needs rather than results already obtained here. The Gram-matrix bootstrap calculation was independently checked against explicit finite-feature operator sums, with maximum absolute error below $6\times10^{-15}$.

'''.replace('% ROWS','\n'.join(rows))
s=s.replace('\\section{A reproducible financial application}',diag+'\\section{A reproducible financial application}')
(S/'main.tex').write_text(s);(P/'abstract.txt').write_text(abstract)
print('Weak-dependence and component-bound results integrated; abstract words',len(abstract.split()))

# This supplementary output was computed after the primary results were known.
financial=json.loads((P/'finance/simultaneous_bounds_results.json').read_text())
assert financial['n_retained']==120 and financial['block_length']==4
assert sum(x['dii_lower']>0 for x in financial['comparisons'].values())==0
extra=r"""
\paragraph{Secondary simultaneous bounds.} After the primary results were known, the new procedure was applied to the same 12 cells, using one common path for all 24 operators, four-row nonoverlapping blocks, and 1,999 draws. No lower DII bound is positive. No forward-validity tolerance was chosen after seeing the bounds. Point estimates agree with the original implementation to within $4\times10^{-18}$; the new bounds therefore reflect a different inferential output rather than changed fitted models. All component endpoints appear in the reproducibility files. This post-results analysis does not replace the fixed primary tests, and inherits the unresolved empirical assumptions.

"""
s=s.replace('\\section{Financial transmission and the use of DII}',extra+'\\section{Financial transmission and the use of DII}')
(S/'main.tex').write_text(s)
# Keep proof cross-references accurate and references in author order.
s=s.replace('Appendix A proves the theorem by',r'Appendix~\ref{app:joint} proves the theorem by')
s=s.replace(r'\clearpage'+'\n'+r'\section{An optional decision-use illustration}',r'\section{An optional decision-use illustration}')
m=re.search(r'(\\begin\{thebibliography\}\{99\})(.*?)(\\end\{thebibliography\})',s,re.S)
entries=re.findall(r'\\bibitem\{[^}]+\}.*?(?=\\bibitem|$)',m.group(2),re.S)
entries.sort(key=lambda x:x.split('}',1)[1].strip())
s=s[:m.start()]+m.group(1)+'\n'+'\n'.join(x.strip() for x in entries)+'\n'+m.group(3)+s[m.end():]
(S/'main.tex').write_text(s)
