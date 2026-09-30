"""Final narrative stage: state observability as a foundational DII result."""
from pathlib import Path
import re,json
S=Path(__file__).resolve().parent;P=S.parent;p=S/'main.tex';t=p.read_text()
needle=r'\subsection{From directional diagnostics to the cost of a model-based decision}'
assert needle in t
t=t.replace(needle,(S/'state_observation_theory.tex').read_text()+'\n'+needle,1)
needle='The central inferential difficulty is that equality'
paragraph="A third distinction concerns what is observed. We construct a state-dependent exposure with positive DII in each state and in the state-conditioned representation, yet exactly zero DII after pooling. The pooled law is observationally equivalent to an independent system: additional pooled observations cannot repair this loss. A state-error perturbation bound supplies a local preservation condition, and a paired simulation study quantifies the effect of imperfect states and fitted conditional means. This makes the information set part of the directional estimand. Together with the predictability ceiling, the result separates a lack of driver innovation from a lack of state information.\n\n"
t=t.replace(needle,paragraph+needle,1)
t=t.replace('The restriction matters when a fitted transmission model is used to evaluate a contingent exposure or protection decision by reusing residuals across scenarios. Independent reuse can change an expected payoff and reverse an action ranking. We connect the residual-dependence components to a bound on that decision loss, while keeping the directional contrast distinct from the adequacy of either representation.','The primary contribution is a foundation for measuring directional transmission: its target, its observability, and inference for its fitted implementation. A subsequent model-use implication connects the dependence components to decision error when fitted residuals are reused across scenarios.',1)
abstract='''What can directional structure reveal about financial shock transmission, and when can the available data reveal it? The Directional Irreversibility Index (DII) compares residual dependence in opposing, history-conditioned representations. We establish two distinct limits: predictability restricts the available directional signal, while omitting an exposure state can erase positive state-conditioned DII entirely. The pooled law can be observationally equivalent to independence; additional pooled observations cannot resolve that ambiguity. A perturbation bound describes local preservation under imperfect state measurement. Joint inference propagates regression and preprocessing uncertainty across dependent horizons and distinct null configurations. Simulations examine calibration, specification sensitivity, and estimation under state errors. Component bounds also support a scoped interpretation for model-based decisions. A monetary-shock illustration finds no significant primary directional evidence. The contribution is a methodological foundation for separating directional transmission, its observability, and the economic mechanisms that a substantive financial application must establish.'''
t=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',t,count=1,flags=re.S);(P/'abstract.txt').write_text(abstract+'\n')
needle=r'\section{Conclusion}'
t=t.replace(needle,r'''\subsection{The information set is part of the finance hypothesis}
The state-observation result adds a design requirement: prespecify economically available exposure states and distinguish a state-conditioned DII from a pooled DII. A sign change in exposure need not be a reversal of structural direction, and averaging observations is not equivalent to averaging state-specific indices. The independent pooled benchmark rules out interpreting a null pooled DII as proof of no state-dependent transmission. Conversely, the noisy-state calculation rules out selecting a state measure solely because it maximizes the index. These conclusions concern the measurement of direction and do not identify a particular state variable or financial mechanism.

'''+needle,1)
t=t.replace('The predictability bound explains why driver selection matters:', 'The state-observation theorem establishes that positive directional structure can be invisible in pooled data even when the driver retains full innovation variance. Its perturbation bound and estimation experiment characterize a separate source of fragility: the information available about the exposure state. The predictability bound explains why driver selection matters:',1)
# Attribute established conditional-versus-marginal dependence without claiming a new general CI test.
needle='The dependence measure is the kernel criterion'
t=t.replace(needle,'Conditional-independence methods explicitly distinguish conditioning from marginalization; examples include Zhang et al. (2011) and Petersen and Hansen (2021). Our state-observation result uses this established distinction to characterize the original DII target and its loss under state omission. It does not replace a general conditional-independence test.\n\n'+needle,1)
refs=r'''\bibitem{zhangci} Zhang, K., J. Peters, D. Janzing, and B. Sch\"olkopf (2011). Kernel-based Conditional Independence Test and Application in Causal Discovery. \emph{Proceedings of UAI}, 804--813. \url{https://arxiv.org/abs/1202.3775}.
\bibitem{petersenci} Petersen, L., and N. R. Hansen (2021). Testing Conditional Independence via Quantile Regression Based Partial Copulas. \emph{Journal of Machine Learning Research} 22(70), 1--47. \url{https://jmlr.org/papers/v22/20-1074.html}.
'''
t=t.replace(r'\end{thebibliography}',refs+r'\end{thebibliography}',1)
res=json.loads((P/'state_observation_results.json').read_text());pop=json.loads((P/'state_observation_population.json').read_text())
main=r'''\subsection{State observability: a paired estimation experiment}\label{sec:stateexperiment}
The analytical result concerns population DII. A separate experiment measures how accurately its targets can be estimated when the exposure state is imperfect. We use 1,000 independent training/evaluation datasets and four paired state-error variants per dataset, for 4,000 evaluations. The design fixes two noise scales, two evaluation sizes, and four state-error rates before execution. Each reported cell has 250 replications. Oracle and trained conditional means are compared against numerical population DII; these are estimation results, not new rejection or coverage estimates.

\begin{table}[htbp]\centering\small
\caption{DII under imperfect exposure-state information}\label{tab:statepopulation}
\begin{tabular}{rrrrr}\toprule
Noise $\sigma$ & Error $q$ & $H_f$ & $H_b$ & DII\\\midrule
'''
for r in pop:main+=f"{r['sigma']:.0f} & {r['q']:.2f} & {r['forward_hsic']:.6f} & {r['reverse_hsic']:.6f} & {max(0,r['dii']):.6f}"+r'\\'+'\n'
main+=r'''\bottomrule\end{tabular}
\par\vspace{4pt}\raggedright Gaussian bandwidths are one in raw coordinates. Values use Gaussian quadrature checked at three orders. The zero endpoints are analytical. These are distinct information-set targets; a larger DII is not necessarily better state measurement.
\end{table}

Estimation error decreases with the larger sample in every displayed fitted cell. For $\sigma=1$ with the state observed, fitted DII has mean $0.004138$ and RMSE $0.001921$ at $n=120$, versus mean $0.004951$ and RMSE $0.000572$ at $n=480$, against population DII $0.005041$. At $n=480$, the corresponding oracle RMSE is $0.000474$, showing a remaining cost from estimating the means. At complete state loss, small negative average contrasts remain in finite samples despite the zero population target; they are estimation error, not evidence of reverse transmission.

The state-conditioned targets remain positive at the displayed intermediate error rates, but the generic perturbation bound is too conservative to certify those rates. At $\sigma=1$, ten-percent state error increases population DII even though the state is less informative. At complete state loss, both components vanish. These outcomes make DII's information-set dependence empirically visible without changing its definition or claiming monotonicity. Appendix~\ref{app:stateobservation} gives the full design and every oracle/fitted result. The family-specific trained means are consistent but do not establish a generic machine-learning guarantee or extend the earlier bootstrap theorem to their clipped boundary estimators.

'''
needle=r'\section{A reproducible financial application}'
if needle not in t:
 matches=re.findall(r'\\section\{[^}]+\}',t);needle=next(z for z in matches if 'monetary' in z.lower() or 'financial illustration' in z.lower())
t=t.replace(needle,main+needle,1)
app=(S/'state_observation_appendix.tex').read_text()
app+=r'''\begin{table}[htbp]\centering\scriptsize
\caption{All state-observation estimation cells}\label{tab:stateestimation}
\begin{tabular}{rrr|rrr|rrr|r}\toprule
&&&\multicolumn{3}{c|}{Oracle DII}&\multicolumn{3}{c|}{Fitted DII}&\\
$\sigma$ & $n$ & $q$ & Mean & SE & RMSE & Mean & SE & RMSE & Fit--oracle RMS\\\midrule
'''
for r in res['summary']:
 o=r['oracle'];f=r['fitted'];app+=f"{r['sigma']:.0f} & {r['n']} & {r['q']:.2f} & {o['mean']:.5f} & {o['mean_mcse']:.5f} & {o['rmse']:.5f} & {f['mean']:.5f} & {f['mean_mcse']:.5f} & {f['rmse']:.5f} & {r['fitted_oracle_rms']:.5f}"+r'\\'+'\n'
app+=r'''\bottomrule\end{tabular}
\par\vspace{4pt}\raggedright SE is Monte Carlo standard error of the reported mean, not a data-based inferential standard error. RMSE is relative to the corresponding population DII in Table~\ref{tab:statepopulation}. Each cell has 250 replications and training size $2n$. Negative sample contrasts at a zero target are permitted. No rejection-rate interpretation is attached to these means.
\end{table}
\begin{table}[htbp]\centering\small
\caption{Monte Carlo uncertainty of RMSE}\label{tab:statermse}
\begin{tabular}{rrr|rr}\toprule
$\sigma$ & $n$ & $q$ & Oracle RMSE SE & Fitted RMSE SE\\\midrule
'''
for r in res['summary']:app+=f"{r['sigma']:.0f} & {r['n']} & {r['q']:.2f} & {r['oracle']['rmse_mcse']:.6f} & {r['fitted']['rmse_mcse']:.6f}"+r'\\'+'\n'
app+=r'\bottomrule\end{tabular}\end{table}'+'\n'
t=t.replace(r'\end{document}',app+r'\end{document}');p.write_text(t)
(S/'cover_letter.md').write_text('''Dear Editors,

Please consider “Directional Irreversibility in Economic Dynamics: Inference and Shock Transmission” for the journal.

The paper develops a methodological foundation for measuring directional financial transmission through the Directional Irreversibility Index (DII). It connects three questions: what directional asymmetry measures, whether the observed information set can reveal it, and how to infer it with fitted dynamic models.

Two analytical results explain distinct losses of directional signal. A predictability bound relates the signal to the driver's remaining innovation variance. A state-observation theorem establishes that positive DII in each exposure state can disappear completely in pooled observations, whose joint law becomes indistinguishable from independence. A perturbation bound gives a local preservation condition under state errors. These are implications for DII, building on established kernel and conditional-dependence principles.

Joint inference accounts for regression, preprocessing, and dependent-horizon uncertainty under distinct null configurations. In addition to the existing calibration and specification studies, a new experiment reports all 4,000 paired state-error evaluations from 1,000 independent datasets. It compares oracle and trained DII estimation and demonstrates that a larger index need not indicate better state information. Decision-use implications remain secondary to the foundational contribution.

The financial relevance is the design and interpretation of shock-transmission studies, including shock choice, available exposure states, and outcome horizons. The paper does not claim a new market mechanism or investment-performance finding. Its monetary-shock illustration reports no significant primary directional result.

Thank you for considering the manuscript.

Arka Prava Bandyopadhyay

Author note: Unsent draft. Complete submission declarations from the actual circumstances.
''')
(S/'editor_strategy.md').write_text('''---
title: "DII: foundational contribution for the journal"
author: "Prepared for Arka Prava Bandyopadhyay"
date: "13 September 2026"
---

### Central submission argument

DII is the paper's central object. The contribution joins its definition, the information needed to observe it, and inference for fitted dynamic representations. The predictability result and new state-observation result explain different reasons why economically meaningful transmission can leave little measured directionality.

### New theory and evidence

An exact state-dependent exposure has positive DII in each state but an independent pooled law. No increase in pooled sample size resolves that observational equivalence. A Gaussian-feature perturbation bound gives a conservative local condition preserving DII under state errors. The latter is not a monotonicity theorem: numerical population DII can rise when state measurement worsens.

A fixed experiment reports 4,000 paired state-error evaluations from 1,000 independent datasets, with 250 replications per cell, oracle and trained means, population targets, RMSE, and Monte Carlo uncertainty. The new experiment evaluates estimation, not test size or power. Earlier calibration studies and financial findings remain intact.

### Separation from the journal paper

This manuscript supplies definitions, proofs, design implications, synthetic benchmarks, and inference. The journal paper retains its own empirical phenomenon, identification, mechanism, and economic magnitudes. No journal data, findings, tables, or mechanism are used in the new addition. The methods paper should be cited where its actual conditions hold, with related-paper disclosure completed at submission.

### Finance research positioning

The intended contribution is a reusable method for evaluating directional transmission across financial settings. This supports a coherent finance research agenda without making the foundational paper depend on the journal result. Departmental tenure credit remains specific to local standards; FT50 status alone is not a tenure assessment.

The remaining editorial question is whether this methodological advance is sufficiently substantial for the journal. The new results sharpen its purpose, but neither simulated performance nor additional theory establishes a publication probability.
''')
bridge=P/'RFS_foundation_bridge.md';bt=bridge.read_text();mark='\n## State observability and the division of contributions\n';bt=bt.split(mark)[0]+mark+'''
The foundational paper adds an abstract state-dependent exposure theorem, a state-error stability bound, and fully synthetic estimation evidence. These additions use no journal data, estimates, tables, empirical claims, or application-specific mechanisms. They characterize the information set in the original DII estimand; they do not substitute a regime-specific empirical project for the original methods paper.

The journal paper's independent contribution remains its financial phenomenon, identification, economic mechanism, and magnitudes. It may cite the methods result to explain why its information set and shock measure are prespecified, but should not present the same theorem or simulation as a second original contribution. Its actual estimator must meet the theorem invoked. A precise related-paper disclosure and overlap check should be completed against the final journal manuscript before submission; this revision does not claim a text comparison with an unavailable current journal draft.
''';bridge.write_text(bt)
print('State observation integrated; abstract words',len(abstract.split()))
