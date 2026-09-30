"""Connect DII's component inference to bounded financial decision loss."""
from pathlib import Path
import re,json
S=Path(__file__).resolve().parent;P=S.parent;p=S/'main.tex';t=p.read_text()
r=json.loads((P/'decision_regret_results.json').read_text());assert r['true_choice']=='buy' and r['independent_choice']=='do not buy'
a=t.index(r'\subsection{A modeling consequence: reusing residuals across shock scenarios}');b=t.index(r'\subsection{Persistent DII need not mean delayed incorporation}',a)
t=t[:a]+r'''\subsection{From directional diagnostics to the cost of a model-based decision}\label{sec:replay}
A model of shock transmission is often used by combining a shock/history scenario with a draw from the fitted residual distribution. Independent residual reuse replaces the joint law $P_d$ of $(u^d,Z^d)$ by $Q_d=P_{u^d}\otimes P_{Z^d}$. In the tensor-product RKHS $\mathcal K_d$ of the kernels used in DII, the standard mean-embedding identity (Gretton et al., 2012) gives
\begin{equation}\label{eq:replay}
\sup_{\|q\|_{\mathcal K_d}\leq1}|\E_{P_d}q-\E_{Q_d}q|
=\sqrt{H_d},\qquad D_h=H_b-H_f.
\end{equation}
This established identity connects each dependence component to the error in a specified class of model scores. DII compares these squared discrepancies across representations. The representations have different targets, so their discrepancies are not directly comparable portfolio utilities.

The decision consequence follows by fixing one representation and an action set. Write an action's net payoff as $\pi_a(u,z)=q_a(u,z)-c_a$, where $q_a\in\mathcal K$ and the known cost $c_a$ is constant. Suppose $\|q_a\|\leq B$ for every available action. Let $a_Q$ maximize expected net payoff under independent residual reuse, and let $a_P$ maximize it under the joint law. Then
\begin{equation}\label{eq:regret}
0\leq \E_P\pi_{a_P}-\E_P\pi_{a_Q}\leq2B\sqrt H.
\end{equation}
If a computed simulator value has uniform error at most $e$ and the chosen action is within $\eta$ of its computed optimum, the upper bound is $2B\sqrt H+2e+\eta$. Thus a valid component upper confidence bound $U$ supplies the sufficient regret budget $2B\sqrt U+2e+\eta$. The derivation and conditions are in Appendix~\ref{app:decisionregret}. This is an application of the existing discrepancy bound, not a new general optimization principle. Its usefulness is that the paper's inference supplies the component uncertainty needed to use it for these fitted dynamic representations.

A more targeted certificate compares two actions. Their ranking under $Q$ is preserved under $P$ whenever
\[
\E_Q(\pi_a-\pi_b)>\|q_a-q_b\|\sqrt H.
\]
With computed values and a valid $U$, replacing the right side by $\|q_a-q_b\|\sqrt U+2e$ is sufficient. The direct norm of the payoff difference can be much smaller than $2B$. Failure to certify a ranking is inconclusive, rather than a recommendation to reverse it. Neither certificate applies to an unrestricted portfolio loss, discontinuous default indicator, or tail-risk measure without proving that its payoff belongs to the stated class, or controlling its approximation error.

\paragraph{An exact financial decision illustration.}
Let $X$ be uniform on $\{-2,-1,1,2\}$ and $Y=\sqrt{1+X^2}\,\varepsilon$, with $\varepsilon$ independent and uniform on $\{-1,1\}$. Both conditional means vanish. With matched Gaussian kernels of bandwidth $0.5$, the two residual-dependence components equal $0.040052$ and DII is exactly zero. Consider the stylized smooth contingent protection payoff
\[
q(Y,X)=k(Y,-\sqrt5)\{k(X,-2)+k(X,2)\},
\]
where $k$ is that Gaussian kernel, and a fixed purchase price of $0.22$. The payoff concentrates on a negative residual outcome under a large-magnitude shock. It is a constructed bounded contract, not a calibrated or traded security.

\begin{table}[htbp]\centering
\caption{Independent residual reuse reverses a decision in an exact example}\label{tab:decisionexample}
\begin{tabular}{lrr}\toprule
Quantity & Observed joint law $P$ & Independent reuse $Q$\\\midrule
Expected protection payoff & 0.258763 & 0.178675\\
Net value at price 0.22 & 0.038763 & $-0.041325$\\
Optimal action & Buy & Do not buy\\\bottomrule
\end{tabular}
\par\vspace{4pt}\raggedright\noindent Exact sums over the specified finite support; there is no sampling uncertainty or empirical performance claim. DII is zero because both dependence components are equal and positive.
\end{table}

The example makes the distinction operational: zero directional asymmetry does not establish that independent residual reuse is suitable for a financial decision. The component inference is necessary to assess that claim within this framework. Here the generic bound is conservative and does not certify the action ranking; the example demonstrates the cost of ignoring dependence, not a power advantage over a targeted payoff test. The older forward-validity and scenario-score bounds remain in Appendix~\ref{app:replay}. All decision statements concern the specified observed or reweighted law; causal interventions require additional identification and invariance conditions.

'''+t[b:]
# Add the decision problem to the introduction where the modeling restriction is motivated.
needle='This paper studies that question through the Directional Irreversibility Index (DII).'
t=t.replace(needle,'The restriction matters when a fitted transmission model is used to evaluate a contingent exposure or protection decision by reusing residuals across scenarios. Independent reuse can change an expected payoff and reverse an action ranking. We connect the residual-dependence components to a bound on that decision loss, while keeping the directional contrast distinct from the adequacy of either representation.\n\n'+needle,1)
needle='A component upper bound therefore translates into an explicit error budget for residual reuse under a prespecified scenario weight.'
t=t.replace(needle,'A component upper bound therefore translates into a sufficient regret budget for decisions with specified kernel payoffs, including an explicit allowance for simulator and optimization error. A finite-state protection example shows that independent residual reuse can reverse the decision even when DII is zero.',1)
abstract='''How can directional structure in shock transmission inform the use of a financial model? The Directional Irreversibility Index (DII) compares residual dependence in opposing, history-conditioned representations. Financial benchmarks distinguish DII from mean and risk responses, and a predictability bound links its signal to the driver's remaining innovation variance. We develop joint inference for fitted DII across dependent horizons, propagating regression and preprocessing uncertainty through distinct null configurations. Component confidence bounds translate into sufficient decision-regret and action-ranking certificates for a specified class of model payoffs. An exact contingent-protection example shows that independent residual reuse can reverse a decision even when DII is zero, making the components essential to this interpretation. Simulations evaluate calibration, power, and specification sensitivity; a four-market monetary-shock illustration finds no significant primary directional evidence. The framework separates directional structure, uncertainty about model adequacy, and the additional conditions needed to use a transmission model for a financial decision.'''
t=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',t,count=1,flags=re.S)
(P/'abstract.txt').write_text(abstract+'\n')
a=t.index(r'\subsection{From a directional profile to a model-use decision}');b=t.index(r'\section{Conclusion}',a)
t=t[:a]+r'''\subsection{From a directional profile to a model-use decision}
The financial interpretation has three distinct steps. First, DII describes asymmetry between the two representations under a fixed shock, history, and horizon design. Second, a component bound describes uncertainty about independent residual reuse within a representation. Third, the payoff class, error budget, and optimization accuracy determine whether that model is adequate for a particular decision. Equations~\eqref{eq:regret} and the pairwise ranking condition make the third step explicit. A DII p-value cannot replace either the component bound or the payoff specification.

The predictability bound informs the first step by explaining why an economically important but nearly predictable driver can leave a small directional contrast. The protection example informs the second and third: a zero contrast can conceal a dependence-induced decision error. Together, the results explain why measuring direction and using a model are related but different tasks. The same logic can apply to other bounded business payoffs with justified kernel norms; the action and its economic meaning must be supplied by the application.

The monetary-shock exercise establishes neither a decision improvement nor a significant directional finding, and no payoff or tolerance is fitted to its outcomes. A separate financial study can use the methodology to support an independently established transmission pattern and mechanism. It must use inference appropriate to its actual estimator and data, and justify any causal use of scenarios.

'''+t[b:]
appendix=r'''
\section{Decision loss and action-ranking certificates}\label{app:decisionregret}
Fix a representation with joint law $P$, independent-reuse law $Q$, and dependence component $H$. Assume the payoffs $q_a$ lie in its tensor-product RKHS and have uniformly bounded norm $B$, costs $c_a$ are known constants, expectations exist, and the relevant maxima are attained. Finite action sets satisfy the attainment condition. The existing discrepancy identity gives
\[
|V_P(a)-V_Q(a)|\leq B\sqrt H,
\qquad V_R(a)=\E_Rq_a-c_a.
\]
If $a_Q$ maximizes $V_Q$, add and subtract its and $a_P$'s simulator values to obtain
\begin{align*}
V_P(a_P)-V_P(a_Q)
&=[V_P(a_P)-V_Q(a_P)]\\
&\quad+[V_Q(a_P)-V_Q(a_Q)]
+[V_Q(a_Q)-V_P(a_Q)]\\
&\leq 2B\sqrt H.
\end{align*}
If $\sup_a|\widetilde V_Q(a)-V_Q(a)|\leq e$ and the chosen $\widetilde a$ satisfies
$\widetilde V_Q(\widetilde a)\geq\sup_a\widetilde V_Q(a)-\eta$, insert the computed values in the same decomposition. The result is
\[
V_P(a_P)-V_P(\widetilde a)\leq 2B\sqrt H+2e+\eta.
\]
These are deterministic implications on the event of the stated bounds. A probabilistic guarantee needs a valid upper bound $U$ for the same component and a jointly valid simulator-error bound. If their separate failure probabilities are $\alpha$ and $\delta$, the union bound gives coverage at least $1-\alpha-\delta$; independence is not needed. The paper's operator bounds are asymptotic under their maintained sampling assumptions, so a decision certificate using them inherits that qualification.

For action differences, apply the discrepancy identity to $q_a-q_b$:
\[
|[V_P(a)-V_P(b)]-[V_Q(a)-V_Q(b)]|
\leq\|q_a-q_b\|\sqrt H.
\]
Known costs cancel from the discrepancy. The ranking certificates in the main text follow directly. A common component event controls all payoffs in a specified bounded class, including an action selected from that class; learned kernels, an unjustified norm bound, or an unrestricted new payoff class are not covered by this statement. For original conditional-mean residuals, the component bound must include justified mean-approximation error as in Appendix~\ref{app:preprocessing}. Projection-residual coverage cannot be relabeled as original-target coverage.

For a scenario reweighting, use $q_a(u,z)w(z)$ as the relevant score, require a valid bound on that product's RKHS norm, and define both $P^w$ and $Q^w$ with the same fixed normalized weight. The result does not assert invariance of a conditional law to intervention. If the actual payoff is only approximated by a kernel score, its expectation error under both laws must be bounded and added explicitly. No such approximation is assumed for arbitrary financial tail losses.

\subsection{Exact protection example and numerical verification}
The eight joint states in the main text are equally likely. Conditional symmetry gives $\E[Y\mid X]=0$ and $\E[X\mid Y]=0$, so the residuals are $Y$ and $X$. Symmetry of matched-kernel HSIC gives $H_f=H_b$ exactly. Dependence is nonzero because $Y^2=1+X^2$.

The payoff is a product of a residual kernel section and the sum of two shock kernel sections. Its norm is
\[
B=\sqrt{2+2k(-2,2)}=\sqrt{2+2e^{-32}}.
\]
The joint expectation averages over eight states; the independent-reuse expectation averages over the 64 equally weighted pairs of empirical marginal atoms, including repeated atoms. These give the values in Table~\ref{tab:decisionexample}. The simulator rejects the purchase, losing expected value $0.038763$ relative to the optimal action under $P$. The generic score-error bound is approximately $0.283027$, and the generic regret bound is $0.566053$. Their conservatism is explicit: they do not certify the action ranking in this example.

The included exact-sum code verifies conditional means, DII cancellation, the kernel embedding identity, and the reported payoff and norm calculations. It also checks the regret and pairwise discrepancy inequalities on finite libraries of Gaussian-section payoffs. These checks verify implementation and illustrative arithmetic; the inequalities are established by the argument above. The example is selected to demonstrate a logical distinction, with its full distribution and price disclosed. It supplies no estimate of performance in financial markets.
'''
t=t.replace(r'\end{document}',appendix+'\n'+r'\end{document}')
t=t.replace(r'equation~\eqref{eq:scenario} follows',r'$|\E[g(u)w(Z)]-\E[g(u)]|\leq ab\sqrt H$ follows')
p.write_text(t)
(S/'cover_letter.md').write_text('''Dear Editors,

Please consider “Directional Irreversibility in Economic Dynamics: Inference and Shock Transmission” for the journal.

The paper asks what directional structure adds to the analysis and use of a financial transmission model. DII compares residual-dependence restrictions in opposing representations. Analytical benchmarks distinguish that object from mean responses and risk effects, while a predictability bound explains why an economically important but highly predictable driver can produce a small directional signal.

The core methodological contribution is joint inference for the fitted contrast across dependent horizons, including regression and preprocessing uncertainty and distinct null configurations. Component confidence bounds connect this inference to sufficient decision-loss and action-ranking certificates for a specified payoff class. A fully specified protection example demonstrates why a zero directional contrast does not establish that independent residual reuse is adequate for a decision. The decision inequalities apply established kernel discrepancy arguments; the paper does not claim a new general optimization principle.

The paper is intended for researchers evaluating dynamic financial models and their use in scenario-based decisions. Its contribution is methodological and interpretive. The evidence includes unfavorable comparisons with targeted moment tests and a financial illustration with no significant primary directional finding. Those results are reported transparently; the paper does not claim demonstrated investment performance.

Thank you for considering the manuscript.

Arka Prava Bandyopadhyay

Author note: Unsent draft. Complete submission declarations from the actual circumstances.
''')
(S/'editor_strategy.md').write_text('''---
title: "DII: contribution and submission positioning"
author: "Prepared for Arka Prava Bandyopadhyay"
date: "13 September 2026"
---

### The submission argument

DII separates directional structure from mean and risk responses. The original predictability result explains why the driver’s remaining innovation variance limits that signal. Joint inference accounts for fitted models and preprocessing across dependent horizons. The revision now connects component confidence bounds to sufficient bounds on decision loss and certificates for action rankings under a specified payoff class.

The intended reader can distinguish three questions: whether the representations differ, whether independent residual reuse is accurate enough, and whether the resulting uncertainty is acceptable for a particular decision. This supplies a concrete use for the components alongside DII.

### What is new in this revision

A decision-loss derivation includes simulator and optimization error. A pairwise ranking certificate uses the norm of the payoff difference. A fully specified finite-state protection example demonstrates an actual decision reversal under independent residual reuse despite DII being zero. Exact code reproduces the example and checks the inequalities. The generic bound is conservative and does not certify the example’s ranking; that limitation is reported.

The decision inequalities are applications of established kernel discrepancy arguments. The paper's principal originality claim remains the scoped dynamic contrast, its inferential development, and the analytical implications for financial model use. The example is not empirical evidence of investment performance or superiority over targeted tests.

### Remaining editorial question

The [Finance editorial statement](https://pubsonline.informs.org/page/v/editorial-statement) welcomes conceptual and empirical-methodological contributions with substantive relevance. The revised paper makes that relevance more concrete. Whether the incremental advance is large enough remains an editorial judgment. A greater-than-50% R&R estimate is not substantiated.

The current financial illustration still has no significant primary finding. Original DII, its restored predictability result, all prior numerical evidence, and the separate role of the journal paper are preserved. No journal estimates were changed. No submission or reviewer contact has occurred.
''')
(P/'RFS_foundation_bridge.md').write_text((P/'RFS_foundation_bridge.md').read_text()+'''\n## Decision-loss interpretation\n\nThe methods paper connects component bounds to sufficient decision-regret and action-ranking certificates for specified bounded RKHS payoffs. This does not turn DII into an investment return or validate an unrestricted portfolio loss. The exact protection example is illustrative, not journal evidence. The journal contribution remains its own financial phenomenon, identification, and mechanism.\n''')
print('Decision consequence integrated; abstract words',len(abstract.split()))
