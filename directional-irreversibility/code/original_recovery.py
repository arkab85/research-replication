"""Recover the original submission's predictability result and applied design rationale."""
from pathlib import Path
S=Path(__file__).resolve().parent;P=S.parent;p=S/'main.tex';t=p.read_text()
addition=r'''\subsection{Predictable drivers and the innovation available for direction}\label{sec:innovation}
Persistence of the driver and persistence of its directional footprint are different objects. If the observed history already predicts the driver accurately, the reverse regression has little remaining variation to explain. This limits the directional contrast even when the forward mechanism is nonlinear. The following result sharpens the persistence bound by working directly with the Gaussian feature map.

\begin{proposition}[A predictability bound for DII]\label{prop:innovation}
Let $X$ have a finite second moment, let $C$ be the common history, and define
\[
v_X=\E\{[X-\E(X\mid C)]^2\},\qquad
u_b=X-\E[X\mid Y,C].
\]
Use a Gaussian residual kernel of fixed bandwidth $\ell_u>0$ and a regressor kernel satisfying $l(z,z)\leq M^2$. Then
\[
H_b\leq \frac{M^2v_X}{\ell_u^2}.
\]
If the forward residual is independent of its full regressor vector, then
\begin{equation}\label{eq:innovationbound}
0\leq D=H_b\leq \frac{M^2v_X}{\ell_u^2}.
\end{equation}
For a unit-variance AR(1) driver $X_t=\varphi X_{t-1}+\sqrt{1-\varphi^2}\,\eta_t$, with innovations of variance one independent of the past and $X_{t-1}$ in $C_t$, $v_X\leq1-\varphi^2$. The Gaussian regressor kernel has $M=1$, so the upper bound becomes $(1-\varphi^2)/\ell_u^2$.
\end{proposition}

The proof is in Appendix~\ref{app:innovation}. The bound is a population statement at fixed kernel scale; it is not a universal finite-sample power theorem. With standardized $X$ and bandwidth one, the AR(1) ceiling is 0.19 at $\varphi=0.90$ and 0.0591 at $\varphi=0.97$. These are upper bounds, not predicted DII magnitudes or detection probabilities. They show why a small directional footprint need not imply a weak economic mechanism: the reverse target can be nearly determined by its history.

For financial research, the result motivates a design choice before testing. A policy innovation, earnings surprise, or other independently justified news measure asks a different question from a persistent policy level, earnings level, or economic state. An innovation-like driver avoids this particular shrinking bound, but neither guarantees a detectable contrast nor supplies identification. Rescaling a vanishing residual by its own standard deviation, or replacing a level by an estimated innovation, changes the kernel target; it cannot be used to claim that the original index has escaped the bound. Such choices and their training uncertainty must be specified as part of the research design.

'''
a=r'\subsection{A modeling consequence: reusing residuals across shock scenarios}';assert a in t;t=t.replace(a,addition+a,1)
needle='The central inferential difficulty is that equality of the two dependence components has different sampling behavior depending on what is equal.'
paragraph=r'''A second distinction concerns the driver itself. Persistence of directional structure across horizons is not the same as persistence of the cause. When the history already predicts the cause well, the reverse residual has little variation left. We show that, with fixed Gaussian kernels, the reverse dependence component is bounded by the cause's remaining conditional variance, up to the kernel scale. This supplies a direct reason to examine economically justified innovations and surprises, rather than interpreting a weak contrast for a persistent level as absence of transmission. It also links the choice of shock measure to the information available for directional inference.

'''
t=t.replace(needle,paragraph+needle,1)
# Restore the generic first-stage route without claiming arbitrary learned models satisfy the fast rate.
needle='There are two covered boundary configurations.'
text=r'''There are two routes for the generated-residual step. The generic first-stage route bounds the effect of prediction error and requires $\sqrt n\,r_n\to_p0$ on a separated evaluation sample; it is not tied to a polynomial learner, but each learner must deliver that rate under the maintained dependence conditions. The finite-dimensional route in Section~\ref{sec:training} instead propagates training uncertainty explicitly when $n/m$ has a positive limit. Its fixed-space assumptions are narrower, but it does not discard training uncertainty as negligible. These routes address different training allocations and should not be interchanged when citing a guarantee.

'''
t=t.replace(needle,text+needle,1)
# Replace generic wording in finance with the original shock-vs-state motivation.
needle='It is most interpretable when the shock design, outcome transformation, time aggregation, and information set have a clear economic justification.'
t=t.replace(needle,needle+' Proposition~\\ref{prop:innovation} adds a specific consideration: the shock should retain economically meaningful innovation variance after conditioning on the common history. Persistence of the driver can reduce the contrast even when transmission exists. The horizon profile then concerns the persistence of the directional footprint, not the autoregressive persistence of the input.')
needle='The paper supplies joint inference for fitted DII and fixed horizon comparisons, including regression and preprocessing uncertainty.'
t=t.replace(needle,'The predictability bound explains why driver selection matters: at a fixed kernel scale, the remaining innovation variance limits the reverse dependence component. '+needle,1)
proof=r'''
\section{Proof of the predictability bound}\label{app:innovation}
Write $Z=(Y,C)$ and let $\phi$ and $\psi$ denote the residual and regressor feature maps. For the Gaussian residual kernel,
\[
\|\phi(u)-\phi(0)\|^2
=2\{1-\exp[-u^2/(2\ell_u^2)]\}\leq u^2/\ell_u^2.
\]
The residual-regressor covariance operator satisfies
\[
C_b=\E[(\phi(\nu_b)-\phi(0))\otimes(\psi(Z)-\E\psi(Z))],
\]
because subtracting a constant from the first feature does not change this covariance. Jensen's inequality and Cauchy--Schwarz give
\begin{align*}
\|C_b\|^2
&\leq \E\|\phi(\nu_b)-\phi(0)\|^2\,
             \E\|\psi(Z)-\E\psi(Z)\|^2\\
&\leq \frac{\E\nu_b^2}{\ell_u^2}\,M^2
\leq \frac{v_X}{\ell_u^2}\,M^2.
\end{align*}
The final inequality uses the $L^2$ optimality of $\E[X\mid Y,C]$: the predictor $\E[X\mid C]$ is available to the reverse regression. Since $H_b=\|C_b\|^2$, the first assertion follows. Forward independence gives $H_f=0$ and hence $D=H_b$. For the AR(1) driver, $\varphi X_{t-1}$ is an available history predictor with mean squared error $1-\varphi^2$, proving the final assertion. The bound does not depend on the forward nonlinear mechanism and requires no Gaussian innovation assumption; Gaussianity is imposed on the residual kernel. If forward independence fails, the bound on $H_b$ still holds but $D$ need not be nonnegative.

This sharpens the earlier first-order Lipschitz bound of order $\sqrt{1-\varphi^2}$ for fixed kernels to a bound of order $1-\varphi^2$ under the Gaussian feature map. It does not establish a lower bound, a local-power rate, or a uniform bootstrap result as $\varphi\to1$. In particular, the time-series sampling conditions must still be checked separately along changing-persistence sequences. A bandwidth that shrinks with the residual's own dispersion removes the stated fixed-scale comparison and defines a different sequence of targets.
'''
t=t.replace(r'\end{document}',proof+'\n'+r'\end{document}')
t=t.replace('Analytical financial exposures show that a linear response can coexist with zero DII, a nonlinear exposure can have positive DII despite a zero linear response, and risk dependence can cancel in the contrast.', "Financial benchmarks distinguish DII from mean and risk responses. A predictability bound links directional signal to the driver's remaining innovation variance.",1)
p.write_text(t)
import re
(P/'abstract.txt').write_text(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',t,re.S).group(1).strip()+'\n')
# Author-facing provenance records exactly which original file and claims were reviewed.
(P/'Econometrica_to_Management_Science.md').write_text('''# What was recovered from the the journal manuscript

Source reviewed: `econometrica-submission (6).pdf`, the latest file with that submission name available in the materials (73 pages; uploaded September 6, 2026). This identifies the source; it does not independently confirm which version the journal received. The original PDF was not edited.

| Original contribution | Change in the the journal revision |
|---|---|
| Proposition 4, pp. 22–23: predictable causes leave little reverse residual variation | Restored as a general conditional-innovation-variance bound. For fixed Gaussian residual kernels, the feature-map proof sharpens the AR(1) upper bound from order square-root(1-phi²) to order (1-phi²). |
| Sections 3.2–3.3 and 6.6: persistence is a horizon-specific object | Explicitly separate persistence of the driver from persistence of its directional footprint. Retain existing conjunction, average, and horizon-shape distinctions. |
| Section 6.4: generic first stages subject to a sufficient prediction-error rate | Clarify the generic negligible-error route versus the newer fixed-dimensional training-uncertainty route. Neither is presented as covering arbitrary machine learning automatically. |
| Sections 8.1–8.3: externally identified shocks as a deliberate application class | Restore the shock-versus-state design rationale in the introduction and financial interpretation. |

## Empirical material reviewed and retained as provenance

The original domestic six-pair pool reports DII=0.0028 at 12 months, wild p=0.047 and paired p=0.020; its intersection p is 0.047. Across the three tested horizons, the Bonferroni-adjusted intersection p is 0.141. The horizon-average tests do not reject. The G7 oil-news illustration reports all seven estimates positive at six months, wild p=0.056 and paired p=0.073; the intersection does not reject at five percent. These are reported results in the original PDF, not newly reproduced results. They are not substituted for the current financial experiment or used to upgrade journal claims.

The proprietary mortgage exercise in Appendix D reports information measures rather than a loan-level DII test and explicitly lacks loan identifiers in the available file. It is not evidence validating DII and is not restored as such. Monotonicity of information as predictors are added is not a distinctive directional finding.

## What is preserved

The original DII idea remains the comparison of opposing residual-dependence restrictions over horizons. The sharper predictability result develops an original theoretical contribution rather than creating a new empirical project. Existing financial estimates, simulations, inferential conditions, and journal results are unchanged. The new theorem does not claim universal power collapse: it bounds the population signal at a fixed kernel scale. The original claims about labeling, general causal identification, and near-unit-root inference are not imported without their necessary conditions.
''')
cover=S/'cover_letter.md';c=cover.read_text();c=c.replace('The methodological contribution is','A predictability bound links the method to shock selection: at fixed Gaussian kernel scale, the driver’s remaining conditional variance bounds the reverse dependence component. This explains why an economically important but highly predictable level can yield little directional separation, and why independently justified innovations provide a distinct design.\n\nThe methodological contribution is');cover.write_text(c)
(P/'RFS_foundation_bridge.md').write_text((P/'RFS_foundation_bridge.md').read_text()+'''\n## Restored result from the original the journal manuscript\n\nThe predictability bound separates persistence of the driver from persistence of its directional footprint. At fixed Gaussian residual bandwidth, the reverse dependence component is bounded by the driver’s conditional innovation variance times the regressor-kernel diagonal bound divided by bandwidth squared. This motivates independently justified shock or surprise measures; it does not guarantee power, identify those shocks, or make a nonsignificant DII profile evidence of no transmission. Innovation standardization changes the target and must be specified rather than used as a post-results repair.\n''')
(S/'editor_strategy.md').write_text('''---
title: "DII: restoring the original contribution for the journal"
author: "Prepared for Arka Prava Bandyopadhyay"
date: "13 September 2026"
---

### What strengthens the submission

The strongest recovered result is the original the journal paper's predictability argument. When history already predicts the driver, little residual variation remains for the reverse model. The revision restores this result and sharpens it: with fixed Gaussian kernels, the reverse dependence component is bounded by the driver's conditional innovation variance. For a standardized AR(1) driver the bound is proportional to 1-phi², improving the earlier square-root bound.

This gives the methods a concrete financial implication: persistent levels and economically justified innovations are different research designs. It also separates driver persistence from persistence of a directional footprint across horizons. The original DII object stays central.

### Positioning

Lead with the financial question, the analytical distinctions, the predictability bound, and joint inference for the fitted contrast. The generic first-stage route and the fixed-dimensional training-uncertainty route are now explicitly distinguished. The recent scenario-score interpretation remains a supporting use of the components.

This matches the [Finance editorial statement](https://pubsonline.informs.org/page/v/editorial-statement)'s interest in conceptual and empirical-methodological contributions with substantive relevance. Whether the advance is sufficiently large remains an editorial judgment. No numerical R&R probability is justified by the available evidence.

### Empirical and journal boundaries

The original pooled macro result is significant at one unadjusted horizon, but not across the three-horizon family; the G7 result does not reject under the intersection test. Those original reported results have not been re-estimated and are documented in the author comparison note. They are not substituted for current financial results. The original mortgage exercise is not a loan-level DII test.

The journal paper retains responsibility for its own financial finding, shock identification, and mechanism. This revision supplies a stronger theoretical rationale for the method and shock choice, without changing journal data or claims. No submission or reviewer contact has occurred.
''')
print('Original predictability contribution restored and sharpened.')
