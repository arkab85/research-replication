from pathlib import Path
import json,re
S=Path(__file__).resolve().parent;P=S.parent;s=(S/'main.tex').read_text()
r=json.loads((P/'joint_nuisance_study_results.json').read_text());assert r['datasets']==3000 and r['seed']==20260918
expected={(0.,'double'):0,(.6,'double'):2,(0.,'equal_positive'):9,(.6,'equal_positive'):13,(0.,'weak'):77,(.6,'weak'):104,(0.,'strong'):214,(.6,'strong'):211}
assert all(x['rejections']==expected[(x['rho'],x['design'])] for x in r['summary'] if x['method']=='joint_scales')
assert [x['rejections'] for x in r['summary'] if x['method']=='misspecified_joint']==[113,102]
assert all(x['rejections']==0 for x in r['summary'] if x['method']=='correct_basis_joint')
# Keep the historical 800-dataset validation and attribution audit in the companion.
a=s.index(r'\subsection{Training-aware calibration: new-seed validation}');b=s.index(r'\section{A reproducible financial application}',a)
historical=s[a:b];s=s[:a]+s[b:]
s=s.replace(r'\section{An optional decision-use illustration}',historical+'\n'+r'\section{An optional decision-use illustration}')
s=s.replace(r'\section{Supplementary diagnostic designs}',(S/'joint_nuisance_theory.tex').read_text()+'\n'+r'\section{Supplementary diagnostic designs}')
a=s.index(r'\subsection{Keeping training uncertainty in the original DII}');b=s.index(r'\section{Simulation evidence:',a)
main=r'''
\subsection{Inference for trained models and fitted units}\label{sec:training}
A financial exposure is uncertain because its prediction function and normalization were learned from a finite training period. Freezing both when resampling the evaluation period omits that uncertainty. We retain the original sample DII and propagate a joint training influence containing regression coefficients, log standard deviations, and regressor centers. Common draws across directions and horizons retain their dependence; independent training and evaluation draws represent the two sources of sampling variation.

For a fixed polynomial space and comparable training and evaluation samples, write $C_P$ for the covariance operator using population projection residuals, $G$ for the evaluation fluctuation, and $Z_\eta$ for the joint coefficient-and-preprocessing fluctuation. Appendix~\ref{app:preprocessing} establishes
\[
\sqrt n(\widehat C-C_P)\Rightarrow G+\sqrt\lambda\mathcal J Z_\eta,
\qquad n/m\to\lambda\in(0,\infty).
\]
The derivative $\mathcal J$ acts on both residual and regressor features. It is computed directly from Gaussian kernel derivatives. The bootstrap calibrates squared operator fluctuations at all-zero operators and a linear contrast at regular laws, then takes the maximum of the two p-values. The guarantee remains pointwise on the stated null strata, under explicit moments, dependence, and training/evaluation coupling conditions.

If the fitted spaces contain both conditional means, $C_P=C$ and the result applies to original DII. Under misspecification the same algorithm instead estimates a projection-residual contrast. A bootstrap does not repair that target discrepancy. To preserve the original DII interpretation, let $r_f,r_b$ bound population mean-approximation errors in standardized outcome units, and let $t$ be a joint bootstrap radius for operator estimation error. Proposition~\ref{prop:meansensitivity} gives
\[
D\geq\bigl[\max\{0,\sqrt{\widehat H_b}-t-2r_b\}\bigr]^2
       -\bigl(\sqrt{\widehat H_f}+t+2r_f\bigr)^2.
\]
A positive lower bound supports the original-DII sign conditional on the stated approximation bounds. With a common error bound $r$, the tolerable error is
$(\sqrt{\widehat H_b}-\sqrt{\widehat H_f}-2t)/4$ when this quantity is positive. This is a sensitivity threshold, not an estimate of actual specification error. The software reports no original-DII sign certificate if no approximation bound is supplied.

This distinction matters for financial model validation: residual dependence can reflect an omitted nonlinear mean, rather than an irreducible directional feature. Sen and Sen (2014) already study residual-HSIC tests of regression specification and error independence. Our contribution is the joint dynamic contrast across directions and horizons, with its mixed null rates, fitted-scale propagation, and explicit translation back to the original conditional-mean target. We do not claim the general principle of residual testing or two-step inference as new.

The orthogonal estimator remains an extension with a second-order nuisance-rate condition. Its substantial power cost, including with known nuisance functions, is documented in Appendices~\ref{app:orthogonal} and \ref{app:diagnostics}. The preferred implementation retains original sample DII and includes joint coefficient, scale, and centering uncertainty.

'''
s=s[:a]+main+s[b:]
rows=[]
for rho in (0.,.6):
 for design,label in [('double','Double zero'),('equal_positive','Equal positive'),('weak','Weak exposure'),('strong','Strong exposure')]:
  d={x['method']:x for x in r['summary'] if x['rho']==rho and x['design']==design};j=d['joint_scales'];ci=j['wilson95']
  rows.append(f"{rho:.1f} & {label} & {100*d['coefficient_only']['rate']:.1f} & {100*j['rate']:.1f} & [{100*ci[0]:.1f}, {100*ci[1]:.1f}]"+r' \\')
new=r'''
\subsection{Joint coefficient and scale uncertainty: 3,000-dataset validation}\label{sec:jointvalidation}
A new validation fixes its design before execution and estimates every scale on the training rows. There are 300 replications for each of four designs at persistence zero and 0.6, plus 600 misspecification stress datasets described below. Each dataset uses 204 training rows, a 39-row gap, 120 evaluation rows, 399 bootstrap draws, and nonoverlapping blocks of four. The four laws are the double-zero, equal-positive, weak quadratic, and strong quadratic designs detailed in Appendix~\ref{app:diagnostics}. Coefficient-only and joint procedures share all point estimates and evaluation draws; the latter also propagates scale and centering uncertainty in the common training draw.

\begin{table}[htbp]\centering
\caption{Estimated-scale validation: rejection percentages}
\begin{tabular}{rlrrl}\toprule
$\rho$ & Design & Coefficients only & Joint scales & Joint Wilson 95\% CI\\\midrule
% ROWS
\bottomrule\end{tabular}
\par\vspace{5pt}\raggedright\noindent Each row uses 300 datasets; intervals measure Monte Carlo uncertainty, not empirical effect uncertainty. The two methods use identical DII estimates. The primary earlier 4,800-dataset study and the separate 800-dataset fixed-scale validation are retained, with the latter in the companion.
\end{table}

Joint equal-positive null rejection is 3.0\% under independence and 4.3\% with persistence, with Wilson intervals of approximately [1.6\%,5.6\%] and [2.5\%,7.3\%]. These results are compatible with nominal five-percent size in these designs; they do not establish uniform size or exact finite-sample calibration. Double-zero rejection is 0\% and 0.7\%. Weak-exposure detection is 25.7\% and 34.7\%, and strong-exposure detection is 71.3\% and 70.3\%. Including scale uncertainty is required by the fitted-unit expansion; a universal power gain is neither expected nor claimed.

\paragraph{An omitted mean can manufacture apparent direction.}
Let $A_t,B_t$ be independent stationary Gaussian AR(1) sequences and let $C_t$ be independent iid standard normal. Set
\[
X_t=0.3(C_t^4-6C_t^2+3)+A_t,\qquad Y_t=B_t,
\]
and include $C_t$ with the focal predictor in both regressions. The true forward and reverse residuals are $B_t$ and $A_t$, each independent of its full contemporaneous regressor vector, so original DII is zero. A basis $(1,z,z^2,C,C^2)$ omits the nonlinear reverse mean. Joint calibration nevertheless rejects in 113 of 300 iid datasets (37.7\%) and 102 of 300 persistent datasets (34.0\%). These are erroneous directional conclusions if attributed to original DII; they are not tests of a true zero projection contrast. Adding the prespecified $C^3,C^4$ terms yields zero rejections in both cells, with Wilson upper bounds of approximately 1.3\%.

The experiment demonstrates a target error that more bootstrap draws or scale propagation cannot cure. The approximation bound in Section~\ref{sec:training} exposes the additional requirement for an original-DII claim. It does not infer that a quartic regression is generally correct or that agreement across polynomial specifications identifies a financial mechanism.

'''.replace('% ROWS','\n'.join(rows))
s=s.replace(r'\section{A reproducible financial application}',new+r'\section{A reproducible financial application}')
f=json.loads((P/'finance/joint_nuisance_results.json').read_text());assert not any(v['p_intersection']<=.05 for v in f['comparisons'].values())
a=s.index(r'\paragraph{Secondary training-aware audit.}');b=s.index(r'\section{Financial transmission and the use of DII}',a)
s=s[:a]+r'''
\paragraph{Joint coefficient-and-scale audit.}
A secondary analysis propagates the fitted coefficients, training-derived standard deviations, and regressor centers in all 12 cells. Original sample DII values are reproduced exactly. With 999 bootstrap draws, no cell rejects before or after Holm adjustment, and no simultaneous directional lower bound is positive even with zero mean-approximation error. Full p-values and bounds are archived. This resolves the earlier omission of fitted-scale uncertainty, but does not establish that polynomial projections equal the financial conditional means or that the stability and dependence assumptions hold. The result therefore supplies neither a new original-DII sign finding nor a positive robustness radius. The primary protocol and results remain unchanged.

'''+s[b:]
abstract="""An economic shock can alter an outcome's mean, change its risk, or generate an asymmetric relationship between opposing dynamic models. The Directional Irreversibility Index (DII) separates this directional structure from mean responses and general dependence. Financial benchmarks distinguish nonlinear exposure from risk dependence and a persistent footprint from delayed information absorption. We develop joint block inference that retains sample DII while propagating uncertainty in fitted regressions and training-derived scales across dependent horizons. Its quadratic and regular calibrations address distinct null configurations. A sensitivity bound states how accurately the regressions must approximate conditional means before a projection-residual contrast supports an original-DII claim. In 3,000 new simulated datasets, the method accommodates estimated scales, while an omitted nonlinear mean exposes why bootstrap calibration alone cannot establish directionality. Simultaneous component bounds separate asymmetry from approximate model validity. A fixed four-market application yields no primary directional rejection. The framework supports financial model validation by distinguishing sampling uncertainty, mean misspecification, and the additional assumptions needed to interpret a statistical footprint as economic transmission."""
s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',s,flags=re.S)
s=s.replace('including a training-aware bootstrap that retains the original estimator at comparable sample sizes','including a joint bootstrap for regression coefficients and training-derived scales at comparable sample sizes')
s=s.replace('The training-aware implementation improves weak-exposure detection in a new-seed validation; simultaneous bounds and orthogonal correction have substantial small-sample costs.','A 3,000-dataset validation evaluates estimated scales and exposes misspecification-induced directionality; an explicit bound states what is needed to recover the original-DII interpretation. Simultaneous bounds and orthogonal correction have substantial small-sample costs.')
s=s.replace('For correctly specified fixed-dimensional means, training-aware calibration retains the original estimator and includes coefficient uncertainty at comparable sample sizes. Its new-seed validation improves weak-exposure detection in the studied designs.','For fixed polynomial means, joint calibration retains sample DII and includes coefficient and scale uncertainty at comparable sample sizes. The original conditional-mean interpretation requires correct means or a justified approximation bound. The new validation reports both the estimated-scale performance and a sharp misspecification failure.')
s=s.replace('The empirical application provides a reproducible test of the proposed financial use, with null primary results and no demonstrated positive predictive gain.','The empirical application provides a reproducible test of the proposed financial use, with null primary results, no positive simultaneous directional lower bound, and no demonstrated positive predictive gain.')
s=s.replace('Standardization estimated from training is additional nuisance estimation and needs an augmented expansion or separately justified treatment; the validation experiment therefore uses fixed population scales.', 'Standardization estimated from training is additional nuisance estimation. Appendix~'+r'\ref{app:preprocessing}'+' supplies the augmented expansion; the earlier coefficient-only validation deliberately uses fixed population scales.')
# Credit a close antecedent explicitly in the related-work discussion and references.
needle='The full joint law is essential because independence across directions does not establish consistency for a difference.'
s=s.replace(needle,needle+' Sen and Sen (2014) derive residual-HSIC specification and independence tests with bootstrap calibration. That antecedent rules out claiming residual-dependence testing as our innovation; our focus is the joint signed dynamic contrast and its target interpretation.')
s=s.replace(r'\end{thebibliography}',r'\bibitem{SenSen} Sen, A. and B. Sen (2014). On Testing Independence and Goodness-of-fit in Linear Models. arXiv:1302.5831v2. \url{https://arxiv.org/abs/1302.5831}.'+'\n'+r'\end{thebibliography}')

# Keep the conclusion compact and prevent single-line paragraph spillovers.
a=s.index(r'\section{Conclusion}');b=s.index(r'\begin{thebibliography}',a)
s=s[:a]+r'''
\section{Conclusion}
DII measures directional irreversibility in competing dynamic residual representations. It is distinct from a linear response and from general distributional dependence. The financial examples explain why each distinction matters, including why a persistent footprint need not imply delayed information absorption.

Joint block inference propagates fitted regression, scale, and centering uncertainty across dependent horizons and distinct null regimes. Its original-DII interpretation requires correct conditional means or justified approximation bounds. The new validation evaluates estimated preprocessing and demonstrates the directional error an omitted mean can create. The orthogonal alternative and conservative component bounds remain documented extensions. The financial application supplies no primary rejection or demonstrated positive predictive gain. The paper's contribution is an explicit account of what directional evidence measures, how it is estimated, and which additional assumptions connect it to a financial mechanism.

\clearpage
'''+s[b:]
s=s.replace(r'\begin{document}',r'\widowpenalty=10000\clubpenalty=10000'+'\n'+r'\begin{document}')

(S/'main.tex').write_text(s);(P/'abstract.txt').write_text(abstract+'\n');print('Joint nuisance revision integrated; abstract words',len(abstract.split()))
