"""the journal submission stage, round 3.

Adds (i) a main-text subsection and figure showing each proposition in a toy
design with known answers, with a fitted-inference table applying Theorem 1;
(ii) a companion appendix with the designs; (iii) a companion appendix with a
simple public-data illustration of the predictability bound (U.S. T-bill rate
and unemployment, statsmodels macrodata). Numbers are read from
theory_demos_results.json. No identified shock or asset-price outcome is used,
so nothing here overlaps the separate journal application. Existing results are
unchanged.
"""
from pathlib import Path
import json

S = Path(__file__).resolve().parent
P = S.parent
T = S / 'main.tex'
tex = T.read_text()
d = json.load(open(P / 'theory_demos_results.json'))


def rep(old, new, count=1):
    global tex
    n = tex.count(old)
    assert n == count, (n, old[:100])
    tex = tex.replace(old, new)


A = {r['phi']: r for r in d['A_predictability']['rows']}
B = d['B_state_omission']['rows']
pop = [r['dii'] for r in d['C_state_error_population'] if r['sigma'] == 1.0 and r['q'] == 0.0][0]
Dp = {r['h']: r for r in d['D_incorporation']['population_by_h']}
Dr = d['D_incorporation']['fitted_joint_rejection']
share95 = A[0.95]['reverse_hsic']['mean'] / A[0.0]['reverse_hsic']['mean']

# ---------------------------------------------------------------- main-text subsection
rows = ''.join(
    f"{r['h']} & {Dp[r['h']]['dii_inclusive']['mean']:.4f} & {100*r['reject_inclusive']:.1f} & {100*r['reject_post_impact']:.1f}\\\\\n"
    for r in Dr['rows'])
SUB = rf"""
\subsection{{The propositions at work: toy examples with known answers}}\label{{sec:toys}}
Figure~\ref{{fig:toys}} shows each population result in a design whose answer is known. Appendix~\ref{{app:toys}} gives the designs; all use fixed seeds and the kernels of Section~\ref{{sec:object}}.

Panel A draws a unit-variance AR(1) driver with $\varphi$ between 0 and 0.99 and the quadratic exposure $Y_{{t+1}}=X_t^2-1+0.5\varepsilon_{{t+1}}$, whose forward residual is independent of $(X_t,X_{{t-1}})$. The estimated reverse component, which here equals DII up to estimation error, falls from {A[0.0]['reverse_hsic']['mean']:.4f} at $\varphi=0$ to {A[0.99]['reverse_hsic']['mean']:.5f} at $\varphi=0.99$, in line with the rate $1-\varphi^2$ in Proposition~\ref{{prop:innovation}}. The level of the bound is conservative in this design; its content is the rate. A driver with $\varphi=0.95$ carries about {100*share95:.0f} percent of the directional signal of an unpredictable driver with the same exposure.

Panel B draws the design of Proposition~\ref{{prop:stateerasure}} with $\sigma=1$. As the sample grows from {B[0]['n']} to {B[-1]['n']}, state-conditioned DII converges to its population value {pop:.4f}. Each pooled component shrinks toward zero at the rate of its estimation bias, and pooled DII is identically zero. More pooled data do not recover the directional structure. Panel C plots population values from the state-observation experiment. DII need not fall monotonically with misclassification---at $\sigma=1$, ten-percent errors raise it---but it vanishes when the observed state is uninformative. The perturbation bound of Proposition~\ref{{prop:statestability}} is sufficient but conservative: in this family it certifies positive DII only for error rates below about 0.3 percent.

Panel D simulates the martingale price of Proposition~\ref{{prop:incorporation}} with $\gamma=0.5$. Impact-inclusive DII is positive at every horizon and decays slowly, whereas post-impact DII is zero: the persistent footprint reflects only the retained impact. Table~\ref{{tab:toyinference}} applies Theorem~\ref{{thm:main}} to the same design with fitted quadratic means, {Dr['m']} training and {Dr['n']} evaluation observations, and {Dr['draws']} bootstrap draws. The joint test never rejects for post-impact outcomes, a double-independence null at which the procedure is conservative. It detects the inclusive footprint in {100*Dr['rows'][0]['reject_inclusive']:.0f} percent of samples at $h=1$ but only {100*Dr['rows'][-1]['reject_inclusive']:.1f} percent at $h=6$. The design warning therefore comes with a power warning: at financial sample sizes, a long-horizon footprint can be present and still hard to detect.

\begin{{figure}}[htbp]\centering
\includegraphics[width=\linewidth]{{../dii_theory_demos.pdf}}
\caption{{Toy examples with known answers}}\label{{fig:toys}}
\par\raggedright\footnotesize Panel A: Proposition~\ref{{prop:innovation}}; means and 5--95 percent ranges over {d['A_predictability']['reps']} samples of {d['A_predictability']['n']} with the bound $1-\varphi^2$ (log scales). Panel B: Proposition~\ref{{prop:stateerasure}}; oracle residuals, 40--100 samples per size. Panel C: Proposition~\ref{{prop:statestability}}; population values computed by quadrature in the state-observation experiment. Panel D: Proposition~\ref{{prop:incorporation}}; oracle residuals, samples of 2,000.
\end{{figure}}

\begin{{table}}[htbp]\centering\small
\caption{{Theorem~\ref{{thm:main}} in the immediate-incorporation design: rejection percentages at nominal 5\%}}\label{{tab:toyinference}}
\begin{{tabular}}{{rccc}}\toprule
$h$ & Population inclusive DII & Inclusive outcome & Post-impact outcome\\\midrule
{rows}\bottomrule\end{{tabular}}
\par\vspace{{4pt}}\raggedright\noindent {Dr['rows'][0]['reps']} samples per horizon; fitted total-degree-two means in both directions (correctly specified), joint coefficient and scale uncertainty. Post-impact outcomes satisfy the double-independence null; inclusive outcomes have positive DII. Monte Carlo standard errors are at most 3.5 percentage points.
\end{{table}}
"""
anchor = "\\section{A reproducible financial application}\\label{sec:application}"
rep(anchor, SUB.strip() + "\n\n" + anchor)

rep("These experiments establish design-specific performance, not general diagnostic dominance.",
    "Toy examples with known answers show each proposition at work (Figure~\\ref{fig:toys}). These experiments establish design-specific performance, not general diagnostic dominance.")

# ---------------------------------------------------------------- application: pointer to macro illustration
E = d['E_macro']['results']
pmin = min(v['p_intersection'] for g in E.values() for v in g.values())
rep("The application therefore demonstrates how the proposed contrast can be implemented and reported alongside conventional financial questions.",
    f"Appendix~\\ref{{app:macro}} adds a second, simple public-data illustration unrelated to identified shocks: the level of the three-month Treasury bill rate and its innovation as drivers of later unemployment changes. It shows how to compute and report the predictability ceiling of Proposition~\\ref{{prop:innovation}}; no contrast rejects (smallest p-value {pmin:.3f}). "
    "The application therefore demonstrates how the proposed contrast can be implemented and reported alongside conventional financial questions.")

# ---------------------------------------------------------------- companion appendices
Arows = ''.join(f"{r['phi']:.2f} & {r['bound']:.4f} & {r['mean_sq_reverse_residual']['mean']:.4f} & {r['reverse_hsic']['mean']:.6f} & {r['forward_hsic']['mean']:.6f} & {r['dii']['mean']:.6f}\\\\\n"
                for r in d['A_predictability']['rows'])
Brows = ''.join(f"{r['n']} & {r['reps']} & {r['dii_state_conditioned']['mean']:.5f} & [{r['dii_state_conditioned']['q05']:.5f}, {r['dii_state_conditioned']['q95']:.5f}] & {r['pooled_component']['mean']:.5f}\\\\\n"
                for r in B)
Drows = ''.join(f"{h} & {Dp[h]['dii_inclusive']['mean']:.5f} & [{Dp[h]['dii_inclusive']['q05']:.5f}, {Dp[h]['dii_inclusive']['q95']:.5f}] & {Dp[h]['dii_post_impact']['mean']:.5f}\\\\\n" for h in sorted(Dp))
names = {'level': 'Level', 'innovation': 'AR(2) innovation'}
Erows = ''
for g, res in E.items():
    for k, v in res.items():
        Erows += (f"{names[g]} & {k[1:]} & {v['forward_hsic']:.4f} & {v['reverse_hsic']:.4f} & {v['dii']:.4f} & {v['p_intersection']:.3f} & "
                  f"{v['mean_sq_reverse_residual']:.3f} & {v['history_only_share']:.3f}\\\\\n")
EC = rf"""
\section{{Designs for the toy examples}}\label{{app:toys}}
All experiments are produced by \texttt{{theory\_demos.py}} with seed {20260919}. Kernels are Gaussian with bandwidth one, in the coordinates stated for each design. HSIC is the biased V-statistic of the squared Hilbert--Schmidt norm used throughout the paper.

\paragraph{{Predictability.}} $X_t=\varphi X_{{t-1}}+\sqrt{{1-\varphi^2}}\eta_t$ with standard normal $\eta_t$, $C_t=X_{{t-1}}$, and $Y_{{t+1}}=X_t^2-1+0.5\varepsilon_{{t+1}}$. The forward residual is the true $\varepsilon$ (standardized by the sample deviation of $Y$), so $H_f=0$ in population. The reverse residual is the projection residual of $X_t$ on a total-degree-two polynomial in standardized $(Y_{{t+1}},C_t)$, kept in the unit-variance units of $X$. Because the basis contains $C_t$ linearly, its mean square is at most $1-\varphi^2$ in population, so the proof of Proposition~\ref{{prop:innovation}} applies to it. Each row averages {d['A_predictability']['reps']} samples of {d['A_predictability']['n']}.

\begin{{table}}[htbp]\centering\small
\caption{{Predictability design}}
\begin{{tabular}}{{rrrrrr}}\toprule
$\varphi$ & Bound $1-\varphi^2$ & Mean $\hat u_b^2$ & $\widehat H_b$ & $\widehat H_f$ & $\widehat D$\\\midrule
{Arows}\bottomrule\end{{tabular}}
\end{{table}}

\paragraph{{State omission.}} The design of Proposition~\ref{{prop:stateerasure}} with $\sigma=1$ and oracle residuals $u_f=\varepsilon$, $u_b=X-S\tanh(Y)$, regressors $(X,S)$ and $(Y,S)$. Pooled conditional means are zero, so pooled residuals are $Y$ and $X$ and both pooled components equal $\widehat{{\mathrm{{HSIC}}}}(X,Y)$.

\begin{{table}}[htbp]\centering\small
\caption{{State-omission design}}
\begin{{tabular}}{{rrrcr}}\toprule
$n$ & Samples & State-conditioned DII & 5--95\% range & Pooled component\\\midrule
{Brows}\bottomrule\end{{tabular}}
\par\vspace{{4pt}}\raggedright\noindent Population state-conditioned DII is {pop:.5f}; pooled population components are zero.
\end{{table}}

\paragraph{{Imperfect states.}} Panel C of Figure~\ref{{fig:toys}} plots the frozen population values from Appendix~\ref{{app:stateobservation}}. With $\ell_u=\ell_S=1$, $\delta(q)\approx10.6q$ for small $q$, so the sufficient condition $\delta(q)<\sqrt{{D_0}}/2$ holds only for $q$ below about 0.003.

\paragraph{{Immediate incorporation.}} $\Delta P_t=0.5(S_t^2-1)+e_t$ with independent standard normal $S_t,e_t$. The population panel uses oracle residuals in samples of 2,000 (ten per horizon): the inclusive forward residual removes $0.5(S_t^2-1)$ and the reverse residual is $S_t$; both post-impact residuals are the raw variables, so post-impact DII is zero by symmetry. The inference experiment fits total-degree-two means in both directions with {Dr['m']} training rows, a {Dr['gap']}-row gap, and {Dr['n']} evaluation rows, uses the joint procedure of Theorem~\ref{{thm:main}} with {Dr['draws']} draws, and repeats {Dr['rows'][0]['reps']} times per horizon.

\begin{{table}}[htbp]\centering\small
\caption{{Immediate-incorporation design: population DII by horizon}}
\begin{{tabular}}{{rrcr}}\toprule
$h$ & Inclusive DII & 5--95\% range & Post-impact DII\\\midrule
{Drows}\bottomrule\end{{tabular}}
\end{{table}}

\section{{A public-data illustration of the predictability ceiling}}\label{{app:macro}}
\paragraph{{Question and data.}} Does a highly predictable driver leave room for directional structure? The illustration uses the public-domain quarterly U.S. series distributed with \texttt{{statsmodels}} (\texttt{{macrodata}}, 1959Q1--2009Q3; FRED and BLS sources). The driver is the three-month Treasury bill rate, either its level or its innovation relative to a training-sample AR(2). The outcome is the one-quarter change in the unemployment rate at $t+h$, $h\in\{{1,2,4\}}$. The history is the lagged driver and the current unemployment change. This is not an identified-shock design and makes no causal claim; it involves no asset-price outcome.

\paragraph{{Design.}} The specification was fixed before estimation and not altered afterwards. Origins run from 1959Q4: 96 training quarters, a four-quarter gap, and 96 evaluation quarters, common to all horizons. Both directions use total-degree-two polynomial means and the joint coefficient-and-preprocessing procedure of Theorem~\ref{{thm:main}} with 999 draws and common paths across horizons for each driver.

\paragraph{{The ceiling in sample.}} The proof of Proposition~\ref{{prop:innovation}} applies verbatim to the empirical measure: with Gaussian kernels of bandwidth one, the estimated reverse component never exceeds the evaluation mean square of the reverse residual. The column ``Ceiling'' reports that quantity; ``History share'' is the evaluation mean square of the standardized driver after removing its training-sample projection on the history alone, the empirical counterpart of $v_X$.

\begin{{table}}[htbp]\centering\small
\caption{{T-bill rate and unemployment changes: DII and the predictability ceiling}}\label{{tab:macro}}
\begin{{tabular}}{{lrrrrrrr}}\toprule
Driver & $h$ & $\widehat H_f$ & $\widehat H_b$ & $\widehat D$ & Joint $p$ & Ceiling & History share\\\midrule
{Erows}\bottomrule\end{{tabular}}
\par\vspace{{4pt}}\raggedright\noindent Standardized kernel units using training-sample scales. Evaluation quarters have lower interest-rate volatility than training quarters, so the innovation's history share is well below one in training units.
\end{{table}}

\paragraph{{Reading.}} For the level, the history leaves only about 5 percent of the driver's training-period variance, so its reverse component cannot exceed roughly 0.05 whatever the unemployment response. In this sample both ceilings far exceed the estimated components, no contrast rejects, and the evidence is uninformative about direction. The illustration shows how to compute and report the ceiling alongside DII; it does not show that the ceiling binds in these data.
"""
rep("\\end{document}", EC.strip() + "\n\\end{document}")

T.write_text(tex)
print('revision_r3: applied')
