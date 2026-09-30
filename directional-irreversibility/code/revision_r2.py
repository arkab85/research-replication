"""the journal submission stage, round 2 (editorial only).

Runs after submission_revision.py. It
  * leads the abstract and introduction with the calibration failure that the
    inference repairs (frozen baseline-study rates, no new simulation);
  * states the main inferential result as Theorem 1 in the main text (a
    restatement of the companion's joint coefficient/preprocessing theorem);
  * moves the research-design table into the introduction;
  * adds a two-panel calibration figure built by figures_main.py from frozen output;
  * rewrites the conclusion as a finance agenda.
No proof, estimand, simulation value, or financial estimate changes.
"""
from pathlib import Path

S = Path(__file__).resolve().parent
T = S / 'main.tex'
tex = T.read_text()


def rep(old, new, count=1):
    global tex
    n = tex.count(old)
    assert n == count, (n, old[:100])
    tex = tex.replace(old, new)


# ------------------------------------------------------------------ abstract
rep("For inference, a joint block "
    "bootstrap for fitted horizon profiles propagates regression and preprocessing uncertainty and "
    "accommodates the nonregular null in which both dependence components vanish. In a 3,000-dataset "
    "validation the procedure rejects equal-positive nulls in 3.0--4.3 percent of replications and detects "
    "the stronger quadratic alternatives in about 70 percent at 120 evaluation observations. A prespecified four-market "
    "monetary-shock illustration finds no significant primary directional evidence and shows how an "
    "inconclusive profile should be reported.",
    "Inference is the central difficulty: calibrating the index as if both dependence components vanished "
    "rejects a true null of equal, positive dependence in 41--47 percent of simulated samples at a nominal "
    "5 percent level. We develop a joint block bootstrap for fitted horizon profiles that accounts for both "
    "null configurations and propagates regression and preprocessing uncertainty; in a 3,000-dataset "
    "validation it rejects in 3.0--4.3 percent and detects strong quadratic exposures in about 70 percent at "
    "120 evaluation observations. In a prespecified four-market monetary-shock illustration, seven naive "
    "rejections do not survive joint inference.")
rep("Three "
    "analytical results translate the index into research-design guidance.",
    "Three analytical results yield research-design guidance.")

rep("studies the transmission of identified shocks, such as monetary-policy surprises, mainly through",
    "studies the transmission of identified shocks mainly through")
rep("a stability bound gives a local preservation condition under state misclassification.",
    "a stability bound covers state misclassification.")

# ------------------------------------------------------------------ intro: evidence paragraph
rep("The evidence separates population arguments from sampling performance. Existing simulations evaluate calibration, detection, and specification sensitivity: at the 120-observation evaluation size, the joint procedure rejects",
    "The inferential problem is practically important. A difference of two kernel dependence measures looks like a degenerate statistic, and calibrating it that way is natural. Yet when both representations are imperfect---the typical case in financial data---that calibration rejects a true null of equal dependence in 41--47 percent of simulated samples at a nominal five-percent level (Figure~\\ref{fig:calibration}A). Accounting for both null configurations restores size. With fitted coefficients and scales, and at the 120-observation evaluation size, the joint procedure rejects")
rep("None of its 12 primary intersection tests, "
    "which require rejection under both null configurations, rejects at five percent, although the "
    "degenerate-null component alone falls below 0.05 in seven cells.",
    "None of its 12 primary intersection tests, "
    "which require rejection under both null configurations, rejects at five percent, although the "
    "degenerate-null component alone falls below 0.05 in seven cells (Figure~\\ref{fig:calibration}B).")

# ------------------------------------------------------------------ move design table into the introduction
t0 = tex.index("\\begin{table}[htbp]\\centering\\small\n\\caption{Financial research choices informed by DII}")
t1 = tex.index("\\end{table}", t0) + len("\\end{table}\n")
table = tex[t0:t1]
tex = tex[:t0] + tex[t1:]
rep("Complete proofs and supplementary evidence appear in the electronic companion.\n",
    "Complete proofs and supplementary evidence appear in the electronic companion. Table~\\ref{tab:financedesign} summarizes how each result changes a choice in a financial shock-transmission study.\n\n" + table)
rep("Table~\\ref{tab:financedesign} connects those choices to the paper's results.",
    "Table~\\ref{tab:financedesign} in the introduction connects those choices to the paper's results.")

# ------------------------------------------------------------------ main theorem in the main text
rep("For a fixed polynomial space and comparable training and evaluation samples, write $C_P$ for the covariance operator using population projection residuals, $G$ for the evaluation fluctuation, and $Z_\\eta$ for the joint coefficient-and-preprocessing fluctuation. Appendix~\\ref{app:preprocessing} establishes\n\\[\n\\sqrt n(\\widehat C-C_P)\\Rightarrow G+\\sqrt\\lambda\\mathcal J Z_\\eta,\n\\qquad n/m\\to\\lambda\\in(0,\\infty).\n\\]",
    "Write $C_P$ for the vector of covariance operators built from population projection residuals, $G$ for the evaluation fluctuation, $Z_\\eta$ for the joint coefficient-and-preprocessing fluctuation, $\\mathcal J$ for the derivative of the operators with respect to those nuisance parameters, and $n$ and $m$ for the evaluation and training sizes. The main inferential result is the following.\n\n"
    "\\begin{theorem}[Joint inference for fitted DII profiles]\\label{thm:main}\n"
    "Suppose the data are strictly stationary and absolutely regular with the summability and independent-block coupling conditions of Appendix~\\ref{app:training}; each representation uses a fixed polynomial space, with a fixed finite set of directions and horizons, positive definite population design matrices, positive scales, and finite eighth moments; Gaussian residual and regressor kernels have fixed bandwidth after training-sample standardization; and $n/m\\to\\lambda\\in(0,\\infty)$. Then:\n"
    "\\begin{enumerate}\\itemsep0pt\n"
    "\\item jointly across components, $\\sqrt n(\\widehat C-C_P)\\Rightarrow G+\\sqrt\\lambda\\mathcal J Z_\\eta$;\n"
    "\\item the nonoverlapping block bootstrap with common evaluation and common training multipliers consistently estimates this joint limit;\n"
    "\\item for any fixed signed contrast $D_P=\\sum_d a_d\\|C_{P,d}\\|^2$, quadratic calibration at vanishing operators and linear calibration at regular laws are asymptotically valid under the corresponding nondegeneracy conditions, and the test based on the larger of the two p-values has pointwise asymptotic level at most $\\alpha$ on the covered null strata.\n"
    "\\end{enumerate}\n"
    "If the fitted spaces contain the conditional means, $C_P=C$ and the conclusions apply to the original DII.\n"
    "\\end{theorem}\n"
    "Theorem~\\ref{thm:main} restates Theorem~\\ref{thm:joint}, proved in Appendix~\\ref{app:preprocessing} together with the coefficient-only case. The first term of the limit is the evaluation-period fluctuation; the second is the price of learning the exposure and its units from a finite training period.")

# ------------------------------------------------------------------ figure in Section 4.1
rep("At an equal-positive null, wild-only rejection is 40.7--44.3 percent at a nominal five-percent level; paired and intersection rejection are both 5.0 percent.",
    "At an equal-positive null, wild-only rejection is 40.7--44.3 percent at a nominal five-percent level with large training samples and 41.3--47.3 percent with equal training and evaluation sizes; paired and intersection rejection are 4.7--6.0 percent (Figure~\\ref{fig:calibration}A).\n\n"
    "\\begin{figure}[htbp]\\centering\n\\includegraphics[width=\\linewidth]{../dii_calibration.pdf}\n"
    "\\caption{Calibration under the two null configurations}\\label{fig:calibration}\n"
    "\\par\\raggedright\\footnotesize Panel A: feasible estimators in the 4,800-dataset baseline study at the equal-positive null; bars are rejection rates at nominal 5 percent with Wilson 95 percent intervals over 300 replications. Paired and intersection decisions coincide in this study. Panel B: the 12 primary cells of Section~\\ref{sec:application}; open circles are degenerate-null (wild) p-values and squares are joint intersection p-values; the dashed line marks 0.05. Without the population restrictions, Panel B does not establish that the seven wild rejections are false positives.\n"
    "\\end{figure}\n")
rep("The wild component alone has seven p-values below 0.05.",
    "The wild component alone has seven p-values below 0.05 (Figure~\\ref{fig:calibration}B).")

# ------------------------------------------------------------------ conclusion
c0 = tex.index("\\section{Conclusion}\n") + len("\\section{Conclusion}\n")
c1 = tex.index("\\clearpage", c0)
tex = tex[:c0] + (
    "Mean exposure, conditional risk, and directional structure are distinct features of how a financial shock is transmitted. DII measures the third as a horizon-specific contrast between opposing residual representations. The paper shows when that contrast is observable and how to conduct inference on it with fitted models.\n\n"
    "Three design conclusions follow. Directional evidence should be sought in surprises rather than predictable levels, because the driver's innovation bounds the signal. Exposure states should be defined before estimation, because pooling can erase directional structure entirely. Outcome windows should separate retained impact from later adjustment, because an impact-inclusive footprint persists even when prices incorporate a shock immediately. On the inferential side, a difference of dependence measures cannot be calibrated as a degenerate statistic without severe over-rejection; joint inference that propagates training uncertainty restores size in the designs studied, at a documented cost in power.\n\n"
    "These results give applied work a disciplined way to add a distributional restriction to shock-response studies of monetary policy, credit conditions, institutional flows, and risk transmission. The monetary-shock illustration shows what responsible reporting looks like when the evidence is inconclusive. Establishing a particular directional mechanism remains the task of a design that identifies it; the framework here supplies the measurement and the inference such a design requires.\n\n"
) + tex[c1:]

rep("One horizon or a profile & Joint uncertainty differs across null configurations &",
    "One horizon or a profile & Degenerate-only calibration rejects a true equal-dependence null 41--47\\% of the time &")
rep("\\subsection{Inference for trained models and fitted units}\\label{sec:training}",
    "\\subsection{Main result: inference for trained models and fitted units}\\label{sec:training}")
tex = tex.replace("(Jord\\`a, 2005)", "(Jord\\`a 2005)").replace("(Gretton et al., 2012)", "(Gretton et al. 2012)")

T.write_text(tex)
ab = tex[tex.index('\\begin{abstract}') + 17:tex.index('\\end{abstract}')].strip().replace('--', '–')
(S / 'cover_letter.md').write_text((S / 'v_cover_letter.md').read_text().replace('@@ABSTRACT@@', ab))
(S / 'editor_strategy.md').write_text((S / 'v_editor_strategy.md').read_text())
print('revision_r2: applied')
