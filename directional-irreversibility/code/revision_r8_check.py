"""External-review corrections (runs after revision_r7_singular.py; both versions inherit).
1. Two positive components do not imply a near-zero contrast; common news is only a possible explanation.
2. Component rejections concern the specified polynomial representations, not independence under correct means.
3. Theorem 1 states the full moment and basis conditions of the appendix theorem.
4. Proposition 7 inherits those conditions for the augmented variables.
5. A paragraph stating what a researcher can conclude and decide from each application's bounds.
"""
from pathlib import Path
import json
import pandas as pd
S = Path(__file__).resolve().parent; P = S.parent
T = S / 'main.tex'; tex = T.read_text()


def rep(old, new, count=1):
    global tex
    n = tex.count(old)
    assert n == count, (n, old[:100])
    tex = tex.replace(old, new)


# 1-2: oil-VIX interpretation
rep("the data reject residual independence in the oil-to-VIX representation and in its reverse. Neither representation is admissible within the maintained class, so the contrast has no ordering interpretation, and a DII near zero is what one would expect under common news that moves both markets on the same day.",
    "both fitted representations reject residual independence, and the contrast itself does not reject equality. The rejections concern the specified polynomial representations; without an external mean-approximation bound they do not rule out independence under correctly specified conditional means. Two positive components place no restriction on their difference, so the data do not order the directions at impact. Same-day common news is one possible explanation, not an implication of these results.")
rep("so neither direction is supported and a near-zero DII is the expected reading.",
    "and the contrast does not reject equality, so the data do not order the directions at impact.")
rep("and oil and VIX data reject independent disturbances in both directions at impact.",
    "and oil and VIX data reject residual independence in both fitted representations at impact.")

# 3: Theorem 1 conditions
rep("each representation uses a fixed polynomial space, with a fixed finite set of directions and horizons, positive definite population design matrices, positive scales, and finite eighth moments;",
    "each representation uses a fixed polynomial space invariant under coordinatewise nonsingular affine transformations, with a fixed finite set of directions and horizons, positive definite population design matrices, and positive scales; the raw variables, the basis coordinates, and the projection residuals have finite eighth moments (for quadratic or cubic bases this requires sixteenth or twenty-fourth moments of the raw variables);")
# 4: Proposition 7
rep("Suppose the conditions of Theorem~\\ref{thm:main} hold with an observed factor $f$ that has finite eighth moments.",
    "Suppose the conditions of Theorem~\\ref{thm:main} hold for both representations with the observed factor $f$ included, so that $f$, the basis coordinates in $Y-\\lambda f$, and the resulting projection residuals have finite eighth moments.")

# 5: what a researcher can conclude and decide
b = pd.read_csv(P / 'finance/simultaneous_bounds.csv')
lo, hi = b.dii_lower.min(), b.dii_upper.max()
cu = max(b.forward_upper.max(), b.reverse_upper.max())
td = json.load(open(P / 'theory_demos_results.json'))
bench = [td['A_predictability']['rows'][0]['reverse_hsic']['mean'], 0.00504,
         td['D_incorporation']['population_by_h'][0]['dii_inclusive']['mean']]
DEC = (f"\\paragraph{{What a researcher can conclude and decide.}} The two applications support different conclusions. In the monetary application, the simultaneous "
       f"bands place every primary contrast between ${lo:.3f}$ and ${hi:.3f}$ in standardized kernel units, and every component below {cu:.3f}. These bands are wider than "
       f"the contrasts in the paper's benchmark designs ({min(bench):.3f}--{max(bench):.3f}; Figure~\\ref{{fig:toys}}). The sample therefore cannot rule out directional "
       f"structure of that size. The design implication is concrete: more evaluation periods, pooled defactored panels (Corollary~\\ref{{cor:panel}}), or a sharper surprise "
       f"measure are needed before a nonrejection can inform a transmission claim. In the oil and VIX application, the impact-day rejections bear on model validation: "
       f"a scenario analysis that draws the fitted VIX residual independently of the same-day oil move is inconsistent with the data for this specification. The component "
       f"bounds of Section~\\ref{{sec:replay}} quantify the error of such independent reuse for bounded payoffs.\n\n")
anchor = "\\section{Financial transmission and the use of DII}"
rep(anchor, DEC + anchor)
T.write_text(tex)

cl = S / 'cover_letter.md'; c = cl.read_text()
c = c.replace("at impact, both representations reject residual independence, so the index correctly reports no directional ordering.",
              "at impact, both fitted representations reject residual independence while the contrast does not reject equality, so the data do not order the directions and independent reuse of the fitted residuals in scenario analysis is ruled out.")
a = tex[tex.index('\\begin{abstract}') + 16:tex.index('\\end{abstract}')]
assert len(a.split()) <= 250
st = c.index('Abstract:\n') + len('Abstract:\n'); en = c.index('\n\nSuggested Associate Editors')
c = c[:st] + a.strip().replace('--', '–') + c[en:]
cl.write_text(c)
print('revision_r8_check: applied')
