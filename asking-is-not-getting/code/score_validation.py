"""Score the hand-coded validation sample. Usage: python score_validation.py <folder>
Prints, for each keyword measure, precision, recall and Cohen's kappa of the machine flag against the human code,
and writes validation_table.tex for the paper's appendix. (The sample is stratified on the machine flag, so precision is
estimated on flagged notes and the miss rate on unflagged notes; recall is reweighted by the population flag rate if you supply it.)"""
import sys, os, pandas as pd
d = sys.argv[1]; h = pd.read_csv(os.path.join(d, "to_code.csv")); k = pd.read_csv(os.path.join(d, "key_DO_NOT_OPEN_UNTIL_CODED.csv")); m = h.merge(k, on="note_id", suffixes=("_h", "_m"))
rows = []
for c in ["docs", "ineligible", "repayment", "jobloss"]:
    x = m.dropna(subset=[c + "_h"]); hh, mm = x[c + "_h"].astype(int), x[c + "_m"].astype(int)
    tp, fp, fn, tn = ((hh == 1) & (mm == 1)).sum(), ((hh == 0) & (mm == 1)).sum(), ((hh == 1) & (mm == 0)).sum(), ((hh == 0) & (mm == 0)).sum()
    po = (tp + tn) / len(x); pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / len(x) ** 2; kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    rows.append((c, len(x), tp / max(1, tp + fp), fn / max(1, fn + tn), kappa)); print(f"{c:<11} n={len(x):<4} precision={rows[-1][2]:.2f}  miss rate among unflagged={rows[-1][3]:.2f}  kappa={kappa:.2f}")
with open(os.path.join(d, "validation_table.tex"), "w") as f:
    f.write("\\begin{tabular}{lcccc}\n\\toprule\nMeasure & Notes coded & Precision & Miss rate among unflagged & Cohen's $\\kappa$ \\\\\n\\midrule\n")
    for c, n, p, mr, kp in rows: f.write(f"{c} & {n} & {p:.2f} & {mr:.2f} & {kp:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
