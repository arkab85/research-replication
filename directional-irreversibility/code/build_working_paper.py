from __future__ import annotations

import argparse
import csv
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


BLUE = "1F4E79"
PALE_BLUE = "EAF2F8"
LIGHT_GRAY = "D9D9D9"
BLACK = "000000"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color: str = LIGHT_GRAY, size: str = "6") -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_keep_with_next(paragraph, value=True) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    element = p_pr.find(qn("w:keepNext"))
    if value and element is None:
        p_pr.append(OxmlElement("w:keepNext"))
    elif not value and element is not None:
        p_pr.remove(element)


def set_font(run, name="Times New Roman", size=11, bold=False, italic=False, color=BLACK) -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def add_runs(paragraph, pieces) -> None:
    for text, kwargs in pieces:
        run = paragraph.add_run(text)
        set_font(run, **kwargs)


def add_body(doc, text: str, first_line=True, space_after=6) -> None:
    p = doc.add_paragraph()
    p.style = doc.styles["Normal"]
    p.paragraph_format.line_spacing = 1.08
    p.paragraph_format.space_after = Pt(space_after)
    if first_line:
        p.paragraph_format.first_line_indent = Inches(0.25)
    run = p.add_run(text)
    set_font(run, size=11)


def add_lead_body(doc, lead: str, text: str) -> None:
    p = doc.add_paragraph()
    p.style = doc.styles["Normal"]
    p.paragraph_format.line_spacing = 1.08
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.first_line_indent = Inches(0.25)
    add_runs(p, [(lead, {"size": 11, "bold": True}), (text, {"size": 11})])


def add_heading(doc, text: str, level=1) -> None:
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(5)
    set_keep_with_next(p)
    run = p.add_run(text)
    set_font(run, size=15 if level == 1 else 12.5, bold=True)


def add_bullet(doc, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    set_font(run, size=10.5)


def add_formula(doc, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(text)
    set_font(run, name="Consolas", size=10.2)


def add_table(doc, headers, rows, widths, font_size=8.8) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header = table.rows[0]
    set_repeat_table_header(header)
    for idx, text in enumerate(headers):
        cell = header.cells[idx]
        cell.width = Inches(widths[idx])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, BLUE)
        set_cell_margins(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(str(text))
        set_font(r, size=font_size, bold=True, color="FFFFFF")
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            cell = cells[idx]
            cell.width = Inches(widths[idx])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if ridx % 2 == 1:
                set_cell_shading(cell, PALE_BLUE)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(str(value))
            set_font(r, size=font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_caption(doc, label: str, caption: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(5)
    add_runs(p, [(label, {"size": 9.5, "bold": True}), (caption, {"size": 9.5, "italic": True})])


def add_figure(doc, path: Path, label: str, caption: str, width=6.4) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Inches(width))
    add_caption(doc, label, caption)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: str, digits=3) -> str:
    return f"{float(value):.{digits}f}"


def setup_document() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.78)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.82)
    section.right_margin = Inches(0.82)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(11)
    for level, size in [(1, 15), (2, 12.5), (3, 11.5)]:
        style = doc.styles[f"Heading {level}"]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(BLACK)
    footer = section.footer
    footer_p = footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer_p.add_run("Persistent Directional Dependence in Household Finance")
    set_font(run, size=8.5, color="555555")
    return doc


def add_title_page(doc: Document) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(52)
    p.paragraph_format.space_after = Pt(20)
    run = p.add_run("Persistent Directional Dependence in Household Finance")
    set_font(run, size=20, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(26)
    run = p.add_run("A Synthetic Validation of Employment Credit Housing Diagnostics")
    set_font(run, size=13, bold=True)
    author_lines = [
        ("Working Paper", 11, False),
        ("September 2026", 11, False),
        ("Lilia Maliar", 11.5, True),
        ("Professor, The Graduate Center, CUNY", 10.5, False),
        ("Principal Investigator", 9.8, False),
        ("Arka Prava Bandyopadhyay", 11.5, True),
        ("Adjunct Professor, Columbia University", 10.5, False),
        ("Co-Investigator", 9.8, False),
    ]
    for text, size, bold in author_lines:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(8)
        run = p.add_run(text)
        set_font(run, size=size, bold=bold)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(26)
    p.paragraph_format.left_indent = Inches(0.55)
    p.paragraph_format.right_indent = Inches(0.55)
    run = p.add_run(
        "All empirical results in this paper are generated from a transparent synthetic data-generating process. "
        "The paper is a methods and feasibility draft and does not report estimates from ADP, CoreLogic, Experian, "
        "Equifax, or any other proprietary consumer dataset."
    )
    set_font(run, size=10.5, italic=True)
    doc.add_page_break()


def build_paper(out_path: Path, result_dir: Path) -> None:
    direction = read_csv(result_dir / "directional_results.csv")
    soft = read_csv(result_dir / "soft_information_results.csv")
    doc = setup_document()
    add_title_page(doc)

    add_heading(doc, "Abstract", 1)
    add_body(doc, "This paper develops and stress-tests a directional-dependence diagnostic for household financial dynamics. The motivating setting is a sequence in which employment stress may affect credit stress, credit conditions may feed back into employment, and both may influence housing outcomes. The diagnostic compares the out-of-sample predictive value of lagged employment for future credit with the reverse predictive value of lagged credit for future employment. A positive persistent difference is informative about temporal direction, but it is not by itself a causal estimate.")
    add_body(doc, "We evaluate the diagnostic using a synthetic panel of 2,500 households observed monthly for 48 months under three known data-generating processes: direct employment-to-credit transmission, bidirectional feedback, and a persistent common confounder with no direct employment-to-credit effect. The diagnostic correctly identifies the direct channel and the reverse channel in the first two designs. It also produces a positive employment-to-credit asymmetry under the confounded design, demonstrating why the statistic must be paired with external-shock validation and design-based causal inference. A synthetic servicing signal improves next-month delinquency prediction, but its value is smaller when the underlying state is dominated by feedback or confounding. The results provide a reproducible feasibility benchmark for a future secure analysis combining payroll, credit, housing, and servicing data.")
    add_lead_body(doc, "Keywords. ", "Household finance; credit risk; housing; causal direction; time-series prediction; synthetic data; soft information; data fusion.")
    add_lead_body(doc, "JEL codes. ", "C14; C32; C55; D14; G21; R21.")

    add_heading(doc, "1 Introduction", 1)
    add_body(doc, "Household financial distress is a dynamic process rather than a single event. A reduction in hours or earnings can change liquidity, credit utilization, and payment behavior before a conventional delinquency indicator appears. Credit deterioration can then limit refinancing and mobility, while housing conditions can alter both household balance sheets and labor-market choices. Mortgage servicing communications add a further measurement layer: conversations may record hardship, employment loss, property condition, and workout choices before these states are fully represented in structured data.")
    add_body(doc, "The empirical opportunity is also an identification problem. If lagged employment predicts future credit, the finding may reflect a genuine transmission mechanism, a common persistent shock, measurement timing, selection into observed records, or feedback that is not adequately modeled. The central question of this paper is therefore narrower than whether employment causes credit distress. We ask whether a persistent out-of-sample asymmetry in cross-lag predictive content can serve as a useful diagnostic for prioritizing causal designs and detecting failure modes in a multi-domain household panel.")
    add_body(doc, "The paper makes four contributions. First, it defines a simple, auditable directional statistic based on incremental out-of-sample forecasting gains. The statistic is deliberately scale normalized and estimated under chronological holdouts. Second, it gives a dynamic benchmark in which the population direction is known, allowing power and sign behavior to be checked without confusing a simulation with evidence about households. Third, it includes a persistent-confounder design that generates a false positive, making the limitation part of the contribution rather than an afterthought. Fourth, it connects directional measurement to a synthetic soft-information exercise in which text-like signals improve the prediction of delinquency without being treated as causal treatments.")
    add_body(doc, "The intended empirical extension is a secure research design using ADP employment measures, Experian credit outcomes, CoreLogic or Cotality property and mortgage measures, and an approved mortgage servicing text panel. The current paper is a working paper because it uses no proprietary records. Its purpose is to establish the estimand, simulation benchmark, code path, and pre-analysis safeguards before any restricted-data acquisition or linkage.")
    add_heading(doc, "Research questions and hypotheses", 2)
    add_bullet(doc, "When employment stress directly precedes credit stress, does the forward predictive gain exceed the reverse gain across multiple horizons?")
    add_bullet(doc, "When employment and credit affect one another, does the diagnostic reveal a competing reverse channel rather than mechanically selecting employment as the source?")
    add_bullet(doc, "Can a persistent common confounder create an apparently durable directional signal even when no direct employment-to-credit effect exists?")
    add_bullet(doc, "Does a soft-information signal improve out-of-sample delinquency prediction after credit and housing histories are included?")

    add_heading(doc, "2 Relation to the Literature", 1)
    add_body(doc, "The diagnostic is rooted in the insight that temporal prediction can be informative about dynamic dependence without automatically establishing structural causality. Granger (1969) formalized a predictive notion of causality in dynamic systems. This paper adapts that intuition to a household-finance setting with explicit out-of-sample evaluation, multi-horizon persistence, and a confounding benchmark designed to show where the interpretation fails.")
    add_body(doc, "The paper also builds on the modern causal-inference literature that separates prediction from identification. Callaway and Sant'Anna (2021) show how treatment effects can be estimated in staggered-adoption settings under transparent assumptions, while Chernozhukov et al. (2018) provide a framework for using flexible nuisance prediction without allowing regularization to substitute for an identified causal parameter. Our proposed workflow follows the same discipline: use prediction to measure information and prioritize channels, then reserve causal language for designs supported by external variation, pre-trends, placebo tests, and sensitivity analysis.")
    add_body(doc, "The household-finance application is motivated by recent work showing that credit and mortgage records reveal economically important mechanisms. Adelino, Gerardi, and Hartman-Glaser (2019) use mortgage-market performance and time to sale to study dynamic signaling. Fonseca and Liu (2024) use individual credit records and mortgage-origination timing to identify effects of mortgage lock-in on mobility and labor reallocation. These studies illustrate why longitudinal credit and housing data can answer questions that are difficult to address with a single cross section. The present paper asks how an additional employment layer and a soft-information layer can be integrated without overstating what the resulting predictive asymmetry proves.")
    add_body(doc, "Finally, the paper treats text measurement as an economic measurement problem. Gentzkow, Kelly, and Taddy (2019) provide a general framework for text as data, while the proposed synthetic signal exercise emphasizes chronological holdouts, explicit exclusion of future-outcome language, and decision-relevant loss. A text signal that forecasts delinquency may be useful for measurement, but the forecasting result does not establish that exposing a lender or servicer to that signal would improve welfare.")

    add_heading(doc, "3 A Directional Diagnostic for Dynamic Household Finance", 1)
    add_heading(doc, "3.1 Dynamic state representation", 2)
    add_body(doc, "Let E_it denote employment stress for household i at month t, C_it denote credit stress, H_it denote housing stress, and S_it denote a soft-information signal extracted from a contemporaneous communication. The variables are intentionally generic. In an actual restricted-data project, E_it could combine payroll continuity, earnings changes, and employment transitions; C_it could combine score movement, utilization, delinquency, collections, and mortgage payment behavior; and H_it could combine property value, equity, refinancing, sale, mobility, and distressed resolution.")
    add_body(doc, "The state vector is allowed to be persistent, heterogeneous, and partially observed. That matters because a persistent state can transmit information across time even when it is not a causal treatment. The diagnostic therefore targets directional dependence in the observed state process and is not labeled a causal estimator.")
    add_heading(doc, "3.2 Forward and reverse predictive gains", 2)
    add_body(doc, "For a candidate ordered pair X to Y and a horizon h, we estimate two forecasting models on an early time window and evaluate them on a later chronological holdout. The baseline model predicts Y at t plus h from three lags of Y. The full model adds three lags of X. The normalized gain is the reduction in holdout mean squared error relative to the holdout variance of Y.")
    add_formula(doc, "G_h(X -> Y) = [MSE_h(baseline) - MSE_h(full)] / Var(Y_test)")
    add_body(doc, "The directional asymmetry for employment and credit is the forward gain minus the reverse gain. We report it at horizons of one, three, and six months and attach a household bootstrap interval. A positive value indicates that lagged employment contributes more to predicting future credit than lagged credit contributes to predicting future employment under the chosen forecasting class and holdout design.")
    add_formula(doc, "I_h = G_h(E -> C) - G_h(C -> E)")
    add_body(doc, "The persistence summary is the sign and magnitude pattern across h. A stable positive pattern is a reason to investigate employment-to-credit transmission with design-based variation. It is not a license to describe employment as a cause. In particular, a persistent confounder can make E_t a useful proxy for an unobserved state that also predicts C_t.")
    add_heading(doc, "3.3 A benchmark implication", 2)
    add_body(doc, "Consider a triangular linear benchmark in which employment follows its own lags and credit follows its own lags plus a lagged employment term. Suppose the innovations are serially independent and the forecasting model includes the relevant lags. At the population level, adding lagged employment reduces the forecast error for future credit when the transmission coefficient is nonzero, while adding lagged credit does not improve the forecast of future employment. Under those restrictive conditions, I_h is positive for the relevant horizons.")
    add_body(doc, "The benchmark implication is intentionally modest. It says what the statistic should do in a controlled triangular system; it does not say that household financial systems are triangular. The simulation includes reverse feedback and confounding precisely because those conditions are plausible in the application and because a useful methods paper should show how the diagnostic behaves when its interpretation is threatened.")

    add_heading(doc, "4 Synthetic Data and Empirical Design", 1)
    add_heading(doc, "4.1 Panel construction", 2)
    add_body(doc, "The synthetic panel contains 2,500 households in each of three scenarios and 48 monthly observations per household. The random seed is fixed at 20260914 and is recorded in the replication script. Each household has a persistent risk draw that affects initial credit and housing states. The process generates employment stress, credit stress, housing stress, soft information, a credit score, utilization, a home-value index, delinquency, and a move indicator.")
    add_body(doc, "The three scenarios are not intended to reproduce the distribution of any vendor dataset. They are controlled stress tests with interpretable mechanisms. The first scenario contains a direct lagged employment-to-credit channel. The second adds a lagged credit-to-employment channel. The third removes the direct employment-to-credit term and instead lets both processes respond to a highly persistent latent state with different timing and persistence.")
    add_table(doc, ["Scenario", "Employment process", "Credit process", "Purpose"], [
        ["Direct", "0.65 E(t-1) + common shock + noise", "0.85 C(t-1) + 0.45 E(t-1) + noise", "Known forward channel"],
        ["Feedback", "0.55 E(t-1) + 0.20 C(t-1) + noise", "0.80 C(t-1) + 0.45 E(t-1) + noise", "Competing reverse channel"],
        ["Confounded", "0.20 E(t-1) + 0.85 latent(t) + noise", "0.92 C(t-1) + 0.33 latent(t-1) + noise", "No direct E to C term"],
    ], widths=[1.0, 2.0, 2.45, 1.35], font_size=8.6)
    add_caption(doc, "Table 1. ", "Synthetic data-generating processes. Coefficients describe the simulation equations and are not estimated from real consumer records.")
    add_heading(doc, "4.2 Outcomes and soft information", 2)
    add_body(doc, "Credit score is a noisy decreasing transformation of credit stress. Utilization and delinquency are generated from logistic transformations of current stress and lagged soft information. Home value is a positive index that declines with housing stress and has a small time trend. The soft-information signal is generated from current employment, credit, and housing states plus measurement noise; it is not allowed to use future outcomes. This construction makes the signal predictive by design but still exposes the difference between incremental prediction and causal interpretation.")
    add_body(doc, "The benchmark uses a monthly forecasting split. Observations with time indices through month 29 form the training window. Observations from month 30 onward form the holdout. For each household and horizon, features are formed from lags one through three, so future values are not used in feature construction. The diagnostic is estimated with ordinary least squares to keep the mechanism auditable; the same design can later be re-run with lasso, random forests, or neural sequence models as robustness exercises.")
    add_heading(doc, "4.3 Inference and diagnostics", 2)
    add_body(doc, "The interval around I_h is a household bootstrap interval based on the difference in household-level holdout improvements. It is a descriptive simulation interval, not a formal confidence interval for a population parameter under a real-world sampling design. The simulation also records Brier loss for next-month delinquency to measure the incremental value of soft information after credit and housing histories.")
    add_bullet(doc, "Chronological holdouts protect against look-ahead bias and mimic a future prediction exercise.")
    add_bullet(doc, "The confounded scenario is a negative-control-style stress test for interpretation, not a claim about a particular market or vendor.")
    add_bullet(doc, "No direct linkage across synthetic domains is required; all variables are generated within a controlled household panel.")

    add_heading(doc, "5 Results", 1)
    add_heading(doc, "5.1 Dynamic path", 2)
    add_body(doc, "Figure 1 shows the standardized mean path in the direct-transmission scenario. The figure is descriptive: it visualizes persistence and co-movement in the generated panel, not an impulse response estimated from an identified shock. The staggered movement across the three states is the environment in which a temporal diagnostic can be informative but also vulnerable to confounding.")
    add_figure(doc, result_dir / "figure_dynamic_path.png", "Figure 1. ", "Standardized mean stress by month in the direct-transmission synthetic panel. The series are generated outcomes, not observations from commercial data.")
    add_heading(doc, "5.2 Directional asymmetry", 2)
    add_body(doc, "Table 2 reports the main diagnostic. In the direct scenario, the forward gain is 0.083 at one month, 0.114 at three months, and 0.074 at six months, while the reverse gain is essentially zero. The resulting asymmetry is positive at every horizon and the household bootstrap intervals exclude zero. This is the expected behavior when the data-generating process contains a direct lagged employment-to-credit channel.")
    add_body(doc, "The feedback scenario reverses the interpretation. The forward gain is positive but smaller, while the reverse gain is larger, producing a negative asymmetry of approximately -0.014 to -0.015 across horizons. The statistic therefore does not automatically select employment as the origin of the process; it reflects the relative contribution of the two lag structures under the forecasting design.")
    add_body(doc, "The confounded scenario is the critical limitation. There is no direct employment-to-credit coefficient in that design, yet the asymmetry is positive and grows from 0.007 at one month to 0.057 at six months. Employment stress is a noisy signal of a persistent latent state, and the latent state also drives credit stress. A persistent positive statistic can therefore be generated without the proposed direct causal arrow.")
    result_rows = []
    for row in direction:
        label = {"direct": "Direct", "feedback": "Feedback", "confounded": "Confounded"}[row["scenario"]]
        result_rows.append([label, row["horizon"], fmt(row["forward_gain"]), fmt(row["reverse_gain"]), fmt(row["asymmetry"]), f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]"])
    add_table(doc, ["Scenario", "h", "Forward gain", "Reverse gain", "Asymmetry", "95 percent interval"], result_rows, widths=[1.05, 0.42, 1.13, 1.13, 1.0, 1.55], font_size=8.5)
    add_caption(doc, "Table 2. ", "Out-of-sample directional gains. Gain is normalized by the holdout variance of the target. Intervals bootstrap households while holding fitted models fixed.")
    add_figure(doc, result_dir / "figure_directional_asymmetry.png", "Figure 2. ", "Directional asymmetry by horizon. The confounded design produces a positive signal despite the absence of a direct employment-to-credit term.")
    add_heading(doc, "5.3 Soft information and delinquency prediction", 2)
    add_body(doc, "The synthetic soft-information signal improves next-month delinquency prediction in all three scenarios, but the gain is heterogeneous. The normalized Brier gain is 0.031 in the direct scenario, 0.007 in the feedback scenario, and 0.002 in the confounded scenario. The result illustrates a useful measurement role for servicing communications: the signal can add information beyond structured histories. It does not establish that a real servicer could use such information without selection, privacy, fairness, or policy concerns.")
    soft_rows = []
    for row in soft:
        label = {"direct": "Direct", "feedback": "Feedback", "confounded": "Confounded"}[row["scenario"]]
        soft_rows.append([label, fmt(row["baseline_brier"]), fmt(row["soft_info_brier"]), fmt(row["soft_info_gain"])])
    add_table(doc, ["Scenario", "Baseline Brier", "With soft information", "Normalized gain"], soft_rows, widths=[1.25, 1.45, 1.65, 1.45], font_size=8.8)
    add_caption(doc, "Table 3. ", "Incremental predictive value of the synthetic soft-information signal for next-month delinquency.")
    add_figure(doc, result_dir / "figure_soft_information.png", "Figure 3. ", "Incremental soft-information gain for next-month delinquency. Positive bars indicate lower holdout Brier loss after adding the signal.", width=5.7)

    add_heading(doc, "6 Interpretation, Limitations, and Empirical Safeguards", 1)
    add_body(doc, "The results support a narrow claim. A persistent out-of-sample asymmetry can be a useful diagnostic for deciding which channels deserve a design-based causal investigation. In the direct scenario it identifies the intended direction; in the feedback scenario it detects that the reverse channel is stronger; and in the confounded scenario it warns that persistence alone is not enough. The diagnostic is thus most useful as part of a staged research workflow rather than as a standalone causal test.")
    add_heading(doc, "6.1 What the statistic cannot establish", 2)
    add_bullet(doc, "It cannot distinguish a causal path from a persistent omitted state when one observed variable proxies the omitted state.")
    add_bullet(doc, "It cannot turn county-level or ZIP-level co-movement into an individual causal effect.")
    add_bullet(doc, "It cannot correct for selective coverage, missing households, vendor definition changes, or unobserved linkage failures.")
    add_bullet(doc, "It cannot show that a predictive soft-information signal improves welfare or should be used in underwriting or servicing decisions.")
    add_heading(doc, "6.2 Pre-analysis safeguards for restricted data", 2)
    add_body(doc, "A future analysis using commercial records should freeze the variable ontology, timing conventions, sample restrictions, feature windows, and primary estimands before inspecting treatment effects. Each source should be versioned by product, vintage, refresh cadence, unit of observation, geography, and retention term. The secure environment should generate linkage quality flags and keep raw identifiers with the data custodian where possible. Analysts should receive derived features rather than raw identifiers or individual scores unless a data-use agreement and institutional review explicitly permit otherwise.")
    add_body(doc, "The preferred causal designs are external-shock event studies, difference-in-differences with transparent treatment timing, or credible instruments whose exclusion restrictions can be defended. The persistent directional statistic should be reported alongside pre-trends, placebo outcomes, negative controls, alternative windows, and a confounding sensitivity analysis. When direct linkage is unavailable, the project should use a geography-by-time design, clearly label the estimand as aggregate, and report bounds for aggregation error.")
    add_heading(doc, "6.3 Relevance to decision systems", 2)
    add_body(doc, "A high-performing predictor is not necessarily a good decision rule. A servicing or credit system chooses among a menu of interventions, and errors may have different costs across states. Future work should therefore report predictive loss alongside menu-level outcome sets, disclosure constraints, and welfare-relevant error. The present paper does not estimate a policy effect and does not recommend automated underwriting, surveillance, or benefits decisions.")

    add_heading(doc, "7 Roadmap for a Restricted-Data Extension", 1)
    add_body(doc, "The synthetic benchmark defines a disciplined first stage for the broader employment-credit-housing project. The next stage would replace each generated layer with an approved, de-identified source while preserving the same data dictionary and holdout logic. A secure custodian would perform linkage or construct approved crosswalks, and the research team would compare person-level, matched-sample, and geography-by-time estimands rather than assuming they are interchangeable.")
    add_table(doc, ["Stage", "Scientific task", "Required evidence before advancing"], [
        ["1. Measurement", "Validate employment, credit, housing, and servicing state variables", "Versioned dictionary; leakage audit; chronological and entity holdouts"],
        ["2. Direction", "Estimate forward and reverse gains across horizons", "Confounding sensitivity; external-shock candidates; placebo tests"],
        ["3. Causality", "Estimate an employment-to-credit channel", "Pre-trends; treatment timing; credible comparison or instrument"],
        ["4. Integration", "Estimate credit-to-housing transmission and counterfactuals", "Approved linkage; disclosure review; model-based labels for projections"],
    ], widths=[1.05, 2.8, 2.65], font_size=8.7)
    add_caption(doc, "Table 4. ", "Proposed progression from synthetic validation to approved restricted-data research.")
    add_body(doc, "This roadmap preserves the dependency structure of the four-paper series. Soft servicing information supplies a measurement layer. The directional diagnostic supplies a bounded temporal tool. The decision-theoretic component specifies why model validation must be aligned with intervention menus. The integrated paper then uses approved employment, credit, housing, and servicing states to estimate economically meaningful transmission mechanisms.")

    add_heading(doc, "8 Conclusion", 1)
    add_body(doc, "This paper develops a reproducible, synthetic benchmark for a persistent causal-direction diagnostic in household financial dynamics. The statistic behaves as intended in a triangular employment-to-credit design and detects reverse feedback when credit also predicts employment. Its failure under a persistent common confounder is equally important: a stable predictive asymmetry is not a causal estimate. The appropriate use is diagnostic and design-prioritizing.")
    add_body(doc, "The results justify a next phase of research using approved payroll, credit, property, mortgage, and servicing records. That phase should preserve the paper's discipline: synthetic and public benchmarks first, secure linkage only when authorized, causal language only for identified variation, and transparent separation of observed facts, predictions, causal effects, and model-based projections.")

    doc.add_page_break()
    add_heading(doc, "Appendix A Algorithm and Reproducibility", 1)
    add_heading(doc, "A.1 Estimation algorithm", 2)
    for text in [
        "1. Generate the synthetic household panel using simulate_persistent_causality.py with the fixed seed and scenario parameters.",
        "2. For each scenario and horizon h in {1, 3, 6}, construct three lags of the target and three lags of the candidate predictor.",
        "3. Fit the baseline and full linear forecasting models through time index 29.",
        "4. Evaluate both models from time index 30 onward and compute normalized holdout gains.",
        "5. Repeat the calculation in the reverse direction and subtract reverse gain from forward gain.",
        "6. Bootstrap household-level improvement differences to form descriptive 95 percent intervals.",
        "7. Estimate the next-month delinquency Brier-loss comparison with and without soft information.",
    ]:
        add_bullet(doc, text)
    add_heading(doc, "A.2 Files supplied with this paper", 2)
    add_body(doc, "The replication package contains the simulation script, the generated synthetic panel, scenario definitions, directional-results table, soft-information results table, three figures, this working paper in DOCX and PDF form, and a README with execution instructions. The generated panel contains no real person, employer, property, account, or credit-record identifiers.")
    add_heading(doc, "A.3 Reproducibility note", 2)
    add_body(doc, "The numeric values reported in Tables 2 and 3 are read directly from the CSV outputs generated by the supplied script. Because the data are synthetic and the seed is fixed, another researcher can reproduce the exact values with the bundled command. The code uses NumPy, pandas, Matplotlib, and the Python standard library; no vendor API or restricted-data credential is required.")

    add_heading(doc, "References", 1)
    refs = [
        "Adelino, Manuel, Kristopher Gerardi, and Barney Hartman-Glaser. 2019. Are Lemons Sold First? Dynamic Signaling in the Mortgage Market. Journal of Financial Economics 132(1): 1-25. https://doi.org/10.1016/j.jfineco.2018.09.005",
        "Callaway, Brantly, and Pedro H. C. Sant'Anna. 2021. Difference-in-Differences with Multiple Time Periods. Journal of Econometrics 225(2): 200-230. https://doi.org/10.1016/j.jeconom.2020.12.001",
        "Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. 2018. Double/Debiased Machine Learning for Treatment and Structural Parameters. The Econometrics Journal 21(1): C1-C68. https://doi.org/10.1111/ectj.12097",
        "Fonseca, Julia, and Lu Liu. 2024. Mortgage Lock-In, Mobility, and Labor Reallocation. the journal 79(6): 3729-3772. https://doi.org/10.1111/jofi.13398",
        "Gentzkow, Matthew, Bryan Kelly, and Matt Taddy. 2019. Text as Data. Journal of Economic Literature 57(3): 535-574. https://doi.org/10.1257/jel.20181020",
        "Granger, C. W. J. 1969. Investigating Causal Relations by Econometric Models and Cross-Spectral Methods. the journal 37(3): 424-438. https://www.jstor.org/stable/1912791",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_after = Pt(5)
        run = p.add_run(ref)
        set_font(run, size=9.3)

    doc.core_properties.title = "Persistent Directional Dependence in Household Finance"
    doc.core_properties.subject = "Synthetic validation working paper"
    doc.core_properties.author = "Lilia Maliar and Arka Prava Bandyopadhyay"
    doc.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    build_paper(Path(args.out), Path(args.results))
    print(args.out)


if __name__ == "__main__":
    main()
