"""
Build the comprehensive English PDF report covering the ENTIRE DeLeAn / ADeLe
demand-level annotation project, from the beginning:
  Part I   - Algebra/Calculus pilot (3 annotator models, 18 rubrics)
  Part II  - Knowledge distillation (Qwen2.5-7B QLoRA) and its validation
  Part III - ADeLe held-out task subset generalization test (6 non-math tasks)
  Part IV  - Novel external benchmark evaluation (5 benchmarks + distilled model)
"""
import pandas as pd
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 PageBreak, ListFlowable, ListItem, KeepTogether)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

RUBRICS = json.load(open('rubrics_cache.json'))
FULLNAME = {k: v['subdomain'] for k, v in RUBRICS.items()}
RUBRIC_ORDER = ['AS','AT','CEc','CEe','CL','KNa','KNc','KNf','KNn','KNs',
                'MCr','MCt','MCu','MS','QLl','QLq','SNs','VO']

OUT = "/home/administrador/Downloads/DeLeAn_Full_Project_Report.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Justify', parent=styles['Normal'], alignment=TA_JUSTIFY, fontSize=9.5, leading=13))
styles.add(ParagraphStyle(name='SmallCenter', parent=styles['Normal'], alignment=TA_CENTER, fontSize=8, textColor=colors.grey))
styles.add(ParagraphStyle(name='H1', parent=styles['Heading1'], fontSize=16, spaceBefore=18, spaceAfter=8, textColor=colors.HexColor('#1a1a2e')))
styles.add(ParagraphStyle(name='H2', parent=styles['Heading2'], fontSize=13, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor('#16213e')))
styles.add(ParagraphStyle(name='H3', parent=styles['Heading3'], fontSize=11, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor('#0f3460')))
styles.add(ParagraphStyle(name='TableCell', parent=styles['Normal'], fontSize=7.3, leading=9))
styles.add(ParagraphStyle(name='TableHeader', parent=styles['Normal'], fontSize=7.5, leading=9, textColor=colors.white, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='Caption', parent=styles['Normal'], fontSize=8.5, textColor=colors.HexColor('#444444'), spaceBefore=2, spaceAfter=10, fontName='Helvetica-Oblique'))
styles.add(ParagraphStyle(name='Finding', parent=styles['Normal'], fontSize=9.5, leading=13, backColor=colors.HexColor('#eef3fb'), borderPadding=8, spaceBefore=6, spaceAfter=10))

story = []

def h1(t): story.append(Paragraph(t, styles['H1']))
def h2(t): story.append(Paragraph(t, styles['H2']))
def h3(t): story.append(Paragraph(t, styles['H3']))
def p(t): story.append(Paragraph(t, styles['Justify']))
def cap(t): story.append(Paragraph(t, styles['Caption']))
def finding(t): story.append(Paragraph(t, styles['Finding']))
def sp(h=8): story.append(Spacer(1, h))

def make_table(df, col_widths=None, header_bg='#0f3460', fontsize=7.3):
    data = [[Paragraph(str(c), styles['TableHeader']) for c in df.columns]]
    for _, row in df.iterrows():
        data.append([Paragraph('' if pd.isna(v) else str(v), styles['TableCell']) for v in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor(header_bg)),
        ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f4f6fa')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]
    t.setStyle(TableStyle(style))
    return t

MODEL3 = {'Qwen':'Qwen2.5-72B','Llama':'Llama-3.3-70B','Mistral':'Mistral-Large-123B'}
MODEL3b = {'qwen':'Qwen2.5-72B','llama33':'Llama-3.3-70B','mistral':'Mistral-Large-123B'}
MODEL4 = {'qwen':'Qwen2.5-72B','llama33':'Llama-3.3-70B','mistral':'Mistral-Large-123B','distilled':'Distilled-7B'}

# ============================== TITLE PAGE ==============================
story.append(Spacer(1, 1.4*inch))
story.append(Paragraph("DeLeAn Demand-Level Annotation Project", ParagraphStyle(
    name='Title', parent=styles['Title'], fontSize=24, textColor=colors.HexColor('#0f3460'), alignment=TA_CENTER)))
story.append(Paragraph("Full Project Report: Algebra Pilot, Knowledge Distillation, and Generalization Studies",
    ParagraphStyle(name='Subtitle', parent=styles['Normal'], fontSize=13, alignment=TA_CENTER,
                   textColor=colors.HexColor('#444444'), spaceBefore=14)))
sp(30)
story.append(Paragraph("Annotator models: Qwen2.5-72B-Instruct-AWQ &middot; Llama-3.3-70B-Instruct-AWQ &middot; "
                        "Mistral-Large-Instruct-2411-AWQ (123B)<br/>"
                        "Distilled model: Qwen2.5-7B, QLoRA fine-tuned on the official ADeLe v1.0 battery",
                        styles['SmallCenter']))
sp(20)
story.append(Paragraph(
    "Part I &mdash; Algebra/Calculus Pilot (18 rubrics, 3 models)<br/>"
    "Part II &mdash; Knowledge Distillation and Validation<br/>"
    "Part III &mdash; ADeLe Held-Out Task Subset Generalization (6 non-math tasks)<br/>"
    "Part IV &mdash; Novel External Benchmark Evaluation (5 benchmarks + distilled model)",
    styles['SmallCenter']))
sp(40)
story.append(Paragraph("Report generated 2026-07-19", styles['SmallCenter']))
story.append(PageBreak())

# ============================== EXECUTIVE SUMMARY ==============================
h1("Executive Summary")
p("""This report documents the full arc of the DeLeAn / ADeLe v1.0 demand-level annotation project, from the
initial algebra-domain pilot through knowledge distillation into a small model and two independent generalization
studies designed to stress-test that distilled model on data it had never seen.""")
p("""<b>Part I</b> establishes the baseline: three large instruction-tuned models (Qwen2.5-72B-Instruct-AWQ,
Llama-3.3-70B-Instruct-AWQ, and Mistral-Large-Instruct-2411-AWQ, 123B) independently annotate 1,520
algebra/calculus problems against all 18 ADeLe demand-level rubrics. Agreement with the gold reference is
moderate-to-strong on most rubrics (mean &kappa;-quad in the 0.4-0.6 range), and a shared upward annotation bias
across all three models means that a simple <i>minimum</i>-across-models ensemble is the best de-noising strategy
on 6 of 17 usable rubrics.""")
p("""<b>Part II</b> distills this annotation competence into a much smaller, cheaper model: Qwen2.5-7B, QLoRA
fine-tuned on the full official ADeLe v1.0 battery (173,682 training examples spanning 51 tasks). Early validation
on held-out algebra items and a small sample of the official ADeLe test split showed strong agreement with the
gold labels and with the large annotator models &mdash; but because the training data included algebra and
calculus items, this raised a legitimate concern: was the model's competence genuine, or an artifact of
training-domain overlap ("trucado")?""")
p("""<b>Part III</b> tests generalization within the official ADeLe ecosystem, on 350 instances from six held-out
ADeLe test tasks containing no algebra or calculus content (MenatQA-Scope, Physics, chemistry, cta, engineering,
zebra_puzzle). The algebra-domain ensemble finding does <b>not</b> transfer: no shared bias exists on this diverse
subset, so median/mean ensembling wins instead of minimum. A previously undocumented Mistral failure mode is also
identified: on tasks with clear "right answer" framing, Mistral frequently answers the underlying task question
instead of the demand-level rubric question, cutting its usable-response rate to 74-88% (vs. 91-100% for Qwen and
Llama).""")
p("""<b>Part IV</b> settles the distillation-legitimacy question directly: the distilled model, alongside the
three large annotators, is evaluated on five external benchmarks (MBPP, TheoremQA, Omega, RelBench,
SWE-bench-Verified) that are completely absent from the official ADeLe v1.0 battery and were never seen during
fine-tuning in any form. The result is unambiguous: the 7B distilled model matches or exceeds all three 70-123B
annotator models, winning on mean quadratic-weighted kappa (0.628, tied with Llama-3.3's 0.629, ahead of
Mistral's 0.609 and Qwen's 0.601) and on mean exact-match accuracy (66.6%, clearly ahead of the next-best model
at 61.0%), and achieving the single best per-dataset exact-match score in all five benchmarks.""")
finding("""<b>Bottom line:</b> (1) the annotation pipeline reliably captures demand-level structure across very
different task domains, but (2) the best way to combine multiple annotators' scores is domain-dependent and must
be re-validated per task family, not assumed from a single pilot; (3) Mistral has a genuine, previously
undocumented instruction-following weakness distinct from context-length overflow; and (4) the distilled 7B
model's competence is genuine and portable &mdash; it was not memorizing algebra-domain surface patterns, and
generalizes cleanly to benchmark families it has never encountered in any form.""")
story.append(PageBreak())

# ============================== METHODOLOGY ==============================
h1("Methodology")

h2("The ADeLe Framework and the 18 Demand-Level Rubrics")
p("""ADeLe (Annotated Demand Levels) v1.0 is an 18-rubric framework for characterizing the cognitive and
knowledge demands a task instance places on a solver, independent of whether the solver answers correctly. Each
rubric defines six discrete demand levels (0-5) with detailed level descriptions and worked examples. The 18
rubrics span attention/search (AS), atypicality (AT), comprehension (CEc), expression (CEe), conceptualization/
learning/abstraction (CL), four knowledge sub-domains (KNa: applied sciences, KNc: everyday knowledge, KNf:
formal sciences, KNn: natural sciences, KNs: social sciences/humanities), three metacognitive rubrics
(MCr: identifying relevant information, MCt: critical thinking, MCu: calibrating knowns/unknowns), mind modelling
(MS), logical reasoning (QLl), quantitative reasoning (QLq), spatial-physical reasoning (SNs), and volume (VO).
The official ADeLe v1.0 battery comprises 16,108 items across 63 tasks drawn from 20 published benchmarks.""")

h2("Annotation Procedure")
p("""For every (task instance, rubric) pair, each model receives a fixed prompt template: the full rubric text
(description + all six level definitions with examples), the task instance itself, and a chain-of-thought
instruction asking the model to reason step by step and conclude with the statement "Thus, the level of *X*
demanded by the given TASK INSTANCE is: SCORE". The three large annotator models were served via vLLM 0.6.1
(offline batch inference, tensor-parallel size 4, AWQ 4-bit quantization, max context length 4096 tokens) on a
shared SLURM GPU cluster, using greedy/deterministic decoding so results are reproducible. Model responses are
parsed with a regex that extracts the final integer score; responses where no valid score in [0,5] can be
extracted are marked unusable and excluded from that model's per-rubric metrics (reported as "% usable").""")

h2("Evaluation Metrics")
story.append(ListFlowable([
    ListItem(Paragraph("<b>MAE</b> &mdash; mean absolute error between predicted and gold (reference) level.", styles['Justify'])),
    ListItem(Paragraph("<b>Exact</b> &mdash; percentage of predictions matching the gold level exactly.", styles['Justify'])),
    ListItem(Paragraph("<b>Within-1</b> &mdash; percentage of predictions within &plusmn;1 level of gold.", styles['Justify'])),
    ListItem(Paragraph("<b>&kappa;-quad</b> &mdash; quadratic-weighted Cohen's kappa, the primary ranking metric, "
                        "since it penalizes larger disagreements more and corrects for chance agreement.", styles['Justify'])),
    ListItem(Paragraph("<b>Pearson / Spearman</b> &mdash; used in Part II's small-sample distillation validation, "
                        "as linear and rank correlation between distilled-model and gold levels.", styles['Justify'])),
], bulletType='bullet', start='circle'))
p("""<b>Caveat:</b> &kappa;-quad is uninformative (near 0 or undefined) for rubrics with very low label variance
in the gold reference (e.g. KNa, KNc, KNn, KNs, MS often have a large majority class). For these, Exact and MAE
are used as the practical tie-breaker instead of trusting &kappa; alone &mdash; this logic is applied consistently
throughout the "best ensemble candidate" tables below: if the spread of &kappa;-quad across candidates for a given
rubric is &le;0.02, ranking falls back to (Exact desc, MAE asc).""")

h2("Ensemble Strategies")
p("""In addition to each model's solo predictions, we evaluate simple, training-free ensembles that combine
per-instance predictions across models: <b>mean</b>, <b>median</b>, <b>min</b>, and <b>max</b>, computed over
various model subsets (pairs, triples, and, in Part IV, the full quartet including the distilled model). These
require no additional training and test whether simple statistical combination of independent annotators improves
reliability, and if so, which combination rule is most robust for a given task family.""")
story.append(PageBreak())

# ============================================================================
# PART I - ALGEBRA PILOT
# ============================================================================
h1("Part I &mdash; Algebra/Calculus Pilot")

h2("Dataset")
p("""The pilot dataset comprises 1,520 algebra and calculus problems (e.g. counting monotonic intervals of
piecewise-defined trigonometric functions, symbolic derivative analysis) drawn from the OmniMath-style algebra and
calculus problem pool. Each instance was scored against all 18 ADeLe rubrics by all three annotator models
(1,520 &times; 18 &approx; 27,360 requests per model), against the official ADeLe gold reference level. This was
the first large-scale run of the pipeline and established the baseline annotation-quality profile used to design
later experiments.""")

h2("Per-Rubric Results (3 Models)")
c0 = pd.read_csv('comparison_18rubrics_3models_FULL.csv')
c0d = c0.copy()
c0d['rubric'] = c0d['rubric'].map(lambda r: f"{r} ({FULLNAME.get(r,r)})")
c0d['model'] = c0d['model'].map(MODEL3)
for col in ['mae','exact','within1','kappa_quad']:
    c0d[col] = c0d[col].round(3)
c0d['note'] = c0d['note'].fillna('')
c0d = c0d.rename(columns={'rubric':'Rubric','model':'Model','n':'n','mae':'MAE','exact':'Exact%','within1':'Within-1%','kappa_quad':'k-quad','note':'Note'})
c0d = c0d[['Rubric','Model','n','MAE','Exact%','Within-1%','k-quad','Note']]
tbl0 = make_table(c0d, col_widths=[1.55*inch, 0.9*inch, 0.35*inch, 0.5*inch, 0.45*inch, 0.55*inch, 0.5*inch, 1.0*inch])
story.append(tbl0)
cap("Table 0.1 &mdash; Per-rubric annotation quality, 3 annotator models, algebra/calculus pilot (n up to 1,520 instances per rubric before usability filtering). Exact/Within-1 reported as fractions (0-1) in the source data; MCr for Mistral had 0% usable responses due to prompts exceeding the 4096-token context limit.")

h3("Key Finding: A Shared Upward Annotation Bias")
p("""On this domain, all three models tend to over-estimate demand levels relative to the gold reference on
several rubrics (notably AS, CEc, MCr, MCt, MCu, QLq, SNs, VO), producing a consistent upward bias shared across
models rather than independent, zero-mean noise. Several rubrics also show near-zero gold-label variance
(KNa, KNc, KNn, KNs, MS), where almost every algebra/calculus instance receives the same reference level, making
&kappa;-quad uninformative there (MAE and Exact are the relevant metrics instead). A context-length limitation was
also identified here for the first time: Mistral's prompts for MCr and, to a lesser extent, QLl occasionally
exceeded the 4096-token context window, causing 0% and reduced usable-response rates respectively.""")

h2("Ensemble Analysis by Rubric")
p("""For each of the 17 rubrics with usable Mistral output, all pairwise and triple-wise combination strategies
across Qwen, Llama, and Mistral were evaluated and the best-performing candidate selected.""")
b0 = pd.read_csv('ensemble_17rubrics_best.csv')
b0d = b0.copy()
b0d['rubric'] = b0d['rubric'].map(lambda r: f"{r} ({FULLNAME.get(r,r)})")
b0d['exact'] = b0d['exact'].round(3)
b0d['mae'] = b0d['mae'].round(3)
b0d['kappa_quad'] = b0d['kappa_quad'].round(3)
b0d = b0d.rename(columns={'rubric':'Rubric','candidate':'Best Candidate','n':'n','mae':'MAE','exact':'Exact%','kappa_quad':'k-quad'})
tbl0b = make_table(b0d, col_widths=[1.7*inch, 1.15*inch, 0.5*inch, 0.6*inch, 0.6*inch, 0.6*inch])
story.append(tbl0b)
cap("Table 0.2 &mdash; Best-performing prediction source per rubric (solo model or ensemble combination), algebra/calculus pilot. Q=Qwen, L=Llama, M=Mistral.")

p("""Win tally: <b>min(Q,L,M) wins 6/17 rubrics</b> and <b>min(Q,L) wins 4/17</b> &mdash; together, some form of
minimum-across-models wins 10 of 17 rubrics, directly reflecting the shared upward bias documented above: taking
the minimum systematically pulls the ensemble estimate back down toward the gold reference. This is the pattern
that motivated the generalization question addressed in Part III: does "use the minimum" hold on task domains
without a shared upward bias?""")

h2("Discussion")
p("""The algebra pilot establishes that the three-model annotation pipeline produces useful, well-correlated
demand-level scores on a mathematically dense domain, but also that (a) minimum-based ensembling is specifically
effective here because of a domain-specific shared bias, not a general property of the pipeline, and (b) Mistral's
practical usability is constrained on this domain by context-length overflow on verbose rubrics (MCr, QLl) rather
than by the instruction-confusion failure mode later identified in Part III. These two domain-specific artifacts
(shared bias, context-length pressure) are the reason later experiments deliberately test task families where
neither is expected to hold.""")
story.append(PageBreak())

# ============================================================================
# PART II - DISTILLATION
# ============================================================================
h1("Part II &mdash; Knowledge Distillation and Validation")

h2("Distillation Setup")
p("""To reduce the cost of large-scale demand-level annotation, a small model, <b>Qwen2.5-7B-Instruct</b>, was
fine-tuned via <b>QLoRA</b> (4-bit NF4 quantization, LoRA rank 16, alpha 32, bf16 compute, per-device batch size 1
with gradient accumulation 16, 4&times;GPU multi-process training) to reproduce ADeLe demand-level annotation
directly, using the full <b>official ADeLe v1.0 battery</b> (16,108 items, 63 tasks, 20 benchmarks) as supervision.
The battery was split by <b>task</b> (not by instance) into 51 training tasks (173,682 examples after
rubric-expansion), 6 validation tasks (15,714 examples: TimeQA-implicit, biology, dosage, olympiad,
reaction_prediction, spatial), and 6 held-out test tasks (17,226 examples: MenatQA-Scope, Physics, chemistry, cta,
engineering, zebra_puzzle) &mdash; the same six tasks later used for the Part III generalization study. Training
data included, among many others, Algebra, Calculus, AMPS_Hard, and Precalculus task splits, i.e. the same
mathematical domain as the Part I pilot.""")

h2("Initial Validation: Algebra Domain")
p("""The distilled model was first validated on a small sample of algebra-domain items (25-30 per rubric,
overlapping in domain with its training data), both against the gold reference directly and head-to-head against
the three large annotator models on the identical items.""")
d1 = pd.read_csv('distilled_vs_algebra_metrics.csv')
d1d = d1.copy()
d1d['rubric'] = d1d['rubric'].map(lambda r: f"{r} ({FULLNAME.get(r,r)})" if r in FULLNAME else r)
for col in ['pearson','spearman','mae','exact','within1','kappa_quad']:
    d1d[col] = d1d[col].astype(float).round(3)
d1d = d1d.rename(columns={'rubric':'Rubric','n':'n','pearson':'Pearson','spearman':'Spearman','mae':'MAE','exact':'Exact%','within1':'Within-1%','kappa_quad':'k-quad'})
d1d = d1d[['Rubric','n','Pearson','Spearman','MAE','Exact%','Within-1%','k-quad']]
tbl_d1 = make_table(d1d, col_widths=[1.55*inch, 0.35*inch, 0.6*inch, 0.6*inch, 0.5*inch, 0.5*inch, 0.6*inch, 0.5*inch])
story.append(tbl_d1)
cap("Table 0.3 &mdash; Distilled model (Qwen2.5-7B) vs. gold reference, algebra-domain validation sample (n up to 30 per rubric).")

d2 = pd.read_csv('distilled_vs_bigmodels_algebra.csv')
d2d = d2.copy()
d2d['rubric'] = d2d['rubric'].map(lambda r: f"{r} ({FULLNAME.get(r,r)})" if r in FULLNAME else r)
for col in ['kq_Llama','kq_Mistral','kq_Qwen','mae_Llama','mae_Mistral','mae_Qwen','mae_dist','exact_dist','kappa_quad_dist']:
    d2d[col] = d2d[col].astype(float).round(3)
d2d = d2d.rename(columns={'rubric':'Rubric','kq_Qwen':'k-q Qwen','kq_Llama':'k-q Llama','kq_Mistral':'k-q Mistral',
                           'kappa_quad_dist':'k-q Dist','mae_dist':'MAE Dist','exact_dist':'Exact% Dist','n_dist':'n'})
d2d = d2d[['Rubric','n','k-q Qwen','k-q Llama','k-q Mistral','k-q Dist','MAE Dist','Exact% Dist']]
tbl_d2 = make_table(d2d, col_widths=[1.5*inch, 0.3*inch, 0.65*inch, 0.65*inch, 0.7*inch, 0.6*inch, 0.6*inch, 0.65*inch])
story.append(tbl_d2)
cap("Table 0.4 &mdash; Distilled model's kappa/MAE/Exact head-to-head against each large annotator's kappa/MAE, same algebra-domain items.")

p("""On these same-domain items, the distilled model is broadly competitive with, and on several rubrics
(CEc, CEe, KNf, MCu) better than, all three large annotators. However, since Algebra and Calculus splits were
part of the distillation training data, this result alone <b>cannot rule out training-domain overlap</b> as the
explanation &mdash; the model may simply have learned to reproduce algebra-specific annotation patterns rather
than a general demand-level annotation skill.""")

h2("Initial Validation: Official ADeLe Held-Out Test Split")
p("""As a first (small-scale) generalization check, the distilled model was also evaluated on a 15-instance-per-
rubric sample from the six officially held-out ADeLe test tasks (the same tasks later used for the larger Part III
study).""")
d3 = pd.read_csv('distilled_vs_adele_test_metrics.csv')
d3d = d3.copy()
d3d['rubric'] = d3d['rubric'].map(lambda r: f"{r} ({FULLNAME.get(r,r)})" if r in FULLNAME else r)
for col in ['pearson','spearman','mae','exact','within1','kappa_quad']:
    d3d[col] = d3d[col].astype(float).round(3)
d3d = d3d.sort_values('kappa_quad', ascending=False)
d3d = d3d.rename(columns={'rubric':'Rubric','n':'n','pearson':'Pearson','spearman':'Spearman','mae':'MAE','exact':'Exact%','within1':'Within-1%','kappa_quad':'k-quad'})
d3d = d3d[['Rubric','n','Pearson','Spearman','MAE','Exact%','Within-1%','k-quad']]
tbl_d3 = make_table(d3d, col_widths=[1.55*inch, 0.35*inch, 0.6*inch, 0.6*inch, 0.5*inch, 0.5*inch, 0.6*inch, 0.5*inch])
story.append(tbl_d3)
cap("Table 0.5 &mdash; Distilled model vs. gold reference, small sample (n=15/rubric) from the six officially held-out ADeLe test tasks. Sorted by k-quad descending.")

finding("""Encouragingly, agreement remains strong on this small held-out sample (e.g. CEe and KNs reach k-quad
= 1.0, KNf = 0.939, KNa = 0.921), but with only 15 instances per rubric this is not statistically decisive, and it
still draws from the same official ADeLe ecosystem the model was fine-tuned within (only the specific task, not
the underlying benchmark family, was held out). This motivated the two larger, more rigorous generalization
studies in Parts III and IV.""")

h2("Discussion")
p("""Part II's initial validation results were promising but insufficient to fully resolve the training-domain-
overlap concern: strong algebra-domain performance is expected regardless of whether the model learned a genuine
skill (since Algebra/Calculus were training tasks), and the small n=15/rubric held-out-task sample, while
reassuring, was too small to be conclusive and still originated from the same benchmark ecosystem. This is why
Part III scales up the held-out-task test to 350 instances, and Part IV goes further still, testing on benchmark
families with zero presence in the ADeLe v1.0 battery in any split.""")
story.append(PageBreak())

# ============================================================================
# PART III - ADELE SUBSET
# ============================================================================
h1("Part III &mdash; ADeLe Held-Out Task Subset Generalization (6 Non-Math Tasks)")

h2("Dataset")
p("""To test whether the ensemble-strategy findings from Part I generalize, and to more rigorously test the
distilled model's held-out-task performance than the small n=15/rubric sample in Part II, we sampled 350 instances
(capped at 80 per task, using all available instances where a task had fewer) from the same six ADeLe v1.0
held-out test tasks used to validate the distilled model: <b>MenatQA-Scope</b>, <b>Physics</b>, <b>chemistry</b>,
<b>cta</b> (column-type annotation / classification), <b>engineering</b>, and <b>zebra_puzzle</b>
(constraint-satisfaction logic puzzles) &mdash; none of which contain algebra or calculus content. Each instance
was scored against all 18 rubrics by all three annotator models (350 &times; 18 = 6,300 requests per model),
against the official ADeLe gold reference level.""")

h2("Per-Rubric Results (3 Models)")
c1 = pd.read_csv('comparison_adele_subset_18rubrics.csv')
c1d = c1.copy()
c1d['rubric'] = c1d['rubric'].map(lambda r: f"{r} ({FULLNAME.get(r,r)})")
c1d['model'] = c1d['model'].map(MODEL3b)
c1d = c1d.rename(columns={'rubric':'Rubric','model':'Model','n':'n','pct_usable':'%Usable','mae':'MAE','exact':'Exact%','within1':'Within-1%','kappa_quad':'k-quad'})
c1d = c1d[['Rubric','Model','n','%Usable','MAE','Exact%','Within-1%','k-quad']]
tbl1 = make_table(c1d, col_widths=[1.55*inch, 0.95*inch, 0.35*inch, 0.55*inch, 0.5*inch, 0.5*inch, 0.6*inch, 0.55*inch])
story.append(tbl1)
cap("Table I.1 &mdash; Per-rubric annotation quality, 3 annotator models, on the 6-task diverse subset (n=350 instances per rubric before usability filtering). Sorted by rubric, then model.")

h3("Key Finding: A Mistral-Specific Instruction-Confusion Failure Mode")
p("""Mistral's usable-response rate ranges 74.0%-87.7% across rubrics on this subset, markedly lower than Qwen
(93.4-100%) and Llama-3.3 (91.7-100%). Manual inspection of the unparseable completions revealed that this is
<b>not</b> primarily the context-length overflow issue documented on the algebra pilot (where MCr and QLl prompts
occasionally exceeded the 4096-token limit). Instead, on task types with a clear "right answer" framing
(chemistry QA, cta classification, engineering MCQ), Mistral frequently (roughly 15-25% of responses on affected
rubrics) answers the <i>underlying task question</i> ("Thus, the correct answer is: X") instead of the
<i>rubric demand-level question</i> ("Thus, the level of *X* demanded... is: N") that the prompt actually asks
for. Qwen and Llama exhibit this confusion far less often (roughly 1-7% of responses).""")

h2("Ensemble Analysis by Rubric")
p("""For each rubric, all pairwise and triple-wise combination strategies were evaluated and the best-performing
candidate selected per the tie-break rule above. The table below reports only the winning candidate per rubric;
full candidate-level results are in Appendix A.""")
b1 = pd.read_csv('ensemble_adele_subset_18_best.csv')
b1d = b1.copy()
b1d['rubric'] = b1d['rubric'].map(lambda r: f"{r} ({FULLNAME.get(r,r)})")
b1d = b1d.rename(columns={'rubric':'Rubric','best_candidate':'Best Candidate','n':'n','mae':'MAE','exact':'Exact%','within1':'Within-1%','kappa_quad':'k-quad'})
tbl1b = make_table(b1d, col_widths=[1.7*inch, 1.15*inch, 0.4*inch, 0.55*inch, 0.55*inch, 0.65*inch, 0.55*inch])
story.append(tbl1b)
cap("Table I.2 &mdash; Best-performing prediction source per rubric (solo model or ensemble combination), diverse 6-task subset.")

p("""Win tally across the 18 rubrics: <b>median(Q,L,M) wins 5</b>, max(Q,L) and min(Q,L) win 3 each, qwen-solo
and mean(Q,L,M) win 2 each, and min(Q,L,M), llama33-solo, and mean(Q,L) win 1 each. This is qualitatively
different from the algebra pilot (Part I), where min(Q,L,M)/min(Q,L) dominated (6/17 and 4/17 rubrics
respectively) because all three models shared a common upward annotation bias on algebra/calculus problems. On
this diverse subset, no single directional bias is shared across models, so <i>minimum</i> is no longer specially
favored &mdash; <i>median</i>, which is robust to a single outlier model regardless of direction, becomes the most
frequent winner instead.""")

h2("Per-Task Breakdown")
t1 = pd.read_csv('comparison_adele_subset_18_by_task.csv')
t1d = t1.copy()
t1d = t1d.rename(columns={'task':'Task','candidate':'Candidate','n':'n','mae':'MAE','exact':'Exact%','within1':'Within-1%','kappa_quad':'k-quad'})
tbl1c = make_table(t1d, col_widths=[1.15*inch, 1.15*inch, 0.5*inch, 0.55*inch, 0.55*inch, 0.65*inch, 0.55*inch])
story.append(tbl1c)
cap("Table I.3 &mdash; Solo models vs. triple ensembles, pooled across all 18 rubrics, broken down by task (n = number of usable (instance, rubric) pairs for that task).")

p("""Median(Q,L,M) is the top or near-top candidate in 5 of the 6 tasks (all but zebra_puzzle, where mean(Q,L,M)
edges it out marginally). Mistral is the weakest solo model in 5 of 6 tasks, consistent with its elevated
unusable-response rate; it is competitive only on Physics.""")

h2("Discussion")
p("""The central conclusion of Part III is that <b>ensemble strategy choice is task-family dependent and cannot
be generalized from a single pilot domain</b>. The Part I algebra pilot's headline recommendation (use
min(Q,L,M) to correct a shared upward bias) does not hold here; a practitioner deploying this pipeline on a new
task type should re-run this kind of ensemble sweep rather than assume the algebra-domain best-practice
transfers. Separately, Mistral's elevated failure rate on this subset is a genuine, previously undocumented
instruction-following weakness (confusing "solve the task" with "annotate the task's demand level") rather than a
context-length artifact, and should be accounted for when weighting Mistral's contribution to any production
ensemble.""")
story.append(PageBreak())

# ============================================================================
# PART IV - NEW BENCHMARKS
# ============================================================================
h1("Part IV &mdash; Novel External Benchmark Evaluation (Including the Distilled Model)")

h2("Motivation")
p("""Part II's initial distillation validation and Part III's larger held-out-task study both still drew on the
official ADeLe v1.0 ecosystem, so a residual concern remained: the distilled model's fine-tuning data included
Algebra, Calculus, and dozens of other ADeLe tasks, and even the "held-out" test tasks are part of the same
overall battery design. Part IV directly addresses this by testing the distilled model, alongside the three large
annotators, on five benchmarks that are <b>completely absent</b> from the official ADeLe v1.0 battery (verified
against the battery's full benchmark list) and were never encountered by the distilled model during fine-tuning in
any form.""")

h2("Dataset")
p("""We sampled 300 instances from five external benchmarks with independently-collected ADeLe-style gold
annotations: <b>MBPP</b> (60 instances, Python code generation), <b>TheoremQA</b> (60, theorem-grounded
quantitative QA), <b>Omega</b> (60, split 30/30 between geometry and logic-puzzle sub-tasks), <b>RelBench</b> (60,
split 12/12/12/12/12 across addition, locality, science, anagram, and transforms sub-tasks), and
<b>SWE-bench-Verified</b> (60, real GitHub issue-to-patch software engineering tasks). Each instance was scored
against all 18 rubrics by Qwen2.5-72B, Llama-3.3-70B, Mistral-Large-123B, and the distilled Qwen2.5-7B model
(300 &times; 18 = 5,400 requests per model). Rubric prompt templates and rubric content were reconstructed
byte-for-byte from the original training data to guarantee identical prompting to all four models.""")

h2("Per-Rubric Results (4 Models)")
c2 = pd.read_csv('comparison_newset_18rubrics_4models.csv')
c2d = c2.copy()
c2d['rubric'] = c2d['rubric'].map(lambda r: f"{r} ({FULLNAME.get(r,r)})")
c2d['model'] = c2d['model'].map(MODEL4)
c2d = c2d.rename(columns={'rubric':'Rubric','model':'Model','n':'n','pct_usable':'%Usable','mae':'MAE','exact':'Exact%','within1':'Within-1%','kappa_quad':'k-quad'})
c2d = c2d[['Rubric','Model','n','%Usable','MAE','Exact%','Within-1%','k-quad']]
tbl2 = make_table(c2d, col_widths=[1.55*inch, 0.9*inch, 0.35*inch, 0.55*inch, 0.5*inch, 0.5*inch, 0.6*inch, 0.55*inch])
story.append(tbl2)
cap("Table II.1 &mdash; Per-rubric annotation quality, 4 models (3 large annotators + distilled 7B), on the 5-benchmark novel-task set (n=300 instances per rubric before usability filtering).")

h2("Aggregate Summary and the Distillation Question")
agg2 = c2.groupby('model').agg(mean_kappa=('kappa_quad','mean'), mean_exact=('exact','mean'),
                                 mean_within1=('within1','mean'), mean_pct_usable=('pct_usable','mean')).round(3)
agg2 = agg2.reset_index()
agg2['model'] = agg2['model'].map(MODEL4)
agg2 = agg2.sort_values('mean_kappa', ascending=False)
agg2 = agg2.rename(columns={'model':'Model','mean_kappa':'Mean k-quad','mean_exact':'Mean Exact%','mean_within1':'Mean Within-1%','mean_pct_usable':'Mean %Usable'})
tbl2b = make_table(agg2, col_widths=[1.6*inch, 1.1*inch, 1.1*inch, 1.2*inch, 1.1*inch])
story.append(tbl2b)
cap("Table II.2 &mdash; Aggregate (mean across 18 rubrics) annotation quality, 5-benchmark novel-task set.")

finding("""<b>The distilled 7B model achieves the highest mean exact-match accuracy (66.6%) of all four models,
and a mean quadratic-weighted kappa (0.628) statistically tied with the best 70B-scale annotator (Llama-3.3:
0.629), ahead of Mistral-Large-123B (0.609) and Qwen2.5-72B (0.601).</b> Counting per-rubric kappa wins across
the 18 rubrics: the distilled model wins outright on <b>10/18</b>, Llama-3.3 on 4/18, and Qwen/Mistral on 2/18
each. This is on a benchmark family (code generation, theorem QA, geometry/logic, arithmetic/relational
reasoning, and real bug-fix patches) with zero overlap with the distilled model's fine-tuning data &mdash; a
genuinely out-of-domain test. This result directly counters the concern raised in Part II that the distilled
model's strong algebra-domain performance was a training-domain-overlap artifact: on data it has never seen in
any form, a model 10-17x smaller than the annotators it was distilled from matches or beats them.""")

h2("Ensemble Analysis by Rubric")
b2 = pd.read_csv('ensemble_newset_best.csv')
b2d = b2.copy()
b2d['rubric'] = b2d['rubric'].map(lambda r: f"{r} ({FULLNAME.get(r,r)})")
b2d = b2d.rename(columns={'rubric':'Rubric','best_candidate':'Best Candidate','n':'n','mae':'MAE','exact':'Exact%','within1':'Within-1%','kappa_quad':'k-quad'})
tbl2c = make_table(b2d, col_widths=[1.6*inch, 1.25*inch, 0.4*inch, 0.55*inch, 0.55*inch, 0.65*inch, 0.55*inch])
story.append(tbl2c)
cap("Table II.3 &mdash; Best-performing prediction source per rubric (solo model or ensemble, including the distilled model), 5-benchmark novel-task set. Q=Qwen, L=Llama-3.3, M=Mistral, D=Distilled.")

p("""Win tally: <b>median(Q,L,M,D) wins 5/18</b> and <b>the distilled model wins solo on 4/18</b> (i.e. adding
it to an ensemble does not always help &mdash; on AS, CEc, MCr, and MCt it is already the single best predictor),
mean(Q,L,M,D) wins 3/18, median(Q,L,M) (without the distilled model) wins 2/18, and the remainder split
1 win each. Combining all four models continues to help on most rubrics, but the marginal gain over the
distilled model alone is far smaller here than the gain ensembling provided on the algebra pilot (Part I) or the
diverse ADeLe subset (Part III) &mdash; consistent with the distilled model already being a strong,
largely-independent predictor rather than a weak one that benefits heavily from correction.""")

h2("Per-Dataset Breakdown")
d2b = pd.read_csv('comparison_newset_by_dataset.csv')
d2d2 = d2b.copy()
label_map = {'qwen (solo)':'Qwen2.5-72B (solo)','llama33 (solo)':'Llama-3.3-70B (solo)',
             'mistral (solo)':'Mistral-Large-123B (solo)','distilled (solo)':'Distilled-7B (solo)',
             'mean(Q,L,M)':'mean(Q,L,M)','median(Q,L,M,D)':'median(Q,L,M,D)','mean(Q,L,M,D)':'mean(Q,L,M,D)'}
d2d2['candidate'] = d2d2['candidate'].map(lambda x: label_map.get(x,x))
d2d2 = d2d2.rename(columns={'dataset':'Dataset','candidate':'Candidate','n':'n','mae':'MAE','exact':'Exact%','within1':'Within-1%','kappa_quad':'k-quad'})
tbl2d = make_table(d2d2, col_widths=[0.95*inch, 1.35*inch, 0.5*inch, 0.55*inch, 0.55*inch, 0.65*inch, 0.55*inch])
story.append(tbl2d)
cap("Table II.4 &mdash; Solo models and ensembles pooled across all 18 rubrics, broken down by external benchmark. n = number of usable (instance, rubric) pairs for that benchmark.")

finding("""<b>The distilled model achieves the best solo exact-match accuracy in all 5 benchmarks</b>
(mbpp: 66.4% vs 58.6-61.8% for the large models; omega: 68.5% vs 59.1-60.2%; relbench: 63.1% vs 60.1-61.2%;
swebenchver: 68.7% vs 47.9-64.9%; theoremqa: 68.4% vs 57.0-60.6%), and the best solo k-quad in 4 of 5
(all but relbench, where it is a close second to mean(Q,L,M)). Combining all four models (mean/median) still
yields the lowest MAE and highest within-1 agreement per benchmark, but the solo distilled model is already
highly competitive &mdash; further evidence that its ADeLe-annotation competence is genuine and not
domain-overlap-dependent.""")

h2("Discussion")
p("""Part IV provides a clean, direct answer to the training-domain-overlap question raised in Part II: tested on
five benchmark families with zero presence in its training data, spanning code generation, theorem-grounded
mathematics, geometric/logical puzzles, arithmetic/relational reasoning, and real-world software-engineering bug
patches, the 7B distilled model is not merely competitive but frequently the single best solo annotator of the
four models tested, at a fraction of the parameter count of the 70-123B annotators it was distilled from. This is
strong evidence that the distillation captured a genuine, transferable demand-level annotation skill rather than
memorized surface statistics from the training tasks. The practical implication is that the distilled model is a
credible low-cost substitute for the large annotator ensemble on novel task types, though combining it with even
one large model (e.g. median or mean with Llama-3.3) still provides a modest additional accuracy gain in most
rubrics.""")
story.append(PageBreak())

# ============================== OVERALL CONCLUSIONS ==============================
h1("Overall Conclusions")
story.append(ListFlowable([
    ListItem(Paragraph("<b>The annotation pipeline is robust across very different task domains</b> &mdash; from "
        "dense algebra/calculus problems (Part I) to diverse non-math ADeLe tasks (Part III) to entirely external "
        "benchmarks (Part IV) &mdash; producing meaningful, gold-correlated demand-level scores throughout, "
        "though absolute agreement levels vary substantially by rubric and domain.", styles['Justify'])),
    ListItem(Paragraph("<b>Ensemble strategy must be re-validated per task family, not assumed from a single "
        "pilot.</b> The algebra-domain recommendation (min(Q,L,M) to correct a shared upward bias, winning 10/17 "
        "rubrics in Part I) does not hold on diverse non-math tasks (Part III), where median/mean-based "
        "ensembling wins instead, because the shared directional bias observed on algebra is not present in "
        "general.", styles['Justify'])),
    ListItem(Paragraph("<b>Mistral-Large exhibits two distinct, domain-dependent reliability issues:</b> "
        "context-length overflow on verbose rubrics in the algebra pilot (Part I, e.g. 0% usable on MCr), and a "
        "separate, previously undocumented instruction-following weakness on the diverse ADeLe subset (Part "
        "III) &mdash; confusing the demand-level annotation question with the underlying task question &mdash; "
        "on tasks with clear \"correct answer\" framing (chemistry, cta, engineering).", styles['Justify'])),
    ListItem(Paragraph("<b>The distilled 7B model's competence is genuine and generalizes fully out-of-domain.</b> "
        "Initial validation (Part II) on algebra-domain and small held-out-task samples was promising but "
        "inconclusive given training-domain overlap. Tested on five external benchmarks with zero overlap with "
        "its training data (Part IV), it matches or beats all three 70-123B annotator models on mean kappa and "
        "exact-match, and is the best solo model in the majority of individual rubrics and benchmarks tested "
        "&mdash; directly resolving the earlier concern that its strong results were an artifact of "
        "training-domain overlap.", styles['Justify'])),
    ListItem(Paragraph("<b>Practical recommendation:</b> for cost-sensitive deployment on novel task types, the "
        "distilled model is a credible standalone substitute for the large annotator ensemble; where compute "
        "budget allows, combining it with one or two large models via median/mean ensembling still yields a "
        "further, if modest, accuracy improvement. Ensemble weights and strategy should be re-tuned per task "
        "family rather than reused from the algebra pilot.", styles['Justify'])),
], bulletType='bullet', start='circle', bulletFontSize=9))
story.append(PageBreak())

# ============================== APPENDIX ==============================
h1("Appendix A &mdash; Full Ensemble Candidate Tables")
h2("A.1 Part I: All Ensemble Candidates by Rubric (Algebra Pilot)")
eA0 = pd.read_csv('ensemble_17rubrics_3models.csv')
eA0d = eA0.copy()
keep_cols = [c for c in ['rubric','candidate','n','mae','exact','kappa_quad'] if c in eA0d.columns]
eA0d = eA0d[keep_cols]
for col in ['mae','exact','kappa_quad']:
    if col in eA0d.columns:
        eA0d[col] = eA0d[col].astype(float).round(3)
eA0d = eA0d.rename(columns={'rubric':'Rubric','candidate':'Candidate','n':'n','mae':'MAE','exact':'Exact%','kappa_quad':'k-quad'})
tblA0 = make_table(eA0d, col_widths=[0.6*inch, 1.4*inch, 0.5*inch, 0.6*inch, 0.6*inch, 0.6*inch], fontsize=6.8)
story.append(tblA0)
story.append(PageBreak())

h2("A.2 Part III: All Ensemble Candidates by Rubric (Diverse ADeLe Subset)")
e1 = pd.read_csv('ensemble_adele_subset_18rubrics.csv')
e1d = e1.copy()
e1d = e1d.rename(columns={'rubric':'Rubric','candidate':'Candidate','n':'n','mae':'MAE','exact':'Exact%','within1':'Within-1%','kappa_quad':'k-quad'})
tblA1 = make_table(e1d, col_widths=[0.6*inch, 1.2*inch, 0.4*inch, 0.5*inch, 0.5*inch, 0.6*inch, 0.5*inch], fontsize=6.8)
story.append(tblA1)
story.append(PageBreak())

h2("A.3 Part IV: All Ensemble Candidates by Rubric (Novel Benchmarks)")
e2 = pd.read_csv('ensemble_newset_18rubrics.csv')
e2d = e2.copy()
e2d = e2d.rename(columns={'rubric':'Rubric','candidate':'Candidate','n':'n','mae':'MAE','exact':'Exact%','within1':'Within-1%','kappa_quad':'k-quad'})
tblA2 = make_table(e2d, col_widths=[0.6*inch, 1.3*inch, 0.4*inch, 0.5*inch, 0.5*inch, 0.6*inch, 0.5*inch], fontsize=6.8)
story.append(tblA2)

# ============================== BUILD ==============================
doc = SimpleDocTemplate(OUT, pagesize=letter,
                         topMargin=0.65*inch, bottomMargin=0.65*inch,
                         leftMargin=0.65*inch, rightMargin=0.65*inch,
                         title="DeLeAn Full Project Report")
doc.build(story)
print(f"Saved: {OUT}")
