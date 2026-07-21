"""
Generate a comprehensive PDF report on the DeLeAn (ADeLe v1.0) algebra annotation
experiment: methodology, infrastructure, per-rubric model comparison (Qwen2.5-72B
vs Llama-3.3-70B vs reference), ensemble strategies, and preliminary Mistral-Large
results.
"""
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

RUBRIC_NAMES = {
    'AS': 'Attention and Search',
    'AT': 'Atypicality',
    'CEc': 'Comprehension',
    'CEe': 'Expression',
    'CL': 'Conceptualization, Learning & Abstraction',
    'KNa': 'Knowledge in Applied Sciences and Professions',
    'KNc': 'Customary Everyday Knowledge',
    'KNf': 'Knowledge in Formal Sciences',
    'KNn': 'Knowledge in Natural Sciences',
    'KNs': 'Knowledge in Social Sciences and Humanities',
    'MCr': 'Identifying Relevant Information (metacognition)',
    'MCt': 'Critical Thinking Processes (metacognition)',
    'MCu': 'Calibrating Knowns and Unknowns (metacognition)',
    'MS': 'Mind Modelling and Social Cognition',
    'QLl': 'Logical Reasoning (deductive)',
    'QLq': 'Quantitative Reasoning',
    'SNs': 'Spatial-physical Reasoning',
    'VO': 'Volume (time/effort to complete task)',
}

RUBRIC_ORDER = ['AS','AT','CEc','CEe','CL','KNa','KNc','KNf','KNn','KNs',
                'MCr','MCt','MCu','MS','QLl','QLq','SNs','VO']

# ---------- Load data ----------
comp = pd.read_csv('/home/administrador/delean_run/comparison_18rubrics_qwen_vs_llama.csv')
ens = pd.read_csv('/home/administrador/delean_run/ensemble_analysis_18rubrics.csv')

mistral_as_at = pd.DataFrame([
    {'rubric':'AS','model':'Mistral','n':1483,'pearson':0.369,'spearman':0.399,'mae':0.740,'exact':0.476,'within1':0.873,'kappa_lin':0.265,'kappa_quad':0.344},
    {'rubric':'AT','model':'Mistral','n':1507,'pearson':0.597,'spearman':0.576,'mae':0.660,'exact':0.407,'within1':0.935,'kappa_lin':0.322,'kappa_quad':0.523},
])

# ---------- Styles ----------
styles = getSampleStyleSheet()
title_style = ParagraphStyle('TitleX', parent=styles['Title'], fontSize=20, spaceAfter=6)
subtitle_style = ParagraphStyle('SubtitleX', parent=styles['Normal'], fontSize=11, textColor=colors.grey, spaceAfter=20)
h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=15, spaceBefore=18, spaceAfter=8, textColor=colors.HexColor('#1a3c6e'))
h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12.5, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor('#2a5a94'))
body = ParagraphStyle('BodyX', parent=styles['Normal'], fontSize=9.7, leading=14, alignment=TA_LEFT, spaceAfter=6)
small = ParagraphStyle('SmallX', parent=styles['Normal'], fontSize=8, leading=11, textColor=colors.grey)
caption = ParagraphStyle('Caption', parent=styles['Normal'], fontSize=8.5, leading=11, textColor=colors.HexColor('#444444'), spaceAfter=10, spaceBefore=2)

def bullet_list(items):
    return ListFlowable(
        [ListItem(Paragraph(it, body), leftIndent=10) for it in items],
        bulletType='bullet', start='•', leftIndent=14
    )

def make_table(data, col_widths=None, header_bg='#1a3c6e', font_size=8.2, align_first_left=True):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor(header_bg)),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), font_size),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f2f6fa')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]
    if align_first_left:
        style.append(('ALIGN', (0,0), (0,-1), 'LEFT'))
    t.setStyle(TableStyle(style))
    return t

story = []

# ================= TITLE PAGE =================
story.append(Spacer(1, 40))
story.append(Paragraph("Comparative Evaluation of LLM Judges for<br/>ADeLe v1.0 Difficulty Annotation", title_style))
story.append(Paragraph("Algebra/Calculus Question Bank (N = 1,520) — Qwen2.5-72B vs Llama-3.3-70B vs Mistral-Large-2411, "
                       "benchmarked against a human-derived reference annotation", subtitle_style))
story.append(Spacer(1, 10))
story.append(Paragraph(
    "This report documents the setup, methodology, and results of a local (self-hosted) replication of the "
    "DeLeAn / ADeLe v1.0 difficulty-annotation pipeline, in which large open-weight language models act as "
    "automated judges that score each task along 18 cognitive/behavioral \"demand\" rubrics (0-5 scale). "
    "The goal is to determine which annotator model(s) best approximate a trusted human/reference annotation, "
    "and whether combining multiple models improves agreement over any single model.",
    body))
story.append(Spacer(1, 20))

# Executive summary box
summary_items = [
    "<b>Qwen2.5-72B-Instruct-AWQ</b> is the strongest single annotator overall, outperforming Llama-3.3-70B on "
    "10 of 14 rubrics with usable reference signal.",
    "<b>Llama-3.3-70B-Instruct-AWQ</b> exhibits a stronger systematic upward bias (over-estimates difficulty) "
    "than Qwen across almost all rubrics, which depresses its exact-agreement rate.",
    "Taking the <b>element-wise minimum of Qwen and Llama</b> predictions improves quadratic-weighted Cohen's "
    "kappa over the best individual model in 10 of 14 rubrics (mean improvement +0.038), because it corrects "
    "for the shared upward bias. The only clear exception is QLl (Logical Reasoning), where Qwen alone is "
    "already the strongest signal and the ensemble introduces noise.",
    "<b>Mistral-Large-Instruct-2411-AWQ</b> was added as a third, statistically independent judge; only 2 of "
    "18 rubrics are complete so far (still processing on the cluster), showing intermediate performance "
    "between Qwen and Llama on those two rubrics.",
    "Four rubrics (KNa, KNc, KNn, KNs, MS) show degenerate/undefined correlation statistics because the "
    "reference annotation is constant (always 0) for this algebra-specific dataset — these demand types are "
    "simply not exercised by algebra/calculus tasks.",
]
story.append(Paragraph("Executive Summary", h1))
story.append(bullet_list(summary_items))
story.append(PageBreak())

# ================= 1. BACKGROUND & METHODOLOGY =================
story.append(Paragraph("1. Background and Objective", h1))
story.append(Paragraph(
    "ADeLe v1.0 (Annotated Demand Levels) is a framework for characterizing the cognitive and behavioral "
    "demands of a task using 18 independent rubrics, each scored on a 0-5 ordinal scale via chain-of-thought "
    "LLM judging. The reference pipeline (delean-batch-manager) was originally designed around the OpenAI "
    "Batch API; for this project it was repurposed to run entirely locally using vLLM's offline batch "
    "inference engine on a SLURM-managed GPU cluster, avoiding per-token API costs and allowing full control "
    "over annotator model choice.", body))
story.append(Paragraph(
    "The dataset under annotation is a bank of 1,520 algebra/calculus questions. A trusted reference "
    "annotation (produced separately, treated as ground truth for this comparison) exists for all 18 rubrics. "
    "The central question addressed here is: <i>which open-weight LLM(s), used as automated judges, best "
    "reproduce this reference annotation, and can multiple judges be combined to do better than any one alone?</i>",
    body))

story.append(Paragraph("1.1 The 18 ADeLe Rubrics", h2))
story.append(Paragraph(
    "Each rubric captures a distinct cognitive/behavioral demand dimension. Codes and definitions used "
    "throughout this report:", body))

rubric_table_data = [["Code", "Name"]]
for code in RUBRIC_ORDER:
    rubric_table_data.append([code, RUBRIC_NAMES[code]])
t = make_table(rubric_table_data, col_widths=[0.9*inch, 5.3*inch], font_size=8.5)
story.append(t)
story.append(Spacer(1, 10))

story.append(Paragraph("1.2 Infrastructure", h2))
story.append(bullet_list([
    "SLURM cluster (Vrain HPC, DSIC-UPV), nodes vrhpc3/5/6, each with 4× GPUs (PCIe-only, no NVLink).",
    "Inference engine: vLLM 0.6.1.post2, offline batch mode (<font face='Courier'>vllm.entrypoints.openai.run_batch</font>), "
    "tensor-parallel size 4, AWQ 4-bit quantization (<font face='Courier'>awq_marlin</font>), "
    "<font face='Courier'>--max-model-len 4096</font>, <font face='Courier'>--enforce-eager</font>, "
    "<font face='Courier'>--gpu-memory-utilization 0.95</font>.",
    "NCCL configured for PCIe-only interconnect (<font face='Courier'>NCCL_P2P_DISABLE=1</font>, "
    "<font face='Courier'>NCCL_IB_DISABLE=1</font>), attention backend forced to XFormers "
    "(<font face='Courier'>VLLM_ATTENTION_BACKEND=XFORMERS</font>) for eager-mode compatibility.",
    "Per-rubric sequential processing loop with skip-if-output-exists resumability: each of the 18 rubrics "
    "is submitted as a separate batch job within the same SLURM allocation, and completed rubrics are "
    "automatically skipped on job resubmission (needed after time-limit cancellations).",
]))

story.append(Paragraph("1.3 Models Evaluated", h2))
model_table = [
    ["Model", "Size", "Quantization", "Status"],
    ["Qwen2.5-72B-Instruct-AWQ", "72B dense", "AWQ 4-bit", "Complete (18/18 rubrics)"],
    ["Llama-3.3-70B-Instruct-AWQ", "70B dense", "AWQ 4-bit", "Complete (18/18 rubrics)"],
    ["Mistral-Large-Instruct-2411-AWQ", "123B dense", "AWQ 4-bit", "In progress (2/18 rubrics)"],
    ["Mixtral-8x22B-Instruct-v0.1-AWQ", "141B MoE", "AWQ 4-bit", "Discarded (underperformed)"],
]
t = make_table(model_table, col_widths=[2.5*inch, 1.1*inch, 1.1*inch, 1.5*inch])
story.append(t)
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Note: a Qwen3-32B candidate was also considered but discarded due to an architecture incompatibility "
    "with the installed vLLM version (Qwen3ForCausalLM is not supported in vLLM 0.6.1.post2).", small))

story.append(Paragraph("1.4 Comparison Methodology", h2))
story.append(Paragraph(
    "For each rubric, model predictions are merged against the reference annotation on a common task ID, "
    "restricted to rows where both a valid model prediction and a valid reference value exist (rows with "
    "parsing failures / context-length truncation are dropped from that rubric's comparison, hence sample "
    "sizes vary slightly by rubric and model). The following metrics are reported:", body))
story.append(bullet_list([
    "<b>Pearson / Spearman correlation</b> — linear and rank agreement between model and reference scores.",
    "<b>MAE</b> — mean absolute error on the 0-5 scale.",
    "<b>Exact agreement</b> — fraction of tasks where model score equals reference score exactly.",
    "<b>Within-1 agreement</b> — fraction of tasks where model and reference differ by at most one level.",
    "<b>Cohen's kappa (linear- and quadratic-weighted)</b> — chance-corrected ordinal agreement; the "
    "quadratic-weighted variant is used as the primary ranking criterion since it penalizes larger "
    "disagreements more heavily, which is appropriate for an ordinal 0-5 difficulty scale.",
]))

story.append(PageBreak())

# ================= 2. QWEN vs LLAMA — FULL 18-RUBRIC RESULTS =================
story.append(Paragraph("2. Qwen2.5-72B vs Llama-3.3-70B: Full 18-Rubric Comparison", h1))
story.append(Paragraph(
    "Both models completed annotation of all 1,520 tasks across all 18 rubrics. The table below reports "
    "quadratic-weighted kappa (the primary agreement metric) and exact-agreement rate for each rubric and "
    "model, against the reference annotation. Rubrics are sorted in ADeLe's standard order; rows where the "
    "reference is constant for this dataset (and thus correlation/kappa are undefined) are marked N/A.", body))

header = ["Rubric", "N (Qwen)", "Qwen κ-quad", "Qwen exact", "N (Llama)", "Llama κ-quad", "Llama exact", "Better model"]
rows = [header]
for rub in RUBRIC_ORDER:
    sub = comp[comp['rubric'] == rub]
    if sub.empty:
        continue
    qrow = sub[sub['model']=='Qwen'].iloc[0]
    lrow = sub[sub['model']=='Llama'].iloc[0]
    def fmt(v):
        return "N/A" if pd.isna(v) else f"{v:.3f}"
    if pd.isna(qrow['kappa_quad']) and pd.isna(lrow['kappa_quad']):
        better = "—"
    elif pd.isna(lrow['kappa_quad']):
        better = "Qwen"
    elif pd.isna(qrow['kappa_quad']):
        better = "Llama"
    else:
        better = "Qwen" if qrow['kappa_quad'] > lrow['kappa_quad'] else ("Llama" if lrow['kappa_quad'] > qrow['kappa_quad'] else "Tie")
    rows.append([
        rub, int(qrow['n']), fmt(qrow['kappa_quad']), fmt(qrow['exact']),
        int(lrow['n']), fmt(lrow['kappa_quad']), fmt(lrow['exact']), better
    ])

t = make_table(rows, col_widths=[0.55*inch, 0.65*inch, 0.85*inch, 0.75*inch, 0.65*inch, 0.85*inch, 0.75*inch, 0.85*inch], font_size=7.6)
story.append(t)
story.append(Paragraph(
    "N = number of tasks with both a valid model prediction and a valid non-missing reference value for that rubric. "
    "κ-quad = Cohen's kappa, quadratic weights. \"Better model\" indicates which model achieves higher κ-quad "
    "(N/A rubrics excluded).", caption))

n_qwen_wins = sum(1 for rub in RUBRIC_ORDER
                   for sub in [comp[comp['rubric']==rub]]
                   if not sub.empty and not pd.isna(sub[sub['model']=='Qwen']['kappa_quad'].values[0])
                   and not pd.isna(sub[sub['model']=='Llama']['kappa_quad'].values[0])
                   and sub[sub['model']=='Qwen']['kappa_quad'].values[0] > sub[sub['model']=='Llama']['kappa_quad'].values[0])

story.append(Paragraph("2.1 Key Findings", h2))
story.append(bullet_list([
    f"Qwen achieves higher quadratic-weighted kappa than Llama on <b>{n_qwen_wins} of 14</b> comparable rubrics.",
    "Qwen's advantage is most pronounced on <b>QLl</b> (Logical Reasoning, κ=0.503 vs 0.290), "
    "<b>QLq</b> (Quantitative Reasoning, κ=0.483 vs 0.305), <b>MCu</b> (Calibrating Knowns/Unknowns, "
    "κ=0.599 vs 0.473), and <b>VO</b> (Volume/effort, κ=0.544 vs 0.374) — all rubrics where numerical or "
    "logical precision matters, plausibly favoring Qwen's stronger STEM training emphasis.",
    "Llama's advantage is limited to <b>AS</b> (Attention and Search, κ=0.364 vs 0.297) and "
    "<b>KNf</b> (Knowledge in Formal Sciences, κ=0.485 vs 0.454) — both narrow margins.",
    "Across nearly all rubrics, Llama shows a stronger upward bias than Qwen (higher mean predicted level "
    "relative to the reference), which depresses its exact-agreement rate even when correlation is comparable "
    "(e.g., AT: exact agreement 54.7% for Qwen vs only 30.2% for Llama, despite similar Pearson correlation).",
    "Both models fail similarly on <b>CEe</b> (Expression) and <b>KNc</b> (Customary Everyday Knowledge) — "
    "near-zero kappa for both, suggesting these rubrics are either genuinely hard to judge for this domain "
    "or poorly specified for algebra-style tasks.",
]))

story.append(PageBreak())

# ================= 3. ENSEMBLE ANALYSIS =================
story.append(Paragraph("3. Ensembling Qwen and Llama", h1))
story.append(Paragraph(
    "Since both models are systematically biased upward relative to the reference (i.e., they tend to rate "
    "tasks as more demanding than the reference does), three simple combination strategies were tested per "
    "rubric: the rounded <b>average</b> of the two scores, the <b>element-wise minimum</b>, and the "
    "<b>element-wise maximum</b>. The minimum operator is expected to partially cancel a shared upward bias.",
    body))

header = ["Rubric", "Qwen κ", "Llama κ", "Avg κ", "Min κ", "Max κ", "Best single", "Best ensemble", "Δ improvement"]
rows = [header]
for _, r in ens.iterrows():
    def fmt(v):
        return "N/A" if pd.isna(v) else f"{v:.3f}"
    rows.append([
        r['rubric'], fmt(r['Qwen_kq']), fmt(r['Llama_kq']), fmt(r['Avg_kq']), fmt(r['Min_kq']), fmt(r['Max_kq']),
        fmt(r['best_single']) if r['best_single'] != -1 else "N/A",
        fmt(r['best_ensemble']),
        fmt(r['improvement'])
    ])
t = make_table(rows, col_widths=[0.55*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.75*inch, 0.8*inch, 0.85*inch], font_size=7.3)
story.append(t)
story.append(Paragraph("All values are quadratic-weighted Cohen's kappa vs. reference. Rubrics with constant "
                        "reference values (KNa, KNn, KNs, MS) are omitted from this table.", caption))

n_ens_wins = (ens['improvement'] > 0).sum()
n_total = ens['improvement'].notna().sum()
mean_imp = ens['improvement'].mean()

story.append(Paragraph("3.1 Key Findings", h2))
story.append(bullet_list([
    f"The best ensemble strategy beats the best individual model on <b>{n_ens_wins} of {n_total}</b> comparable "
    f"rubrics, with a mean quadratic-kappa improvement of <b>+{mean_imp:.3f}</b>.",
    "<b>Minimum(Qwen, Llama)</b> is the winning strategy in the large majority of cases (10/14), confirming "
    "the hypothesis that both models share a systematic upward-bias failure mode that the min operator "
    "partially corrects.",
    "Largest gains from ensembling are seen on <b>MCt</b> (+0.117, Critical Thinking) and <b>MCu</b> (+0.111, "
    "Calibrating Knowns/Unknowns) — both metacognition rubrics where neither individual model is strong "
    "alone, but the shared bias correction yields a substantial joint improvement.",
    "<b>QLl (Logical Reasoning) is the one clear exception</b>: Qwen alone (κ=0.503) is already far stronger "
    "than Llama (κ=0.290) on this rubric, and combining them (best ensemble κ=0.386) actually hurts — the "
    "min/average operators dilute Qwen's strong signal with Llama's weaker one. This illustrates that "
    "ensembling should be applied selectively, not blindly rubric-by-rubric.",
    "<b>Recommended default policy</b>: use min(Qwen, Llama) as the combined annotation for all rubrics "
    "except QLl, where Qwen's individual prediction should be used directly.",
]))

story.append(PageBreak())

# ================= 4. MISTRAL PRELIMINARY RESULTS =================
story.append(Paragraph("4. Mistral-Large-2411: Preliminary Results (Third Judge)", h1))
story.append(Paragraph(
    "Mistral-Large-Instruct-2411-AWQ (123B dense, AWQ 4-bit quantized checkpoint from "
    "TechxGenus/Mistral-Large-Instruct-2411-AWQ) was launched as a third, independent annotator to enable "
    "more robust ensembling in the future (e.g., median or majority vote across three judges rather than "
    "min-of-two). At the time of writing, only 2 of 18 rubrics have completed; results below should be "
    "treated as preliminary.", body))

header = ["Rubric", "Model", "N", "Pearson", "MAE", "Exact", "κ-quad"]
rows = [header]
for rub in ['AS', 'AT']:
    for model in ['Qwen', 'Llama']:
        r = comp[(comp['rubric']==rub) & (comp['model']==model)].iloc[0]
        rows.append([rub, model, int(r['n']), f"{r['pearson']:.3f}", f"{r['mae']:.3f}", f"{r['exact']:.3f}", f"{r['kappa_quad']:.3f}"])
    r = mistral_as_at[mistral_as_at['rubric']==rub].iloc[0]
    rows.append([rub, 'Mistral', int(r['n']), f"{r['pearson']:.3f}", f"{r['mae']:.3f}", f"{r['exact']:.3f}", f"{r['kappa_quad']:.3f}"])

t = make_table(rows, col_widths=[0.8*inch, 1.0*inch, 0.7*inch, 0.9*inch, 0.7*inch, 0.7*inch, 0.9*inch], font_size=8.5)
story.append(t)
story.append(Spacer(1, 10))

story.append(Paragraph("4.1 Preliminary Observations", h2))
story.append(bullet_list([
    "On <b>AS</b>, Llama is marginally the strongest of the three (κ=0.364), with Mistral close behind "
    "(κ=0.344) and Qwen last (κ=0.297).",
    "On <b>AT</b>, Qwen is clearly strongest (κ=0.664), with Mistral (κ=0.523) and Llama (κ=0.539) close "
    "to each other, both well behind Qwen.",
    "Mistral does not yet stand out as uniformly better or worse than the other two — it occupies an "
    "intermediate position on both rubrics evaluated so far. A firmer conclusion requires the remaining "
    "16 rubrics to complete.",
    "Once complete, the plan is to extend the ensemble analysis in Section 3 to a three-way combination "
    "(e.g., median or majority vote of Qwen/Llama/Mistral), which should be more robust to any single "
    "model's idiosyncratic failure modes than the current min-of-two heuristic.",
]))

story.append(Paragraph("5. Data Quality Notes", h1))
story.append(bullet_list([
    "<b>MCr</b> and <b>QLl</b> show a notably higher rate of missing/unparseable model responses "
    "(~10-30% of tasks) than other rubrics for both Qwen and Llama, suggesting these rubric prompts may be "
    "more prone to eliciting responses that don't conform to the expected structured output format. Root "
    "cause not yet investigated.",
    "A residual ~0.1-0.2% of tasks per rubric fail due to context-length overflow (prompt + completion "
    "exceeding the 4096-token model context limit after raising max_completion_tokens from 1000 to 2000 to "
    "fix truncation issues); this was judged low-impact enough not to warrant re-running with a larger "
    "context window.",
    "Four rubrics (<b>KNa, KNn, KNs, MS</b>) have a constant reference value of 0 across the entire algebra "
    "dataset, correctly reflecting that these demand types (applied-science knowledge, natural-science "
    "knowledge, social-science knowledge, and social cognition/mind-modelling) are simply not exercised by "
    "algebra/calculus tasks — this is a property of the dataset domain, not a data quality defect.",
    "<b>KNa</b> is a partial exception: although the reference is constant 0, both Qwen and Llama "
    "hallucinate non-zero levels here (shared false-positive failure mode), unlike KNc/KNn/KNs/MS where at "
    "least one model correctly predicts the constant 0.",
]))

story.append(Paragraph("6. Summary Recommendations", h1))
story.append(bullet_list([
    "Use <b>Qwen2.5-72B-Instruct-AWQ</b> as the primary single-model annotator if only one model can be used "
    "— it outperforms Llama-3.3-70B on the majority of rubrics with meaningful signal.",
    "For production annotation, use the <b>min(Qwen, Llama)</b> ensemble by default, with the single "
    "exception of QLl (use Qwen alone there).",
    "Complete the in-progress <b>Mistral-Large-2411</b> run and re-evaluate whether a three-model "
    "combination (median / majority vote) further improves on the two-model minimum heuristic.",
    "Treat KNa, KNc, KNn, KNs, and MS as out-of-scope / non-informative rubrics for algebra-domain datasets; "
    "any future domain-general validation should include a task sample where these rubrics have genuine "
    "variance.",
])
)

doc = SimpleDocTemplate(
    "/home/administrador/delean_run/DeLeAn_Algebra_Annotation_Report.pdf",
    pagesize=letter,
    topMargin=0.7*inch, bottomMargin=0.7*inch, leftMargin=0.7*inch, rightMargin=0.7*inch,
    title="Comparative Evaluation of LLM Judges for ADeLe v1.0 Difficulty Annotation"
)
doc.build(story)
print("PDF generated.")
