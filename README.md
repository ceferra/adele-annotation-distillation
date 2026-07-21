# DeLeAn Multi-Model Annotation & Distillation

Multi-model comparison, ensembling and knowledge-distillation study for
**ADeLe v1.0 / DeLeAn** demand-level annotation, plus a distilled 7B model
that can be used as a cheap drop-in substitute for large-LLM (or GPT-4o)
annotators on new tasks.

This repository is **built on top of the official ADeLe project**:
https://github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation

We reuse its 18-rubric demand-level annotation framework, its official
task battery (ADeLe v1.0), and its `delean_batch_manager` batching/pricing
tooling as the basis for everything done here. See
["Based on ADeLe"](#based-on-adele) below for exactly what was reused vs.
what is original to this repo.

## What's here

1. **A 3-model annotation pilot** (Qwen2.5-72B-Instruct-AWQ,
   Llama-3.3-70B-Instruct-AWQ, Mistral-Large-Instruct-2411-AWQ) run locally
   via vLLM on a SLURM/GPU cluster, annotating 1,520 algebra/calculus
   instances across all 18 ADeLe rubrics, and a study of which
   ensemble-of-models best approximates a reliable consensus label.
2. **A QLoRA distillation** of a much smaller and cheaper Qwen2.5-7B-Instruct
   model, trained to reproduce the demand-level annotation behaviour of the
   three large models on the full official ADeLe v1.0 battery (16,108 items,
   63 tasks), plus its validation against the large models on held-out data.
3. **A generalization study** on a 350-item, 6-task subset of ADeLe's own
   held-out tasks (tasks the distilled model never trained on), comparing
   all four models (3 large + 1 distilled).
4. **A novel-benchmark study**: the same 4 models re-run on 300 fresh items
   drawn from 5 benchmarks that are *not part of ADeLe at all*
   (MBPP, TheoremQA, Omega, RelBench, SWE-bench-Verified), to test true
   out-of-domain generalization.
5. **A rWG (within-group agreement) analysis** against GPT-4o's own
   annotations, directly addressing the practical question: *can the
   distilled 7B model (or an ensemble of the open models) be used instead of
   GPT-4o to annotate new tasks?*

## Headline findings

- **Shared upward bias across all 3 large models** on the algebra/calculus
  pilot: all three models tend to over-estimate demand levels relative to
  the reference labels, but they are highly correlated with each other. This
  motivated an ensembling study.
- **Ensembling helps**: taking `min(Qwen, Llama, Mistral)` per item was the
  single best simple ensemble candidate for 6 of 17 usable rubrics
  (Mistral had 0% usable outputs for MCr, context length exceeded);
  `min(Qwen, Llama)` won on another 4/17. See `results/pilot_algebra/`.
- **QLoRA distillation of a 7B model successfully reproduces annotator
  behaviour**: on both the training-domain algebra data and a small
  (n≈15-30/rubric) held-out-task sample, the distilled model tracks the
  three large models with Pearson/Spearman correlations and quadratic-weighted
  kappa comparable to the agreement between the large models themselves. See
  `results/distillation_validation/`.
- **Generalization confirmed on a larger held-out sample** (350 items, 6
  ADeLe tasks never seen in training) and on **300 completely novel,
  non-ADeLe benchmark items** (5 external benchmarks) — the distilled model's
  agreement with the large models does not collapse out-of-domain. See
  `results/adele_subset_generalization/` and `results/novel_benchmarks/`.
- **rWG vs. GPT-4o (the "can we replace GPT-4o?" question)**: using the
  within-group agreement index (rWG; James, Demaree & Wolf, 1984 — the same
  metric used in the ADeLe/DeLeAn Nature paper to compare its Delphi human
  consensus against GPT-4o) against the actual GPT-4o annotations already
  present in the benchmark data (GPT-4o is the `delean_batch_manager`
  default annotator model), the **distilled 7B model alone scores rWG=0.934**
  (mean over 18 rubrics, range 0.682-0.994) against GPT-4o — **higher than
  any of the three large annotator models individually** (Qwen2.5-72B=0.892,
  Llama-3.3-70B=0.908, Mistral-Large=0.918), and close to the full 4-model
  ensemble consensus (0.944). The one rubric where all candidates, including
  the full ensemble, struggle is **KNa** (all candidates < 0.78). See
  `results/rwg_gpt4o_analysis/`.
  - Practical answer: **yes**, for 17 of 18 rubrics the cheap distilled
    7B model is a viable stand-in for GPT-4o (or the large open models) when
    annotating brand-new tasks, at a fraction of the inference cost.

Full narrative writeups with all tables are in `reports/` (PDF):
`DeLeAn_Algebra_Annotation_Report.pdf` (pilot only),
`DeLeAn_Multi_Model_Annotation_Report.pdf` (ADeLe subset + novel benchmarks),
`DeLeAn_Full_Project_Report.pdf` (the whole project, all 4 parts + appendix).

## Repository structure

```
scripts/                         Python + SLURM scripts used at every stage
  run_annotation*.slurm          vLLM batch-annotation jobs (one per model)
  merge_inputs.py                merges per-rubric vLLM outputs into wide CSVs
  prepare_distillation_dataset.py  builds the QLoRA train/val/test split from
                                    the official ADeLe v1.0 battery
  train_distill_lora.py/.slurm   QLoRA fine-tuning of Qwen2.5-7B-Instruct
  merge_lora.py                 merges the trained LoRA adapter into a
                                 standalone model for fast vLLM inference
  eval_distilled.py             runs the distilled/merged model and compares
                                 it against the large models
  compute_rwg*.py                rWG index analyses (symmetric, leave-one-out,
                                 vs. GPT-4o, substitution ranking)
  build_report.py / build_full_report.py / generate_report.py
                                 reportlab PDF report generators

results/
  pilot_algebra/                 3-model comparison + ensembles, 1,520-item
                                  algebra/calculus pilot (18 rubrics)
  distillation_validation/       distilled model vs. large models, small
                                  validation samples (train-domain + held-out)
  adele_subset_generalization/   4-model comparison on 350 held-out ADeLe
                                  items, 6 tasks, 18 rubrics (+ ensembles)
  novel_benchmarks/               4-model comparison on 300 items from 5
                                  external benchmarks (+ ensembles)
  rwg_gpt4o_analysis/            rWG index tables: inter-model agreement,
                                  Delphi-style consensus, and agreement vs.
                                  GPT-4o's own annotations, per rubric

reports/                        Full PDF reports (see above)
docs/                           split_manifest.json (train/val/test task
                                 split used for distillation) and other notes
model_adapter/                  see "Distilled model" below
```

Raw per-item vLLM outputs, the full 16k-item distillation training set, and
intermediate batch-run artifacts (several GB) are **not** included here to
keep the repository lightweight; they can be regenerated with the scripts
above from the official ADeLe v1.0 battery, or are available on request.

## Distilled model

The distilled QLoRA adapter (`Qwen2.5-7B-Instruct` base, LoRA r=16, alpha=32,
dropout=0.05, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,
up_proj,down_proj`, trained for 1 epoch on 51 of the 63 official ADeLe v1.0
tasks — 6 tasks held out for validation, 6 for test, seed 42) is **~161MB**.

**Canonical location — Hugging Face Hub:**
https://huggingface.co/ceferra/qwen2.5-7b-adele-annotator

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "ceferra/qwen2.5-7b-adele-annotator")
tok = AutoTokenizer.from_pretrained("ceferra/qwen2.5-7b-adele-annotator")
```

It is also mirrored as a **GitHub Release asset** on this repository (a
`.tar.gz` containing the same PEFT adapter + tokenizer files, useful if you
don't want a Hugging Face dependency) — see the Releases page and load it
the same way, pointing `from_pretrained` at the extracted local folder
instead of the HF repo id.

Either way, merge the adapter into a standalone checkpoint with
`scripts/merge_lora.py` before serving it with vLLM (this is how it was
evaluated in this project).

## Methodology summary

- **Framework**: ADeLe v1.0's 18 demand-level rubrics (AS, AT, CEc, CEe, CL,
  KNa, KNc, KNf, KNn, KNs, MCr, MCt, MCu, MS, QLl, QLq, SNs, VO), each scored
  0-5.
- **Inference**: vLLM 0.6.1.post2, offline batch mode, tensor-parallel=4,
  AWQ 4-bit quantization for the large models, on a SLURM GPU cluster.
- **Comparison metrics**: MAE, exact agreement, within-1 agreement,
  quadratic-weighted Cohen's kappa, and (for the GPT-4o substitution
  question) the **rWG within-group agreement index**
  (`rWG = 1 - S_x^2 / sigma_EU^2`, uniform-null variance
  `sigma_EU^2 = (K^2-1)/12` for K=6 categories), following the same metric
  used in the ADeLe/DeLeAn Nature paper's human-vs-GPT-4o validation.
- **Ensembling**: simple order-statistic ensembles (min/max/median/mean)
  across model subsets, and a median-based "Delphi-style" consensus as a
  practical proxy for the paper's iterative human panel process.

## Based on ADeLe

This work would not exist without, and explicitly builds on:

- **ADeLe v1.0 framework and paper**: Zhou, L. et al., *"Predictable
  performance from a general-purpose evaluation instrument"* (Nature, 2026),
  and its arXiv preprint (arXiv:2503.06378) — the demand-level rubric
  taxonomy, the human-annotation validation methodology (Delphi consensus +
  rWG vs. GPT-4o), and the official ADeLe v1.0 task battery used to train
  the distilled model here.
- **Official ADeLe GitHub repository**:
  https://github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation (MIT
  License) — we reuse its `delean_batch_manager` package for batch-cost
  estimation/annotation orchestration, and its official rubric definitions
  and task battery as ground truth / training data.
- Related project page: https://github.com/Kinds-of-Intelligence-CFI/ADELE

Everything in this repository beyond that (the multi-model vLLM annotation
pipeline, the ensembling study, the QLoRA distillation, the generalization
studies on held-out ADeLe tasks and novel external benchmarks, and the rWG
substitution analysis vs. GPT-4o) is original work carried out for this
project.

## License

MIT — see `LICENSE`. Please also cite the ADeLe paper and repository above
if you use the annotation framework or task battery.
