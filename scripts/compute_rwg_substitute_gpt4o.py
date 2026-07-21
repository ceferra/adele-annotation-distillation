"""
Question: can any of our models (or simple ensembles of them) substitute for
GPT-4o as the annotator on NEW tasks, i.e. tasks where we don't already have
GPT-4o labels?

We test this using the rWG index (same metric as the ADeLe/Nature paper),
computed between GPT-4o ('true' column, the actual gold annotator used to
build these 5 new benchmarks) and each candidate substitute:
  - each of the 4 models solo
  - simple ensembles (median) of subsets of the 4 models

This is a fair substitution test because GPT-4o's annotations were produced
independently by delean_batch_manager -- none of our 4 models saw or derived
from GPT-4o's outputs.
"""
import pandas as pd
import numpy as np

K = 6
SIGMA_EU2 = (K**2 - 1) / 12

df = pd.read_csv('newset_merged_18rubrics.csv')
MODELS = ['qwen', 'llama33', 'mistral', 'distilled']
LABELS = {'qwen':'Qwen2.5-72B', 'llama33':'Llama-3.3-70B', 'mistral':'Mistral-Large-123B', 'distilled':'Distilled-7B'}

def rwg_pair(a, b):
    """rWG per item for a 2-rater comparison (candidate vs GPT-4o), then averaged."""
    vals = np.stack([a.values, b.values], axis=1).astype(float)
    var = vals.var(axis=1, ddof=0)
    return np.mean(1 - var / SIGMA_EU2)

RUBRIC_ORDER = ['AS','AT','CEc','CEe','CL','KNa','KNc','KNf','KNn','KNs',
                'MCr','MCt','MCu','MS','QLl','QLq','SNs','VO']

candidates = {
    'Qwen2.5-72B (solo)': ['qwen'],
    'Llama-3.3-70B (solo)': ['llama33'],
    'Mistral-Large-123B (solo)': ['mistral'],
    'Distilled-7B (solo)': ['distilled'],
    'median(Q,L,M)': ['qwen','llama33','mistral'],
    'median(Q,L,M,D)': MODELS,
    'median(Distilled + best large)': None,  # handled specially below (Llama)
}

rows = []
for rubric in RUBRIC_ORDER:
    sub = df[df['rubric'] == rubric]
    row = {'rubric': rubric, 'n': len(sub)}
    for name, cols in candidates.items():
        if name == 'median(Distilled + best large)':
            pred = sub[['distilled','llama33']].median(axis=1)
        elif len(cols) == 1:
            pred = sub[cols[0]]
        else:
            pred = sub[cols].median(axis=1)
        row[name] = round(rwg_pair(pred, sub['true']), 3)
    rows.append(row)

out = pd.DataFrame(rows)
out.to_csv('rwg_substitute_gpt4o.csv', index=False)

pd.set_option('display.width', 200)
print(out.to_string(index=False))
print()
print("=== Mean rWG vs GPT-4o across 18 rubrics, per candidate substitute ===")
means = out.drop(columns=['rubric','n']).mean().sort_values(ascending=False)
for name, val in means.items():
    lo = out[name].min()
    hi = out[name].max()
    print(f"  {name:32s}: mean={val:.3f}  range={lo:.3f}-{hi:.3f}")
print()
print("Paper reference (Nature): rWG(Delphi human consensus vs GPT-4o) = 0.75-0.94, avg 0.86")
print("Paper reference (Nature): rWG(5 humans vs each other, pre-Delphi) = 0.70-0.91, avg 0.83")
