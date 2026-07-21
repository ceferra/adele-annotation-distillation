"""
Symmetric rWG analysis: treat all 4 models (Qwen, Llama-3.3, Mistral, Distilled)
as equal peers, rather than singling out the distilled model as the one being
"validated" against a consensus of the other three (as the previous script did,
mirroring the ADeLe paper's human-consensus-vs-GPT-4o asymmetric design).

Two views:
  A) rWG among all 4 models jointly, per item, per rubric (fully symmetric).
  B) Leave-one-out: for each of the 4 models M, build consensus = median of the
     OTHER 3, then compute rWG(consensus_others, M). Repeated for all 4 models,
     so each one gets a turn being "the odd one out" -- letting us see whether
     the distilled model is any more or less of an outlier than any large model.
"""
import pandas as pd
import numpy as np

K = 6
SIGMA_EU2 = (K**2 - 1) / 12

df = pd.read_csv('newset_merged_18rubrics.csv')
MODELS = ['qwen', 'llama33', 'mistral', 'distilled']
LABELS = {'qwen': 'Qwen2.5-72B', 'llama33': 'Llama-3.3-70B', 'mistral': 'Mistral-Large-123B', 'distilled': 'Distilled-7B'}

def item_rwg(vals):
    vals = np.asarray(vals, dtype=float)
    var = vals.var(ddof=0)
    return 1 - var / SIGMA_EU2

RUBRIC_ORDER = ['AS','AT','CEc','CEe','CL','KNa','KNc','KNf','KNn','KNs',
                'MCr','MCt','MCu','MS','QLl','QLq','SNs','VO']

# ---------- A) Symmetric rWG among all 4 models ----------
rows_a = []
for rubric in RUBRIC_ORDER:
    sub = df[df['rubric'] == rubric]
    rwg_items = sub[MODELS].apply(item_rwg, axis=1)
    rows_a.append({'rubric': rubric, 'n': len(sub), 'rWG(Q,L,M,D)': round(rwg_items.mean(), 3)})
out_a = pd.DataFrame(rows_a)

# ---------- B) Leave-one-out: consensus of other 3 vs each single model ----------
rows_b = []
for rubric in RUBRIC_ORDER:
    sub = df[df['rubric'] == rubric]
    row = {'rubric': rubric, 'n': len(sub)}
    for held_out in MODELS:
        others = [m for m in MODELS if m != held_out]
        consensus = sub[others].median(axis=1)
        pair = pd.DataFrame({'consensus': consensus, 'held_out': sub[held_out]})
        rwg_items = pair.apply(item_rwg, axis=1)
        row[f'rWG(rest vs {LABELS[held_out]})'] = round(rwg_items.mean(), 3)
    rows_b.append(row)
out_b = pd.DataFrame(rows_b)

out_a.to_csv('rwg_newset_symmetric_4models.csv', index=False)
out_b.to_csv('rwg_newset_leaveoneout.csv', index=False)

print("=== A) Symmetric rWG among all 4 models (Q,L,M,D), per rubric ===")
print(out_a.to_string(index=False))
print(f"\nMean across 18 rubrics: {out_a['rWG(Q,L,M,D)'].mean():.3f}")
print(f"Range: {out_a['rWG(Q,L,M,D)'].min():.3f} - {out_a['rWG(Q,L,M,D)'].max():.3f}")

print("\n\n=== B) Leave-one-out: rWG(consensus of other 3, held-out model) ===")
print(out_b.to_string(index=False))
print("\nMean across 18 rubrics, per held-out model:")
for held_out in MODELS:
    col = f'rWG(rest vs {LABELS[held_out]})'
    print(f"  {LABELS[held_out]:22s}: {out_b[col].mean():.3f}  (range {out_b[col].min():.3f}-{out_b[col].max():.3f})")
