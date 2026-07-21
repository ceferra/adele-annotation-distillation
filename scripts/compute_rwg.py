"""
Compute the rWG (within-group inter-rater agreement) index, following the
ADeLe paper's methodology (James, Demaree & Wolf 1984; used in ADeLe Table 7),
on Table II.2's underlying data: the 5-benchmark novel-task set, 18 rubrics.

Step 1: rWG among the three large annotator models (Qwen, Llama-3.3, Mistral),
        playing the role of the paper's "5 human raters" -> per-item rWG,
        averaged per rubric.
Step 2: Build a Delphi-style consensus per instance (median of the three large
        models, as a practical proxy for the paper's iterative Delphi
        reconciliation of human raters into one consensus rating).
Step 3: rWG between that consensus and the distilled model (playing the role
        of the paper's single "LLM annotator", i.e. GPT-4o, being validated
        against the consensus) -> per-item rWG (J=2), averaged per rubric.
"""
import pandas as pd
import numpy as np

K = 6  # levels 0-5
SIGMA_EU2 = (K**2 - 1) / 12  # uniform-null expected variance = 2.9167

df = pd.read_csv('newset_merged_18rubrics.csv')

def item_rwg(row_vals):
    """rWG for a single item given an array of ratings from J raters."""
    row_vals = np.asarray(row_vals, dtype=float)
    j = len(row_vals)
    if j < 2:
        return np.nan
    var = row_vals.var(ddof=0)  # population variance across the J raters for this item
    rwg = 1 - var / SIGMA_EU2
    return rwg

RUBRIC_ORDER = ['AS','AT','CEc','CEe','CL','KNa','KNc','KNf','KNn','KNs',
                'MCr','MCt','MCu','MS','QLl','QLq','SNs','VO']

rows = []
for rubric in RUBRIC_ORDER:
    sub = df[df['rubric'] == rubric].copy()

    # --- Step 1: rWG among the 3 large models (Qwen, Llama, Mistral) ---
    m3 = sub.dropna(subset=['qwen','llama33','mistral'])
    rwg_items_3 = m3[['qwen','llama33','mistral']].apply(item_rwg, axis=1)
    rwg_models = rwg_items_3.mean()
    rwg_models_trunc = rwg_items_3.clip(lower=0).mean()  # LeBreton & Senter: negative -> 0

    # --- Step 2: Delphi-style consensus = median(Qwen, Llama, Mistral) ---
    m3['consensus'] = m3[['qwen','llama33','mistral']].median(axis=1)

    # --- Step 3: rWG between consensus and distilled model ---
    both = m3.dropna(subset=['consensus','distilled'])
    rwg_items_cd = both[['consensus','distilled']].apply(item_rwg, axis=1)
    rwg_consensus_dist = rwg_items_cd.mean()
    rwg_consensus_dist_trunc = rwg_items_cd.clip(lower=0).mean()

    rows.append({
        'rubric': rubric,
        'n_models': len(m3),
        'rWG_models(Q,L,M)': round(rwg_models, 3),
        'rWG_models_trunc': round(rwg_models_trunc, 3),
        'n_consensus_vs_dist': len(both),
        'rWG_consensus_vs_distilled': round(rwg_consensus_dist, 3),
        'rWG_consensus_vs_distilled_trunc': round(rwg_consensus_dist_trunc, 3),
    })

out = pd.DataFrame(rows)
out.to_csv('rwg_newset_18rubrics.csv', index=False)
print(out.to_string(index=False))
print()
print("=== Averages across 18 rubrics ===")
print(f"Mean rWG(Q,L,M)              : {out['rWG_models(Q,L,M)'].mean():.3f}  (truncated: {out['rWG_models_trunc'].mean():.3f})")
print(f"Range rWG(Q,L,M)             : {out['rWG_models(Q,L,M)'].min():.3f} - {out['rWG_models(Q,L,M)'].max():.3f}")
print(f"Mean rWG(consensus,distilled): {out['rWG_consensus_vs_distilled'].mean():.3f}  (truncated: {out['rWG_consensus_vs_distilled_trunc'].mean():.3f})")
print(f"Range rWG(consensus,distilled): {out['rWG_consensus_vs_distilled'].min():.3f} - {out['rWG_consensus_vs_distilled'].max():.3f}")
