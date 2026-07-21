"""
Prepare a SFT/LoRA training dataset for distilling ADeLe v1.0 demand-level
annotation into a small open-weight instruct model.

Source: official ADeLe v1.0 battery (16,108 items, 63 tasks, 20 benchmarks),
downloaded from HuggingFace (CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0).

For each (task instance, rubric) pair, builds the exact same prompt format used
by delean-batch-manager (get_full_instruction), with the target completion being
a short justification + the final "Thus, the level of *X* ... is: SCORE" line,
using the official annotated level as the label.

Output: JSONL files (train/val/test), split by `task` (not randomly) so that
validation/test can measure generalization to held-out task types, not just
held-out instances of already-seen tasks.
"""
import sys
sys.path.insert(0, '/tmp/delean-batch-manager/src')

import json
import random
import pandas as pd
from pathlib import Path
from delean_batch_manager.core.utils.rubrics import RubricsCatalog
from delean_batch_manager.core.batching.files import get_full_instruction

RUBRICS_FOLDER = '/tmp/delean-batch-manager/rubrics'
DATASET_CSV = '/home/administrador/delean_run/adele_official_dataset/ADeLe_battery_v1dot0.csv'
OUT_DIR = Path('/home/administrador/delean_run/distillation_dataset')
OUT_DIR.mkdir(exist_ok=True)

RUBRIC_CODES = ['AS','AT','CEc','CEe','CL','KNa','KNc','KNf','KNn','KNs',
                'MCr','MCt','MCu','MS','QLl','QLq','SNs','VO']

# Only use rows verified by the original human-in-the-loop process, for higher
# label confidence. Set to False to use all rows (larger but noisier).
ONLY_VERIFIED = True

SEED = 42
VAL_FRAC = 0.10
TEST_FRAC = 0.10  # held-out TASKS, not just held-out rows


def build_target_completion(subdomain: str, level: int) -> str:
    """
    Short synthetic completion matching the expected output format.
    Since the official dataset does not include the original CoT reasoning
    text (only the final level), we generate a minimal, honest placeholder
    reasoning that still teaches the model the expected output *format* and
    the correct final score. This keeps SFT targets truthful (no fabricated
    justification content) while preserving the required output structure.
    """
    return (
        f"Based on the rubric's level descriptions, the task instance's demands "
        f"most closely match the criteria described for level {level}.\n\n"
        f'Thus, the level of *{subdomain}* demanded by the given TASK INSTANCE is: {level}'
    )


def main():
    print("Loading rubrics catalog...")
    catalog = RubricsCatalog(RUBRICS_FOLDER)
    rubrics = {code: {'full_name': catalog.get_rubric(code).full_name, 'content': catalog.get_rubric(code).content}
               for code in RUBRIC_CODES}

    print("Loading ADeLe official dataset...")
    df = pd.read_csv(DATASET_CSV)
    print(f"  Total rows: {len(df)}")

    if ONLY_VERIFIED:
        df = df[df['verification_final'] == 1].reset_index(drop=True)
        print(f"  Verified rows only: {len(df)}")

    # Split by task (not by row) so val/test measure generalization to
    # held-out task types.
    tasks = sorted(df['task'].unique())
    rng = random.Random(SEED)
    rng.shuffle(tasks)

    n_tasks = len(tasks)
    n_test = max(1, int(n_tasks * TEST_FRAC))
    n_val = max(1, int(n_tasks * VAL_FRAC))

    test_tasks = set(tasks[:n_test])
    val_tasks = set(tasks[n_test:n_test + n_val])
    train_tasks = set(tasks[n_test + n_val:])

    print(f"  Tasks: {n_tasks} total -> {len(train_tasks)} train / {len(val_tasks)} val / {len(test_tasks)} test")

    split_of = {}
    for t in train_tasks:
        split_of[t] = 'train'
    for t in val_tasks:
        split_of[t] = 'val'
    for t in test_tasks:
        split_of[t] = 'test'

    df['split'] = df['task'].map(split_of)

    writers = {s: open(OUT_DIR / f'{s}.jsonl', 'w') for s in ('train', 'val', 'test')}
    counts = {s: 0 for s in ('train', 'val', 'test')}

    for _, row in df.iterrows():
        split = row['split']
        prompt_text = row['prompt']
        for code in RUBRIC_CODES:
            level = row[code]
            if pd.isna(level):
                continue
            level = int(level)
            subdomain = rubrics[code]['full_name']
            user_prompt = get_full_instruction(
                subdomain=subdomain,
                rubric_content=rubrics[code]['content'],
                prompt=prompt_text,
            )
            target = build_target_completion(subdomain, level)

            example = {
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": target},
                ],
                "meta": {
                    "rubric": code,
                    "level": level,
                    "task": row['task'],
                    "benchmark": row['benchmark'],
                    "source": row['source'],
                    "instance_id": row['instance_id'],
                }
            }
            writers[split].write(json.dumps(example) + '\n')
            counts[split] += 1

    for w in writers.values():
        w.close()

    print("\nExamples per split (task x rubric pairs):")
    for s in ('train', 'val', 'test'):
        print(f"  {s}: {counts[s]}")

    # Save split manifest for reference
    manifest = {
        'train_tasks': sorted(train_tasks),
        'val_tasks': sorted(val_tasks),
        'test_tasks': sorted(test_tasks),
        'only_verified': ONLY_VERIFIED,
        'seed': SEED,
    }
    with open(OUT_DIR / 'split_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Files written to {OUT_DIR}")


if __name__ == '__main__':
    main()
