"""
Batched generation + level extraction for the distilled model, given a JSONL
file of {"custom_id": ..., "messages": [...], "meta": {...}} examples
(assistant turn, if present, is ignored/stripped before generation).

Writes a CSV with one row per example: custom_id, predicted_level, plus all
meta fields flattened.

Robustness features (added after an OOM crash lost several hours of progress
on a mixed-length dataset that included long SWE-bench-verified prompts):
  - Examples are sorted by tokenized prompt length before batching, so each
    batch has similar-length prompts (much less padding waste, far less OOM
    risk than random-order batches mixing very long and very short prompts).
  - Results are flushed to the output CSV incrementally (append mode) after
    every batch, instead of only once at the very end.
  - If --output-csv already exists, already-processed custom_ids are skipped
    on restart (safe to re-run the same command after a crash).
  - Per-batch OOM recovery: on CUDA OOM, empty the cache and retry the same
    batch split into halves (recursively down to size 1) instead of crashing
    the whole job.
"""
import argparse
import csv
import json
import os
import re
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

LEVEL_PATTERN = re.compile(r"is:?\s*\**\s*(\d+)\.?\s*\**\s*\.?\s*$")


def extract_level(content):
    m = LEVEL_PATTERN.search(content.strip())
    if m:
        return int(m.group(1))
    nums = re.findall(r"\b([0-5])\b", content[-100:])
    return int(nums[-1]) if nums else None


def load_done_ids(output_csv):
    if not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv)
        return set(df["custom_id"].astype(str))
    except Exception:
        return set()


def generate_batch(model, tokenizer, batch, max_new_tokens):
    """Generate for a batch, recursively halving on CUDA OOM."""
    prompts = []
    for ex in batch:
        msgs = [m for m in ex["messages"] if m["role"] != "assistant"]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=3900).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_only = out[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        return list(zip(batch, decoded))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) == 1:
            print(f"  !! OOM on single example {batch[0].get('custom_id')}, skipping.")
            return [(batch[0], "")]
        mid = len(batch) // 2
        print(f"  !! OOM on batch of {len(batch)}, retrying split into {mid} + {len(batch) - mid}")
        return (generate_batch(model, tokenizer, batch[:mid], max_new_tokens)
                + generate_batch(model, tokenizer, batch[mid:], max_new_tokens))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="/data/$USER/delean_run/distill_merged")
    p.add_argument("--input-file", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=150)
    args = p.parse_args()

    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()

    examples = []
    with open(args.input_file) as f:
        for line in f:
            examples.append(json.loads(line))

    done_ids = load_done_ids(args.output_csv)
    if done_ids:
        print(f"Resuming: {len(done_ids)} examples already done, skipping those.")
        examples = [ex for ex in examples if str(ex.get("custom_id")) not in done_ids]

    # Sort by prompt length (proxy: total char length of all message contents)
    # so batches have homogeneous length -> less padding, less OOM risk.
    examples.sort(key=lambda ex: sum(len(m.get("content", "")) for m in ex["messages"]))

    print(f"{len(examples)} examples left to process. Running generation in batches of {args.batch_size}...")

    write_header = not os.path.exists(args.output_csv) or len(done_ids) == 0
    fieldnames = None
    fout = open(args.output_csv, "a", newline="")
    writer = None

    n_done = 0
    for i in range(0, len(examples), args.batch_size):
        batch = examples[i:i + args.batch_size]
        pairs = generate_batch(model, tokenizer, batch, args.max_new_tokens)

        rows = []
        for ex, text in pairs:
            pred = extract_level(text)
            row = {"custom_id": ex.get("custom_id"), "predicted_level": pred, "raw_completion": text[:300]}
            meta = ex.get("meta", {})
            row.update(meta)
            rows.append(row)

        if writer is None:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
        for row in rows:
            writer.writerow(row)
        fout.flush()

        n_done += len(batch)
        if (i // args.batch_size) % 5 == 0:
            print(f"  {n_done}/{len(examples)} done")

    fout.close()
    print(f"Done. Appended results to {args.output_csv}")


if __name__ == "__main__":
    main()
