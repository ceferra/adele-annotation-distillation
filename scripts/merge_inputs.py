import json
from pathlib import Path

base = Path("/home/administrador/delean_run/batch_runs/algebra")
out_path = base / "combined_input.jsonl"

n = 0
with open(out_path, "w") as out_f:
    for sub in sorted(base.iterdir()):
        infile = sub / "input.jsonl"
        if not infile.exists():
            continue
        acronym = sub.name
        with open(infile) as f:
            for line in f:
                req = json.loads(line)
                req["custom_id"] = f"{acronym}::{req['custom_id']}"
                out_f.write(json.dumps(req) + "\n")
                n += 1

print(f"Wrote {n} requests to {out_path}")
