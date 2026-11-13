"""Concatenate per-experiment result files into one queryable view.

Each experiment writes its own JSONL (exp_random.jsonl,
exp_random_k_sweep.jsonl, ...) because resume, sharding and re-running
operate per experiment. Records already carry method, manifest, model,
platform and compute, so the concatenation is lossless and this view can
be regenerated at any time. The output adds "_src" (source file) to every
record; all_records.jsonl itself is excluded from the inputs so
re-aggregating never duplicates.

Run: python src/aggregate_results.py [--out results/all_records.jsonl]
"""

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(HERE, "results", "all_records.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    out_abs = os.path.abspath(args.out)

    files = sorted(glob.glob(os.path.join(args.results_dir, "*.jsonl")))
    files = [f for f in files if os.path.abspath(f) != out_abs]
    total = 0
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_abs, "w") as dst:
        for f in files:
            n = 0
            src_name = os.path.basename(f)
            with open(f) as src:
                for line in src:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    rec["_src"] = src_name
                    dst.write(json.dumps(rec) + "\n")
                    n += 1
            print(f"  {src_name}: {n} records")
            total += n
    print(f"wrote {out_abs}: {total} records from {len(files)} files")


if __name__ == "__main__":
    main()
