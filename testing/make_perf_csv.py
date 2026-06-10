#!/usr/bin/env python3
"""
make_perf_csv.py — extract a flat perf.csv from a sweep's results dir.

One row per test case (driven by summary.csv), pulling the headline metrics out
of each test's sglang.bench_serving JSON (bench_<label>_r<rate>.json). Tests that
were skipped / failed (no JSON) still get a row, with blank metric columns and
their status carried through, so the CSV mirrors summary.csv 1:1.

Use it two ways:
  - standalone:  python3 make_perf_csv.py <model_results_dir>
                 e.g. .../results/20260610_074550_Qwen_Qwen3-VL-32B-Instruct-FP8
  - imported:    from make_perf_csv import write_perf_csv; write_perf_csv(root)
    (run_matrix.py calls this at the end of a sweep)

Writes <model_results_dir>/perf.csv and returns its path.
"""

import csv
import json
import os
import sys

# CSV column header -> bench JSON key. Order here == column order in perf.csv.
# 'rate' is taken from summary.csv (always present, even for skipped tests).
METRIC_COLUMNS = [
    ("Request throughput (req/s)", "request_throughput"),
    ("Mean TTFT (ms)", "mean_ttft_ms"),
    ("Mean TPOT (ms)", "mean_tpot_ms"),
    ("Mean ITL (ms)", "mean_itl_ms"),
    ("Mean E2E Latency (ms)", "mean_e2e_latency_ms"),
    ("Successful requests", "completed"),
    ("Total input vision tokens", "total_input_vision_tokens"),
    ("Total input text tokens", "total_input_text_tokens"),
    ("Total generated tokens (retokenized)", "total_output_tokens_retokenized"),
]

HEADER = (["case", "label", "status", "Traffic request rate"]
          + [name for name, _ in METRIC_COLUMNS])


def _fmt(v):
    """Render a metric for CSV: blank for missing, int kept as int, float to 2dp."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        # integral floats (e.g. token counts) print without a trailing .0
        return int(v) if v.is_integer() else round(v, 2)
    return v


def _read_summary(root):
    """Yield (case, label, rate, status) rows from summary.csv, in file order.
    Falls back to walking case*/<label>_r<rate>/ dirs if summary.csv is absent."""
    summary = os.path.join(root, "summary.csv")
    if os.path.exists(summary):
        with open(summary, newline="") as f:
            for row in csv.DictReader(f):
                yield row["case"], row["label"], row["rate"], row.get("status", "")
        return
    # Fallback: reconstruct from directory layout (no status available).
    for case in sorted(os.listdir(root)):
        cdir = os.path.join(root, case)
        if not os.path.isdir(cdir) or not case.startswith("case"):
            continue
        for sub in sorted(os.listdir(cdir)):
            # sub looks like "<label>_r<rate>"
            if "_r" not in sub:
                continue
            label, _, rate = sub.rpartition("_r")
            yield case, label, rate, ""


def _load_bench(root, case, label, rate):
    """Return the parsed bench JSON dict for one test, or None if missing/bad."""
    path = os.path.join(root, case, f"{label}_r{rate}",
                        f"bench_{label}_r{rate}.json")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def write_perf_csv(root, out_path=None):
    """Build perf.csv for the sweep rooted at `root`. Returns the output path."""
    out_path = out_path or os.path.join(root, "perf.csv")
    rows_written = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for case, label, rate, status in _read_summary(root):
            data = _load_bench(root, case, label, rate)
            row = [case, label, status, rate]
            if data is None:
                row += [""] * len(METRIC_COLUMNS)
            else:
                row += [_fmt(data.get(key)) for _, key in METRIC_COLUMNS]
            w.writerow(row)
            rows_written += 1
    print(f"[perf] wrote {rows_written} rows -> {out_path}")
    return out_path


def main(argv):
    if len(argv) != 1:
        print(f"usage: {os.path.basename(sys.argv[0])} <model_results_dir>",
              file=sys.stderr)
        return 2
    root = os.path.abspath(os.path.expanduser(argv[0]))
    if not os.path.isdir(root):
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2
    write_perf_csv(root)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
