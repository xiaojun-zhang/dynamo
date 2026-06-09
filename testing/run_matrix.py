#!/usr/bin/env python3
"""
run_matrix.py — trigger a sweep of benchmark tests over a parameter matrix.

OPT-IN: nothing runs unless you enable a case. Each enabled case sweeps the full
cross-product of its instance counts x request rates. Resumable: a test whose
result JSON already exists is skipped.

Cases (enable by passing the flag; absent = disabled):
  --case1-agg     "1,2,4"              aggregation: N agg instances
  --case2-epd-gpu "E=1,2,4;PD=1,2,4"   E on GPU x PD on GPU
  --case3-epd-xpu "E=1,2,4;PD=1,2,4"   E on XPU x PD on GPU

Shared:
  --model (required)        --rates "0.2,0.4,...,2.0"
  --num-prompts 32          --image-count 8       --image-resolution 1080p
  --gpus 0,1,2,3,4,5        --xpus 0,1,2,3        (xpus only needed for case3)
  --results-root DIR        --dry-run             --max-tests N

Example:
  python3 run_matrix.py --model Qwen/Qwen3-VL-8B-Instruct \\
      --gpus 0,1,2,3,4,5 --rates 0.2,0.4,0.6,0.8,1.0 \\
      --case1-agg 1,2,4
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import bench_lib as B

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_counts(s):
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def parse_epd(s):
    """'E=1,2,4;PD=1,2,4' -> ([1,2,4],[1,2,4])."""
    e = pd = None
    for part in s.split(";"):
        k, _, v = part.partition("=")
        k = k.strip().upper()
        if k == "E":
            e = parse_counts(v)
        elif k == "PD":
            pd = parse_counts(v)
    if not e or not pd:
        raise SystemExit(f"bad --case*-epd spec '{s}', want 'E=1,2;PD=1,2'")
    return e, pd


def build_plan(args):
    """Return list of test dicts (mode + instance counts + rate)."""
    rates = [r for r in args.rates.replace(" ", "").split(",") if r]
    plan = []
    if args.case1_agg:
        for n in parse_counts(args.case1_agg):
            for r in rates:
                plan.append(dict(case="case1_agg", mode="agg",
                                 agg=n, e=0, pd=0, rate=r, label=f"{n}AGG"))
    if args.case2_epd_gpu:
        es, pds = parse_epd(args.case2_epd_gpu)
        for e in es:
            for pd in pds:
                for r in rates:
                    plan.append(dict(case="case2_epd_gpu", mode="epd_gpu",
                                     agg=0, e=e, pd=pd, rate=r, label=f"{e}E{pd}PD"))
    if args.case3_epd_xpu:
        es, pds = parse_epd(args.case3_epd_xpu)
        for e in es:
            for pd in pds:
                for r in rates:
                    plan.append(dict(case="case3_epd_xpu", mode="epd_xpu",
                                     agg=0, e=e, pd=pd, rate=r, label=f"{e}E{pd}PD"))
    return plan


def result_done(out_dir, label, rate):
    j = os.path.join(out_dir, f"bench_{label}_r{rate}.json")
    return os.path.exists(j) and os.path.getsize(j) > 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--case1-agg", default="")
    ap.add_argument("--case2-epd-gpu", default="")
    ap.add_argument("--case3-epd-xpu", default="")
    ap.add_argument("--rates", default="0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0")
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--image-count", type=int, default=8)
    ap.add_argument("--image-resolution", default="1080p")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--xpus", default="")
    ap.add_argument("--results-root", default=os.path.join(HERE, "results"))
    ap.add_argument("--mm-attn-backend", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-tests", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    plan = build_plan(args)
    if not plan:
        B.log("No test cases enabled. Pass --case1-agg / --case2-epd-gpu / "
              "--case3-epd-xpu to enable. Nothing to do.")
        return 0

    B.model_info(args.model)  # validate model early
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "_")
    root = os.path.join(args.results_root, f"{stamp}_{safe_model}")
    summary_csv = os.path.join(root, "summary.csv")

    B.log(f"Planned {len(plan)} tests over model={args.model}")
    B.log(f"  gpus={args.gpus or '(none)'}  xpus={args.xpus or '(none)'}")
    B.log(f"  results -> {root}")
    if args.case3_epd_xpu and not args.xpus:
        B.log("  WARN: case3 (E on XPU) enabled but --xpus is empty; those tests will skip.")

    if args.dry_run:
        for i, t in enumerate(plan, 1):
            print(f"  [{i}/{len(plan)}] {t['case']} {t['label']} rate={t['rate']}")
        return 0

    os.makedirs(root, exist_ok=True)
    new_summary = not os.path.exists(summary_csv)
    sf = open(summary_csv, "a", newline="")
    w = csv.writer(sf)
    if new_summary:
        w.writerow(["case", "label", "rate", "status", "out_dir"])

    ran = 0
    for i, t in enumerate(plan, 1):
        if args.max_tests and ran >= args.max_tests:
            B.log(f"Reached --max-tests={args.max_tests}; stopping.")
            break
        out_dir = os.path.join(root, t["case"], f"{t['label']}_r{t['rate']}")
        if result_done(out_dir, t["label"], t["rate"]):
            B.log(f"[{i}/{len(plan)}] SKIP (done) {t['label']} r{t['rate']}")
            w.writerow([t["case"], t["label"], t["rate"], "done(resumed)", out_dir])
            sf.flush()
            continue

        B.log(f"[{i}/{len(plan)}] RUN {t['case']} {t['label']} rate={t['rate']}")
        cmd = ["python3", os.path.join(HERE, "orchestrator.py"),
               "--mode", t["mode"], "--model", args.model,
               "--agg-instances", str(t["agg"]),
               "--e-instances", str(t["e"]), "--pd-instances", str(t["pd"]),
               "--num-prompts", str(args.num_prompts),
               "--image-count", str(args.image_count),
               "--image-resolution", args.image_resolution,
               "--request-rate", t["rate"],
               "--gpus", args.gpus, "--xpus", args.xpus,
               "--out-dir", out_dir]
        if args.mm_attn_backend:
            cmd += ["--mm-attn-backend", args.mm_attn_backend]
        # Capture the orchestrator's narration per test (device selection,
        # readiness waits, skip/fail reasons, bench progress) so the CSV status
        # is explainable after the fact.
        os.makedirs(out_dir, exist_ok=True)
        orch_log = os.path.join(out_dir, "orchestrator.log")
        with open(orch_log, "w") as lf:
            rc = subprocess.run(cmd, stdout=lf,
                                stderr=subprocess.STDOUT).returncode
        status = {0: "ok", 2: "skipped", 3: "bench_failed"}.get(rc, f"rc={rc}")
        B.log(f"[{i}/{len(plan)}] -> {status}  (log: {orch_log})")
        w.writerow([t["case"], t["label"], t["rate"], status, out_dir])
        sf.flush()
        ran += 1

    sf.close()
    B.log(f"Done. Summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
