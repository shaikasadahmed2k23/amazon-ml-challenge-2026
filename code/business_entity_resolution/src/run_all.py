"""Run the pipeline end to end, with checkpoints, a progress log and a final summary.

Pipeline position: driver of the CPU pipeline (stages 1-7); the GPU cross-encoders and stack.py run afterwards
(README).

Steps (each its own subprocess, so memory is released between stages):
  normalize  raw TSVs -> cache/normalized/*.parquet            (skips files already built)
  probe      blocking scale probe on test (full S1 index, 20k-record slice per country)
  blocking   train and test, both full S1 index x full S2/S3 pool   (skip: blocking/<split>/_DONE.json)
  features   pairwise features for train and test candidates         (skip: features/<split>.parquet)
  train      LightGBM, 5-fold GroupKFold by S1, OOF thresholds       (skip: model/meta.json)
  train_pw   optional, unused experiment: matcher without config.CONTEXT_FEATURES (pairwise-only)
                                                                     (skip: model/pairwise/meta.json)
  prefilter  stage-1 model on blocking signals + cheap flags: cutoff from the OOF trade-off, test keep set
                                                                     (skip: prefilter/test_keep.parquet)
  tune       distractor-weighted OOF threshold tuning + rule C check   (skip: model/threshold.json)
  predict    test -> output/matching_results.tsv + candidate_pairs.tsv + variants/ + pair probabilities
                                                                     (skip: both TSVs exist)
  models     optional, unused experiment: base vs pairwise vs ensemble on OOF (+ dense quartile), variants
                                                                     (skip: output/test_pair_probs_models.parquet)
  validate   utils/validate_submission.py on the output files and every variant (always runs)
  eval       blocking recall on the 10% held-out train S1s (diagnostic, not in the default steps)

Every checkpoint is written only when its stage completes and counts only if it is newer than the
checkpoints it was built from, so a rerun resumes after the last
finished stage; --force reruns everything. All output is also appended to --log.

With --sample N only the train side runs (normalize, blocking, features, train, eval).

Examples (from student_resource/):
    python code/business_entity_resolution/src/run_all.py --sample 50000          # local smoke test
    python code/business_entity_resolution/src/run_all.py --data-dir <dataset> --cache-dir <cache> \\
        --output-dir <output> --log <run.log>
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ("normalize", "probe", "blocking", "features", "train", "train_pw", "prefilter", "tune", "predict", "models",
         "validate", "eval")
# The submitted package (run 4). train_pw / models are an optional, unused experiment (run 5).
DEFAULT_STEPS = ("normalize", "blocking", "features", "train", "prefilter", "tune", "predict", "validate")

# Throughput measured on a 4-core laptop (Kaggle CPU sessions are similar or up to ~1.5x slower).
RATE = {
    "normalize_s_per_10k_rows": 0.67,     # full test: 11.7M rows in ~13 min
    "blocking_s_per_10k_records": 5.0,    # France full index, pre_cap 150 (4.8-6.9 measured)
    "tfidf_s_per_10k_s1": 2.7,            # S1-side TF-IDF fit + transform
    "features_s_per_10k_pairs": 0.6,      # run 2 features (France run 1: 1.43M pairs in 53 s)
    "train_s_per_1m_pairs": 360,          # 5 folds LightGBM (lr 0.1), scaled from the sample
    "probe_s": 600,
    "predict_s": 4500,                    # Kaggle run 3: 5 fold models on ~10M test pairs + 5 TSVs (75 min)
    "tune_s": 180,
    "prefilter_s": 1800,
    "models_s": 3300,                     # OOF comparison + pairwise scoring of ~7.7M test pairs (~run 4 predict)                  # stage-1 5 folds on ~5M pairs + scoring ~10M test pairs (scaled from local)
}
# Row counts from the EDA, used when the normalised parquet does not exist yet.
KNOWN_ROWS = {("train", "S1"): 2_206_821, ("train", "S2"): 5_034_616, ("train", "S3"): 5_285_603,
              ("test", "S1"): 1_732_544, ("test", "S2"): 4_887_273, ("test", "S3"): 5_082_316}


class Tee:
    """File-like object writing to stdout and, if a path is given, appending to a log file."""
    def __init__(self, path):
        """Open `path` for appending (line-buffered), or log to stdout only when path is empty."""
        self.f = open(path, "a", encoding="utf-8", buffering=1) if path else None

    def write(self, line):
        """Write `line` to stdout and the log file."""
        sys.stdout.write(line)
        sys.stdout.flush()
        if self.f:
            self.f.write(line)


def run(tee, script, *args, capture=False):
    """Run a pipeline script in a subprocess, streaming its output to `tee`.

    Returns (exit code, captured output if capture=True). A failure stops the run unless capture=True.
    """
    cmd = [sys.executable, "-u", str(script if Path(script).is_absolute() else SRC / script), *map(str, args)]
    tee.write(f"\n>>> [{time.strftime('%H:%M:%S')}] {' '.join(cmd)}\n")
    t = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    lines = []
    for line in proc.stdout:
        tee.write(line)
        if capture:
            lines.append(line)
    rc = proc.wait()
    tee.write(f"<<< {Path(script).name} exit {rc} after {time.time() - t:.0f}s\n")
    if rc != 0 and not capture:
        raise SystemExit(f"{Path(script).name} failed with exit code {rc}")
    return rc, "".join(lines)


def row_counts(config):
    """Rows per split / source (from the parquet metadata when present, else the known counts)."""
    import pyarrow.parquet as pq
    out = {}
    for (split, src), n in KNOWN_ROWS.items():
        p = config.norm_path(split, src)
        out[(split, src)] = pq.ParquetFile(p).metadata.num_rows if p.exists() else n
    return out


def estimate(config, steps, splits):
    """Estimated minutes per step from row counts and the measured throughput."""
    rows = row_counts(config)
    rec = {s: rows[(s, "S2")] + rows[(s, "S3")] for s in splits}
    est = {}
    if "normalize" in steps:
        est["normalize"] = sum(n for (s, _), n in rows.items() if s in splits) / 1e4 * RATE["normalize_s_per_10k_rows"]
    if "probe" in steps and "test" in splits:
        est["probe"] = RATE["probe_s"]
    if "blocking" in steps:
        est["blocking"] = sum(rec[s] / 1e4 * RATE["blocking_s_per_10k_records"]
                              + rows[(s, "S1")] / 1e4 * RATE["tfidf_s_per_10k_s1"] for s in splits)
    if "features" in steps:   # M=1: about one candidate pair per record
        est["features"] = sum(rec[s] for s in splits) / 1e4 * RATE["features_s_per_10k_pairs"]
    if "train" in steps:
        est["train"] = min(rec["train"], config.TRAIN_MAX_PAIRS or rec["train"]) / 1e6 * RATE["train_s_per_1m_pairs"]
    if "train_pw" in steps:
        est["train_pw"] = min(rec["train"], config.TRAIN_MAX_PAIRS or rec["train"]) / 1e6 * RATE["train_s_per_1m_pairs"]
    if "prefilter" in steps:
        est["prefilter"] = RATE["prefilter_s"]
    if "tune" in steps:
        est["tune"] = RATE["tune_s"]
    if "predict" in steps and "test" in splits:
        est["predict"] = RATE["predict_s"]
    if "models" in steps and "test" in splits:
        est["models"] = RATE["models_s"]
    return {k: v / 60 for k, v in est.items()}


TRUST_CACHE = False   # --trust-cache: copied checkpoints count as fresh (cache of another run)
RUN_START = time.time()


def fresh(out, *deps):
    """Checkpoint `out` is usable: it exists and is newer than every checkpoint it was built from.

    With --trust-cache only dependencies rebuilt during THIS run can make it stale (the timestamps
    of a copied cache are not trusted to be ordered)."""
    if not out.exists():
        return False
    if TRUST_CACHE:
        return not any(d.exists() and d.stat().st_mtime > max(RUN_START, out.stat().st_mtime) for d in deps)
    return all(d.exists() and out.stat().st_mtime >= d.stat().st_mtime for d in deps)


def find_validator(config):
    """utils/validate_submission.py: $BER_VALIDATOR, next to the repo, or next to the dataset."""
    cands = [os.environ.get("BER_VALIDATOR", ""), config.REPO_ROOT / "utils" / "validate_submission.py",
             config.DATA_DIR.parent / "utils" / "validate_submission.py"]
    return next((Path(c) for c in cands if c and Path(c).is_file()), None)


def summary(tee, config, validation):
    """Write the final summary: OOF macro F0.5, thresholds, blocking recall and the validator's verdict."""
    tee.write("\n" + "=" * 70 + "\nSUMMARY\n")
    meta_path = config.model_dir() / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        f = meta["oof_macro_f05"]
        tee.write(f"OOF macro F0.5: global threshold {meta['global']} -> {f['global']:.5f}; per-source "
                  f"{meta['per_source']} -> {f['per_source']:.5f}\n")
        tee.write(f"chosen thresholds: {meta['thresholds']}; fold AUC: "
                  f"{', '.join(f'{a:.5f}' for a in meta['fold_auc'])}\n")
    else:
        tee.write("no trained model (train step not run)\n")
    tpath = config.model_dir() / "threshold.json"
    if tpath.exists():
        t = json.loads(tpath.read_text())
        tee.write("distractor-weighted OOF: " + "; ".join(
            f"w={w}: best threshold {v['threshold']:.2f} -> {v['f05']:.5f}" for w, v in t["weights"].items())
            + f"; chosen w={t['chosen_w']} -> {t['threshold']:.2f}; rule C {'on' if t['rule_c'] else 'off'}\n")
    ppath = config.prefilter_dir() / "prefilter.json"
    if ppath.exists():
        p = json.loads(ppath.read_text())
        tee.write(f"pre-filter (OOF): p1 >= {p['cutoff']:g}; candidates per S1 {p['avg_cands_before']:.2f} -> "
                  f"{p['avg_cands_after']:.2f} (p95 {p['p95_cands_before']:.0f} -> {p['p95_cands_after']:.0f}); "
                  f"true pairs lost {p['true_lost_pct']:.3f}%, accepted pairs lost {p['accepted_lost_pct']:.3f}%; "
                  f"macro F0.5 w={p['w']:g} @ {p['threshold']:.2f}: {p['f05_before']:.5f} -> {p['f05_after']:.5f}; "
                  f"{'ON' if config.PREFILTER else 'OFF (config.PREFILTER)'}\n")
    mpath = config.model_dir() / "pairwise" / "models.json"
    if mpath.exists():
        mj = json.loads(mpath.read_text())
        for r in mj["oof"]:
            tee.write(f"model {r['model']}: OOF w=1 {r['F0.5_w1']:.5f} @ {r['thr_w1']:.2f}; w=2 {r['F0.5_w2']:.5f} @ "
                      f"{r['thr_w2']:.2f}; dense quartile w=2 {r['dense_F0.5_w2@thr_w2']:.5f} "
                      f"(best {r['dense_best_F0.5']:.5f} @ {r['dense_best_thr']:.2f})\n")
        tee.write(f"model thresholds: {mj['thresholds']}; matching_results.tsv model: {mj['final_model']}\n")
    tee.write(f"final threshold: {config.FINAL_THRESHOLD if config.FINAL_THRESHOLD is not None else 'OOF-optimal'}\n")
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        p = config.OUTPUT_DIR / name
        tee.write(f"{p}: {'%.1f MB' % (p.stat().st_size / 2 ** 20) if p.exists() else 'MISSING'}\n")
    for name, (rc, out) in (validation or {}).items():
        verdict = next((l for l in out.splitlines() if l.startswith(("PASS", "FAIL"))), f"exit code {rc}")
        tee.write(f"validator [{name}]: {verdict}\n")
        if rc != 0:
            tee.write(out[-3000:] + "\n")
    tee.write("=" * 70 + "\n")


def main():
    """Parse args, set the data / cache / output environment, run each step unless its checkpoint exists,
    validate the outputs and print the summary.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="directory containing train/ and test/ (sets BER_DATA_DIR)")
    ap.add_argument("--cache-dir", help="parquet cache directory (sets BER_CACHE_DIR)")
    ap.add_argument("--output-dir", help="submission output directory (sets BER_OUTPUT_DIR)")
    ap.add_argument("--log", help="append all progress output to this file")
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(DEFAULT_STEPS))
    ap.add_argument("--sample", type=int, default=0, help="smoke test on N train S1s (train side only)")
    ap.add_argument("--pools", nargs="+", choices=("heldout", "all"), default=["all"],
                    help="eval pools: heldout = index held-out S1s only; all = index every train S1")
    ap.add_argument("--force", action="store_true", help="ignore checkpoints and rerun every step")
    ap.add_argument("--trust-cache", action="store_true",
                    help="a checkpoint only has to exist (use after copying the cache of an earlier run)")
    args, extra = ap.parse_known_args()   # extra args (BlockingParams, e.g. --pre_cap 100) go to blocking/probe/eval

    for flag, var in (("data_dir", "BER_DATA_DIR"), ("cache_dir", "BER_CACHE_DIR"), ("output_dir", "BER_OUTPUT_DIR")):
        if getattr(args, flag):
            os.environ[var] = getattr(args, flag)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.path.insert(0, str(SRC))
    import config  # noqa: E402  (imported after the environment is set)
    global TRUST_CACHE
    TRUST_CACHE = args.trust_cache
    config.use_sample(args.sample)

    tee = Tee(args.log)
    sample = ["--sample", args.sample] if args.sample else []
    splits = ["train"] if args.sample else ["train", "test"]
    t0 = time.time()
    tee.write(f"\n##### run_all started {time.strftime('%Y-%m-%d %H:%M:%S')}: steps {args.steps}, "
              f"extra {extra}, cache {config.CACHE_DIR}, output {config.OUTPUT_DIR}\n")
    est = estimate(config, args.steps, splits)
    tee.write("estimated runtime (min, before checkpoints): "
              + ", ".join(f"{k} {v:.0f}" for k, v in est.items())
              + f"; TOTAL ~{sum(est.values()):.0f} min ({sum(est.values()) / 60:.1f} h), "
                f"up to ~{1.5 * sum(est.values()) / 60:.1f} h on a slower CPU\n")

    def skip(step, done):
        """True (and logged) when the step's checkpoint exists and --force is not set."""
        if done and not args.force:
            tee.write(f"\n--- {step}: checkpoint found, skipping (--force to rerun)\n")
            return True
        return False

    validation, failed = None, None
    try:
        for step in STEPS:
            if step not in args.steps:
                continue
            if args.sample and step in ("probe", "predict", "validate"):
                tee.write(f"\n--sample: skipping {step} (test side)\n")
            elif step == "normalize":
                run(tee, "normalize.py", *sample, *(["--force"] if args.force else []))
            elif step == "probe":
                out = config.CACHE_DIR / "probe_test.txt"
                if not skip("probe", out.exists()):
                    _, text = run(tee, "probe_scale.py", "--split", "test", *extra, capture=True)
                    out.write_text(text, encoding="utf-8")
            elif step == "blocking":
                for split in splits:
                    if not skip(f"blocking {split}", fresh(config.blocking_done(split),
                                                         *(config.norm_path(split, s) for s in config.SOURCES))):
                        run(tee, "blocking.py", "--split", split, *sample, *extra)
            elif step == "features":
                for split in splits:
                    if not skip(f"features {split}", fresh(config.features_path(split), config.blocking_done(split))):
                        run(tee, "features.py", "--split", split, *sample)
            elif step == "train":
                if not skip("train", fresh(config.model_dir() / "meta.json", config.features_path("train"))):
                    run(tee, "train_matcher.py", *sample)
            elif step == "train_pw":
                if not skip("train_pw", fresh(config.model_dir() / "pairwise" / "meta.json",
                                              config.features_path("train"))):
                    run(tee, "train_matcher.py", *sample, "--drop_features", *config.CONTEXT_FEATURES,
                        "--out_subdir", "pairwise")
            elif step == "prefilter":
                if not config.PREFILTER:
                    tee.write("\n--- prefilter: off (config.PREFILTER = False)\n")
                    continue
                out = config.prefilter_keep_path("train" if args.sample else "test")
                deps = [config.model_dir() / "meta.json"] + [config.features_path(s) for s in splits]
                if not skip("prefilter", fresh(out, *deps)):
                    run(tee, "prefilter.py", *sample)
            elif step == "tune":
                deps = [config.model_dir() / "meta.json"]
                if config.PREFILTER:
                    deps.append(config.prefilter_dir() / "prefilter.json")
                if not skip("tune", fresh(config.model_dir() / "threshold.json", *deps)):
                    run(tee, "tune_threshold.py", *sample)
            elif step == "predict":
                deps = [config.features_path("test"), config.model_dir() / "meta.json"]
                if (config.model_dir() / "threshold.json").exists():
                    deps.append(config.model_dir() / "threshold.json")
                if config.PREFILTER:
                    deps.append(config.prefilter_keep_path("test"))
                done = all(fresh(config.OUTPUT_DIR / f, *deps) for f in ("matching_results.tsv", "candidate_pairs.tsv"))
                if not skip("predict", done):
                    run(tee, "predict.py")
            elif step == "models":
                if not (config.model_dir() / "pairwise" / "meta.json").exists():
                    tee.write("\n--- models: no pairwise model (train_pw not run), skipping\n")
                    continue
                deps = [config.model_dir() / "meta.json", config.model_dir() / "pairwise" / "meta.json"]
                if config.PREFILTER:
                    deps.append(config.prefilter_keep_path("train" if args.sample else "test"))
                out = (config.model_dir() / "pairwise" / "models.json" if args.sample
                       else config.OUTPUT_DIR / "test_pair_probs_models.parquet")
                if not skip("models", fresh(out, *deps)):
                    run(tee, "models.py", *sample)
            elif step == "validate":
                validator = find_validator(config)
                if validator is None:
                    tee.write("\nvalidate_submission.py not found (set BER_VALIDATOR): skipping validation\n")
                    continue
                validation = {}
                files = [config.OUTPUT_DIR / "matching_results.tsv"] + sorted(
                    (config.OUTPUT_DIR / "variants").glob("matching_results_*.tsv"))
                for f in files:
                    validation[f.name] = run(tee, validator, "--matching", f,
                                             "--candidate", config.OUTPUT_DIR / "candidate_pairs.tsv",
                                             "--test-dir", config.DATA_DIR / "test", capture=True)
            elif step == "eval":
                for pool in args.pools:
                    run(tee, "eval_blocking.py", "--pool", pool, *sample, *extra)
    except SystemExit as e:          # a stage failed: still print the summary, then exit non-zero
        failed = str(e)
        tee.write(f"\n!!! {failed}\n")
    tee.write(f"\nsteps finished in {(time.time() - t0) / 60:.1f} min{' (FAILED)' if failed else ''}\n")
    summary(tee, config, validation)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
