"""Score test candidates with the trained matcher and write the submission files.

Pipeline position: stage 7 (run-4 submission: LightGBM probabilities on the pre-filter keep set); its
test_probs_pf.parquet is p_lgbm for stack.py.

    output/candidate_pairs.tsv      every candidate the matcher scored, grouped by S1: the blocking
                                    pairs kept by the pre-filter (prefilter.py; all of them when
                                    config.PREFILTER is False)
    output/matching_results.tsv     candidates kept by the matcher (many-to-one + threshold)
    output/test_pair_probs.parquet  s1_id, rec_id, prob for every candidate pair (<split>_pair_probs)
    output/variants/matching_results_t{thr}.tsv   one submission per variant threshold

Every S1 of test_source1.tsv gets a row in every file (empty list when nothing is kept),
whatever its country.

Threshold of matching_results.tsv, first that applies:
    --threshold T  >  config.FINAL_THRESHOLD  >  OOF-optimal threshold(s) in model/meta.json
Variant thresholds: --variants, else the list chosen by tune_threshold.py (model/threshold.json).
Rule C (empty address + exact unique name, see tune_threshold.py) is applied when threshold.json
enables it. Pair probabilities are cached (cache/predict/<split>_probs.parquet), so re-running
with other thresholds only re-writes the TSVs.

Usage (from student_resource/):
    python code/business_entity_resolution/src/predict.py [--threshold 0.80] [--variants 0.7 0.8]
"""

import argparse
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config
from memory import peak_mem_gb
from train_matcher import keep_mask, many_to_one
from tune_threshold import rule_c_mask


def all_s1_ids(split):
    """Every S1 id of a split, read from the raw TSV (the output must list all of them)."""
    return (pl.scan_csv(config.source_path(split, "S1"), separator="\t", quote_char=None, infer_schema=False)
              .select(s1_id="entity_id").collect())


def write_lists(s1_ids, pairs, col, path):
    """Write one row per S1 (all of `s1_ids`) with its comma-joined cand_ids in column `col`.

    S1s without pairs get an empty list. Written to a .tmp file and renamed, so an existing file is
    always complete. Returns the written frame (s1_id, col).
    """
    lists = pairs.group_by("s1_id").agg(pl.col("cand_id").sort().str.join(",").alias(col))
    out = s1_ids.join(lists, on="s1_id", how="left").with_columns(pl.col(col).fill_null(""))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")        # renamed when complete: the final file is a checkpoint
    out.rename({"s1_id": "source1_entity_id"}).write_csv(tmp, separator="\t", quote_style="never")
    tmp.replace(path)
    return out


def pair_probs(split, meta):
    """Average fold-model probability for every candidate pair (cached next to the features).

    With the pre-filter on, only the pairs in its keep file are loaded and scored: that set IS the
    candidate set. Its probabilities are cached under a separate name (<split>_probs_pf.parquet), so
    the cache of a run without the pre-filter is never mistaken for it."""
    keep = config.prefilter_keep_path(split)
    use_pf = config.PREFILTER and keep.exists()
    if config.PREFILTER and not use_pf:
        print(f"!!! pre-filter on but {keep} missing: scoring every blocking pair")
    cache = config.CACHE_DIR / "predict" / f"{split}_probs{'_pf' if use_pf else ''}.parquet"
    feat_path, meta_path = config.features_path(split), config.model_dir() / "meta.json"
    deps = (feat_path, meta_path) + ((keep,) if use_pf else ())
    if cache.exists() and all(cache.stat().st_mtime >= p.stat().st_mtime for p in deps):
        print(f"pair probabilities: cached {cache}")
        return pl.read_parquet(cache)
    feats = meta["features"]
    pairs = pl.scan_parquet(feat_path)
    if use_pf:
        pairs = pairs.join(pl.scan_parquet(keep), on=["s1_id", "cand_id"], how="semi")
    pairs = pairs.collect()
    print(f"scoring {len(pairs):,} candidate pairs" + (f" (pre-filter keep set {keep.name})" if use_pf else ""))
    missing = [f for f in feats if f not in pairs.columns]
    assert not missing, f"test features lack {missing}: block test with the same top_m as train"
    X = pairs.select(feats).to_numpy()
    prob = np.zeros(len(pairs))
    for k in range(meta["n_folds"]):
        prob += lgb.Booster(model_file=str(config.model_dir() / f"fold{k}.txt")).predict(X) / meta["n_folds"]
    del X
    out = pairs.select("s1_id", "cand_id", "country", "source", "a_empty_rec", "n_exact", "legal_conflict") \
               .with_columns(prob=pl.Series(prob))
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(cache, compression="zstd")
    return out


def summarize(res, s1_country, label):
    """Print, per country, the number of S1s, % with an empty match list and average matches per S1."""
    s = (res.join(s1_country, on="s1_id", how="left")
            .with_columns(n=pl.when(pl.col("matched_entity_ids") == "").then(0)
                            .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1))
            .group_by("country").agg(S1=pl.len(), empty_pct=(pl.col("n") == 0).mean() * 100,
                                     avg_matches=pl.col("n").mean()).sort("country"))
    print(f"\n{label}")
    print(s)


def main():
    """Score the test candidates with the matcher, apply many-to-one and the threshold, and write
    matching_results.tsv, candidate_pairs.tsv, variants and the pair probabilities.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=config.SPLITS, default="test")
    ap.add_argument("--sample", type=int, default=0, help="use the normalize.py --sample N cache")
    ap.add_argument("--threshold", type=float, help="global threshold for matching_results.tsv")
    ap.add_argument("--variants", type=float, nargs="*", help="extra thresholds written to output/variants/")
    args = ap.parse_args()
    config.use_sample(args.sample)
    t0 = time.time()

    meta = json.loads((config.model_dir() / "meta.json").read_text())
    tpath = config.model_dir() / "threshold.json"
    tuned = json.loads(tpath.read_text()) if tpath.exists() else {}
    if args.threshold is not None:
        thr, why = {"global": args.threshold}, "--threshold"
    elif config.FINAL_THRESHOLD is not None:
        thr, why = {"global": config.FINAL_THRESHOLD}, "config.FINAL_THRESHOLD"
    else:
        thr, why = meta["thresholds"], "OOF-optimal (model/meta.json)"
    variants = args.variants if args.variants is not None else (
        tuned.get("variants", []) if config.BASE_VARIANTS else [])
    use_rc = bool(tuned.get("rule_c", False))

    pairs = pair_probs(args.split, meta)
    kept = many_to_one(pairs)
    prob, source = kept["prob"].to_numpy(), kept["source"].to_numpy()
    forced = rule_c_mask(kept, args.split) if use_rc else np.zeros(len(kept), bool)

    s1_ids = all_s1_ids(args.split)
    s1_country = pl.read_parquet(config.norm_path(args.split, "S1"), columns=["entity_id", "country"]).rename(
        {"entity_id": "s1_id"})
    out = config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    cand = write_lists(s1_ids, pairs, "candidate_entity_ids", out / "candidate_pairs.tsv")
    pairs.select("s1_id", rec_id="cand_id", prob="prob").write_parquet(
        out / f"{args.split}_pair_probs.parquet", compression="zstd")

    for t in sorted(set(variants)):
        m = kept.filter(pl.Series(keep_mask(prob, source, {"global": t}) | forced))
        res = write_lists(s1_ids, m, "matched_entity_ids", out / "variants" / f"matching_results_t{t:.2f}.tsv")
        summarize(res, s1_country, f"variant t={t:.2f}: {len(m):,} matches")

    matches = kept.filter(pl.Series(keep_mask(prob, source, thr) | forced))
    res = write_lists(s1_ids, matches, "matched_entity_ids", out / "matching_results.tsv")
    summarize(res, s1_country, f"matching_results.tsv: threshold {thr} ({why}), rule C {'on' if use_rc else 'off'}; "
                               f"{len(matches):,} matches of {len(pairs):,} candidate pairs")
    print(f"-> {out}; {len(cand):,} rows per file; runtime {time.time() - t0:.0f}s, peak memory {peak_mem_gb():.2f} GB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
