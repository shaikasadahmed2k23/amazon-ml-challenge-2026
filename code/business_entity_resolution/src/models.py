"""Compare the base matcher, the pairwise-only matcher and their average; write their test submissions.

Pipeline position: optional comparison after stage 5 (base vs pairwise vs ensemble); not part of the final pipeline.

Models (all score exactly the same candidate pairs, the pre-filter keep set, so candidate_pairs.tsv
is shared):
  base       model_dir()/            every feature (train_matcher.py)
  pairwise   model_dir()/pairwise/   without the features whose distribution depends on the number of
                                     candidates around an S1 or a record (config.CONTEXT_FEATURES)
  ensemble   mean of the two probabilities

OOF (train universe, pre-filtered pairs count as rejected): macro F0.5 at w = 1 and w = 2 (kept negatives
count w times, see tune_threshold.py), best threshold per model, and the same on the S1s in the top
quartile of candidates per S1 (a proxy for test's denser neighbourhoods).

Test: base probabilities from predict.py's cache (scored if missing), pairwise scored here, then the
variant submissions output/variants/matching_results_{model}_t{thr}.tsv:
  pairwise @ its w = 2 OOF-best threshold, ensemble @ its w = 2 OOF-best, and both @ best - LOWER_BY.
Thresholds: config.MODEL_THRESHOLDS[model] when set, else the OOF-best at w = 2. If config.FINAL_MODEL is
not "base", output/matching_results.tsv is rewritten with that model at its threshold.
Also exports output/test_pair_probs_models.parquet and output/oof_pair_probs_models.parquet.

Usage (from student_resource/):
    python code/business_entity_resolution/src/models.py [--sample N] [--no_test]
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
from predict import all_s1_ids, summarize, write_lists
from train_matcher import load_universe, many_to_one
from tune_threshold import best_threshold, per_s1_f05_w

MODELS = ("base", "pairwise", "ensemble")
LOWER_BY = 0.15


def model_path(name):
    """Model folder of 'base' (the matcher) or 'pairwise' (the pairwise-only matcher)."""
    return config.model_dir() if name == "base" else config.model_dir() / "pairwise"


def score(pairs, name):
    """Average probability of a model's fold boosters over `pairs` (its meta.json feature list)."""
    meta = json.loads((model_path(name) / "meta.json").read_text())
    X = pairs.select(meta["features"]).to_numpy()
    prob = np.zeros(len(pairs))
    for k in range(meta["n_folds"]):
        prob += lgb.Booster(model_file=str(model_path(name) / f"fold{k}.txt")).predict(X) / meta["n_folds"]
    return prob


def evaluate(kept, universe, dense):
    """Per model (its own many-to-one frame): best thresholds at w = 1 / 2, all S1s and dense quartile."""
    n_true = universe["n_true"].to_numpy().astype(float)
    rows, best = [], {}
    for m in MODELS:
        df = kept[m]
        code, label, p = df["code"].to_numpy(), df["label"].to_numpy().astype(float), df["prob"].to_numpy()
        t1, f1, _ = best_threshold(code, p, label, n_true, 1.0)
        t2, f2, _ = best_threshold(code, p, label, n_true, 2.0)
        best[m] = t2
        fw2 = per_s1_f05_w(code, p >= t2, label, n_true, 2.0)
        fw2_low = per_s1_f05_w(code, p >= t2 - LOWER_BY, label, n_true, 2.0)
        fd = [per_s1_f05_w(code, p >= t, label, n_true, 2.0)[dense].mean() for t in np.arange(0.05, 0.99, 0.01)]
        rows.append({"model": m, "thr_w1": t1, "F0.5_w1": f1, "thr_w2": t2, "F0.5_w2": f2,
                     f"F0.5_w2@thr-{LOWER_BY}": fw2_low.mean(),
                     "dense_F0.5_w2@thr_w2": fw2[dense].mean(),
                     "dense_best_thr": round(0.05 + 0.01 * int(np.argmax(fd)), 2), "dense_best_F0.5": max(fd)})
    return pl.DataFrame(rows), best


def oof_part(args):
    """Compare base / pairwise / ensemble on the OOF pairs (pre-filter applied, many-to-one per model).

    Returns: (table of macro F0.5 per model and density slice, best threshold per model at w = 2).
    """
    base = pl.read_parquet(config.model_dir() / "oof.parquet").select("s1_id", "cand_id", "label", "source",
                                                                      p_base="prob")
    pw = pl.read_parquet(model_path("pairwise") / "oof.parquet").select("s1_id", "cand_id", p_pairwise="prob")
    n_pairs = pl.scan_parquet(config.features_path("train")).select(pl.len()).collect().item()
    universe = load_universe(n_pairs, config.TRAIN_MAX_PAIRS)
    df = base.join(pw, on=["s1_id", "cand_id"]).join(universe.select("s1_id", "code"), on="s1_id")
    assert len(df) == len(base), "base and pairwise OOF cover different pairs (different universes?)"
    # candidates per S1 over the full blocking list, before any pruning (= s1_n_cands)
    n_c = df.group_by("code").agg(n_cands=pl.len())
    universe = universe.join(n_c, on="code", how="left").with_columns(pl.col("n_cands").fill_null(0)).sort("code")
    q75 = float(universe["n_cands"].quantile(0.75))
    dense = (universe["n_cands"] >= q75).to_numpy()
    df = df.with_columns(p_ensemble=(pl.col("p_base") + pl.col("p_pairwise")) / 2)
    kp = config.prefilter_keep_path("train")
    if config.PREFILTER and kp.exists():
        df = df.join(pl.read_parquet(kp).with_columns(pf=pl.lit(True)), on=["s1_id", "cand_id"], how="left")
        df = df.with_columns(pl.col("pf").fill_null(False))
    else:
        df = df.with_columns(pf=pl.lit(True))
    # pruned pairs are never scored (prob 0); each model does its own many-to-one
    df = df.with_columns(pl.when("pf").then(pl.col(f"p_{m}")).otherwise(0.0).alias(f"p_{m}") for m in MODELS)
    kept = {m: many_to_one(df.with_columns(prob=pl.col(f"p_{m}"))).select("code", "label", "prob")
            for m in MODELS}
    print(f"OOF: {len(df):,} pairs over {len(universe):,} S1s; dense quartile = S1s with >= {q75:.0f} "
          f"candidates ({int(dense.sum()):,} S1s, {universe.filter(pl.Series(dense))['n_cands'].mean():.2f} "
          f"candidates/S1 vs {universe['n_cands'].mean():.2f} overall)")
    table, best = evaluate(kept, universe, dense)
    pl.Config.set_float_precision(5)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(12)
    print("\n=== OOF macro F0.5 by model (w = kept-negative weight; dense = top quartile of candidates per S1) ===")
    print(table)
    export = config.CACHE_DIR / "oof_export" if args.sample else config.OUTPUT_DIR
    export.mkdir(parents=True, exist_ok=True)
    df.select("s1_id", rec_id="cand_id", label="label", p_base="p_base", p_pairwise="p_pairwise",
              p_ensemble="p_ensemble", prefilter_kept="pf").write_parquet(export / "oof_pair_probs_models.parquet", compression="zstd")
    universe.select("s1_id", "country", "n_true", "n_cands").write_parquet(export / "oof_universe_models.parquet")
    return table, best


def thresholds(best):
    """Threshold per model: config.MODEL_THRESHOLDS where pinned, else the OOF best in `best`."""
    return {m: (config.MODEL_THRESHOLDS.get(m) if config.MODEL_THRESHOLDS.get(m) is not None else best[m])
            for m in MODELS}


def test_part(thr):
    """Score the test candidates with each model (cached probabilities reused) and write their submissions."""
    keep = config.prefilter_keep_path("test")
    use_pf = config.PREFILTER and keep.exists()
    pairs = pl.scan_parquet(config.features_path("test"))
    if use_pf:
        pairs = pairs.join(pl.scan_parquet(keep), on=["s1_id", "cand_id"], how="semi")
    pairs = pairs.collect()
    print(f"\ntest: {len(pairs):,} candidate pairs" + (" (pre-filter keep set)" if use_pf else ""))
    cache = config.CACHE_DIR / "predict" / f"test_probs{'_pf' if use_pf else ''}.parquet"
    probs = pairs.select("s1_id", "cand_id", "source")
    if cache.exists():
        base = pl.read_parquet(cache, columns=["s1_id", "cand_id", "prob"]).rename({"prob": "p_base"})
        probs = probs.join(base, on=["s1_id", "cand_id"], how="left")
        assert probs["p_base"].null_count() == 0, f"{cache} does not cover the candidate pairs"
        print(f"base probabilities: cached {cache}")
    else:
        t = time.time()
        probs = probs.with_columns(p_base=pl.Series(score(pairs, "base")))
        print(f"base probabilities scored in {time.time() - t:.0f}s")
    pw_cache = config.CACHE_DIR / "predict" / f"test_probs_pairwise{'_pf' if use_pf else ''}.parquet"
    pw_meta = model_path("pairwise") / "meta.json"
    if pw_cache.exists() and pw_cache.stat().st_mtime >= pw_meta.stat().st_mtime:
        pw = pl.read_parquet(pw_cache)
        probs = probs.join(pw, on=["s1_id", "cand_id"], how="left")
        print(f"pairwise probabilities: cached {pw_cache}")
    else:
        t = time.time()
        probs = probs.with_columns(p_pairwise=pl.Series(score(pairs, "pairwise")))
        pw_cache.parent.mkdir(parents=True, exist_ok=True)
        probs.select("s1_id", "cand_id", "p_pairwise").write_parquet(pw_cache, compression="zstd")
        print(f"pairwise probabilities scored in {time.time() - t:.0f}s, peak mem {peak_mem_gb():.2f} GB")
    del pairs
    probs = probs.with_columns(p_ensemble=(pl.col("p_base") + pl.col("p_pairwise")) / 2)
    out = config.OUTPUT_DIR
    probs.select("s1_id", rec_id="cand_id", p_base="p_base", p_pairwise="p_pairwise",
                 p_ensemble="p_ensemble").write_parquet(out / "test_pair_probs_models.parquet", compression="zstd")

    s1_ids = all_s1_ids("test")
    s1_country = pl.read_parquet(config.norm_path("test", "S1"), columns=["entity_id", "country"]).rename(
        {"entity_id": "s1_id"})
    jobs = [("pairwise", thr["pairwise"]), ("ensemble", thr["ensemble"]),
            ("pairwise", round(thr["pairwise"] - LOWER_BY, 2)), ("ensemble", round(thr["ensemble"] - LOWER_BY, 2))]
    for m, t in jobs:
        k = many_to_one(probs.with_columns(prob=pl.col(f"p_{m}")))
        sel = k.filter(pl.col("prob") >= t)
        res = write_lists(s1_ids, sel, "matched_entity_ids", out / "variants" / f"matching_results_{m}_t{t:.2f}.tsv")
        summarize(res, s1_country, f"variant {m} t={t:.2f}: {len(sel):,} matches")
    if config.FINAL_MODEL != "base":
        m, t = config.FINAL_MODEL, thr[config.FINAL_MODEL]
        k = many_to_one(probs.with_columns(prob=pl.col(f"p_{m}")))
        sel = k.filter(pl.col("prob") >= t)
        res = write_lists(s1_ids, sel, "matched_entity_ids", out / "matching_results.tsv")
        summarize(res, s1_country, f"matching_results.tsv rewritten: {m} t={t:.2f} (config.FINAL_MODEL), "
                                   f"{len(sel):,} matches")


def main():
    """Run the OOF comparison, pick thresholds, write models.json and (full data only) the test submissions."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--no_test", action="store_true")
    args = ap.parse_args()
    config.use_sample(args.sample)
    t0 = time.time()
    table, best = oof_part(args)
    thr = thresholds(best)
    print(f"thresholds used: {thr} (config.MODEL_THRESHOLDS, else OOF-best at w=2); "
          f"final model: {config.FINAL_MODEL}")
    summary = {"oof": table.to_dicts(), "thresholds": thr, "final_model": config.FINAL_MODEL}
    (model_path("pairwise") / "models.json").write_text(json.dumps(summary, indent=2))
    if not args.no_test and not args.sample:
        test_part(thr)
    print(f"-> {model_path('pairwise') / 'models.json'}; {time.time() - t0:.0f}s, peak mem {peak_mem_gb():.2f} GB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
