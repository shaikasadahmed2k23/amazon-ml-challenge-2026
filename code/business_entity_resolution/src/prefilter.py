"""Candidate pre-filter: the last candidate-generation stage, run BEFORE the matcher scores any pair.

Pipeline position: stage 4 (last candidate-generation step, before the matcher); its test keep set IS the submitted
candidate set.

Why: test has ~5.7 candidates per S1 while the matcher keeps ~3.3, so ~40% of the candidate
set is rejected anyway. A smaller candidate set is ranked higher by the organisers, and the rules
say candidate_pairs.tsv must be exactly the set the matcher runs inference on. So pruning cannot
be "drop what the matcher rejected": it must happen before the matcher, on cheap signals only.

Stage 1 = a small LightGBM (19 inputs, vs 50 for the matcher) on
  - the blocking outputs (cache/blocking/<split>/pairs_*.parquet):
        score, name_sim, ph_sim, addr_sim, key_score, n_keys, rec_gap, rec_n, source (S2/S3)
  - cheap equality flags from the features stage:
        house_match, hn_first_eq, hn_conflict, hn_first_dist, hn_s1_missing, hn_rec_missing,
        postal_match, legal_conflict, n_exact, hs_match
No fuzzy string score, no IDF-word feature and no S1-context feature (s1_n_cands, s1_rank, s1_gap:
their scale follows the candidates-per-S1 density, which is ~2x higher on test than on train).
Trained with 5-fold GroupKFold by S1 on the same group-sampled train universe as the matcher,
which gives out-of-fold stage-1 probabilities (p1).

Cutoff: a pair stays a candidate when p1 >= cutoff. For every cutoff on a grid (and, for
reference, simple rules on the blocking score / rank), the OOF trade-off is reported: average /
p95 candidates per S1, % of true pairs lost, % of the pairs the matcher ACCEPTS (OOF prob >=
the final threshold) that are lost, and the distractor-weighted macro F0.5 (w = 2) at the final
threshold with pruned pairs counted as rejected. The chosen cutoff is the largest one that
costs at most --max_loss macro F0.5 against no pruning (or config.PREFILTER_CUTOFF / --cutoff when
pinned, so the package reproduces a submitted candidate file). The matcher is NOT retrained: it scores
the surviving pairs with the features computed on the full blocking list, so the pruned OOF
score above is exactly what the pipeline does.

Writes:
    model/prefilter/fold{k}.txt, model/prefilter/prefilter.json   (features, cutoff, trade-off table)
    prefilter/<split>_keep.parquet   s1_id, cand_id of the surviving pairs (train universe and test)
    <out>/oof_prefilter.parquet      s1_id, rec_id, p1 for every OOF pair
    <out>/test_prefilter.parquet     s1_id, rec_id, p1, kept for every test blocking pair
(<out> = output/, or cache/oof_export in sample mode). If the test probabilities of an earlier
run without the pre-filter are cached (cache/predict/test_probs.parquet), the test pairs the
earlier matcher accepted and the pre-filter would drop are counted (diagnostic only; nothing is
scored).

Usage (from student_resource/):
    python code/business_entity_resolution/src/prefilter.py [--sample N] [--max_loss 0.0001]
    python code/business_entity_resolution/src/prefilter.py --apply_only     (re-apply to test)
"""

import argparse
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

import config
from features import load_truth_pairs
from memory import peak_mem_gb
from train_matcher import load_universe, many_to_one
from tune_threshold import best_threshold, per_s1_f05_w

N_FOLDS = 5
# Pairwise and per-record signals only. The S1-context features (s1_n_cands, s1_rank, s1_gap) are left
# out on purpose: they scale with the number of candidates per S1, which is ~2x higher on test than on
# train, and they bought only ~4% fewer candidates on the sample (DECISIONS.md, run 4).
BLOCK_COLS = ["score", "name_sim", "ph_sim", "addr_sim", "key_score", "n_keys", "rec_gap", "rec_n"]
FLAG_COLS = ["house_match", "hn_first_eq", "hn_conflict", "hn_first_dist", "hn_s1_missing", "hn_rec_missing",
             "postal_match", "legal_conflict", "n_exact", "hs_match"]
STAGE1 = BLOCK_COLS + FLAG_COLS + ["source"]
LGB_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  seed=config.SEED, verbose=-1, num_threads=0)
ROUNDS = 300
CUTOFFS = np.round(np.concatenate([np.geomspace(1e-4, 1e-2, 11), np.arange(0.015, 0.1, 0.005),
                                   np.arange(0.1, 0.31, 0.02)]), 5)
DEFAULT_W = 2.0
DEFAULT_THRESHOLD = 0.79
MAX_LOSS = 0.0001      # knee of the OOF trade-off curve (see DECISIONS.md, run 4)


def stage1_pairs(split, s1_filter=None):
    """Blocking pairs of a split with the stage-1 inputs (blocking columns + the cheap flags of the
    feature file) and, for the reference rules only, the S1 rank of the blocking score."""
    files = sorted(config.blocking_dir(split).glob("pairs_*.parquet"))
    assert files, f"no blocking pairs in {config.blocking_dir(split)}"
    lf = pl.concat([pl.scan_parquet(f).select(["s1_id", "cand_id", "country"] + BLOCK_COLS) for f in files])
    if s1_filter is not None:
        lf = lf.join(s1_filter.lazy(), on="s1_id", how="semi")
    flags = pl.scan_parquet(config.features_path(split)).select(["s1_id", "cand_id"] + FLAG_COLS)
    lf = lf.join(flags, on=["s1_id", "cand_id"], how="left")
    return (lf.with_columns(
        pl.col(FLAG_COLS).cast(pl.Float32),
        s1_n_cands=pl.len().over("s1_id").cast(pl.Float32),
        s1_rank=pl.col("score").rank("ordinal", descending=True).over("s1_id").cast(pl.Float32),
        s1_top1=pl.col("score").max().over("s1_id"),
        s1_top2=pl.col("score").sort(descending=True).get(1, null_on_oob=True).over("s1_id"),
        source=(pl.col("cand_id").str.starts_with("S3-")).cast(pl.Float32),
    ).with_columns(
        s1_gap=(pl.col("score") - pl.when(pl.col("s1_rank") == 1).then(pl.col("s1_top2").fill_null(0))
                                    .otherwise(pl.col("s1_top1"))).cast(pl.Float32),
    ).drop("s1_top1", "s1_top2").collect())


def predict_p1(pairs, model_dir=None, features=None):
    """Average stage-1 fold-model probability (models in model_dir, default config.prefilter_dir())."""
    model_dir = model_dir or config.prefilter_dir()
    X = pairs.select(features or STAGE1).to_numpy().astype(np.float32)
    p = np.zeros(len(pairs))
    for k in range(N_FOLDS):
        p += lgb.Booster(model_file=str(model_dir / f"fold{k}.txt")).predict(X) / N_FOLDS
    return p


def cands_stats(code, keep, n_s1):
    """Mean and 95th percentile of kept candidates per S1."""
    n = np.bincount(code[keep], minlength=n_s1)
    return n.mean(), float(np.percentile(n, 95))


def tradeoff(code, prob, label, p1, blk, n_true, thr, w, extra_cutoffs=()):
    """One row per pruning rule: candidate counts, losses and macro F0.5 after pruning."""
    n_s1 = len(n_true)
    acc = prob >= thr
    base = per_s1_f05_w(code, acc, label, n_true, w).mean()
    rules = [("none", np.ones(len(prob), bool))]
    rules += [(f"p1 >= {c:g}", p1 >= c) for c in sorted(set(CUTOFFS) | set(extra_cutoffs))]
    rules += [(f"score >= {c:g}", blk["score"] >= c) for c in (0.2, 0.3, 0.4, 0.5)]
    rules += [(f"s1_rank <= {k}", blk["s1_rank"] <= k) for k in (3, 5, 8, 10)]
    rows = []
    for name, keep in rules:
        avg, p95 = cands_stats(code, keep, n_s1)
        f = per_s1_f05_w(code, acc & keep, label, n_true, w).mean()
        rows.append({"rule": name, "avg_cands": avg, "p95_cands": p95, "pairs_kept_%": 100 * keep.mean(),
                     "true_lost_%": 100 * (~keep & (label > 0)).sum() / max((label > 0).sum(), 1),
                     "accepted_lost_%": 100 * (~keep & acc).sum() / max(acc.sum(), 1),
                     f"F0.5_w{w:g}@{thr:.2f}": f, "delta": f - base})
    return pl.DataFrame(rows), base


def fit(args):
    """Train the stage-1 model (5-fold GroupKFold by S1, cheap blocking features), choose the p1 cutoff
    with at most --max_loss macro F0.5 loss on OOF, save models + prefilter.json and the train keep set.
    """
    t0 = time.time()
    n_pairs = pl.scan_parquet(config.features_path("train")).select(pl.len()).collect().item()
    universe = load_universe(n_pairs, config.TRAIN_MAX_PAIRS)          # same S1s as the matcher
    pairs = stage1_pairs("train", universe.select("s1_id"))
    truth = load_truth_pairs().with_columns(label=pl.lit(1, pl.Int8))
    pairs = (pairs.join(truth, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0))
                  .join(universe.select("s1_id", "code"), on="s1_id"))
    print(f"stage 1: {len(pairs):,} train blocking pairs over {len(universe):,} S1s, "
          f"positives {int(pairs['label'].sum()):,}")

    X = pairs.select(STAGE1).to_numpy().astype(np.float32)
    y = pairs["label"].to_numpy()
    p1 = np.zeros(len(pairs))
    config.prefilter_dir().mkdir(parents=True, exist_ok=True)
    folds = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.SEED)
    for k, (tr, va) in enumerate(folds.split(X, y, pairs["code"].to_numpy())):
        model = lgb.train(LGB_PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=STAGE1), ROUNDS)
        p1[va] = model.predict(X[va])
        model.save_model(str(config.prefilter_dir() / f"fold{k}.txt"))
    del X
    pairs = pairs.with_columns(p1=pl.Series(p1))
    print(f"stage-1 folds trained in {time.time() - t0:.0f}s, peak mem {peak_mem_gb():.2f} GB")

    # join the matcher's OOF probabilities (same universe) and evaluate every pruning rule
    oof = many_to_one(pl.read_parquet(config.model_dir() / "oof.parquet").drop("code"))
    d = (oof.join(universe.select("s1_id", "code"), on="s1_id")
            .join(pairs.select("s1_id", "cand_id", "p1", "score", "s1_rank"), on=["s1_id", "cand_id"], how="left"))
    assert d["p1"].null_count() == 0, "matcher OOF pairs missing from the train blocking pairs"
    code, prob, label = d["code"].to_numpy(), d["prob"].to_numpy(), d["label"].to_numpy().astype(float)
    n_true = universe["n_true"].to_numpy().astype(float)
    table, base = tradeoff(code, prob, label, d["p1"].to_numpy(),
                           {"score": d["score"].to_numpy(), "s1_rank": d["s1_rank"].to_numpy()},
                           n_true, args.threshold, args.w, [args.cutoff] if args.cutoff is not None else [])
    pl.Config.set_tbl_rows(80)
    pl.Config.set_float_precision(5)
    pl.Config.set_tbl_width_chars(220)
    print(f"\n=== Pre-filter trade-off on OOF ({len(universe):,} S1s; macro F0.5 at w={args.w:g}, "
          f"threshold {args.threshold:.2f}, pruned pairs = rejected) ===")
    print(table)

    if args.cutoff is not None:                        # pinned (config.PREFILTER_CUTOFF or --cutoff)
        best = table.filter(pl.col("rule") == f"p1 >= {args.cutoff:g}").row(0, named=True)
        why = "pinned"
    else:
        ok = table.filter(pl.col("rule").str.starts_with("p1 >= ") & (pl.col("delta") >= -args.max_loss))
        best = ok.sort("avg_cands").row(0, named=True)
        why = f"largest with loss <= {args.max_loss}"
    cutoff = float(best["rule"].split(">= ")[1])
    keep = d["p1"].to_numpy() >= cutoff
    thr_after, f_after, _ = best_threshold(code, prob * keep, label, n_true, args.w)
    thr_before, f_before, _ = best_threshold(code, prob, label, n_true, args.w)
    print(f"\nchosen cutoff p1 >= {cutoff:g} ({why}): avg candidates "
          f"{table['avg_cands'][0]:.2f} -> {best['avg_cands']:.2f} per S1, p95 {table['p95_cands'][0]:.0f} -> "
          f"{best['p95_cands']:.0f}; true pairs lost {best['true_lost_%']:.3f}%, accepted pairs lost "
          f"{best['accepted_lost_%']:.3f}%; macro F0.5 w={args.w:g} @ {args.threshold:.2f}: {base:.5f} -> "
          f"{best[f'F0.5_w{args.w:g}@{args.threshold:.2f}']:.5f}")
    print(f"best w={args.w:g} threshold: without pre-filter {thr_before:.2f} -> {f_before:.5f}; "
          f"with {thr_after:.2f} -> {f_after:.5f}")

    meta = {"features": STAGE1, "n_folds": N_FOLDS, "rounds": ROUNDS, "lgb_params": LGB_PARAMS,
            "cutoff": cutoff, "max_loss": args.max_loss, "w": args.w, "threshold": args.threshold,
            "f05_before": base, "f05_after": best[f"F0.5_w{args.w:g}@{args.threshold:.2f}"],
            "avg_cands_before": table["avg_cands"][0], "avg_cands_after": best["avg_cands"],
            "p95_cands_before": table["p95_cands"][0], "p95_cands_after": best["p95_cands"],
            "true_lost_pct": best["true_lost_%"], "accepted_lost_pct": best["accepted_lost_%"],
            "best_threshold_after": thr_after, "best_f05_after": f_after, "table": table.to_dicts()}
    (config.prefilter_dir() / "prefilter.json").write_text(json.dumps(meta, indent=2))

    out = export_dir(args)
    pairs.select("s1_id", rec_id="cand_id", p1="p1").write_parquet(out / "oof_prefilter.parquet", compression="zstd")
    kp = config.prefilter_keep_path("train")
    kp.parent.mkdir(parents=True, exist_ok=True)
    pairs.filter(pl.col("p1") >= cutoff).select("s1_id", "cand_id").write_parquet(kp, compression="zstd")
    return meta


def match_loss(acc, s1):
    """acc: earlier matched test pairs (s1_id, country, kept). Per country: matches, how many the
    pre-filter drops, and 1 - macro F0.5 of the kept matches against the earlier ones over all S1s
    of the country (precision 1, recall = kept / earlier; S1s without earlier matches score 1)."""
    per = acc.group_by("s1_id").agg(n=pl.len(), k=pl.col("kept").sum())
    r = per["k"] / per["n"]
    per = per.with_columns(f=(1.25 * r / (0.25 + r)).fill_nan(0.0))
    d = s1.join(per, on="s1_id", how="left").with_columns(pl.col("f").fill_null(1.0), pl.col("n", "k").fill_null(0))
    return (pl.concat([d, d.with_columns(country=pl.lit("ALL"))])
              .group_by("country").agg(S1=pl.len(), matched=pl.col("n").sum(),
                                       dropped=(pl.col("n") - pl.col("k")).sum(),
                                       S1_affected=(pl.col("k") < pl.col("n")).sum(),
                                       loss_bound=1 - pl.col("f").mean()).sort("country"))


def export_dir(args):
    """Where the OOF export is written: the cache in sample mode, else config.OUTPUT_DIR."""
    out = config.CACHE_DIR / "oof_export" if args.sample else config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def apply_test(args, meta):
    """Stage-1 probabilities for every test blocking pair; write the surviving set."""
    t0 = time.time()
    pairs = stage1_pairs("test")
    p1 = predict_p1(pairs)
    cutoff = meta["cutoff"]
    pairs = pairs.select("s1_id", "cand_id", "country").with_columns(p1=pl.Series(p1), kept=pl.Series(p1 >= cutoff))
    kp = config.prefilter_keep_path("test")
    kp.parent.mkdir(parents=True, exist_ok=True)
    tmp = kp.with_suffix(".tmp")
    pairs.filter("kept").select("s1_id", "cand_id").write_parquet(tmp, compression="zstd")
    tmp.replace(kp)                                    # the keep file is the stage checkpoint
    pairs.select("s1_id", rec_id="cand_id", p1="p1", kept="kept").write_parquet(
        export_dir(args) / "test_prefilter.parquet", compression="zstd")

    s1 = pl.read_parquet(config.norm_path("test", "S1"), columns=["entity_id", "country"]).rename(
        {"entity_id": "s1_id"})
    per = (s1.join(pairs.group_by("s1_id").agg(before=pl.len(), after=pl.col("kept").sum()), on="s1_id", how="left")
             .fill_null(0))
    summary = per.group_by("country").agg(S1=pl.len(), avg_before=pl.col("before").mean(),
                                          avg_after=pl.col("after").mean(),
                                          p95_before=pl.col("before").quantile(0.95),
                                          p95_after=pl.col("after").quantile(0.95)).sort("country")
    print(f"\n=== Test candidates per S1, pre-filter p1 >= {cutoff:g} ===")
    print(summary)
    print(f"test pairs kept: {int(pairs['kept'].sum()):,} of {len(pairs):,} ({100 * pairs['kept'].mean():.1f}%)")

    prev = config.CACHE_DIR / "predict" / "test_probs.parquet"      # an earlier run WITHOUT the pre-filter
    if prev.exists():
        thr = config.FINAL_THRESHOLD or meta["threshold"]
        acc = (pl.read_parquet(prev, columns=["s1_id", "cand_id", "prob"]).filter(pl.col("prob") >= thr)
                 .join(pairs.select("s1_id", "cand_id", "country", "kept"), on=["s1_id", "cand_id"], how="inner"))
        print(f"earlier run's accepted test pairs (prob >= {thr:.2f}, cached {prev.name}) dropped by the pre-filter; "
              f"loss_bound = 1 - macro F0.5 of the kept vs the earlier matches (the loss if those were all correct):")
        print(match_loss(acc, s1))
    print(f"-> {kp}; {time.time() - t0:.0f}s, peak mem {peak_mem_gb():.2f} GB")


def main():
    """Fit (or with --apply_only reload) the stage-1 model and write the test keep set = candidate set."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--max_loss", type=float, default=MAX_LOSS, help="max macro F0.5 loss vs no pruning (OOF)")
    ap.add_argument("--drop_features", nargs="+", default=[], help="ablation: stage 1 without these features")
    ap.add_argument("--w", type=float, default=DEFAULT_W, help="distractor weight of the F0.5 used for the cutoff")
    ap.add_argument("--threshold", type=float, help="matcher threshold of the F0.5 used for the cutoff "
                                                    "(default: config.FINAL_THRESHOLD, else 0.79)")
    ap.add_argument("--cutoff", type=float, help="pin the p1 cutoff (default: config.PREFILTER_CUTOFF, "
                                                 "else chosen on OOF with --max_loss)")
    ap.add_argument("--apply_only", action="store_true", help="re-apply the saved stage-1 model to test")
    ap.add_argument("--no_test", action="store_true", help="fit only (no test blocking pairs, e.g. sample mode)")
    args = ap.parse_args()
    config.use_sample(args.sample)
    global STAGE1
    STAGE1 = [f for f in STAGE1 if f not in args.drop_features]
    if args.threshold is None:
        args.threshold = config.FINAL_THRESHOLD or DEFAULT_THRESHOLD
    if args.cutoff is None:
        args.cutoff = config.PREFILTER_CUTOFF
    if args.apply_only:
        meta = json.loads((config.prefilter_dir() / "prefilter.json").read_text())
        STAGE1 = meta["features"]
        if args.cutoff is not None:
            meta["cutoff"] = args.cutoff
    else:
        meta = fit(args)
    if not args.no_test and any(config.blocking_dir("test").glob("pairs_*.parquet")):
        apply_test(args, meta)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
