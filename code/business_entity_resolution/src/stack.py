"""Stacking: a second-level LightGBM on fold B (the cross-encoder's held-out S1s) over every pair feature of the
matcher + p_lgbm (OOF) + p_ce, compared with the linear blend on the same fold-B pairs.

Pipeline position: final stage: the submitted file is stack.py --ce2_dir (stack_ctx2 @ config.STACK_THRESHOLD =
0.83, public LB 0.978359).

Inputs (read only):
  cache (config.CACHE_DIR, the run-4 cache): features/{train,test}.parquet, model/meta.json (feature list),
      prefilter/test_keep.parquet, predict/test_probs_pf.parquet (test p_lgbm)
  ce_dir (cross_encoder.py output): pairs_B.parquet (label, p_lgbm OOF, pf flag), ce_B.parquet, universe_B.parquet,
      ce_test.parquet

Two variants, each 5-fold GroupKFold by S1 within fold B, trained on the pre-filter survivors (the rest count as
rejected, as on test):
  stack      matcher features + p_lgbm + p_ce + blend
  stack_ctx  + S1-group context of the two probabilities (rank inside the S1's candidates, S1 max, number of
             candidates >= 0.5), computed over the same candidate set as on test
Each is scored like the blend: many-to-one, macro F0.5 on every fold-B S1, best threshold at distractor weight 2
(w = 1 reported). The best of {blend, stack, stack_ctx} at w = 2 is printed with a verdict; the test output is
written for the better stack variant (the blend file already exists: run_ce/).

Outputs: <out>/matching_results.tsv, <out>/candidate_pairs.tsv (the run-4 candidate set, byte-identical),
<out>/variants/matching_results_stack_t<thr_w1>.tsv, <out>/test_pair_probs_stack.parquet, <out>/stack.json,
fold models in <stack_dir>/.

Usage (from student_resource/):
    python code/business_entity_resolution/src/stack.py --ce_dir cache_ce --out output_stack
"""

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

import config
from memory import peak_mem_gb

N_FOLDS = 5
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              seed=config.SEED, verbose=-1, num_threads=0)
MAX_ROUNDS, EARLY_STOP = 1500, 100   # bounds test scoring (7.73M pairs x 5 folds)
PRED_CHUNK = 1_000_000
CTX = ["lg_rank", "ce_rank", "lg_max", "ce_max", "lg_n_hi", "ce_n_hi", "lg_gap_max", "ce_gap_max"]
# second cross-encoder (--ce2_dir): its probability, blends and S1 context
CE2_FEATS = ["p_ce2", "blend2", "ce_mean", "blend12"]
CTX2 = ["ce2_rank", "ce2_max", "ce2_n_hi", "ce2_gap_max"]
# third cross-encoder (--ce3_dir, needs --ce2_dir): stack_ctx3 = stack_ctx2 + these, verdict vs --ref
CE3_FEATS = ["p_ce3", "blend3", "ce_mean3"]
CTX3 = ["ce3_rank", "ce3_max", "ce3_n_hi", "ce3_gap_max"]


def log(msg):
    """Print a time-stamped message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_probs(df, ce2=False, ce3=False):
    """Add the blend features: blend = w * p_lgbm + (1 - w) * p_ce (w = config.CE_W_LGBM) and, with the
    second / third cross-encoder, blend2, ce_mean, blend12 / blend3, ce_mean3.
    """
    w = config.CE_W_LGBM
    df = df.with_columns(blend=w * pl.col("p_lgbm") + (1 - w) * pl.col("p_ce"))
    if ce2:
        df = df.with_columns(blend2=w * pl.col("p_lgbm") + (1 - w) * pl.col("p_ce2"),
                             ce_mean=(pl.col("p_ce") + pl.col("p_ce2")) / 2) \
               .with_columns(blend12=w * pl.col("p_lgbm") + (1 - w) * pl.col("ce_mean"))
    if ce3:
        df = df.with_columns(blend3=w * pl.col("p_lgbm") + (1 - w) * pl.col("p_ce3"),
                             ce_mean3=(pl.col("p_ce") + pl.col("p_ce2") + pl.col("p_ce3")) / 3)
    return df


def add_ctx(df, ce2=False, ce3=False):
    """S1-group context of p_lgbm / p_ce (/ p_ce2) over the rows present (= the candidate set)."""
    out = []
    for p, n in (("p_lgbm", "lg"), ("p_ce", "ce")) + ((("p_ce2", "ce2"),) if ce2 else ())             + ((("p_ce3", "ce3"),) if ce3 else ()):
        g = pl.col(p)
        # S1-group context: a pair's probability relative to the S1's other candidates (rank, S1 max, number of
        # confident candidates, gap to the max), computed over the pre-filter survivors exactly as on test
        out += [g.rank("ordinal", descending=True).over("s1_id").cast(pl.Float32).alias(f"{n}_rank"),
                g.max().over("s1_id").alias(f"{n}_max"),
                (g >= 0.5).sum().over("s1_id").cast(pl.Float32).alias(f"{n}_n_hi"),
                (g.max().over("s1_id") - g).alias(f"{n}_gap_max")]
    return df.with_columns(out)


def evaluate(b, universe, prob_col):
    """Fold-B macro F0.5 of `prob_col` (many-to-one, pre-filtered pairs rejected).

    Returns: dict with the best threshold and score at distractor weight 2 and at weight 1.
    """
    from train_matcher import many_to_one
    from tune_threshold import best_threshold
    n_true = universe["n_true"].to_numpy().astype(float)
    k = many_to_one(b.with_columns(prob=pl.when("pf").then(pl.col(prob_col)).otherwise(0.0)))
    code, label, prob = k["code"].to_numpy(), k["label"].to_numpy().astype(float), k["prob"].to_numpy()
    t2, f2, _ = best_threshold(code, prob, label, n_true, 2.0)
    t1, f1, _ = best_threshold(code, prob, label, n_true, 1.0)
    return {"thr_w2": t2, "F0.5_w2": f2, "thr_w1": t1, "F0.5_w1": f1}


def fit_oof(b, feats, name, stack_dir):
    """Train a 5-fold GroupKFold (by S1) LightGBM on the pre-filter survivors of fold B.

    Saves the fold models to stack_dir and adds the OOF probability as column `name`.
    Returns: (fold-B frame with the new column, best iteration per fold).
    """
    tr_rows = b.filter(pl.col("pf"))
    X, y = tr_rows.select(feats).to_numpy().astype(np.float32), tr_rows["label"].to_numpy()
    groups = tr_rows["code"].to_numpy()
    oof = np.zeros(len(tr_rows))
    imp = np.zeros(len(feats))
    iters = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.SEED)
                                 .split(X, y, groups)):
        t = time.time()
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        m.save_model(str(stack_dir / f"{name}_fold{k}.txt"), num_iteration=m.best_iteration)
        imp += m.feature_importance("gain", iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log(f"  {name} fold {k}: {len(tr):,} train rows, {m.best_iteration} rounds, {time.time() - t:.0f}s")
    top = sorted(zip(feats, imp / imp.sum() * 100), key=lambda x: -x[1])[:8]
    log(f"  {name} top gain%: " + ", ".join(f"{f} {g:.1f}" for f, g in top))
    b = b.join(tr_rows.select("s1_id", "cand_id").with_columns(**{name: pl.Series(oof)}),
               on=["s1_id", "cand_id"], how="left").with_columns(pl.col(name).fill_null(0.0))
    return b, iters


def predict_test(t, feats, name, stack_dir):
    """Average the 5 fold models of variant `name` over the test candidates (chunked)."""
    X = t.select(feats).to_numpy().astype(np.float32)
    prob = np.zeros(len(t))
    for k in range(N_FOLDS):
        m = lgb.Booster(model_file=str(stack_dir / f"{name}_fold{k}.txt"))
        for s in range(0, len(t), PRED_CHUNK):
            prob[s:s + PRED_CHUNK] += m.predict(X[s:s + PRED_CHUNK]) / N_FOLDS
    return prob


def main():
    """Build fold B, train the variants, print the table / verdict, pick the model and threshold, and write
    the test submission.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce_dir", default=str(config.REPO_ROOT / "cache_ce"))
    ap.add_argument("--out", default=str(config.REPO_ROOT / "output_stack"))
    ap.add_argument("--stack_dir", default=None, help="fold models (default <ce_dir>/stack)")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--max_test", type=int, default=0, help="smoke test: only this many test pairs")
    ap.add_argument("--ce2_dir", help="second cross-encoder's ce_dir (cross_encoder.py --from_ce): adds p_ce2 and "
                                      "compares stack_ctx2 with stack_ctx; test output uses stack_ctx2")
    ap.add_argument("--ce3_dir", help="third cross-encoder's ce_dir (with --ce2_dir): trains stack_ctx3 only, "
                                      "verdict vs --ref; test output uses stack_ctx3")
    ap.add_argument("--ref", type=float, default=0.98401, help="fold-B F0.5 (w=2) of stack_ctx2 (LB 0.978359)")
    ap.add_argument("--refit", action="store_true",
                    help="choose variant and threshold on fold B instead of config.STACK_VARIANT / STACK_THRESHOLD")
    args = ap.parse_args()
    config.use_sample(args.sample)
    t_all = time.time()
    ce = Path(args.ce_dir)
    stack_dir = Path(args.stack_dir) if args.stack_dir else ce / "stack"
    stack_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((config.model_dir() / "meta.json").read_text())
    base_feats = meta["features"]
    log(f"stack: cache {config.CACHE_DIR}; ce_dir {ce}; {len(base_feats)} matcher features")

    # ---------------------------------------------------------------- fold B
    t = time.time()
    universe = pl.read_parquet(ce / "universe_B.parquet").sort("s1_id").with_row_index("code")
    b = (pl.read_parquet(ce / "pairs_B.parquet").select("s1_id", "cand_id", "label", "p_lgbm", "pf")
           .join(pl.read_parquet(ce / "ce_B.parquet"), on=["s1_id", "cand_id"])
           .join(universe.select("s1_id", "code"), on="s1_id"))
    keys = ["s1_id", "cand_id"]
    b = b.with_columns(pl.col(keys).cast(pl.String))
    feats_b = (pl.scan_parquet(config.features_path("train")).select(keys + base_feats)
                 .with_columns(pl.col(keys).cast(pl.String))
                 .join(b.lazy().select(keys), on=keys, how="semi").collect()
                 .unique(keys, keep="first").with_columns(_hit=pl.lit(True)))
    # features may be null legitimately (empty name / address): the join is checked with _hit, not feature nulls
    b = b.join(feats_b, on=keys, how="left").with_columns(pl.col("_hit").fill_null(False))
    n_hit = int(b["_hit"].sum())
    log(f"join diagnostic: pairs_B {len(b):,} rows, matched in features/train.parquet {n_hit:,} "
        f"({n_hit / max(len(b), 1):.2%}), unmatched {len(b) - n_hit:,} "
        f"(of which pre-filter survivors {int((~b['_hit'] & b['pf']).sum()):,})")
    log(f"  sample keys pairs_B: {b.select(keys).head(3).rows()}; features: {feats_b.select(keys).head(3).rows()}")
    if len(b) - n_hit:
        log(f"  unmatched sample: {b.filter(~pl.col('_hit')).select(keys).head(3).rows()}")
    assert n_hit >= 0.95 * len(b), "fewer than 95% of fold-B pairs found in features/train.parquet: keys differ"
    # pre-filtered pairs are never scored on test, so on fold B they are not trained on and count as rejected
    b = b.with_columns(pf=pl.col("pf") & pl.col("_hit")).drop("_hit")   # unmatched pairs count as rejected
    ce2 = Path(args.ce2_dir) if args.ce2_dir else None
    ce3 = Path(args.ce3_dir) if args.ce3_dir else None
    assert ce3 is None or ce2 is not None, "--ce3_dir needs --ce2_dir"
    ctx_cols = CTX + (CTX2 if ce2 else []) + (CTX3 if ce3 else [])
    if ce2:
        n0 = len(b)
        b = b.join(pl.read_parquet(ce2 / "ce_B.parquet").rename({"p_ce": "p_ce2"})
                     .with_columns(pl.col(keys).cast(pl.String)), on=keys, how="left")
        miss = b["p_ce2"].null_count()
        log(f"ce2 join: {n0 - miss:,} of {n0:,} fold-B pairs have p_ce2 ({ce2})")
        assert miss == 0 and len(b) == n0, "second cross-encoder did not score the same fold-B pairs"
    if ce3:
        b = b.join(pl.read_parquet(ce3 / "ce_B.parquet").rename({"p_ce": "p_ce3"})
                     .with_columns(pl.col(keys).cast(pl.String)), on=keys, how="left")
        miss = b["p_ce3"].null_count()
        log(f"ce3 join: {n0 - miss:,} of {n0:,} fold-B pairs have p_ce3 ({ce3})")
        assert miss == 0 and len(b) == n0, "third cross-encoder did not score the same fold-B pairs"
    b = add_probs(b, ce2 is not None, ce3 is not None)
    b = pl.concat([add_ctx(b.filter(pl.col("pf")), ce2 is not None, ce3 is not None), b.filter(~pl.col("pf"))],
                  how="diagonal_relaxed").with_columns([pl.col(c).fill_null(0.0) for c in ctx_cols])
    log(f"fold B: {len(universe):,} S1s, {len(b):,} pairs ({int(b['pf'].sum()):,} pre-filter survivors, "
        f"positives {int(b['label'].sum()):,}) in {time.time() - t:.0f}s, peak {peak_mem_gb():.1f} GB")

    variants = {"stack": base_feats + ["p_lgbm", "p_ce", "blend"],
                "stack_ctx": base_feats + ["p_lgbm", "p_ce", "blend"] + CTX}
    if ce2:   # v1 reference (same features as the 0.977 submission) vs + second cross-encoder
        variants = {"stack_ctx": variants["stack_ctx"],
                    "stack_ctx2": variants["stack_ctx"] + CE2_FEATS + CTX2}
    if ce3:   # stack_ctx2 is the submitted model (fold-B --ref): train only the + CE v3 variant
        variants = {"stack_ctx3": variants["stack_ctx2"] + CE3_FEATS + CTX3}
    iters = {}
    for name, feats in variants.items():
        t = time.time()
        b, iters[name] = fit_oof(b, feats, name, stack_dir)
        log(f"{name}: 5 folds in {(time.time() - t) / 60:.1f} min")

    rows = []
    w = config.CE_W_LGBM
    cols = [("LightGBM alone", "p_lgbm"), ("CE alone", "p_ce"), (f"blend {w}/{1 - w:.1f}", "blend")]
    if ce2:
        cols += [("CE2 alone", "p_ce2"), (f"blend2 {w} LGBM + {1 - w:.1f} CE2", "blend2"),
                 (f"blend12 {w} LGBM + {1 - w:.1f} mean(CE, CE2)", "blend12")]
    if ce3:
        cols += [("CE3 alone", "p_ce3"), (f"blend3 {w} LGBM + {1 - w:.1f} CE3", "blend3")]
    cols += [(v, v) for v in variants]
    for name, col in cols:
        rows.append({"model": name, **evaluate(b, universe, col)})
    table = pl.DataFrame(rows)
    pl.Config.set_float_precision(5)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_fmt_str_lengths(60)
    log(f"=== Fold B OOF ({len(universe):,} S1s; stack = 5-fold GroupKFold by S1 inside fold B; "
        f"pre-filtered pairs rejected; many-to-one) ===")
    print(table, flush=True)
    by = {r["model"]: r for r in table.rows(named=True)}
    if ce3:
        v3 = by["stack_ctx3"]
        g3 = v3["F0.5_w2"] - args.ref
        log(f"VERDICT CE v3: stack_ctx3 {v3['F0.5_w2']:.5f} @ {v3['thr_w2']:.2f} (w=2), {g3:+.5f} vs stack_ctx2 "
            f"{args.ref:.5f} (need >= +0.0005) -> "
            + ("SUBMIT output_stack4/matching_results.tsv" if g3 >= 5e-4 else "NO SUBMIT (keep the 0.978359 file)"))
    elif ce2:
        v1, v2 = by["stack_ctx"], by["stack_ctx2"]
        g2 = v2["F0.5_w2"] - v1["F0.5_w2"]
        ok = g2 >= 5e-4
        log(f"VERDICT CE v2: stack_ctx2 {'BEATS' if ok else 'does NOT clearly beat'} stack_ctx (v1) on fold B: "
            f"{g2:+.5f} at w=2 (need >= +0.0005), {v2['F0.5_w1'] - v1['F0.5_w1']:+.5f} at w=1; "
            f"stack_ctx2 {v2['F0.5_w2']:.5f} vs the 0.977 submission's fold-B 0.98342"
            + (" -> candidate for submission: output_stack2/matching_results.tsv" if ok
               else " -> do NOT submit; keep ML_submission.zip (stack_ctx v1, LB 0.977)"))
    blend = by[f"blend {w}/{1 - w:.1f}"]
    ce2_only = ce2 is not None and ce3 is None
    best = max([by[v] for v in variants], key=lambda r: r["F0.5_w2"])
    name = best["model"]
    gain = best["F0.5_w2"] - blend["F0.5_w2"]
    if not ce2:
        log(f"VERDICT: {name} {'BEATS' if gain >= 5e-4 else 'does NOT clearly beat'} the linear blend on fold B: "
            f"{gain:+.5f} at w=2, {best['F0.5_w1'] - blend['F0.5_w1']:+.5f} at w=1"
            + ("" if gain >= 5e-4 else " -> keep the blend (run_ce / ML_submission_ce.zip)"))
    how = "fold-B best at w=2"
    if ce3:
        best, how, name = by["stack_ctx3"], "CE v3 run: stack_ctx3 at its fold-B best (w=2)", "stack_ctx3"
    pinned = None
    if ce2 and not ce3:
        best, how = by["stack_ctx2"], "CE v2 run: stack_ctx2 at its fold-B best (w=2)"
        name = "stack_ctx2"
        if not args.refit and config.STACK_VARIANT == "stack_ctx2":
            pinned, how = config.STACK_THRESHOLD, "pinned: config.STACK_VARIANT / STACK_THRESHOLD"
    elif not ce2 and not args.refit:
        best = table.filter(pl.col("model") == "stack_ctx").row(0, named=True)
        name, how = best["model"], "pinned: stack_ctx @ config.STACK_V1_THRESHOLD (the LB 0.977 run)"
        pinned = config.STACK_V1_THRESHOLD
    thr = best["thr_w2"] if pinned is None else pinned
    thr_w1 = best["thr_w1"]
    log(f"chosen: {name} @ {thr:.2f} ({how}); variant @ {thr_w1:.2f} (best at w=1)")
    (Path(args.out)).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "stack.json").write_text(json.dumps(
        {"table": table.to_dicts(), "chosen": name, "threshold": thr, "threshold_w1": thr_w1,
         "features": variants[name], "rounds": iters, "params": PARAMS}, indent=2))

    # ---------------------------------------------------------------- test
    from predict import all_s1_ids, summarize, write_lists
    from train_matcher import many_to_one
    t = time.time()
    keep = pl.read_parquet(config.prefilter_keep_path("test")).select("s1_id", "cand_id")
    if args.max_test:
        keep = keep.sort(["s1_id", "cand_id"]).head(args.max_test)
    tt = (pl.scan_parquet(config.features_path("test")).select(["s1_id", "cand_id"] + base_feats)
            .join(keep.lazy(), on=["s1_id", "cand_id"], how="semi").collect())
    lg = pl.read_parquet(config.CACHE_DIR / "predict" / "test_probs_pf.parquet", columns=["s1_id", "cand_id", "prob"])
    tt = (tt.join(lg.rename({"prob": "p_lgbm"}), on=["s1_id", "cand_id"], how="left")
            .join(pl.read_parquet(ce / "ce_test.parquet"), on=["s1_id", "cand_id"], how="left"))
    assert len(tt) == len(keep) and tt["p_lgbm"].null_count() == 0 and tt["p_ce"].null_count() == 0, \
        "test features / probabilities do not cover the candidate set"
    if ce2:
        tt = tt.join(pl.read_parquet(ce2 / "ce_test.parquet").rename({"p_ce": "p_ce2"}), on=["s1_id", "cand_id"],
                     how="left")
        assert len(tt) == len(keep) and tt["p_ce2"].null_count() == 0, "p_ce2 does not cover the test candidates"
    if ce3:
        tt = tt.join(pl.read_parquet(ce3 / "ce_test.parquet").rename({"p_ce": "p_ce3"}), on=["s1_id", "cand_id"],
                     how="left")
        assert len(tt) == len(keep) and tt["p_ce3"].null_count() == 0, "p_ce3 does not cover the test candidates"
    tt = add_ctx(add_probs(tt, ce2 is not None, ce3 is not None), ce2 is not None, ce3 is not None)
    log(f"test: {len(tt):,} candidate pairs loaded in {time.time() - t:.0f}s, peak {peak_mem_gb():.1f} GB")
    t = time.time()
    tt = tt.with_columns(p_stack=pl.Series(predict_test(tt, variants[name], name, stack_dir)))
    log(f"test scored with {name} in {(time.time() - t) / 60:.1f} min")

    out = Path(args.out)
    s1_ids = all_s1_ids("test")
    if args.max_test:
        s1_ids = s1_ids.join(tt.select("s1_id").unique(), on="s1_id", how="semi")
    write_lists(s1_ids, tt, "candidate_entity_ids", out / "candidate_pairs.tsv")
    k = many_to_one(tt.with_columns(prob=pl.col("p_stack")))
    sel = k.filter(pl.col("prob") >= thr)
    res = write_lists(s1_ids, sel, "matched_entity_ids", out / "matching_results.tsv")
    (out / "variants").mkdir(exist_ok=True)
    if thr_w1 != thr:
        write_lists(s1_ids, k.filter(pl.col("prob") >= thr_w1), "matched_entity_ids",
                    out / "variants" / f"matching_results_stack_t{thr_w1:.2f}.tsv")
    if ce2_only:   # reference = stack_ctx v1 at the submitted threshold (retrained here: ~ the LB 0.977 file)
        tt = tt.with_columns(p_v1=pl.Series(predict_test(tt, variants["stack_ctx"], "stack_ctx", stack_dir)))
        ref = many_to_one(tt.with_columns(prob=pl.col("p_v1"))).filter(pl.col("prob") >= config.STACK_V1_THRESHOLD)
    else:
        ref = many_to_one(tt.with_columns(prob=pl.col("blend"))).filter(pl.col("prob") >= config.CE_THRESHOLD)
    s1_country = pl.read_parquet(config.norm_path("test", "S1"), columns=["entity_id", "country"]).rename(
        {"entity_id": "s1_id"})
    summarize(res, s1_country,
              f"output_stack matching_results.tsv: {name} @ {thr:.2f}: {len(sel):,} matches of {len(tt):,} candidates "
              f"({'stack_ctx v1 @ ' + str(config.STACK_V1_THRESHOLD) if ce2_only else 'blend file'}: {len(ref):,}; added {len(sel.join(ref, on=['s1_id', 'cand_id'], how='anti')):,}, "
              f"removed {len(ref.join(sel, on=['s1_id', 'cand_id'], how='anti')):,})")
    tt.select(["s1_id", pl.col("cand_id").alias("rec_id"), "p_stack", "p_lgbm", "p_ce"]
              + (["p_ce2", "p_v1"] if ce2_only else ["p_ce2", "p_ce3"] if ce3 else [])).write_parquet(
        out / "test_pair_probs_stack.parquet", compression="zstd")
    log(f"-> {out}; all {(time.time() - t_all) / 60:.1f} min, peak {peak_mem_gb():.1f} GB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
