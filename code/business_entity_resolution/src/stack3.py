"""Stacking v3: seed-bagged stack_ctx2 (+ its average with stack_ctx v1) on fold B, CPU only.

Pipeline position: experiment after stack.py --ce2_dir (seed bagging); not part of the submitted pipeline (public LB
0.978289).

Same inputs as stack.py --ce2_dir (run-4 cache, cache_ce, cache_ce2), same fold-B rows and features, same 5-fold
GroupKFold by S1 (built once, reused by every model). Faster LightGBM settings so the run fits in ~40 min
(learning rate 0.1, early stop 50, at most 500 rounds; bagging_fraction / feature_fraction 0.8):
  bag   stack_ctx2 features, N seeds (LightGBM seed = bagging / feature-fraction seed), OOF and test probs averaged
  v1    stack_ctx features (the 0.977 submission's), one seed
  avg   (v1 + bag) / 2
Each is scored like stack.py (many-to-one, macro F0.5 on every fold-B S1, best threshold at w = 2, w = 1 reported).
VERDICT: the better of {bag, avg} at w = 2 must beat the single stack_ctx2 of the stack2 run (--ref, fold-B 0.98401)
by >= 0.0003, else NO SUBMIT. A time guard stops adding seeds (minimum 3) when the projected run would exceed
--budget_min.

Outputs: <out>/matching_results.tsv (chosen @ its w=2 threshold), <out>/variants/matching_results_<name>_t<thr_w1>.tsv,
<out>/candidate_pairs.tsv, <out>/stack3.json, <out>/test_pair_probs_stack3.parquet; fold models in <stack_dir>/.

Usage (from student_resource/):
    python code/business_entity_resolution/src/stack3.py --ce_dir cache_ce --ce2_dir cache_ce2 --out output_stack3
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
from stack import CE2_FEATS, CTX, CTX2, N_FOLDS, PARAMS, PRED_CHUNK, add_ctx, add_probs, evaluate, log

FAST = dict(PARAMS, learning_rate=0.1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1)
MAX_ROUNDS, EARLY_STOP = 500, 50
SEEDS = [2026, 7, 99, 1234, 31337]


def load_fold_b(ce, ce2, base_feats):
    """Fold-B rows exactly as stack.py --ce2_dir builds them."""
    keys = ["s1_id", "cand_id"]
    universe = pl.read_parquet(ce / "universe_B.parquet").sort("s1_id").with_row_index("code")
    b = (pl.read_parquet(ce / "pairs_B.parquet").select("s1_id", "cand_id", "label", "p_lgbm", "pf")
           .join(pl.read_parquet(ce / "ce_B.parquet"), on=keys)
           .join(universe.select("s1_id", "code"), on="s1_id")
           .with_columns(pl.col(keys).cast(pl.String)))
    feats_b = (pl.scan_parquet(config.features_path("train")).select(keys + base_feats)
                 .with_columns(pl.col(keys).cast(pl.String))
                 .join(b.lazy().select(keys), on=keys, how="semi").collect()
                 .unique(keys, keep="first").with_columns(_hit=pl.lit(True)))
    b = b.join(feats_b, on=keys, how="left").with_columns(pl.col("_hit").fill_null(False))
    n_hit = int(b["_hit"].sum())
    log(f"join diagnostic: pairs_B {len(b):,}, matched in features/train.parquet {n_hit:,} ({n_hit / max(len(b), 1):.2%})")
    assert n_hit >= 0.95 * len(b), "fewer than 95% of fold-B pairs found in features/train.parquet: keys differ"
    b = b.with_columns(pf=pl.col("pf") & pl.col("_hit")).drop("_hit")
    n0 = len(b)
    b = b.join(pl.read_parquet(ce2 / "ce_B.parquet").rename({"p_ce": "p_ce2"})
                 .with_columns(pl.col(keys).cast(pl.String)), on=keys, how="left")
    assert b["p_ce2"].null_count() == 0 and len(b) == n0, "second cross-encoder did not score the same fold-B pairs"
    b = add_probs(b, True)
    b = pl.concat([add_ctx(b.filter(pl.col("pf")), True), b.filter(~pl.col("pf"))],
                  how="diagonal_relaxed").with_columns([pl.col(c).fill_null(0.0) for c in CTX + CTX2])
    return universe, b


def fit_oof(X, y, folds, feats, name, seed, stack_dir):
    """Train one seed of a variant with the shared folds; returns (OOF probabilities, best iterations)."""
    params = dict(FAST, seed=seed)
    oof = np.zeros(len(y))
    iters = []
    t0 = time.time()
    for k, (tr, va) in enumerate(folds):
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        m.save_model(str(stack_dir / f"{name}_fold{k}.txt"), num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
    log(f"  {name}: rounds {iters}, {(time.time() - t0) / 60:.1f} min")
    return oof, iters


def predict_test(X, name, stack_dir):
    """Average the 5 fold models of `name` over the test feature matrix X (chunked)."""
    prob = np.zeros(len(X))
    for k in range(N_FOLDS):
        m = lgb.Booster(model_file=str(stack_dir / f"{name}_fold{k}.txt"))
        for s in range(0, len(X), PRED_CHUNK):
            prob[s:s + PRED_CHUNK] += m.predict(X[s:s + PRED_CHUNK]) / N_FOLDS
    return prob


def main():
    """Train the seeds, print the fold-B table and verdict, and write the chosen test submission."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce_dir", default=str(config.REPO_ROOT / "cache_ce"))
    ap.add_argument("--ce2_dir", default=str(config.REPO_ROOT / "cache_ce2"))
    ap.add_argument("--out", default=str(config.REPO_ROOT / "output_stack3"))
    ap.add_argument("--stack_dir", default=None, help="fold models (default <ce_dir>/stack3)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget_min", type=float, default=38.0, help="stop adding seeds (min 3) past this projection")
    ap.add_argument("--ref", type=float, default=0.98401, help="fold-B F0.5 (w=2) of the single stack_ctx2 (stack2)")
    ap.add_argument("--min_gain", type=float, default=3e-4)
    ap.add_argument("--max_test", type=int, default=0, help="smoke test: only this many test pairs")
    args = ap.parse_args()
    t_all = time.time()
    ce, ce2 = Path(args.ce_dir), Path(args.ce2_dir)
    stack_dir = Path(args.stack_dir) if args.stack_dir else ce / "stack3"
    stack_dir.mkdir(parents=True, exist_ok=True)
    base_feats = json.loads((config.model_dir() / "meta.json").read_text())["features"]
    f_v1 = base_feats + ["p_lgbm", "p_ce", "blend"] + CTX
    f_v2 = f_v1 + CE2_FEATS + CTX2
    log(f"stack3: cache {config.CACHE_DIR}; ce {ce}; ce2 {ce2}; {len(f_v2)} features; params {FAST}")

    # ---------------------------------------------------------------- fold B
    universe, b = load_fold_b(ce, ce2, base_feats)
    tr_rows = b.filter(pl.col("pf"))
    X = tr_rows.select(f_v2).to_numpy().astype(np.float32)
    y = tr_rows["label"].to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.SEED)
                 .split(X, y, tr_rows["code"].to_numpy()))   # same splits as stack.py, reused by every model
    iv1 = [f_v2.index(f) for f in f_v1]
    log(f"fold B: {len(universe):,} S1s, {len(b):,} pairs, {len(tr_rows):,} pre-filter survivors; "
        f"peak {peak_mem_gb():.1f} GB")

    oofs, iters, names, t_model = {}, {}, [], []
    plan = [("v2_s0", f_v2, None, SEEDS[0]), ("v1_s0", f_v1, iv1, SEEDS[0])] + \
           [(f"v2_s{i}", f_v2, None, SEEDS[i]) for i in range(1, args.seeds)]
    for name, feats, idx, seed in plan:
        n_v2 = sum(n.startswith("v2") for n in names)
        if name.startswith("v2") and n_v2 >= 3 and t_model:
            per = float(np.mean(t_model))   # scoring 5 fold models on test ~ training them (measured on stack2)
            proj = (time.time() - t_all) / 60 + per * (len(names) + 2) + 3
            if proj > args.budget_min:
                log(f"time guard: stop at {n_v2} seeds (projected {proj:.0f} min > {args.budget_min:.0f})")
                break
        t = time.time()
        oofs[name], iters[name] = fit_oof(X if idx is None else X[:, idx], y, folds, feats, name, seed, stack_dir)
        t_model.append((time.time() - t) / 60)
        names.append(name)
    bag_names = [n for n in names if n.startswith("v2")]
    oofs["bag"] = np.mean([oofs[n] for n in bag_names], axis=0)
    oofs["avg"] = (oofs["v1_s0"] + oofs["bag"]) / 2
    del X

    b = b.join(tr_rows.select("s1_id", "cand_id").with_columns(**{k: pl.Series(v) for k, v in oofs.items()}),
               on=["s1_id", "cand_id"], how="left").with_columns([pl.col(k).fill_null(0.0) for k in oofs])
    labels = {"v2_s0": "stack_ctx2 single (this run, seed 0)", "v1_s0": "stack_ctx v1 (this run)",
              "bag": f"stack_ctx2 bagged ({len(bag_names)} seeds)", "avg": "avg(stack_ctx v1, stack_ctx2 bagged)"}
    rows = [{"model": "stack_ctx2 single (stack2 run, LB 0.978359)", "F0.5_w2": args.ref}]
    for k in ("v2_s0", "v1_s0", "bag", "avg"):
        rows.append({"model": labels[k], "key": k, **evaluate(b, universe, k)})
    table = pl.DataFrame(rows, strict=False)
    pl.Config.set_float_precision(5)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_fmt_str_lengths(60)
    log(f"=== Fold B OOF ({len(universe):,} S1s; 5-fold GroupKFold by S1; many-to-one) ===")
    print(table.drop("key"), flush=True)
    by = {r["key"]: r for r in rows[1:]}
    best = max((by["bag"], by["avg"]), key=lambda r: r["F0.5_w2"])
    name = best["key"]
    gain = best["F0.5_w2"] - args.ref
    ok = gain >= args.min_gain
    log(f"VERDICT stack3: best = {best['model']} {best['F0.5_w2']:.5f} @ {best['thr_w2']:.2f} (w=2), "
        f"{gain:+.5f} vs stack_ctx2 single {args.ref:.5f} (need >= +{args.min_gain:.4f}) -> "
        + ("SUBMIT output_stack3/matching_results.tsv" if ok else "NO SUBMIT (keep the 0.978359 file)"))
    thr, thr_w1 = best["thr_w2"], best["thr_w1"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stack3.json").write_text(json.dumps(
        {"table": rows, "chosen": name, "threshold": thr, "threshold_w1": thr_w1, "submit": ok, "seeds": bag_names,
         "rounds": iters, "params": FAST, "max_rounds": MAX_ROUNDS, "early_stop": EARLY_STOP}, indent=2))

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
            .join(pl.read_parquet(ce / "ce_test.parquet"), on=["s1_id", "cand_id"], how="left")
            .join(pl.read_parquet(ce2 / "ce_test.parquet").rename({"p_ce": "p_ce2"}), on=["s1_id", "cand_id"],
                  how="left"))
    assert len(tt) == len(keep) and all(tt[c].null_count() == 0 for c in ("p_lgbm", "p_ce", "p_ce2")), \
        "test features / probabilities do not cover the candidate set"
    tt = add_ctx(add_probs(tt, True), True)
    Xt = tt.select(f_v2).to_numpy().astype(np.float32)
    tt = tt.select("s1_id", "cand_id")
    log(f"test: {len(tt):,} candidate pairs loaded in {time.time() - t:.0f}s, peak {peak_mem_gb():.1f} GB")
    t = time.time()
    per_seed = {n: predict_test(Xt, n, stack_dir) for n in bag_names}
    probs = {"v2_s0": per_seed["v2_s0"], "bag": np.mean(list(per_seed.values()), axis=0)}
    if name == "avg":
        probs["v1_s0"] = predict_test(Xt[:, iv1], "v1_s0", stack_dir)
        probs["avg"] = (probs["v1_s0"] + probs["bag"]) / 2
    del Xt
    log(f"test scored ({len(per_seed) + (name == 'avg')} x 5 fold models) in {(time.time() - t) / 60:.1f} min")
    tt = tt.with_columns(**{k: pl.Series(v) for k, v in probs.items()})

    s1_ids = all_s1_ids("test")
    if args.max_test:
        s1_ids = s1_ids.join(tt.select("s1_id").unique(), on="s1_id", how="semi")
    write_lists(s1_ids, tt, "candidate_entity_ids", out / "candidate_pairs.tsv")
    k = many_to_one(tt.with_columns(prob=pl.col(name)))
    sel = k.filter(pl.col("prob") >= thr)
    res = write_lists(s1_ids, sel, "matched_entity_ids", out / "matching_results.tsv")
    (out / "variants").mkdir(exist_ok=True)
    if thr_w1 != thr:
        write_lists(s1_ids, k.filter(pl.col("prob") >= thr_w1), "matched_entity_ids",
                    out / "variants" / f"matching_results_{name}_t{thr_w1:.2f}.tsv")
    ref = many_to_one(tt.with_columns(prob=pl.col("v2_s0"))).filter(pl.col("prob") >= by["v2_s0"]["thr_w2"])
    s1_country = pl.read_parquet(config.norm_path("test", "S1"), columns=["entity_id", "country"]).rename(
        {"entity_id": "s1_id"})
    summarize(res, s1_country,
              f"output_stack3 matching_results.tsv: {name} @ {thr:.2f}: {len(sel):,} matches of {len(tt):,} candidates "
              f"(single stack_ctx2 seed 0 @ {by['v2_s0']['thr_w2']:.2f}: {len(ref):,}; "
              f"added {len(sel.join(ref, on=['s1_id', 'cand_id'], how='anti')):,}, "
              f"removed {len(ref.join(sel, on=['s1_id', 'cand_id'], how='anti')):,})")
    tt.rename({"cand_id": "rec_id"}).write_parquet(out / "test_pair_probs_stack3.parquet", compression="zstd")
    log(f"-> {out}; all {(time.time() - t_all) / 60:.1f} min, peak {peak_mem_gb():.1f} GB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
