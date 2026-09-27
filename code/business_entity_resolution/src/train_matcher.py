"""Train the pair matcher (LightGBM) and tune the decision thresholds for macro F0.5.

Pipeline position: stage 5 (LightGBM matcher, GroupKFold OOF); its OOF is fold A/B for the cross-encoders and the
stacking.

Pairs = blocking candidates on train (full S1 index x full S2/S3 pool, exactly like test),
label = pair in the ground truth. If there are more than --max_pairs pairs, a random set of
S1 GROUPS is kept (every pair of a chosen S1, and chosen S1s without candidates too), so
per-S1 context and the metric stay exact for the chosen S1s. 5-fold GroupKFold by S1 id (fixed seed): out-of-fold (OOF) probabilities
are used to tune the thresholds and report the score; the 5 fold models are saved and
averaged at prediction time.

Post-processing (shared with predict.py):
  1. many-to-one: a record is kept only for its highest-probability S1
  2. keep a pair when prob >= threshold (one global threshold, or one per source S2/S3,
     whichever gives the higher OOF macro F0.5)

Metric: macro F0.5 over EVERY S1 in the (group-sampled) universe (S1s without candidates
included): singleton -> 1 if nothing predicted else 0; otherwise F0.5 of the predicted
list (0 if nothing correct is predicted). Also reported: the blocking ceiling, i.e. the
score of a perfect matcher that keeps exactly the true pairs among the candidates.

Usage (from student_resource/):
    python code/business_entity_resolution/src/train_matcher.py [--sample N]
"""

import argparse
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import config
from features import load_truth_pairs, model_features
from memory import peak_mem_gb

N_FOLDS = 5
BETA2 = 0.25
LGB_PARAMS = dict(
    objective="binary", learning_rate=0.1, num_leaves=127, min_child_samples=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    seed=config.SEED, verbose=-1, num_threads=0,
)
MAX_ROUNDS = 3000
MAX_TRAIN_NEG = 5_000_000   # negatives per fold's training set; more are downsampled (weighted 1/frac)
EARLY_STOP = 100
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


# --------------------------------------------------------------------------------------
# Post-processing and metric
# --------------------------------------------------------------------------------------

def many_to_one(df, prob="prob"):
    """Keep each record only for its highest-probability S1 (ties -> smallest s1_id)."""
    # each S2/S3 record belongs to at most one real business: keep only its best-scoring S1, so a record can never
    # be predicted for two S1s (a guaranteed false positive on one of them)
    return (df.sort([prob, "s1_id"], descending=[True, False])
              .unique("cand_id", keep="first", maintain_order=True))


def keep_mask(prob, source, thr):
    """thr = {"global": t} or {"S2": t2, "S3": t3}; source 0 = S2, 1 = S3."""
    if "global" in thr:
        return prob >= thr["global"]
    return np.where(source == 1, prob >= thr["S3"], prob >= thr["S2"])


def per_s1_f05(code, keep, label, n_true):
    """F0.5 per S1 (array aligned with n_true) for predicted pairs code[keep]."""
    n = len(n_true)
    npred = np.bincount(code[keep], minlength=n)
    tp = np.bincount(code[keep], weights=label[keep], minlength=n)
    p = np.divide(tp, npred, out=np.zeros(n), where=npred > 0)
    r = np.divide(tp, n_true, out=np.zeros(n), where=n_true > 0)
    f = np.divide((1 + BETA2) * p * r, BETA2 * p + r, out=np.zeros(n), where=(p + r) > 0)
    # singletons (no true match) score 1 only when nothing is predicted for them, else 0
    return np.where(n_true == 0, (npred == 0).astype(float), f), npred, tp


def macro_f05(code, keep, label, n_true):
    """Macro F0.5 over all S1s (singletons included) for the predicted pairs code[keep]."""
    return per_s1_f05(code, keep, label, n_true)[0].mean()


def tune_thresholds(code, prob, source, label, n_true):
    """Best global threshold, then per-source thresholds by coordinate ascent from it."""
    scores = [macro_f05(code, prob >= t, label, n_true) for t in GRID]
    g = float(GRID[int(np.argmax(scores))])
    best_global = {"global": g}, float(max(scores))

    thr = {"S2": g, "S3": g}
    best = best_global[1]
    for _ in range(3):
        changed = False
        for src in ("S2", "S3"):
            for t in GRID:
                cand = {**thr, src: float(t)}
                s = macro_f05(code, keep_mask(prob, source, cand), label, n_true)
                if s > best + 1e-9:
                    best, thr, changed = s, cand, True
        if not changed:
            break
    return best_global, (thr, best)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def report(universe, pairs, keep, title):
    """Macro F0.5 / micro precision & recall / singleton accuracy by country, plus ceiling."""
    code, label = pairs["code"].to_numpy(), pairs["label"].to_numpy().astype(float)
    n_true = universe["n_true"].to_numpy().astype(float)
    f, npred, tp = per_s1_f05(code, keep, label, n_true)
    f_ceil, _, _ = per_s1_f05(code, label > 0, label, n_true)
    df = universe.with_columns(f=pl.Series(f), f_ceil=pl.Series(f_ceil), npred=pl.Series(npred),
                               tp=pl.Series(tp))
    rows = []
    for country, g in list(df.group_by("country")) + [(("ALL",), df)]:
        single = g.filter(pl.col("n_true") == 0)
        rows.append({
            "country": country[0], "S1": len(g),
            "macroF0.5": g["f"].mean(),
            "precision": g["tp"].sum() / max(g["npred"].sum(), 1),
            "recall": g["tp"].sum() / max(g["n_true"].sum(), 1),
            "singleton_acc": (single["npred"] == 0).mean() if len(single) else float("nan"),
            "ceilingF0.5": g["f_ceil"].mean(),
        })
    print(f"\n=== {title} ===")
    print(pl.DataFrame(rows).sort("country"))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def load_universe(n_pairs, max_pairs):
    """Train S1s (all, or a random group sample sized to ~max_pairs candidate pairs) with their
    country and number of true matches."""
    s1 = pl.read_parquet(config.norm_path("train", "S1"), columns=["entity_id", "country"]).sort("entity_id")
    if max_pairs and n_pairs > max_pairs:
        frac = max_pairs / n_pairs
        s1 = s1.filter(pl.Series(np.random.default_rng(config.SEED).random(len(s1)) < frac))
        print(f"group sampling: {n_pairs:,} pairs > {max_pairs:,}: keeping {frac:.3f} of the S1s "
              f"({len(s1):,}) with all of their pairs")
    n_true = load_truth_pairs().group_by("s1_id").agg(n_true=pl.len())
    return (s1.rename({"entity_id": "s1_id"}).join(n_true, on="s1_id", how="left")
              .with_columns(pl.col("n_true").fill_null(0)).sort("s1_id").with_row_index("code"))


def main():
    """Train the 5-fold LightGBM matcher (GroupKFold by S1) on the train blocking pairs, write the fold
    models, oof.parquet, feature importance and the OOF-tuned thresholds (meta.json).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="use the normalize.py --sample N cache")
    ap.add_argument("--max_pairs", type=int, default=config.TRAIN_MAX_PAIRS,
                    help="train on random S1 groups up to this many candidate pairs (0 = all)")
    ap.add_argument("--max_train_neg", type=int, default=MAX_TRAIN_NEG,
                    help="negatives per fold's training set before weighted downsampling")
    ap.add_argument("--drop_features", nargs="+", default=[], help="ablation: train without these features")
    ap.add_argument("--out_subdir", default="", help="ablation: write models under model_dir()/<this>")
    args = ap.parse_args()
    config.use_sample(args.sample)
    t0 = time.time()

    feats = pl.scan_parquet(config.features_path("train"))
    n_pairs = feats.select(pl.len()).collect().item()
    universe = load_universe(n_pairs, args.max_pairs)
    pairs = feats.join(universe.lazy().select("s1_id", "code"), on="s1_id", how="inner").collect()
    print(f"universe: {len(universe):,} S1 ({(universe['n_true'] == 0).sum():,} singletons, "
          f"{int(universe['n_true'].sum()):,} true pairs); candidate pairs {len(pairs):,}, "
          f"positives {int(pairs['label'].sum()):,}")

    FEATURES = [f for f in model_features(pairs.columns) if f not in args.drop_features]
    X = pairs.select(FEATURES).to_numpy()
    y = pairs["label"].to_numpy()
    groups = pairs["code"].to_numpy()
    oof = np.zeros(len(pairs), dtype=np.float64)
    out_dir = config.model_dir() / args.out_subdir if args.out_subdir else config.model_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    importance = np.zeros(len(FEATURES))
    best_iters, aucs = [], []
    folds = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.SEED)
    rng = np.random.default_rng(config.SEED)
    for k, (tr, va) in enumerate(folds.split(X, y, groups)):
        t = time.time()
        # Training rows only: keep every positive, downsample negatives to MAX_TRAIN_NEG with weight
        # 1/frac (keeps probabilities calibrated). Validation / OOF always uses every pair.
        n_neg = int((y[tr] == 0).sum())
        frac = min(1.0, args.max_train_neg / max(n_neg, 1))
        keep = (y[tr] == 1) | (rng.random(len(tr)) < frac)
        tr = tr[keep]
        w = np.where(y[tr] == 1, 1.0, 1.0 / frac)
        if frac < 1:
            print(f"fold {k}: training on {len(tr):,} rows (negatives downsampled to {frac:.3f}, weight {1 / frac:.2f})")
        dtr = lgb.Dataset(X[tr], y[tr], weight=w, feature_name=FEATURES, free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        model = lgb.train(LGB_PARAMS, dtr, MAX_ROUNDS, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        oof[va] = model.predict(X[va], num_iteration=model.best_iteration)
        aucs.append(roc_auc_score(y[va], oof[va]))
        best_iters.append(model.best_iteration)
        importance += model.feature_importance("gain", iteration=model.best_iteration)
        model.save_model(str(out_dir / f"fold{k}.txt"), num_iteration=model.best_iteration)
        print(f"fold {k}: AUC {aucs[-1]:.5f}, best iteration {model.best_iteration}, "
              f"{time.time() - t:.0f}s, peak mem {peak_mem_gb():.2f} GB", flush=True)
    print(f"OOF AUC {roc_auc_score(y, oof):.5f} (folds: {', '.join(f'{a:.5f}' for a in aucs)})")

    imp = pl.DataFrame({"feature": FEATURES, "gain%": 100 * importance / importance.sum()}).sort(
        "gain%", descending=True)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_float_precision(4)
    pl.Config.set_tbl_width_chars(200)
    print("\n=== Top feature importances (gain, summed over folds) ===")
    print(imp.head(25))

    pairs = pairs.select("s1_id", "cand_id", "code", "label", "source").with_columns(prob=pl.Series(oof))
    pairs.write_parquet(out_dir / "oof.parquet")
    kept = many_to_one(pairs)
    code, prob = kept["code"].to_numpy(), kept["prob"].to_numpy()
    source, label = kept["source"].to_numpy(), kept["label"].to_numpy().astype(float)
    n_true = universe["n_true"].to_numpy().astype(float)
    (g_thr, g_score), (s_thr, s_score) = tune_thresholds(code, prob, source, label, n_true)
    print(f"\nthreshold search on OOF: global {g_thr} -> macro F0.5 {g_score:.5f}; "
          f"per-source {s_thr} -> {s_score:.5f}")
    thr = s_thr if s_score > g_score + 1e-4 else g_thr
    print(f"chosen thresholds: {thr}")

    report(universe, kept, keep_mask(prob, source, g_thr), f"OOF, global threshold {g_thr}")
    report(universe, kept, keep_mask(prob, source, s_thr), f"OOF, per-source thresholds {s_thr}")

    meta = {"features": FEATURES, "n_folds": N_FOLDS, "best_iters": best_iters, "fold_auc": aucs,
            "thresholds": thr, "global": g_thr, "per_source": s_thr,
            "oof_macro_f05": {"global": g_score, "per_source": s_score}, "lgb_params": LGB_PARAMS}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nmodels + meta -> {out_dir}; runtime {time.time() - t0:.0f}s, peak memory {peak_mem_gb():.2f} GB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
