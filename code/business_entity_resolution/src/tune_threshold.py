"""Distractor-aware threshold tuning on the OOF predictions, plus the empty-address rule check.

Pipeline position: stage 6 (threshold choice on the matcher's OOF); predict.py reads its threshold.json.

Why: test has ~5.8 S2/S3 records per S1 against ~4.7 in train, while true matches per S1 look the
same (~3.46), so a test S1 meets roughly twice as many distractor records. The OOF-optimal
threshold was tuned at train density and is likely too low for test.

Distractor weighting: every kept OOF NEGATIVE pair counts as w false positives in its S1's
precision, i.e. expected false positives per S1 scale by w (w = 1 is the plain metric):
    P = TP / (TP + w * FP)      R = TP / n_true      F0.5 as usual
    singleton S1: 1 if nothing is kept for it, else 0   (w * 0 FP is still 0)
For each w in --weights the best global threshold is found on a 0.01 grid.

Rule C (empty-address rule): force-accept a candidate whose record address is empty, whose
legal-suffix-stripped name equals the S1's exactly (no legal-form conflict), and whose name is
unique among the country's S1 names. Kept only if it raises OOF macro F0.5 at the chosen w.

Writes config.model_dir()/threshold.json:
    {"weights": {w: {"threshold", "f05"}}, "chosen_w", "threshold", "variants": [...],
     "rule_c": bool, ...}
and exports the OOF pair probabilities + labels and the universe to the output dir so the
tuning can be redone locally.

Usage (from student_resource/):
    python code/business_entity_resolution/src/tune_threshold.py [--sample N] [--weights 1 1.5 2]
"""

import argparse
import json
import sys
import time

import numpy as np
import polars as pl

import config
from memory import peak_mem_gb
from train_matcher import GRID, load_universe, many_to_one

BETA2 = 0.25
FIXED_VARIANTS = [0.69, 0.75, 0.80]   # thresholds always written as variant submissions
SPARE_VARIANT = 0.85                  # used when the tuned threshold duplicates a fixed one


def per_s1_f05_w(code, keep, label, n_true, w=1.0):
    """Macro F0.5 per S1 with kept negatives weighted by w (see module docstring)."""
    n = len(n_true)
    pos = keep & (label > 0)
    neg = keep & (label == 0)
    tp = np.bincount(code[pos], minlength=n).astype(float)
    fp = np.bincount(code[neg], minlength=n).astype(float)
    # distractor weighting: test has more records per S1 than train, so each false positive is counted w times
    # (w = 2 chosen); this pushes the threshold up to protect precision on the denser test set
    denom = tp + w * fp
    p = np.divide(tp, denom, out=np.zeros(n), where=denom > 0)
    r = np.divide(tp, n_true, out=np.zeros(n), where=n_true > 0)
    f = np.divide((1 + BETA2) * p * r, BETA2 * p + r, out=np.zeros(n), where=(p + r) > 0)
    return np.where(n_true == 0, ((tp + fp) == 0).astype(float), f)


def best_threshold(code, prob, label, n_true, w, forced=None):
    """Grid-search the threshold maximising the distractor-weighted macro F0.5.

    Args: code (S1 index per pair), prob, label, n_true (true matches per S1), w (weight of each
    false positive), forced (pairs always kept, optional).
    Returns: (best threshold, its macro F0.5, the score of every grid threshold).
    """
    forced = np.zeros(len(prob), bool) if forced is None else forced
    scores = [per_s1_f05_w(code, (prob >= t) | forced, label, n_true, w).mean() for t in GRID]
    i = int(np.argmax(scores))
    return float(GRID[i]), float(scores[i]), scores


def rule_c_mask(pairs, split):
    """Rule C for (s1_id, cand_id) pairs with feature columns a_empty_rec, n_exact, legal_conflict.

    Uniqueness is checked on name_norm over all of the S1's country (normalised S1 parquet)."""
    s1 = pl.read_parquet(config.norm_path(split, "S1"), columns=["entity_id", "country", "name_norm"])
    counts = s1.filter(pl.col("name_norm") != "").group_by("country", "name_norm").agg(n_same=pl.len())
    uniq = s1.join(counts, on=["country", "name_norm"], how="left").select(
        s1_id="entity_id", unique_name=pl.col("n_same") == 1)
    d = pairs.select("s1_id", "a_empty_rec", "n_exact", "legal_conflict").join(uniq, on="s1_id", how="left")
    m = ((d["a_empty_rec"] == 1) & (d["n_exact"] == 1) & (d["legal_conflict"].fill_null(0) == 0)
         & d["unique_name"].fill_null(False))
    return m.fill_null(False).to_numpy()


def variants_for(threshold):
    """Thresholds of the variant submissions: the fixed ones plus `threshold` (a spare if it is already one)."""
    out = list(FIXED_VARIANTS)
    out.append(SPARE_VARIANT if any(abs(threshold - v) < 1e-9 for v in out) else threshold)
    return sorted(set(round(v, 2) for v in out))


def main():
    """Re-score the matcher OOF (many-to-one, pre-filtered pairs rejected) for each distractor weight, check the
    empty-address rule, and write threshold.json (chosen threshold at --chosen_w, variants).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--weights", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--chosen_w", type=float, default=2.0, help="density weight used for the chosen threshold")
    args = ap.parse_args()
    config.use_sample(args.sample)
    t0 = time.time()

    oof = pl.read_parquet(config.model_dir() / "oof.parquet")
    feats = pl.read_parquet(config.features_path("train"),
                            columns=["s1_id", "cand_id", "a_empty_rec", "n_exact", "legal_conflict"])
    universe = load_universe(len(feats), config.TRAIN_MAX_PAIRS)       # same universe as train_matcher
    oof = oof.drop("code").join(universe.select("s1_id", "code", "country"), on="s1_id")
    kept = many_to_one(oof).join(feats, on=["s1_id", "cand_id"], how="left")
    # pre-filter: pairs it drops are never scored by the matcher, i.e. always rejected
    kp = config.prefilter_keep_path("train")
    if config.PREFILTER and kp.exists():
        pf = pl.read_parquet(kp).with_columns(pf=pl.lit(True))
        kept = kept.join(pf, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("pf").fill_null(False))
        print(f"pre-filter: {int(kept['pf'].sum()):,} of {len(kept):,} OOF pairs kept; the others count as rejected")
    else:
        kept = kept.with_columns(pf=pl.lit(True))
    kept = kept.with_columns(prob=pl.when("pf").then("prob").otherwise(0.0))
    code, prob = kept["code"].to_numpy(), kept["prob"].to_numpy()
    label = kept["label"].to_numpy().astype(float)
    n_true = universe["n_true"].to_numpy().astype(float)
    print(f"OOF: {len(kept):,} pairs over {len(universe):,} S1s "
          f"({int((universe['n_true'] == 0).sum()):,} singletons)")

    rows, weights = [], {}
    for w in args.weights:
        t, f, _ = best_threshold(code, prob, label, n_true, w)
        weights[str(w)] = {"threshold": t, "f05": f}
        row = {"w": w, "best_threshold": t, "macroF0.5@best": f}
        for v in FIXED_VARIANTS:
            row[f"F0.5@{v:.2f}"] = per_s1_f05_w(code, prob >= v, label, n_true, w).mean()
        rows.append(row)
    pl.Config.set_float_precision(5)
    pl.Config.set_tbl_width_chars(200)
    print("\n=== Distractor-weighted OOF macro F0.5 (kept negatives count w times) ===")
    print(pl.DataFrame(rows))

    chosen_w = args.chosen_w
    thr = weights[str(chosen_w)]["threshold"] if str(chosen_w) in weights else \
        best_threshold(code, prob, label, n_true, chosen_w)[0]

    # Rule C: evaluated at the chosen density, with the threshold re-tuned with the rule on
    rc = rule_c_mask(kept, "train") & kept["pf"].to_numpy()
    base = per_s1_f05_w(code, prob >= thr, label, n_true, chosen_w).mean()
    t_rc, f_rc, _ = best_threshold(code, prob, label, n_true, chosen_w, forced=rc)
    newly = rc & (prob < t_rc)
    prec_new = label[newly].mean() if newly.any() else float("nan")
    use_rc = f_rc > base + 1e-4
    print(f"\nrule C (empty address + exact unique name): matches {int(rc.sum()):,} pairs, "
          f"{int(newly.sum()):,} would be newly accepted (precision {prec_new:.3f}); "
          f"w={chosen_w}: without {base:.5f} @ {thr:.2f}, with {f_rc:.5f} @ {t_rc:.2f} -> "
          f"{'USE' if use_rc else 'SKIP'}")
    if use_rc:
        thr = t_rc

    by_country = []
    for (country,), g in universe.group_by("country"):
        mask = np.isin(code, g["code"].to_numpy())
        keep = (prob >= thr) | (rc if use_rc else False)
        f_all = per_s1_f05_w(code, keep, label, n_true, chosen_w)
        f_w1 = per_s1_f05_w(code, keep, label, n_true, 1.0)
        idx = g["code"].to_numpy()
        by_country.append({"country": country, f"F0.5 w={chosen_w}": f_all[idx].mean(), "F0.5 w=1": f_w1[idx].mean(),
                           "pairs": int(mask.sum())})
    print(f"\nchosen: w={chosen_w}, threshold {thr:.2f}, rule C {'on' if use_rc else 'off'}")
    print(pl.DataFrame(by_country).sort("country"))

    variants = variants_for(thr)
    out = {"weights": weights, "chosen_w": chosen_w, "threshold": thr, "variants": variants, "rule_c": bool(use_rc),
           "rule_c_eval": {"pairs": int(rc.sum()), "newly_accepted": int(newly.sum()),
                           "newly_precision": None if np.isnan(prec_new) else float(prec_new),
                           "f05_without": float(base), "f05_with": float(f_rc), "threshold_with": t_rc}}
    (config.model_dir() / "threshold.json").write_text(json.dumps(out, indent=2))
    print(f"variants to write: {variants}")

    # export OOF probabilities + labels and the universe for local re-tuning
    export = config.CACHE_DIR / "oof_export" if args.sample else config.OUTPUT_DIR   # output/ is for test runs
    export.mkdir(parents=True, exist_ok=True)
    kept.select("s1_id", rec_id="cand_id", prob="prob", label="label", source="source",
                rule_c=pl.Series(rc), prefilter_kept="pf").write_parquet(export / "oof_pair_probs.parquet", compression="zstd")
    universe.select("s1_id", "country", "n_true").write_parquet(export / "oof_universe.parquet", compression="zstd")
    print(f"-> {config.model_dir() / 'threshold.json'}; OOF probs -> {export}; "
          f"{time.time() - t0:.0f}s, peak {peak_mem_gb():.2f} GB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
