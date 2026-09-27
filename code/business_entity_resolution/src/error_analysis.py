"""Bucket the matcher's OOF errors by pattern and estimate what fixing each bucket is worth.

Pipeline position: diagnostic after stage 5 (OOF error buckets); not needed to reproduce the submission.

Errors at the chosen threshold (model/meta.json):
  FP  kept pair that is not a true match      FN  true candidate pair that was rejected
  MISS  true pair that blocking never produced (the matcher cannot recover it at M=1)
Each error gets every bucket it fits (buckets overlap). "oracle gain" = macro F0.5 increase if
every error of that bucket were fixed (FP dropped / FN kept), all else unchanged.

Usage (from student_resource/):
    python code/business_entity_resolution/src/error_analysis.py [--sample N]
"""

import argparse
import json
import sys

import numpy as np
import polars as pl

import config
from features import load_truth_pairs
from train_matcher import keep_mask, load_universe, macro_f05, many_to_one

GENERIC_TOP = 300   # a name word is "generic" if it is among the most frequent S1 name words


def load_side(split, srcs):
    """Normalized name / address columns of the given sources of a split (one frame)."""
    cols = ["entity_id", "name_norm", "name_full", "name_nonlatin", "addr_norm", "house_nums"]
    return pl.concat([pl.scan_parquet(config.norm_path(split, s)).select(cols) for s in srcs]).collect()


def main():
    """Join the OOF errors (FP / FN at the chosen threshold, after many-to-one) with the record texts, bucket
    them by pattern and print each bucket's size and the macro F0.5 gained if it were fixed.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()
    config.use_sample(args.sample)

    meta = json.loads((config.model_dir() / "meta.json").read_text())
    oof = pl.read_parquet(config.model_dir() / "oof.parquet")
    feats = pl.read_parquet(config.features_path("train"),
                            columns=["s1_id", "cand_id", "n_tfidf", "a_tfidf", "rec_gap", "n_extra_rec",
                                     "n_extra_s1", "legal_conflict", "house_overlap"])
    # Same (deterministic) universe as train_matcher: all train S1s or its S1-group sample.
    universe = load_universe(len(feats), config.TRAIN_MAX_PAIRS)
    oof = oof.drop("code").join(universe.select("s1_id", "code"), on="s1_id")
    kept = many_to_one(oof).join(feats, on=["s1_id", "cand_id"], how="left")
    kept = kept.with_columns(keep=pl.Series(keep_mask(kept["prob"].to_numpy(), kept["source"].to_numpy(),
                                                      meta["thresholds"])))

    s1 = load_side("train", ["S1"]).rename(lambda c: c if c == "entity_id" else c + "_s")
    rec = load_side("train", ["S2", "S3"]).rename(lambda c: c if c == "entity_id" else c + "_r")
    df = (kept.join(s1, left_on="s1_id", right_on="entity_id", how="left")
              .join(rec, left_on="cand_id", right_on="entity_id", how="left"))

    # "generic" name words: the most frequent words across S1 names
    top = (s1.select(w=pl.col("name_norm_s").str.split(" ")).explode("w").filter(pl.col("w") != "")
             .group_by("w").len().sort("len", descending=True).head(GENERIC_TOP)["w"])
    words_r = pl.col("name_norm_r").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    truth = load_truth_pairs()
    rec_true_s1 = truth.select(cand_id="cand_id", true_s1="s1_id")
    df = df.join(rec_true_s1, on="cand_id", how="left")

    B = {
        "rec addr empty": pl.col("addr_norm_r") == "",
        "generic name (<=1 non-generic word)": words_r.list.eval(pl.element().is_in(top).not_()).list.sum() <= 1,
        "extra words in rec name only": (pl.col("n_extra_rec") >= 1) & (pl.col("n_extra_s1") == 0),
        "non-Latin rec name": pl.col("name_nonlatin_r"),
        "house-number conflict": (pl.col("house_nums_s").list.len() > 0) & (pl.col("house_nums_r").list.len() > 0)
                                 & (pl.col("house_overlap") == 0),
        "same addr, different name": (pl.col("a_tfidf") >= 0.8) & (pl.col("n_tfidf") < 0.5),
        "name looks like noise (addr ok)": (pl.col("n_tfidf") < 0.3) & (pl.col("a_tfidf") >= 0.7),
        "domain-style rec name (x.com)": pl.col("name_full_r").str.contains(r"\b(com|net|org|in)$")
                                         & (words_r.list.len() == 1),
        "initials-only rec name (<=3 chars)": pl.col("name_norm_r").str.len_chars() <= 3,
        "legal form conflict": pl.col("legal_conflict") == 1,
        "ambiguous (rec_gap < 0.05)": pl.col("rec_gap") < 0.05,
    }
    df = df.with_columns(**{k: v.fill_null(False) for k, v in B.items()})
    fp = df.filter(pl.col("keep") & (pl.col("label") == 0))
    fn = df.filter(~pl.col("keep") & (pl.col("label") == 1))
    fp = fp.with_columns(fp_kind=pl.when(pl.col("true_s1").is_null()).then(pl.lit("record has no true S1"))
                                   .otherwise(pl.lit("record's true S1 is another S1")))

    code, label = df["code"].to_numpy(), df["label"].to_numpy().astype(float)
    n_true = universe["n_true"].to_numpy().astype(float)
    keep = df["keep"].to_numpy()
    base = macro_f05(code, keep, label, n_true)
    n_miss = int(universe["n_true"].sum() - label.sum())
    print(f"threshold {meta['thresholds']}: macro F0.5 {base:.5f}; kept {keep.sum():,}; "
          f"FP {len(fp):,}, FN {len(fn):,}, blocking MISS {n_miss:,} true pairs")
    print("FP kinds:", fp.group_by("fp_kind").len().rows())

    rows = []
    for name in B:
        is_b = df[name].to_numpy()
        fp_b = keep & (label == 0) & is_b
        fn_b = ~keep & (label == 1) & is_b
        rows.append({"bucket": name, "FP": int(fp_b.sum()), "FN": int(fn_b.sum()),
                     "FP%": 100 * fp_b.sum() / max(len(fp), 1), "FN%": 100 * fn_b.sum() / max(len(fn), 1),
                     "all%": 100 * is_b.mean(),
                     "gain_fix_FP": macro_f05(code, keep & ~fp_b, label, n_true) - base,
                     "gain_fix_FN": macro_f05(code, keep | fn_b, label, n_true) - base})
    fp_all = keep & (label == 0)
    fn_all = ~keep & (label == 1)
    rows.append({"bucket": "ALL errors", "FP": int(fp_all.sum()), "FN": int(fn_all.sum()), "FP%": 100.0,
                 "FN%": 100.0, "all%": 100.0,
                 "gain_fix_FP": macro_f05(code, keep & ~fp_all, label, n_true) - base,
                 "gain_fix_FN": macro_f05(code, keep | fn_all, label, n_true) - base})
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_fmt_str_lengths(45)
    pl.Config.set_float_precision(4)
    print(pl.DataFrame(rows))

    for name, d in (("FP", fp), ("FN", fn)):
        print(f"\n--- 12 random {name} ---")
        for r in d.sample(min(12, len(d)), seed=config.SEED).iter_rows(named=True):
            tags = [k for k in B if r[k]]
            extra = f" [{r['fp_kind']}]" if name == "FP" else ""
            print(f"p={r['prob']:.2f}{extra} {tags}\n   S1 : {r['name_full_s']} | {r['addr_norm_s']}\n"
                  f"   rec: {r['name_full_r']} | {r['addr_norm_r']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
