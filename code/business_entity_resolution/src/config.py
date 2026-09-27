"""Paths and shared settings.

Pipeline position: imported by every stage; pins the final choices (thresholds, pre-filter cutoff, CE / stacking
settings).

Override locations with environment variables (e.g. on Kaggle):
    BER_DATA_DIR    directory containing train/ and test/   (default: <repo>/dataset)
    BER_CACHE_DIR   parquet caches                          (default: <repo>/cache)
    BER_OUTPUT_DIR  submission files                        (default: <repo>/output)
    BER_WORKERS     processes used for normalisation         (default: cpu count)
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", REPO_ROOT / "dataset"))
CACHE_DIR = Path(os.environ.get("BER_CACHE_DIR", REPO_ROOT / "cache"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", REPO_ROOT / "output"))
WORKERS = int(os.environ.get("BER_WORKERS", os.cpu_count() or 1))

SEED = 42
HOLDOUT_FRAC = 0.10
TRAIN_MAX_PAIRS = 5_000_000  # matcher training: random S1 groups (all their pairs) up to this many pairs
# Global threshold of the final submission, chosen from the OOF distractor-weighted tuning
# (tune_threshold.py: best at w = 2) and a public-leaderboard check of the variants
# (run 3: 0.69 and 0.79 both 0.963). None = OOF-optimal.
FINAL_THRESHOLD = 0.79
# Candidate pre-filter (prefilter.py): the matcher only scores the blocking pairs the stage-1
# model keeps, and candidate_pairs.tsv is that kept set. False = score every blocking pair.
PREFILTER = True
# Stage-1 cutoff (p1 >= cutoff stays a candidate). None = chosen on the OOF trade-off (largest cutoff
# losing <= prefilter.MAX_LOSS macro F0.5 at w = 2 and FINAL_THRESHOLD); set it to pin a submitted run.
PREFILTER_CUTOFF = 0.025   # pinned: chosen on the full OOF in Kaggle run 4 (the submitted package)
# Run 5 (optional, unused experiment; not in run_all's default steps): second matcher without the features whose distribution depends on how many candidates surround
# an S1 or a record (test has ~5.8 records per S1 vs ~4.7 in train); models.py compares base / pairwise /
# their mean and writes their variants.
CONTEXT_FEATURES = ["rec_gap", "rec_n", "s1_n_cands", "s1_rank", "s1_gap", "key_score", "n_keys"]
# Model of output/matching_results.tsv: "base" (predict.py at FINAL_THRESHOLD), "pairwise" or "ensemble"
# (models.py rewrites it). Per-model thresholds: None = that model's OOF-best at w = 2.
FINAL_MODEL = "base"
MODEL_THRESHOLDS = {"base": FINAL_THRESHOLD, "pairwise": None, "ensemble": None}
# predict.py's variant submissions (thresholds from tune_threshold.py), as written in run 4.
BASE_VARIANTS = True

# Final submission (run CE, public LB 0.973177): cross-encoder second scorer blended with the LightGBM matcher
# on the run-4 candidates (cross_encoder.py). Model: Apache-2.0, 117.7M parameters (BERT, 12 layers).
CE_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CE_MODEL_LICENSE = "Apache-2.0"
CE_MODEL_PARAMS = 117_654_272
CE_N_TRAIN = 1_200_000     # fold-A training pairs (1 epoch, fp16, seed SEED)
CE_MAX_LEN = 96            # tokens per (S1 text, record text) pair
# Blend p = CE_W_LGBM * p_lgbm + (1 - CE_W_LGBM) * p_ce, match if p >= CE_THRESHOLD (many-to-one kept).
# Chosen by fold-B OOF + public LB: fold B (w = 2) puts 0.4-0.7 within 0.0004 (best 0.6 @ 0.69: 0.98087; 0.4 @ 0.73:
# 0.98054); on the public LB 0.6 @ 0.69 -> 0.972, 0.5 @ 0.69 -> 0.973, 0.4 @ 0.73 -> 0.973177 (submitted). Thresholds are
# each weight's fold-B best at w = 2. cross_encoder.py --refit re-chooses both on fold B.
# CE_W_LGBM also defines the "blend" input feature of the stacking below: keep it at 0.4 to reproduce the final run.
CE_W_LGBM = 0.4
CE_THRESHOLD = 0.73

# FINAL SUBMISSION (public LB 0.978359): stack.py --ce2_dir, a second-level LightGBM trained on fold B over the matcher
# features, p_lgbm (OOF), p_ce (CE v1), p_ce2 (CE v2: same model, other seed / sample / max_len), their blends and
# S1-level context of the probabilities. Variant "stack_ctx2", threshold 0.83 = its fold-B best at distractor weight 2
# (fold B: 0.98401 vs stack_ctx 0.98342). stack.py --refit re-chooses the threshold on fold B.
FINAL_STAGE = "stack"
STACK_VARIANT = "stack_ctx2"
STACK_THRESHOLD = 0.83
# Previous submission (public LB 0.977): stack.py without --ce2_dir, variant "stack_ctx" @ 0.78 (its fold-B best).
# Used as the pinned threshold of that mode and as the reference counted in the --ce2_dir log.
STACK_V1_THRESHOLD = 0.78

SOURCES = ("S1", "S2", "S3")
SPLITS = ("train", "test")


def use_sample(n):
    """Point every cache path at a separate sample cache (see normalize.py --sample)."""
    global CACHE_DIR
    if n:
        CACHE_DIR = CACHE_DIR / f"sample_{n}"


def source_path(split, src):
    """Raw TSV of a split / source, e.g. ('test', 'S2') -> DATA_DIR/test/test_source2.tsv."""
    return DATA_DIR / split / f"{split}_source{src[-1]}.tsv"


def gt_path():
    """Train ground truth TSV (source1_entity_id, matched_entity_ids)."""
    return DATA_DIR / "train" / "train_ground_truth.tsv"


def norm_path(split, src):
    """Normalized parquet of a split / source written by normalize.py."""
    return CACHE_DIR / "normalized" / f"{split}_{src}.parquet"


def holdout_path():
    """Held-out train S1 ids used by eval_blocking.py."""
    return CACHE_DIR / "splits" / "holdout_s1.parquet"


def subset_path(split, n):
    """Random S1 subset of size n (blocking experiments with --s1_subset)."""
    return CACHE_DIR / "splits" / f"{split}_s1_subset_{n}.parquet"


def blocking_dir(split):
    """Folder of blocking.py's per-country pair parquets for a split."""
    return CACHE_DIR / "blocking" / split


def features_path(split):
    """Pair-feature parquet of a split written by features.py."""
    return CACHE_DIR / "features" / f"{split}.parquet"


def model_dir():
    """Folder of the matcher: fold models, meta.json, oof.parquet, threshold.json."""
    return CACHE_DIR / "model"


def prefilter_dir():
    """Folder of the stage-1 pre-filter models (prefilter.json + fold models)."""
    return model_dir() / "prefilter"


def prefilter_keep_path(split):
    """s1_id, cand_id of the blocking pairs kept by the pre-filter (the stage checkpoint)."""
    return CACHE_DIR / "prefilter" / f"{split}_keep.parquet"


def blocking_done(split):
    """Checkpoint marker written when a split's blocking finished for every country."""
    return blocking_dir(split) / "_DONE.json"
