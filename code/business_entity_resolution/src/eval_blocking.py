"""Evaluate blocking recall on a held-out 10% of train S1 entities.

Pipeline position: diagnostic for stage 2 (blocking recall on held-out train S1s); not needed to reproduce the
submission.

The held-out split is made once (fixed seed, split by S1 id) and saved to
config.holdout_path() so later stages reuse the same split.

Pools:
  heldout  S1 index = held-out S1s only; records = ALL train S2/S3 (default).
  all      S1 index = all train S1s; records = ALL train S2/S3; metrics on held-out S1s.
           This mirrors test conditions, where every record's true S1 is in the index.

Usage (from student_resource/):
    python code/business_entity_resolution/src/eval_blocking.py [--pool heldout|all] [--sample N]
"""

import argparse
import sys
import time

import numpy as np
import polars as pl

import config
from blocking import BlockingParams, run_blocking
from memory import peak_mem_gb

N_MISSED_EXAMPLES = 20


def load_holdout():
    """The held-out 10% of train S1 ids (drawn once with config.SEED and cached)."""
    path = config.holdout_path()
    if path.exists():
        return pl.read_parquet(path)
    # Drawn from the normalised train S1 ids, so it also works on a --sample cache.
    ids = pl.read_parquet(config.norm_path("train", "S1"), columns=["entity_id"])["entity_id"].sort()
    mask = np.random.default_rng(config.SEED).random(len(ids)) < config.HOLDOUT_FRAC
    ho = pl.DataFrame({"entity_id": ids.filter(pl.Series(mask))})
    path.parent.mkdir(parents=True, exist_ok=True)
    ho.write_parquet(path)
    return ho


def load_truth(holdout):
    """True (s1_id, cand_id) pairs of the held-out S1s and their number of true matches.

    Returns: (pairs frame, frame s1_id -> n_true).
    """
    gt = pl.read_csv(config.gt_path(), separator="\t", quote_char=None, infer_schema=False).fill_null("")  # empty ID list stays ""
    gt = gt.rename({"source1_entity_id": "s1_id"}).join(holdout.rename({"entity_id": "s1_id"}), on="s1_id")
    n_true = gt.select("s1_id", n_true=pl.when(pl.col("matched_entity_ids") == "").then(0)
                       .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1))
    pairs = (gt.filter(pl.col("matched_entity_ids") != "")
               .select("s1_id", cand_id=pl.col("matched_entity_ids").str.split(",")).explode("cand_id", empty_as_null=True))
    return pairs, n_true


def metrics(cands, truth, s1_info, m_values):
    """Per-country + overall blocking metrics for each M."""
    rows = []
    for m in m_values:
        c = cands.filter(pl.col("rank") <= m).select("s1_id", "cand_id")
        c = c.join(truth.with_columns(t=pl.lit(1)), on=["s1_id", "cand_id"], how="left")
        per = c.group_by("s1_id").agg(n_cand=pl.len(), n_hit=pl.col("t").sum())
        df = s1_info.join(per, on="s1_id", how="left").with_columns(
            pl.col("n_cand").fill_null(0), pl.col("n_hit").fill_null(0))
        for country, g in list(df.group_by("country")) + [(("ALL",), df)]:
            nonsingle = g.filter(pl.col("n_true") > 0)
            single = g.filter(pl.col("n_true") == 0)
            rows.append({
                "country": country[0], "M": m, "S1": len(g),
                "recall": g["n_hit"].sum() / max(g["n_true"].sum(), 1),
                "all_covered%": 100 * (nonsingle["n_hit"] == nonsingle["n_true"]).mean(),
                "avg_cands": g["n_cand"].mean(),
                "p95_cands": float(np.percentile(g["n_cand"].to_numpy(), 95)),
                "max_cands": g["n_cand"].max(),
                "singleton_empty%": 100 * (single["n_cand"] == 0).mean() if len(single) else float("nan"),
            })
    return pl.DataFrame(rows).sort(["country", "M"])


def main():
    """Block the train records, then report per stage (keys, pre-cap, gather, top-M) the share of true
    held-out pairs still reachable, why the rest was lost, and missed examples.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=("heldout", "all"), default="heldout")
    ap.add_argument("--reuse", action="store_true", help="skip blocking, re-evaluate existing pairs")
    ap.add_argument("--sample", type=int, default=0, help="use the normalize.py --sample N cache")
    ap.add_argument("--m_values", type=int, nargs="+", default=[1, 2], help="top-M values to evaluate")
    ap.add_argument("--missed_country", default="India", help="country for the missed-pair examples")
    for f, v in vars(BlockingParams()).items():
        ap.add_argument(f"--{f}", type=type(v), default=v)
    args = ap.parse_args()
    params = BlockingParams(**{f: getattr(args, f) for f in vars(BlockingParams())})
    params.top_m = max(args.m_values)
    config.use_sample(args.sample)

    t0 = time.time()
    holdout = load_holdout()
    truth, n_true = load_truth(holdout)
    s1_country = pl.scan_parquet(config.norm_path("train", "S1")).select(s1_id="entity_id", country="country")
    s1_info = n_true.lazy().join(s1_country, on="s1_id").collect()
    print(f"held-out S1: {len(holdout):,} ({s1_info['country'].value_counts().sort('country').rows()}), "
          f"true pairs: {len(truth):,}, singletons: {(s1_info['n_true'] == 0).sum():,}")

    out_dir = config.CACHE_DIR / "blocking" / f"eval_{args.pool}"
    status_path = out_dir / "truth_status.parquet"
    if not args.reuse:
        _, status = run_blocking("train", out_dir, params,
                                 s1_ids=holdout if args.pool == "heldout" else None, truth=truth)
        status.write_parquet(status_path)
    status = pl.read_parquet(status_path)
    t_block = time.time() - t0

    cands = (pl.scan_parquet(out_dir / "pairs_*.parquet").select("s1_id", "cand_id", "rank")
               .join(holdout.lazy().rename({"entity_id": "s1_id"}), on="s1_id", how="semi").collect())

    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_float_precision(4)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_fmt_str_lengths(90)
    print(f"\n=== Blocking metrics, pool={args.pool}, held-out S1s ===")
    print(metrics(cands, truth, s1_info, args.m_values))

    print("\n=== Stage recall: fraction of true pairs surviving each stage ===")
    print(stage_table(status, params, args.m_values))
    print("key-stage misses by reason (fraction of all true pairs): no_shared_key = the pair shares no key at all, "
          "only_oversized = shares only keys over the cap (lost to caps), not_selected = shares a usable key "
          "the record did not look up")

    print_missed(status, args.missed_country, max(args.m_values))
    print(f"\nruntime: {time.time() - t0:.0f}s (blocking {t_block:.0f}s), peak memory: {peak_mem_gb():.2f} GB")


def stage_table(status, params, m_values):
    """Per-country recall at each blocking stage and the miss reasons, as a table."""
    rows = []
    for country, g in list(status.group_by("country")) + [(("ALL",), status)]:
        n = len(g)
        row = {"country": country[0], "true_pairs": n,
               "keys": g["in_keys"].mean(), f"precap{params.pre_cap}": g["in_precap"].mean(),
               f"gather{params.gather_k}": g["in_gather"].mean()}
        for m in m_values:
            row[f"top{m}"] = (g["rank"] <= m).sum() / n
        for reason in ("no_shared_key", "only_oversized", "not_selected"):
            row[f"miss:{reason}"] = (g["miss_reason"] == reason).sum() / n
        rows.append(row)
    return pl.DataFrame(rows).sort("country")


def raw_rows(ids):
    """Raw train rows (all sources) of the given entity ids."""
    frames = [
        pl.scan_csv(config.source_path("train", s), separator="\t", quote_char=None, infer_schema=False)
          .filter(pl.col("entity_id").is_in(ids)).collect()
        for s in config.SOURCES
    ]
    return pl.concat(frames).fill_null("")


def print_missed(status, country, m):
    """Print sampled true pairs of `country` not selected at top-M, with the stage that lost them."""
    missed = status.filter((pl.col("country") == country) & (pl.col("rank").is_null() | (pl.col("rank") > m)))
    print(f"\n=== {min(N_MISSED_EXAMPLES, len(missed))} of {len(missed):,} {country} true pairs missed at M={m} ===")
    if len(missed) == 0:
        return
    ex = missed.sample(min(N_MISSED_EXAMPLES, len(missed)), seed=config.SEED).with_columns(
        lost_at=pl.when(~pl.col("in_keys")).then(pl.lit("keys: ") + pl.col("miss_reason"))
                  .when(~pl.col("in_precap")).then(pl.lit("pre-cap cut"))
                  .when(~pl.col("in_gather")).then(pl.lit("gather cut"))
                  .otherwise(pl.lit("top-M cut (rank ") + pl.col("rank").cast(pl.String).fill_null(">M") + ")"))
    raw = raw_rows(pl.concat([ex["s1_id"], ex["cand_id"]]).to_list()).select(
        "entity_id", "business_name", "business_address")
    raw = {r[0]: r[1:] for r in raw.iter_rows()}
    for i, r in enumerate(ex.iter_rows(named=True), 1):
        s1n, s1a = raw.get(r["s1_id"], ("?", "?"))
        cn, ca = raw.get(r["cand_id"], ("?", "?"))
        print(f"\n{i:2d}. lost at: {r['lost_at']}")
        print(f"    S1  {r['s1_id']:<14} {s1n:<45} | {s1a}")
        print(f"    rec {r['cand_id']:<14} {cn:<45} | {ca}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
