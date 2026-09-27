"""Write low-threshold LB probe submissions from an earlier run's cached test pair probabilities.

Pipeline position: LB-probe helper after stage 7 (threshold probes); not needed to reproduce the submission.

Reads <run-dir>/test_pair_probs.parquet (s1_id, rec_id, prob; exactly the pairs of
<run-dir>/candidate_pairs.tsv), keeps each record for its best S1 (many-to-one), and writes
<run-dir>/probe/matching_results_t{T}.tsv for every threshold. The candidate file of the probe is the
run's own candidate_pairs.tsv, so matches stay a subset of candidates. Prints matches per S1 and the
share of S2/S3 records assigned, per country (records from the normalized test cache).

Usage (from student_resource/):
    python code/business_entity_resolution/src/probe_thresholds.py --run-dir run3 --thresholds 0.5 0.3 0.15
"""

import argparse
import sys
from pathlib import Path

import polars as pl

import config


def main():
    """Write matching_results_t<thr>.tsv for each --thresholds value from <run-dir>'s cached pair probabilities
    (many-to-one first) into <run-dir>/probe/, with per-country match counts.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="run3")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.3, 0.15])
    ap.add_argument("--probs", default="test_pair_probs.parquet")
    args = ap.parse_args()
    run = Path(args.run_dir)
    out = run / "probe"
    out.mkdir(exist_ok=True)

    s1 = pl.read_parquet(config.norm_path("test", "S1"), columns=["entity_id", "country"]).rename(
        {"entity_id": "s1_id"})
    recs = pl.concat([pl.read_parquet(config.norm_path("test", s), columns=["entity_id", "country"])
                      for s in ("S2", "S3")])
    n_rec = recs.group_by("country").agg(records=pl.len())
    ids = (pl.scan_csv(config.source_path("test", "S1"), separator="\t", quote_char=None, infer_schema=False)
             .select(s1_id="entity_id").collect())

    pairs = pl.read_parquet(run / args.probs)
    kept = (pairs.sort(["prob", "s1_id"], descending=[True, False])
                 .unique("rec_id", keep="first", maintain_order=True))
    print(f"{len(pairs):,} candidate pairs, {len(kept):,} records after many-to-one")
    for t in args.thresholds:
        m = kept.filter(pl.col("prob") >= t)
        lists = m.group_by("s1_id").agg(pl.col("rec_id").sort().str.join(",").alias("matched_entity_ids"))
        res = ids.join(lists, on="s1_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
        path = out / f"matching_results_t{t:.2f}.tsv"
        res.rename({"s1_id": "source1_entity_id"}).write_csv(path, separator="\t", quote_style="never")
        per = (m.join(s1, on="s1_id", how="left").group_by("country").agg(assigned=pl.len())
                .join(s1.group_by("country").agg(S1=pl.len()), on="country", how="right")
                .join(n_rec, on="country", how="left")
                .with_columns(matches_per_S1=pl.col("assigned") / pl.col("S1"),
                              records_per_S1=pl.col("records") / pl.col("S1"),
                              pct_records_assigned=100 * pl.col("assigned") / pl.col("records"))
                .select("country", "S1", "assigned", "matches_per_S1", "records_per_S1", "pct_records_assigned")
                .sort("country"))
        print(f"\n=== t={t:.2f}: {len(m):,} matches -> {path}")
        print(per)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pl.Config.set_tbl_width_chars(160)
    main()
