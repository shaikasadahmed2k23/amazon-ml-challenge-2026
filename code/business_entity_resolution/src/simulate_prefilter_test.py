"""Offline sanity check of the pre-filter on the TEST outputs of an earlier run (no matcher scoring).

Pipeline position: diagnostic for stage 4 on an earlier run's test outputs; not needed to reproduce the submission.

Applies a saved stage-1 model (prefilter.py) to the test blocking pairs available locally and
compares with the earlier run's files: candidates per S1 before / after, and how many of the pairs
that run MATCHED (prob >= threshold) the pre-filter would drop. Also writes the simulated files
<run-dir>/candidate_pairs_prefilter_sim.tsv and matching_results_prefilter_sim.tsv. Countries whose
test blocking pairs are not in the cache keep the earlier candidates. These files are a sanity check,
NOT a submission: the real run re-scores only the kept pairs.

Usage (from student_resource/):
    python code/business_entity_resolution/src/simulate_prefilter_test.py --run-dir run3 \\
        --model-dir cache/sample_50000/model/prefilter [--cutoff 0.006] [--threshold 0.79]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

import config
from prefilter import CUTOFFS, match_loss, predict_p1, stage1_pairs


def lists(df, col):
    """Aggregate pairs into one comma-joined, sorted cand_id list per s1_id in column `col`."""
    return df.group_by("s1_id").agg(pl.col("cand_id").sort().str.join(",").alias(col))


def main():
    """Apply the saved stage-1 model to the test blocking pairs and print, per cutoff, candidates per S1
    before / after and the matches of the earlier run that the cutoff would remove.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="run3")
    ap.add_argument("--model-dir", default=str(config.prefilter_dir()))
    ap.add_argument("--cutoff", type=float, help="default: the cutoff in <model-dir>/prefilter.json")
    ap.add_argument("--threshold", type=float, default=config.FINAL_THRESHOLD or 0.79)
    args = ap.parse_args()
    run, model_dir = Path(args.run_dir), Path(args.model_dir)
    meta = json.loads((model_dir / "prefilter.json").read_text())
    cutoff = args.cutoff if args.cutoff is not None else meta["cutoff"]

    pairs = stage1_pairs("test")
    countries = sorted(pairs["country"].unique().to_list())
    pairs = pairs.select("s1_id", "cand_id", "country").with_columns(
        p1=pl.Series(predict_p1(pairs, model_dir, meta["features"])))
    probs = pl.read_parquet(run / "test_pair_probs.parquet").rename({"rec_id": "cand_id"})
    d = pairs.join(probs, on=["s1_id", "cand_id"], how="left")
    print(f"stage-1 model {model_dir} (trained cutoff {meta['cutoff']:g}); simulated countries: {countries}; "
          f"{len(d):,} test blocking pairs, {d['prob'].null_count():,} without an earlier probability")

    s1 = pl.read_parquet(config.norm_path("test", "S1"), columns=["entity_id", "country"]).rename(
        {"entity_id": "s1_id"}).filter(pl.col("country").is_in(countries))
    rows = []
    matched = d.filter(pl.col("prob").fill_null(0) >= args.threshold)
    for c in sorted(set([cutoff] + [x for x in CUTOFFS if x <= 0.3])):
        cnt = s1.join(d.group_by("s1_id").agg(before=pl.len(), after=(pl.col("p1") >= c).sum()),
                      on="s1_id", how="left").fill_null(0)
        loss = match_loss(matched.with_columns(kept=pl.col("p1") >= c), s1).filter(pl.col("country") == "ALL")
        rows.append({"cutoff": c, "avg_before": cnt["before"].mean(), "avg_after": cnt["after"].mean(),
                     "p95_after": float(np.percentile(cnt["after"].to_numpy(), 95)),
                     "matched": loss["matched"][0], "matched_dropped": loss["dropped"][0],
                     "S1_affected": loss["S1_affected"][0], "loss_bound": loss["loss_bound"][0]})
    pl.Config.set_tbl_rows(100)
    pl.Config.set_float_precision(4)
    pl.Config.set_tbl_width_chars(200)
    table = pl.DataFrame(rows).sort("cutoff")
    print(f"\n=== Test simulation ({countries}): candidates per S1, earlier matches (prob >= {args.threshold:.2f}) dropped, loss_bound = 1 - macro F0.5 of kept vs earlier matches ===")
    print(table)

    # simulated files at the chosen cutoff (other countries: earlier candidates unchanged)
    kept = d.filter(pl.col("p1") >= cutoff)
    cand = pl.read_csv(run / "candidate_pairs.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
    sim_ids = pl.read_parquet(config.norm_path("test", "S1"), columns=["entity_id", "country"]).filter(
        pl.col("country").is_in(countries)).select(source1_entity_id="entity_id")
    new_c = lists(kept, "candidate_entity_ids").rename({"s1_id": "source1_entity_id"})
    new_m = lists(kept.filter(pl.col("prob").fill_null(0) >= args.threshold), "matched_entity_ids").rename(
        {"s1_id": "source1_entity_id"})
    for col, new, name in (("candidate_entity_ids", new_c, "candidate_pairs_prefilter_sim.tsv"),
                           ("matched_entity_ids", new_m, "matching_results_prefilter_sim.tsv")):
        base = cand.select("source1_entity_id")
        if col == "candidate_entity_ids":
            old = cand
        else:
            old = pl.read_csv(run / "matching_results.tsv", separator="\t", quote_char=None,
                              infer_schema=False).fill_null("")
        sim = sim_ids.join(new, on="source1_entity_id", how="left").with_columns(pl.col(col).fill_null(""))
        out = base.join(pl.concat([old.join(sim_ids, on="source1_entity_id", how="anti"), sim]),
                        on="source1_entity_id", how="left", maintain_order="left").with_columns(pl.col(col).fill_null(""))
        out.write_csv(run / name, separator="\t", quote_style="never")
        print(f"-> {run / name}: {len(out):,} rows")
    print("note: run3/matching_results.tsv used threshold 0.69; the simulated matching file uses "
          f"{args.threshold:.2f} on run 3's probabilities, restricted to the kept candidates")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
