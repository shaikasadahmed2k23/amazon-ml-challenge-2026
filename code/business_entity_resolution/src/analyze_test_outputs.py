"""Per-country sanity check of the test outputs against train ground-truth statistics.

Pipeline position: sanity check of the final outputs (per-country structure vs train truth).

No labels exist for test, so predicted match structure is compared with what train's ground
truth looks like (singleton rate, matches per S1, S2/S3 split, share of records with an S1).
A country that deviates strongly from the others (France: no training labels) points at a
recall or precision problem.

Reads only id lists plus the test sources' id / country columns (and raw text for the few
printed examples), so it runs in ~2-3 GB of RAM.

Usage (from student_resource/):
    python code/business_entity_resolution/src/analyze_test_outputs.py \\
        [--output-dir output] [--data-dir dataset] [--country France] [--n-examples 20]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl


def read_ids(path, split, src, cols=("entity_id", "country")):
    """Columns `cols` of a raw source TSV of a split (source given as its number)."""
    return (pl.scan_csv(Path(path) / split / f"{split}_source{src}.tsv", separator="\t", quote_char=None,
                        infer_schema=False).select(list(cols)).collect())


def id_lists(path, col):
    """source1_entity_id -> exploded (s1_id, cand_id) pairs, plus per-S1 counts."""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")
    df = df.rename({"source1_entity_id": "s1_id", col: "ids"})
    pairs = (df.filter(pl.col("ids") != "").select("s1_id", cand_id=pl.col("ids").str.split(","))
               .explode("cand_id", empty_as_null=True))
    return df.select("s1_id"), pairs


def structure(s1, pairs, rec_country, label):
    """Match-structure statistics per country for (S1 list, predicted pairs)."""
    per = pairs.group_by("s1_id").agg(
        n=pl.len(), n_s2=pl.col("cand_id").str.starts_with("S2-").sum(),
        n_s3=pl.col("cand_id").str.starts_with("S3-").sum())
    d = s1.join(per, on="s1_id", how="left").fill_null(0)
    rows = []
    for country, g in sorted(d.group_by("country"), key=lambda x: x[0][0]):
        n = g["n"].to_numpy()
        matched = g.filter(pl.col("n") > 0)
        recs = rec_country.filter(pl.col("country") == country[0])
        assigned = recs.join(pairs.select("cand_id").unique(), left_on="entity_id", right_on="cand_id",
                             how="semi")
        rows.append({
            "set": label, "country": country[0], "S1": len(g),
            "zero%": 100 * (n == 0).mean(),
            "avg": n.mean(), "p50": float(np.percentile(n, 50)), "p95": float(np.percentile(n, 95)),
            "avg|>0": matched["n"].mean() if len(matched) else float("nan"),
            "S2share%": 100 * g["n_s2"].sum() / max(g["n"].sum(), 1),
            "both_src%": 100 * ((g["n_s2"] > 0) & (g["n_s3"] > 0)).mean(),
            "both_src|>0%": 100 * ((matched["n_s2"] > 0) & (matched["n_s3"] > 0)).mean() if len(matched) else 0,
            "recs_assigned%": 100 * len(assigned) / max(len(recs), 1),
        })
    return pl.DataFrame(rows)


def candidates_table(s1, cand, match):
    """Per country: candidates per S1 (mean / p50 / p95), % S1s without candidates and accept rates."""
    per_c = cand.group_by("s1_id").agg(nc=pl.len())
    per_m = match.group_by("s1_id").agg(nm=pl.len())
    d = s1.join(per_c, on="s1_id", how="left").join(per_m, on="s1_id", how="left").fill_null(0)
    rows = []
    for country, g in sorted(d.group_by("country"), key=lambda x: x[0][0]):
        nc = g["nc"].to_numpy()
        rows.append({
            "country": country[0], "S1": len(g),
            "cands_avg": nc.mean(), "cands_p50": float(np.percentile(nc, 50)),
            "cands_p95": float(np.percentile(nc, 95)), "zero_cands%": 100 * (nc == 0).mean(),
            "accepted%": 100 * g["nm"].sum() / max(g["nc"].sum(), 1),
            "zero_match_but_cands%": 100 * ((g["nm"] == 0) & (g["nc"] > 0)).mean(),
            "all_cands_accepted%": 100 * ((g["nm"] == g["nc"]) & (g["nc"] > 0)).mean(),
        })
    return pl.DataFrame(rows)


def examples(data_dir, country, s1, cand, match, n, zero_n, seed):
    """Print `n` random S1s of `country` with their candidates (MATCH marked) plus `zero_n` S1s that
    have candidates but no match.
    """
    s1_c = s1.filter(pl.col("country") == country)
    with_c = s1_c.join(cand.select("s1_id").unique(), on="s1_id", how="semi")
    matched_ids = match.select("s1_id").unique()
    pick = with_c.sample(min(n, len(with_c)), seed=seed)
    zero = with_c.join(matched_ids, on="s1_id", how="anti")
    pick_zero = zero.sample(min(zero_n, len(zero)), seed=seed)
    chosen = pl.concat([pick.select("s1_id"), pick_zero.select("s1_id")])
    c = cand.join(chosen, on="s1_id", how="semi").join(
        match.with_columns(m=pl.lit(True)), on=["s1_id", "cand_id"], how="left")
    ids = pl.concat([chosen["s1_id"], c["cand_id"]]).unique().implode()
    raw = pl.concat([
        pl.scan_csv(Path(data_dir) / "test" / f"test_source{i}.tsv", separator="\t", quote_char=None,
                    infer_schema=False).filter(pl.col("entity_id").is_in(ids))
          .select("entity_id", "business_name", "business_address").collect()
        for i in (1, 2, 3)]).fill_null("")
    text = {r[0]: f"{r[1]} | {r[2]}" for r in raw.iter_rows()}

    def show(title, ids_):
        """Print each S1's text and its candidates, marking the matched ones."""
        print(f"\n=== {title} ===")
        for k, sid in enumerate(ids_, 1):
            rows = c.filter(pl.col("s1_id") == sid).sort("cand_id")
            print(f"\n{k:2d}. S1 {sid}: {text.get(sid, '?')}")
            for r in rows.iter_rows(named=True):
                mark = "MATCH" if r["m"] else "  -  "
                print(f"      [{mark}] {r['cand_id']:<13} {text.get(r['cand_id'], '?')}")
    show(f"{len(pick)} random {country} S1s with candidates ([MATCH] = predicted match)", pick["s1_id"].to_list())
    show(f"{len(pick_zero)} {country} S1s with candidates but ZERO predicted matches "
         f"({len(zero):,} such S1s, {100 * len(zero) / max(len(s1_c), 1):.1f}% of {country})",
         pick_zero["s1_id"].to_list())


def main():
    """Compare predicted match structure per country with train truth and print examples."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--country", default="France", help="country for the printed examples")
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--n-zero", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(2)

    # ---- train ground truth: what real match structure looks like, per country
    tr_s1 = read_ids(args.data_dir, "train", 1).rename({"entity_id": "s1_id"})
    tr_rec = pl.concat([read_ids(args.data_dir, "train", s) for s in (2, 3)])
    _, tr_pairs = id_lists(Path(args.data_dir) / "train" / "train_ground_truth.tsv", "matched_entity_ids")
    train_stats = structure(tr_s1, tr_pairs, tr_rec, "train truth")
    del tr_rec, tr_pairs

    # ---- test predictions
    te_s1_ids = read_ids(args.data_dir, "test", 1).rename({"entity_id": "s1_id"})
    te_rec = pl.concat([read_ids(args.data_dir, "test", s) for s in (2, 3)])
    s1_m, match = id_lists(Path(args.output_dir) / "matching_results.tsv", "matched_entity_ids")
    s1_c, cand = id_lists(Path(args.output_dir) / "candidate_pairs.tsv", "candidate_entity_ids")
    assert len(s1_m) == len(te_s1_ids), "matching_results.tsv does not cover every test S1"
    test_stats = structure(te_s1_ids, match, te_rec, "test predicted")
    cand_stats = structure(te_s1_ids, cand, te_rec, "test candidates")

    print("=== Match structure: train ground truth vs test predictions (and test candidates) ===")
    print("zero% = S1s with no match; avg/p50/p95 = matches per S1; avg|>0 = per S1 with any match;")
    print("S2share% = matches from S2; both_src% = S1s matched in both S2 and S3; recs_assigned% = S2/S3 "
          "records of that country attached to some S1")
    print(pl.concat([train_stats, test_stats, cand_stats]).sort(["country", "set"]))

    print("\n=== Test candidates per S1 and acceptance ===")
    print(candidates_table(te_s1_ids, cand, match))

    examples(args.data_dir, args.country, te_s1_ids, cand, match, args.n_examples, args.n_zero, args.seed)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
