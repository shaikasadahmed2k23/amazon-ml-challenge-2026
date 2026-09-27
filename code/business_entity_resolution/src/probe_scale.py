"""Measure full-scale blocking cost before committing to a full run.

Pipeline position: diagnostic before stage 2 (blocking cost at full scale); not needed to reproduce the submission.

For each country: build the FULL S1 index (optionally a random S1 subset), block a random
slice of `--n_rec` S2+S3 records of that country, and extrapolate to all of its records.
Each country runs in its own subprocess so the reported peak memory is per country.

Reported per country: key hits / record, pairs scored / record (after pre-cap), seconds
per 10k records, index + TF-IDF build time, peak memory, and the extrapolated full runtime
    t_index + t_tfidf + t_records * (records in country / records probed).
Peak memory at full scale is also estimated by adding the in-memory size of all the
country's normalised records (the probe only loads the slice).

Usage (from student_resource/):
    python code/business_entity_resolution/src/probe_scale.py --split test
    python code/business_entity_resolution/src/probe_scale.py --split train
"""

import argparse
import json
import subprocess
import sys
import time

import polars as pl

import config
from blocking import COLUMNS, BlockingParams, block_country, list_countries, load_s1, s1_subset
from memory import peak_mem_gb

RESULT_TAG = "PROBE_RESULT "


def probe_country(args, params):
    """Block a random sample of one country's records against its full S1 index.

    Returns a dict of per-record key hits / scored S1s and timings, extrapolated to all records.
    """
    country = args.country
    s1 = load_s1(args.split, country, s1_subset(args.split, args.s1_subset))
    frames = [pl.scan_parquet(config.norm_path(args.split, s)).filter(pl.col("country") == country)
              for s in ("S2", "S3")]
    ids = pl.concat([f.select("entity_id") for f in frames]).collect()["entity_id"]
    n_all = len(ids)
    pick = pl.DataFrame({"entity_id": ids.sample(min(args.n_rec, n_all), seed=config.SEED)})
    rec = (pl.concat([f.select(COLUMNS) for f in frames]).join(pick.lazy(), on="entity_id", how="semi")
             .collect().with_row_index("ridx"))
    rec_frame_gb = rec.estimated_size() / 2 ** 30 * n_all / max(len(rec), 1)
    del ids, pick

    # Warm-up on a tiny slice so numba JIT compilation is not counted as per-record cost.
    block_country(s1.head(500), rec.head(100), params, log=lambda *a, **k: None)

    t0 = time.time()
    _, stats, _ = block_country(s1, rec, params, log=lambda *a, **k: None)
    n = max(len(rec), 1)
    scale = n_all / n
    return {
        "country": country,
        "S1": len(s1),
        "records": n_all,
        "probed": len(rec),
        "hits/rec": stats.get("key_hits", 0) / n,
        "scored/rec": stats.get("hits_after_precap", 0) / n,
        "rec_with_key%": 100 * stats.get("rec_with_key", 0) / n,
        "t_index_s": stats.get("t_index", 0),
        "t_tfidf_s": stats.get("t_tfidf", 0),
        "s_per_10k": stats.get("t_records", 0) / n * 10_000,
        "probe_s": time.time() - t0,
        "peak_gb": peak_mem_gb(),
        "est_full_peak_gb": peak_mem_gb() + rec_frame_gb,
        "est_full_min": (stats.get("t_index", 0) + stats.get("t_tfidf", 0) + stats.get("t_records", 0) * scale) / 60,
    }


def main():
    """Probe every country in its own subprocess (memory is released between countries) and print the table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=config.SPLITS, default="test")
    ap.add_argument("--sample", type=int, default=0, help="use the normalize.py --sample N cache")
    ap.add_argument("--s1_subset", type=int, default=0, help="index a random N S1s (0 = all)")
    ap.add_argument("--n_rec", type=int, default=20_000, help="records probed per country")
    ap.add_argument("--countries", nargs="+", help="default: every country of the split")
    ap.add_argument("--country", help=argparse.SUPPRESS)   # internal: run one country in-process
    for f, v in vars(BlockingParams()).items():
        ap.add_argument(f"--{f}", type=type(v), default=v)
    args = ap.parse_args()
    params = BlockingParams(**{f: getattr(args, f) for f in vars(BlockingParams())})
    config.use_sample(args.sample)

    if args.country:
        print(RESULT_TAG + json.dumps(probe_country(args, params)), flush=True)
        return

    # When a subset is requested, draw it once here so the subprocesses reuse the same file.
    s1_subset(args.split, args.s1_subset)
    passthrough = [a for a in sys.argv[1:]]
    rows = []
    for country in args.countries or list_countries(args.split):
        print(f"[{country}] probing ...", flush=True)
        out = subprocess.run([sys.executable, "-u", __file__, *passthrough, "--country", country],
                             capture_output=True, text=True, encoding="utf-8")
        line = next((l for l in out.stdout.splitlines() if l.startswith(RESULT_TAG)), None)
        if line is None:
            print(f"[{country}] FAILED (exit {out.returncode}):\n{out.stderr[-3000:]}", flush=True)
            continue
        rows.append(json.loads(line[len(RESULT_TAG):]))
        print(f"[{country}] {rows[-1]}", flush=True)

    if not rows:
        return
    df = pl.DataFrame(rows)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(2)
    print(f"\n=== Blocking scale probe: split={args.split}, s1_subset={args.s1_subset or 'all'}, "
          f"block_cap={params.block_cap}, pre_cap={params.pre_cap}, n_rare={params.n_rare} ===")
    print(df)
    total = df["est_full_min"].sum()
    print(f"extrapolated full blocking runtime: {total:.0f} min ({total / 60:.2f} h) on this machine; "
          f"max est. peak {df['est_full_peak_gb'].max():.1f} GB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
