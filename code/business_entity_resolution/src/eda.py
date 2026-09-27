"""Exploratory data analysis for the business entity resolution challenge.

Pipeline position: exploratory analysis before the pipeline; not needed to reproduce the submission.

Streams every TSV in chunks (the data is ~2.5 GB and RAM is limited), so no file is
ever fully materialised. Run from the student_resource/ directory:

    python code/business_entity_resolution/src/eda.py
"""

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
CHUNK = 500_000
N_SAMPLE = 10
N_GROUPS = 15
TOP_TOKENS = 30

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset"
FILES = {
    ("train", "S1"): DATA / "train" / "train_source1.tsv",
    ("train", "S2"): DATA / "train" / "train_source2.tsv",
    ("train", "S3"): DATA / "train" / "train_source3.tsv",
    ("test", "S1"): DATA / "test" / "test_source1.tsv",
    ("test", "S2"): DATA / "test" / "test_source2.tsv",
    ("test", "S3"): DATA / "test" / "test_source3.tsv",
}
GT_PATH = DATA / "train" / "train_ground_truth.tsv"

# Split on whitespace and punctuation. Deliberately not \w+, which would break
# Devanagari words apart at combining vowel signs.
TOKEN_SPLIT = re.compile(r"[\s,.;:()\[\]{}/\\#&'\"!?*+=|<>_~`@$%^-]+")

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.max_columns", 20)


def read_tsv(path, **kw):
    """Read a challenge TSV as strings (no NA parsing, no quoting); extra kwargs go to pandas."""
    return pd.read_csv(
        path, sep="\t", keep_default_na=False, dtype=str,
        quoting=csv.QUOTE_NONE, encoding="utf-8", **kw,
    )


def header(title):
    """Print a section banner."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def tokenize(text):
    """Lowercase and split on non-alphanumerics; empty tokens dropped."""
    return [t for t in TOKEN_SPLIT.split(text.lower()) if t]


def length_summary(arr):
    """n, % empty, mean and percentiles of an array of lengths."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return {}
    return {
        "n": arr.size,
        "empty%": round(100 * (arr == 0).mean(), 2),
        "mean": round(arr.mean(), 1),
        "p5": int(np.percentile(arr, 5)),
        "p50": int(np.percentile(arr, 50)),
        "p95": int(np.percentile(arr, 95)),
        "max": int(arr.max()),
    }


# --------------------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------------------

def analyse_ground_truth(rng):
    """Print match-list statistics of the train ground truth and draw groups to display.

    Returns: (set of S1 ids, sampled groups, ids whose rows must be fetched).
    """
    header("3. GROUND TRUTH STATS")
    gt = read_tsv(GT_PATH)
    n = len(gt)
    has = gt["matched_entity_ids"].str.len() > 0
    sizes = np.where(has, gt["matched_entity_ids"].str.count(",") + 1, 0)
    n_s2 = np.where(has, gt["matched_entity_ids"].str.count("S2-"), 0)
    n_s3 = np.where(has, gt["matched_entity_ids"].str.count("S3-"), 0)

    print(f"S1 entities in ground truth: {n:,} (unique: {gt['source1_entity_id'].nunique():,})")
    print(f"Singletons (empty match list): {(~has).sum():,} ({100 * (~has).mean():.2f}%)")
    print(f"Total matched ids: {sizes.sum():,}  |  mean per non-singleton: {sizes[has].mean():.2f}")

    print("\nMatch-list size distribution (including 0 = singleton):")
    vc = pd.Series(sizes).value_counts().sort_index()
    dist = pd.DataFrame({"count": vc, "pct": (100 * vc / n).round(2)})
    dist.index.name = "size"
    print(dist.head(15).to_string())
    if len(dist) > 15:
        print(f"  ... sizes > {dist.index[14]}: {dist['count'].iloc[15:].sum():,} entities, max size = {dist.index.max()}")

    print(f"\nMatched ids from S2: {n_s2.sum():,} ({100 * n_s2.sum() / sizes.sum():.2f}%)")
    print(f"Matched ids from S3: {n_s3.sum():,} ({100 * n_s3.sum() / sizes.sum():.2f}%)")
    comp = pd.Series(np.select(
        [~has, (n_s2 > 0) & (n_s3 > 0), n_s2 > 0], ["singleton", "S2+S3", "S2 only"], "S3 only"
    )).value_counts()
    print("\nPer-entity composition:")
    print(pd.DataFrame({"count": comp, "pct": (100 * comp / n).round(2)}).to_string())
    print("\nS2-per-entity distribution (non-singletons):")
    print(pd.Series(n_s2[has]).value_counts().sort_index().head(10).to_string())
    print("S3-per-entity distribution (non-singletons):")
    print(pd.Series(n_s3[has]).value_counts().sort_index().head(10).to_string())

    header("4. S2/S3 IDS MATCHED TO MORE THAN ONE S1")
    exploded = gt.loc[has, "matched_entity_ids"].str.split(",").explode()
    id_counts = exploded.value_counts()
    multi = id_counts[id_counts > 1]
    print(f"Distinct matched S2/S3 ids: {len(id_counts):,}")
    print(f"Ids matched to >1 S1: {len(multi):,} "
          f"(S2: {multi.index.str.startswith('S2-').sum():,}, S3: {multi.index.str.startswith('S3-').sum():,})")
    if len(multi):
        print("Examples:", multi.head(10).to_dict())
    del exploded, id_counts

    # Pick matched groups for the side-by-side view.
    pick = gt[has].sample(N_GROUPS, random_state=SEED)
    groups = {r.source1_entity_id: r.matched_entity_ids.split(",") for r in pick.itertuples()}
    wanted = set(groups) | {m for ms in groups.values() for m in ms}
    return set(gt["source1_entity_id"]), groups, wanted


# --------------------------------------------------------------------------------------
# Streaming pass over source files
# --------------------------------------------------------------------------------------

def scan_sources(wanted_ids, rng):
    """One chunked pass per file collecting counts, samples, lengths and token counts."""
    counts = {}                                   # (split, src) -> Counter(country)
    samples = {}                                  # (split, src) -> DataFrame with rand key
    lengths = defaultdict(lambda: defaultdict(list))  # (split, src, country) -> field -> [arrays]
    tokens = defaultdict(Counter)                 # (split, country, field) -> Counter
    found = []                                    # rows for the side-by-side view
    s1_ids = {}

    for (split, src), path in FILES.items():
        print(f"  scanning {path.relative_to(ROOT)} ...", file=sys.stderr, flush=True)
        cnt = Counter()
        keep = None
        ids = set() if src == "S1" else None
        for chunk in read_tsv(path, chunksize=CHUNK):
            cnt.update(chunk["country"].value_counts().to_dict())
            if ids is not None:
                ids.update(chunk["entity_id"])

            # Reservoir-style sampling: random key per row, keep 10 smallest per country.
            chunk = chunk.assign(_r=rng.random(len(chunk)))
            top = chunk.sort_values("_r").groupby("country", sort=False).head(N_SAMPLE)
            keep = top if keep is None else (
                pd.concat([keep, top]).sort_values("_r").groupby("country", sort=False).head(N_SAMPLE)
            )

            if split == "train":
                hit = chunk[chunk["entity_id"].isin(wanted_ids)]
                if len(hit):
                    found.append(hit.drop(columns="_r"))

            for field, col in (("name", "business_name"), ("address", "business_address")):
                chunk[f"_len_{field}"] = chunk[col].str.len()
            for country, g in chunk.groupby("country", sort=False):
                for field, col in (("name", "business_name"), ("address", "business_address")):
                    lengths[(split, src, country)][field].append(g[f"_len_{field}"].to_numpy())
                    c = tokens[(split, country, field)]
                    for text in g[col]:
                        c.update(tokenize(text))
        counts[(split, src)] = cnt
        samples[(split, src)] = keep.drop(columns="_r")
        if ids is not None:
            s1_ids[split] = ids

    found = pd.concat(found) if found else pd.DataFrame()
    return counts, samples, lengths, tokens, found, s1_ids


# --------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------

def report_counts(counts):
    """Print row counts per split / source / country."""
    header("1. ROW COUNTS PER FILE AND PER COUNTRY")
    rows = []
    for (split, src), cnt in counts.items():
        row = {"split": split, "source": src, "total": sum(cnt.values())}
        row.update(cnt)
        rows.append(row)
    df = pd.DataFrame(rows).fillna(0)
    cols = ["split", "source", "total"] + sorted(c for c in df.columns if c not in ("split", "source", "total"))
    df = df[cols]
    for c in cols[2:]:
        df[c] = df[c].astype(int)
    print(df.to_string(index=False))


def report_samples(samples, split_filter=None, country_filter=None, title=None):
    """Print sampled rows per split / source / country (optionally filtered)."""
    header(title)
    for (split, src), df in samples.items():
        if split_filter and split != split_filter:
            continue
        for country, g in df.groupby("country"):
            if country_filter and country != country_filter:
                continue
            print(f"\n--- {split} {src} | country={country} ---")
            print(g[["entity_id", "business_name", "business_address"]].to_string(index=False))


def report_groups(groups, found):
    """Print sampled S1s next to their true S2 / S3 matches."""
    header(f"5. {N_GROUPS} MATCHED GROUPS: S1 NEXT TO ITS S2/S3 MATCHES")
    lookup = found.drop_duplicates("entity_id").set_index("entity_id")
    for s1, matches in groups.items():
        ids = [s1] + matches
        view = lookup.reindex(ids)[["business_name", "business_address", "country"]].fillna("<missing>")
        view.index = [("  S1 >" if i == s1 else "      ") + " " + i for i in ids]
        print()
        print(view.to_string())


def report_lengths(lengths):
    """Print name / address length statistics per split / source / country."""
    header("7a. NAME / ADDRESS LENGTH STATS (characters) PER SPLIT, SOURCE, COUNTRY")
    for field in ("name", "address"):
        rows = []
        for (split, src, country), d in sorted(lengths.items()):
            s = length_summary(np.concatenate(d[field]))
            rows.append({"split": split, "source": src, "country": country, **s})
        print(f"\n[{field}]")
        print(pd.DataFrame(rows).to_string(index=False))


def report_tokens(tokens):
    """Print the most frequent tokens per split / country / field."""
    header(f"7b. TOP {TOP_TOKENS} TOKENS PER SPLIT AND COUNTRY (all sources pooled, lowercased)")
    for (split, country, field), c in sorted(tokens.items()):
        total = sum(c.values())
        top = ", ".join(f"{t}({100 * n / total:.1f}%)" for t, n in c.most_common(TOP_TOKENS))
        print(f"\n[{split} | {country} | {field}] tokens={total:,} vocab={len(c):,}")
        print("  " + top)


def main():
    """Run every analysis and print the report."""
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)

    gt_s1, groups, wanted = analyse_ground_truth(rng)
    counts, samples, lengths, tokens, found, s1_ids = scan_sources(wanted, rng)

    report_counts(counts)
    train_s1 = s1_ids["train"]
    print(f"\nGT covers {len(gt_s1 & train_s1):,} / {len(train_s1):,} train S1 ids; "
          f"GT ids not in train S1: {len(gt_s1 - train_s1):,}")
    print(f"Train/test S1 id overlap: {len(train_s1 & s1_ids['test']):,}")

    report_samples(samples, split_filter="train", title=f"2. {N_SAMPLE} RANDOM ROWS PER SOURCE PER COUNTRY (train)")
    report_samples(samples, split_filter="test", title=f"2b. {N_SAMPLE} RANDOM ROWS PER SOURCE PER COUNTRY (test)")
    report_groups(groups, found)
    report_samples(samples, split_filter="test", country_filter="France",
                   title=f"6. FRANCE (test): {N_SAMPLE} RANDOM ROWS PER SOURCE")
    report_lengths(lengths)
    report_tokens(tokens)


if __name__ == "__main__":
    main()
