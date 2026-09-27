"""Pairwise features for the matcher, one row per blocking candidate (S1, record).

Pipeline position: stage 3; reads blocking.py's pairs and the normalized records, feeds prefilter.py and
train_matcher.py / predict.py.

Input:  blocking pairs  config.blocking_dir(split)/pairs_*.parquet  (see blocking.py)
        normalised sources  config.norm_path(split, src)
Output: config.features_path(split): s1_id, cand_id, country, [label], FEATURES...
        (`country` is kept for reporting only and is never a model feature: test has an
        unseen country.)

Features (null when a side is empty; `rank` only when blocking kept M>1 S1s per record):
  name   rapidfuzz ratio / partial / token-set / token-sort / Jaro-Winkler on the
         legal-suffix-stripped name, token-sort on the full name, char 3-gram TF-IDF
         cosine (from blocking), double-metaphone phonetic similarity, exact match,
         non-Latin flag, words in the record (S1) name missing from the other side,
         legal form conflict
  addr   the same fuzzy set + TF-IDF cosine, exact house-number match and overlap count,
         token Jaccard, house+street key / postal match, empty flag, lengths
  block  ranking score, key IDF score, keys shared, rank of the S1 for the record,
         gap to the record's best other S1, S1s scored for the record
  S1     candidates of this S1, rank of this pair's score within the S1's candidates
  source 0 = S2, 1 = S3
  words  (run 2) IDF-weighted name-word overlap (IDF from the country's S1 names), IDF mass and
         max IDF of the record's extra words / the S1's missing words, phonetic-code extra words
         (transliteration-robust), no-space name ratio ("earnosethroat.com")
  house  (run 2) first house number equal, house-number conflict (both have numbers, none shared),
         numeric distance of the first numbers, each side's numbers missing from the other

Usage (from student_resource/):
    python code/business_entity_resolution/src/features.py --split train [--sample N]
"""

import argparse
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

import config
from blocking import phonetic_names
from memory import peak_mem_gb

CHUNK = 1_000_000
FIELDS = ["entity_id", "name_full", "name_norm", "name_legal", "name_nonlatin", "addr_norm",
          "postal", "house_nums", "hs_key"]
FUZZY = {
    "ratio": fuzz.ratio,
    "partial": fuzz.partial_ratio,
    "tset": fuzz.token_set_ratio,
    "tsort": fuzz.token_sort_ratio,
    "jw": JaroWinkler.normalized_similarity,
}
BLOCK_FEATURES = ["score", "key_score", "n_keys", "rank", "rec_gap", "rec_n"]
M_DEPENDENT = ["rank"]   # constant (always 1) when blocking keeps M=1 S1 per record: dropped then
FEATURES = (
    [f"n_{k}" for k in FUZZY] + ["nf_tsort", "n_tfidf", "n_phonetic", "n_exact", "n_nonlatin",
                                 "n_extra_rec", "n_extra_s1", "n_len_s1", "n_len_rec", "legal_conflict"]
    + [f"a_{k}" for k in FUZZY] + ["a_tfidf", "house_match", "house_overlap", "a_jaccard", "hs_match",
                                   "postal_match", "a_empty_rec", "a_len_s1", "a_len_rec"]
    + BLOCK_FEATURES
    + ["s1_n_cands", "s1_rank", "s1_gap", "source"]
    + ["n_idf_jacc", "n_idf_extra_rec", "n_idf_extra_s1", "n_maxidf_extra_rec", "n_maxidf_extra_s1",
       "n_ph_extra_rec", "n_ph_extra_s1", "n_nospace_ratio"]
    + ["hn_first_eq", "hn_conflict", "hn_first_dist", "hn_s1_missing", "hn_rec_missing"]
)


def words(col):
    """Polars expression: unique non-empty words of a space-separated column."""
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()


def fuzzy(prefix, a, b, both):
    """RapidFuzz similarities (scaled to 0-1) between aligned lists a / b; NaN where `both` is False.

    Returns: dict {<prefix>_<scorer>: float32 array}.
    """
    out = {}
    for k, scorer in FUZZY.items():
        v = process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
        if k != "jw":
            v = v / 100
        out[f"{prefix}_{k}"] = np.where(both, v, np.nan).astype(np.float32)
    return out


def load_side(split, srcs, ids, country):
    """Normalized records of `srcs` in one country restricted to `ids`, plus phonetic name codes."""
    frames = [pl.scan_parquet(config.norm_path(split, s)).filter(pl.col("country") == country).select(FIELDS)
              for s in srcs]
    df = pl.concat(frames).join(ids.lazy(), on="entity_id", how="semi").collect()
    return df.with_columns(phon=pl.Series(phonetic_names(df["name_norm"].to_list()), dtype=pl.String))


def chunk_features(df, idf):
    """All string features for one chunk of joined pairs (columns suffixed _s = S1, _r = record)."""
    has_n = (df["name_norm_s"].str.len_bytes() > 0) & (df["name_norm_r"].str.len_bytes() > 0)
    has_a = (df["addr_norm_s"].str.len_bytes() > 0) & (df["addr_norm_r"].str.len_bytes() > 0)
    has_n, has_a = has_n.to_numpy(), has_a.to_numpy()
    feats = {}
    feats.update(fuzzy("n", df["name_norm_s"].to_list(), df["name_norm_r"].to_list(), has_n))
    feats["nf_tsort"] = process.cpdist(df["name_full_s"].to_list(), df["name_full_r"].to_list(),
                                       scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32) / 100
    ph_s, ph_r = df["phon_s"].to_numpy(), df["phon_r"].to_numpy()
    ph = process.cpdist(ph_s, ph_r, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32) / 100
    feats["n_phonetic"] = np.where((ph_s != "") & (ph_r != ""), ph, np.nan).astype(np.float32)
    feats.update(fuzzy("a", df["addr_norm_s"].to_list(), df["addr_norm_r"].to_list(), has_a))
    nospace = process.cpdist(df["name_norm_s"].str.replace_all(" ", "").to_list(),
                             df["name_norm_r"].str.replace_all(" ", "").to_list(),
                             scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100
    feats["n_nospace_ratio"] = np.where(has_n, nospace, np.nan).astype(np.float32)

    tok = df.select(
        n_extra_rec=words("name_norm_r").list.set_difference(words("name_norm_s")).list.len(),
        n_extra_s1=words("name_norm_s").list.set_difference(words("name_norm_r")).list.len(),
        a_inter=words("addr_norm_r").list.set_intersection(words("addr_norm_s")).list.len(),
        a_union=words("addr_norm_r").list.set_union(words("addr_norm_s")).list.len(),
        house_overlap=pl.col("house_nums_r").list.unique().list.set_intersection(
            pl.col("house_nums_s").list.unique()).list.len(),
    )
    feats["n_extra_rec"] = tok["n_extra_rec"].to_numpy().astype(np.float32)
    feats["n_extra_s1"] = tok["n_extra_s1"].to_numpy().astype(np.float32)
    feats["house_overlap"] = tok["house_overlap"].to_numpy().astype(np.float32)
    feats["a_jaccard"] = np.where(has_a, tok["a_inter"].to_numpy() / np.maximum(tok["a_union"].to_numpy(), 1),
                                  np.nan).astype(np.float32)
    feats.update(idf_features(df, idf))
    feats.update(house_features(df))
    return feats


def idf_features(df, idf):
    """IDF-weighted name-word overlap and the IDF of each side's unmatched words, plus
    phonetic-code extra words. idf: DataFrame(w, idf) of the country's S1 name words."""
    pid = pl.int_range(pl.len(), dtype=pl.UInt32).alias("pid")
    w = df.select(pid, s=words("name_norm_s"), r=words("name_norm_r"))
    unseen = float(idf["idf"].max()) if len(idf) else 1.0

    def side(col):
        """One row per (pair, word) of column `col` with the word's IDF (unseen words get `unseen`)."""
        return (w.select("pid", w=pl.col(col)).explode("w", empty_as_null=True).drop_nulls()
                 .join(idf, on="w", how="left").with_columns(pl.col("idf").fill_null(unseen)))
    s, r = side("s"), side("r")
    shared = s.join(r.select("pid", "w"), on=["pid", "w"], how="semi")
    agg = lambda d, name: d.group_by("pid").agg(pl.col("idf").sum().alias(name))
    tot = (w.select("pid").join(agg(s, "s_tot"), on="pid", how="left").join(agg(r, "r_tot"), on="pid", how="left")
            .join(agg(shared, "sh"), on="pid", how="left")
            .join(r.join(shared.select("pid", "w"), on=["pid", "w"], how="anti").group_by("pid")
                   .agg(mx_r=pl.col("idf").max()), on="pid", how="left")
            .join(s.join(shared.select("pid", "w"), on=["pid", "w"], how="anti").group_by("pid")
                   .agg(mx_s=pl.col("idf").max()), on="pid", how="left")
            .fill_null(0).sort("pid"))
    s_tot, r_tot, sh = (tot[c].to_numpy() for c in ("s_tot", "r_tot", "sh"))
    has = (s_tot > 0) & (r_tot > 0)
    na = lambda v: np.where(has, v, np.nan).astype(np.float32)
    ph = df.select(
        e_r=pl.col("phon_r").str.split(" ").list.set_difference(pl.col("phon_s").str.split(" "))
              .list.eval(pl.element().filter(pl.element() != "")).list.len(),
        e_s=pl.col("phon_s").str.split(" ").list.set_difference(pl.col("phon_r").str.split(" "))
              .list.eval(pl.element().filter(pl.element() != "")).list.len())
    has_ph = ((df["phon_s"] != "") & (df["phon_r"] != "")).to_numpy()
    return {
        "n_idf_jacc": na(sh / np.maximum(s_tot + r_tot - sh, 1e-9)),
        "n_idf_extra_rec": na(r_tot - sh),
        "n_idf_extra_s1": na(s_tot - sh),
        "n_maxidf_extra_rec": na(tot["mx_r"].to_numpy()),
        "n_maxidf_extra_s1": na(tot["mx_s"].to_numpy()),
        "n_ph_extra_rec": np.where(has_ph, ph["e_r"].to_numpy(), np.nan).astype(np.float32),
        "n_ph_extra_s1": np.where(has_ph, ph["e_s"].to_numpy(), np.nan).astype(np.float32),
    }


def house_features(df):
    """First-number equality / numeric distance, conflict flag, numbers missing on each side."""
    num = lambda c: pl.col(c).list.first().str.extract(r"^(\d+)", 1).cast(pl.Int64, strict=False)
    h = df.select(
        both=(pl.col("house_nums_s").list.len() > 0) & (pl.col("house_nums_r").list.len() > 0),
        first_eq=pl.col("house_nums_s").list.first() == pl.col("house_nums_r").list.first(),
        dist=(num("house_nums_s") - num("house_nums_r")).abs(),
        shared=pl.col("house_nums_s").list.unique().list.set_intersection(pl.col("house_nums_r").list.unique())
                 .list.len(),
        s_miss=pl.col("house_nums_s").list.head(4).list.unique().list.set_difference(pl.col("house_nums_r"))
                 .list.len(),
        r_miss=pl.col("house_nums_r").list.head(4).list.unique().list.set_difference(pl.col("house_nums_s"))
                 .list.len(),
    )
    both = h["both"].fill_null(False).to_numpy()
    na = lambda v: np.where(both, v, np.nan).astype(np.float32)
    return {
        "hn_first_eq": na(h["first_eq"].fill_null(False).to_numpy()),
        "hn_conflict": na(h["shared"].to_numpy() == 0),
        "hn_first_dist": na(np.minimum(h["dist"].fill_null(-1).to_numpy(), 10_000).astype(float)),
        "hn_s1_missing": na(h["s_miss"].to_numpy()),
        "hn_rec_missing": na(h["r_miss"].to_numpy()),
    }


def name_idf(split, country):
    """IDF table (w, idf) of the name words of the country's S1s."""
    names = (pl.scan_parquet(config.norm_path(split, "S1")).filter(pl.col("country") == country)
               .select("name_norm").collect())
    n = max(len(names), 1)
    idf = (names.select(w=words("name_norm")).explode("w", empty_as_null=True).drop_nulls()
                .group_by("w").agg(df=pl.len())
                .select("w", idf=(np.log(n) - pl.col("df").log()).cast(pl.Float32) + 0.1))
    return idf


def country_features(split, pairs, truth):
    """All pair features for one country's blocking pairs.

    Adds S1-level context (candidates, rank, gap to the best other candidate), name / address
    similarities, IDF overlaps, house-number / postal / legal-form agreement and, on train, the
    label. Returns the feature frame.
    """
    country = pairs["country"][0]
    s1 = load_side(split, ["S1"], pairs.select(entity_id="s1_id").unique(), country)
    rec = load_side(split, ["S2", "S3"], pairs.select(entity_id="cand_id").unique(), country)
    idf = name_idf(split, country)

    # S1-level context, computed on the whole candidate list of each S1.
    pairs = pairs.with_columns(
        s1_n_cands=pl.len().over("s1_id").cast(pl.Float32),
        s1_rank=pl.col("score").rank("ordinal", descending=True).over("s1_id").cast(pl.Float32),
        s1_top1=pl.col("score").max().over("s1_id"),
        s1_top2=pl.col("score").sort(descending=True).get(1, null_on_oob=True).over("s1_id"),
        source=(pl.col("cand_id").str.starts_with("S3-")).cast(pl.Float32),
    ).with_columns(
        # this pair's score minus the S1's best OTHER candidate score (0 if it has none)
        s1_gap=(pl.col("score") - pl.when(pl.col("s1_rank") == 1).then(pl.col("s1_top2").fill_null(0))
                                    .otherwise(pl.col("s1_top1"))).cast(pl.Float32),
    ).drop("s1_top1", "s1_top2")

    parts = []
    for lo in range(0, len(pairs), CHUNK):
        df = (pairs.slice(lo, CHUNK)
                   .join(s1.rename(lambda c: c if c == "entity_id" else c + "_s"), left_on="s1_id",
                         right_on="entity_id", how="left")
                   .join(rec.rename(lambda c: c if c == "entity_id" else c + "_r"), left_on="cand_id",
                         right_on="entity_id", how="left"))
        feats = chunk_features(df, idf)
        out = df.select(
            "s1_id", "cand_id", "country", *BLOCK_FEATURES, "s1_n_cands", "s1_rank", "s1_gap", "source",
            n_tfidf=pl.col("name_sim"),
            n_exact=((pl.col("name_norm_s") == pl.col("name_norm_r")) & (pl.col("name_norm_s") != "")),
            n_nonlatin=pl.col("name_nonlatin_s") | pl.col("name_nonlatin_r"),
            n_len_s1=pl.col("name_norm_s").str.len_chars(),
            n_len_rec=pl.col("name_norm_r").str.len_chars(),
            legal_conflict=((pl.col("name_legal_s") != pl.col("name_legal_r")) & (pl.col("name_legal_s") != "")
                            & (pl.col("name_legal_r") != "")),
            a_tfidf=pl.col("addr_sim"),
            house_match=pl.col("house_match"),
            hs_match=((pl.col("hs_key_s") == pl.col("hs_key_r")) & (pl.col("hs_key_s") != "")),
            postal_match=((pl.col("postal_s") == pl.col("postal_r")) & (pl.col("postal_s") != "")),
            a_empty_rec=pl.col("addr_norm_r") == "",
            a_len_s1=pl.col("addr_norm_s").str.len_chars(),
            a_len_rec=pl.col("addr_norm_r").str.len_chars(),
        ).with_columns(**{k: pl.Series(v) for k, v in feats.items()})
        parts.append(out)
    out = pl.concat(parts)
    # Missing values (a field empty on either side) are stored as null, not NaN, so polars
    # aggregations skip them; to_numpy() turns them back into NaN for LightGBM.
    out = out.with_columns(pl.col(FEATURES).cast(pl.Float32).fill_nan(None))
    if truth is not None:
        out = out.join(truth.with_columns(label=pl.lit(1, pl.Int8)), on=["s1_id", "cand_id"], how="left") \
                 .with_columns(pl.col("label").fill_null(0))
    return out.select("s1_id", "cand_id", "country", *(["label"] if truth is not None else []), *FEATURES)


def model_features(columns):
    """FEATURES present in a feature file (M-dependent ones are absent at M=1)."""
    return [f for f in FEATURES if f in columns]


def load_truth_pairs():
    """True (s1_id, cand_id) pairs of the train ground truth (one row per match)."""
    gt = pl.read_csv(config.gt_path(), separator="\t", quote_char=None, infer_schema=False).fill_null("")
    return (gt.filter(pl.col("matched_entity_ids") != "")
              .select(s1_id="source1_entity_id", cand_id=pl.col("matched_entity_ids").str.split(","))
              .explode("cand_id", empty_as_null=True))


def main():
    """Build features for every country's blocking pairs of a split and write features/<split>.parquet."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=config.SPLITS, default="train")
    ap.add_argument("--sample", type=int, default=0, help="use the normalize.py --sample N cache")
    args = ap.parse_args()
    config.use_sample(args.sample)

    t0 = time.time()
    truth = load_truth_pairs() if args.split == "train" else None
    out_path = config.features_path(args.split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(config.blocking_dir(args.split).glob("pairs_*.parquet"))
    assert files, f"no blocking pairs in {config.blocking_dir(args.split)}"
    parts = []
    for f in files:
        t = time.time()
        pairs = pl.read_parquet(f)
        if len(pairs) == 0:
            continue
        feats = country_features(args.split, pairs, truth)
        if pairs["rank"].max() == 1:      # blocking ran with M=1: rank carries no information
            feats = feats.drop(M_DEPENDENT)
        part = out_path.with_name(f"{out_path.stem}_{f.stem}.parquet")
        feats.write_parquet(part, compression="zstd")
        parts.append(part)
        pos = f", positives {int(feats['label'].sum()):,}" if truth is not None else ""
        print(f"  {f.stem}: {len(feats):,} pairs{pos} in {time.time() - t:.0f}s, peak mem {peak_mem_gb():.2f} GB",
              flush=True)
        del pairs, feats
    tmp = out_path.with_suffix(".tmp")    # written then renamed: the final file doubles as a checkpoint
    pl.concat([pl.scan_parquet(p) for p in parts]).sink_parquet(tmp, compression="zstd")
    tmp.replace(out_path)
    for p in parts:
        p.unlink()
    print(f"features -> {out_path} in {time.time() - t0:.0f}s, peak memory {peak_mem_gb():.2f} GB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
