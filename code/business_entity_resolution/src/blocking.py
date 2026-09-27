"""Reverse blocking: for every S2/S3 record, find its most likely S1 entities.

Pipeline position: stage 2 (candidate generation, reverse blocking with top_m = 1); reads normalize.py's cache,
feeds features.py.

Per country (open set, taken from the data):

1. Keys. Every S1 and record yields single keys:
     p   postal code                     h   house number + street word (hs_key)
     s   name signature (sorted folded name tokens, handles word reordering)
     n   folded name tokens (legal suffixes stripped, l/1/i 5/s 0/o 8/b folded)
     a   address tokens (no numbers, street types or function words)
     f   double-metaphone code of each name word (transliteration-robust)
     hn  house numbers (never used alone, only inside combination keys)
   and combination keys:
     hna house number x address token
     na  oversized name token x oversized address token
   A key whose S1 block is larger than `block_cap` is never looked up alone; it only
   contributes through a combination key, which must itself fit under the cap.

2. Lookup. A record looks up its p / h / s keys, its `n_rare` rarest n, a and f keys
   (rarity = S1 block size; tokens unseen in S1 are ignored), house number x its
   `hna_addr` rarest address tokens, and oversized name x oversized address tokens.

3. Ranking. S1s sharing any key are pre-ranked by summed key IDF and cut to `pre_cap`
   (a safety bound on work), then scored:
     name  = char 3-gram TF-IDF cosine (vocab fit on the country's S1); when either name
             was written in a non-Latin script, max(that, phonetic similarity), where
             phonetic = token-sort ratio of the double-metaphone codes of the
             legal-suffix-stripped name words
     addr  = char 3-gram TF-IDF cosine
     score = weighted mean of name and addr over the fields present on both sides
             (w_name, w_addr; an empty field hands its weight to the other)
             + w_house if the two share an exact (normalised) house number
   The best `gather_k` are the gathered set (recall is measured there); the best `top_m`
   are written out.

Output: one parquet per country in `out_dir` with columns
    s1_id, cand_id, country, rank, score, name_sim, ph_sim, addr_sim, house_match,
    key_score, n_keys, rec_gap (score minus the record's best other S1 score), rec_n
    (S1s scored for the record)

Usage (from student_resource/):
    python code/business_entity_resolution/src/blocking.py --split test
    python code/business_entity_resolution/src/blocking.py --split train
Train is blocked exactly like test (full S1 index x full S2/S3 pool) so the matcher sees
test-like candidate distributions; any subsampling happens later, by S1 group.
"""

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import polars as pl
from metaphone import doublemetaphone
import numba
from numba import njit, prange
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from memory import peak_mem_gb
from normalize import HS_SKIP

HASH_SEED = 7
KIND = {"p": 0, "h": 1, "s": 2, "n": 3, "a": 4, "f": 5, "hn": 6, "hna": 7, "na": 8}
KIND_NAME = {v: k for k, v in KIND.items()}
ALWAYS = [KIND["p"], KIND["h"], KIND["s"]]
RARE = [KIND["n"], KIND["a"], KIND["f"]]
ADDR_SKIP = sorted(HS_SKIP)
COLUMNS = ["entity_id", "name_norm", "name_nonlatin", "addr_norm", "postal", "hs_key", "tokens", "house_nums"]
N_HOUSE_MATCH = 4             # house numbers per side compared for the exact-match bonus


@dataclass
class BlockingParams:
    """Blocking hyper-parameters (defaults = the submitted run 4; every field is also a CLI flag)."""
    block_cap: int = 2000        # max S1s in a block for the key to be looked up on its own
    n_rare: int = 2              # rarest name / address / phonetic keys a record looks up
    n_house_s1: int = 3          # house numbers indexed per S1
    n_house_rec: int = 2         # house numbers a record combines with address tokens
    hna_addr: int = 4            # rarest address tokens combined with house numbers
    na_name: int = 2             # oversized name tokens combined with ...
    na_addr: int = 3             # ... oversized address tokens
    pre_cap: int = 150           # S1s per record kept by key score before TF-IDF
    gather_k: int = 150          # S1s per record kept by TF-IDF score (recall stage)
    top_m: int = 1               # S1s per record written out
    w_name: float = 0.4
    w_addr: float = 0.6
    w_house: float = 0.1         # bonus when the two records share an exact house number
    rec_batch: int = 500_000             # records whose keys are built at once
    expansion_budget: int = 20_000_000   # max (record, S1) pairs scored at once (records per chunk = this / pre_cap)


def country_slug(country):
    """File-name-safe version of a country name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", country).strip("_") or "unknown"


def list_countries(split):
    """Every country present in any source of a split (open set, never hard-coded)."""
    frames = [pl.scan_parquet(config.norm_path(split, s)).select("country") for s in config.SOURCES]
    return sorted(pl.concat(frames).unique().collect()["country"].to_list())


def load_s1(split, country, ids=None):
    """S1 rows of one country (optionally only `ids`) with a dense row index s1_idx."""
    lf = pl.scan_parquet(config.norm_path(split, "S1")).filter(pl.col("country") == country)
    if ids is not None:
        lf = lf.join(ids.lazy(), on="entity_id", how="semi")
    return lf.select(COLUMNS).collect().with_row_index("s1_idx")


def load_records(split, country):
    """S2 + S3 rows of one country with a dense row index ridx."""
    frames = [pl.scan_parquet(config.norm_path(split, s)).filter(pl.col("country") == country).select(COLUMNS)
              for s in ("S2", "S3")]
    return pl.concat(frames).collect().with_row_index("ridx")


# --------------------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------------------

_PHONETIC = {}


def phonetic_table(words):
    """DataFrame(w, code): primary double-metaphone code per word (cached across calls)."""
    for w in words:
        if w not in _PHONETIC:
            _PHONETIC[w] = doublemetaphone(w)[0]
    return pl.DataFrame({"w": words, "code": [_PHONETIC[w] for w in words]},
                        schema={"w": pl.String, "code": pl.String}).filter(pl.col("code") != "")


def _single(df, idx, kind, value):
    """Hashed single keys of one kind ('kind|value') for each row index."""
    return df.select(idx, kind=pl.lit(KIND[kind], pl.UInt8),
                     khash=(pl.lit(kind + "|") + value).hash(HASH_SEED))


def _explode(df, idx, expr):
    """One row per (row index, element) of a list expression, nulls dropped."""
    return df.select(idx, v=expr).explode("v", empty_as_null=True).drop_nulls()


def build_items(df, idx, n_house):
    """Long table (idx, kind, khash) of every single key and house number of each row."""
    words = _explode(df, idx, pl.col("name_norm").str.split(" ").list.unique()).filter(
        pl.col("v").str.contains(r"^[a-z]{2,}$"))
    codes = phonetic_table(words["v"].unique().to_list())
    addr = pl.col("addr_norm").str.split(" ").list.eval(pl.element().filter(
        (pl.element().str.len_bytes() >= 2) & ~pl.element().str.contains(r"\d") & ~pl.element().is_in(ADDR_SKIP)
    )).list.unique()
    sig = df.select(idx, v=pl.col("tokens").list.unique().list.sort().list.join(" ")).filter(pl.col("v") != "")
    return pl.concat([
        _single(df.filter(pl.col("postal") != ""), idx, "p", pl.col("postal")),
        _single(df.filter(pl.col("hs_key") != ""), idx, "h", pl.col("hs_key")),
        _single(sig, idx, "s", pl.col("v")),
        _single(_explode(df, idx, pl.col("tokens")), idx, "n", pl.col("v")),
        _single(_explode(df, idx, addr), idx, "a", pl.col("v")),
        _single(words.join(codes, left_on="v", right_on="w"), idx, "f", pl.col("code")),
        _single(_explode(df, idx, pl.col("house_nums").list.head(n_house)), idx, "hn", pl.col("v")),
    ]).unique()


def combine(left, right, idx, kind):
    """Combination keys: every (left key, right key) pair of the same row."""
    return left.join(right, on=idx).select(
        idx, kind=pl.lit(KIND[kind], pl.UInt8),
        khash=pl.struct(a=pl.col("khash"), b=pl.col("khash_right")).hash(HASH_SEED))


def rarest(df, k, by=("ridx", "kind")):
    """The k keys with the smallest S1 block per (row, kind) group (ties by hash, deterministic)."""
    return df.sort([*by, "size", "khash"]).group_by(list(by), maintain_order=True).head(k)


class S1Index:
    """Key -> S1 inverted index of one country, with block sizes and IDF weights."""
    def __init__(self, s1, params):
        """Build single and combination keys of every S1; keys over block_cap are only kept inside
        combination keys that fit under the cap.
        """
        cap, n = params.block_cap, len(s1)
        items = build_items(s1, "s1_idx", params.n_house_s1)
        sizes = items.group_by("khash").agg(size=pl.len(), kind=pl.col("kind").first())
        # keys shared by more than block_cap S1s (common words, big-city postcodes) are too unselective to look up
        # alone; they only contribute through combination keys (house x address, name x address) below the cap
        oversized = sizes.filter(pl.col("size") > cap).select("khash")
        over_items = items.join(oversized, on="khash", how="semi")
        combos = pl.concat([
            combine(items.filter(pl.col("kind") == KIND["hn"]), items.filter(pl.col("kind") == KIND["a"]),
                    "s1_idx", "hna"),
            combine(over_items.filter(pl.col("kind") == KIND["n"]), over_items.filter(pl.col("kind") == KIND["a"]),
                    "s1_idx", "na"),
        ])
        combo_sizes = combos.group_by("khash").agg(size=pl.len(), kind=pl.col("kind").first())

        usable = pl.concat([sizes, combo_sizes]).filter(
            (pl.col("size") <= cap) & (pl.col("kind") != KIND["hn"])
        ).with_columns(w=(math.log(max(n, 2)) - pl.col("size").log()).cast(pl.Float32) + 0.1)
        self.postings = (pl.concat([items.filter(pl.col("kind") != KIND["hn"]), combos])
                           .join(usable.select("khash", "w"), on="khash").select("khash", "s1_idx", "w"))
        # CSR view of the postings for the accumulator: key id -> (s1_idx, w) sorted by s1_idx
        self.keys = self.postings.select("khash").unique().sort("khash").with_row_index("kid")
        csr = self.postings.join(self.keys, on="khash").sort(["kid", "s1_idx"])
        self.p_s1 = csr["s1_idx"].to_numpy().astype(np.int32)
        self.p_w = csr["w"].to_numpy().astype(np.float32)
        self.k_ptr = np.concatenate([[0], np.cumsum(np.bincount(csr["kid"].to_numpy(), minlength=len(self.keys)))]
                                    ).astype(np.int64)
        self.n_s1 = n
        self.sizes = sizes.select("khash", "size")                         # all single keys + hn
        self.combo_sizes = combo_sizes.filter(pl.col("size") <= cap).select("khash", "size")
        self.cap = cap

        single = sizes.filter(pl.col("kind") != KIND["hn"])
        self.stats = {
            "single_keys": len(single),
            "single_keys_oversized": int((single["size"] > cap).sum()),
            "single_postings_dropped_by_cap": int(single.filter(pl.col("size") > cap)["size"].sum()),
            "combo_keys": len(combo_sizes),
            "combo_keys_oversized": int((combo_sizes["size"] > cap).sum()),
            "postings": len(self.postings),
        }


def record_lookups(items, index, params):
    """(ridx, kind, khash, size) keys a record looks up in the S1 index."""
    cap = index.cap
    it = items.join(index.sizes, on="khash")                     # drop keys unseen in S1
    usable = it.filter((pl.col("size") <= cap) & (pl.col("kind") != KIND["hn"]))
    over = it.filter(pl.col("size") > cap)
    hn = it.filter(pl.col("kind") == KIND["hn"])
    a_top = rarest(it.filter(pl.col("kind") == KIND["a"]), params.hna_addr)
    combos = pl.concat([
        combine(hn, a_top, "ridx", "hna"),
        combine(rarest(over.filter(pl.col("kind") == KIND["n"]), params.na_name),
                rarest(over.filter(pl.col("kind") == KIND["a"]), params.na_addr), "ridx", "na"),
    ]).join(index.combo_sizes, on="khash")
    return pl.concat([
        usable.filter(pl.col("kind").is_in(ALWAYS)),
        rarest(usable.filter(pl.col("kind").is_in(RARE)), params.n_rare),
        combos,
    ]).select("ridx", "kind", "khash", "size")


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _pair_dot(a_ptr, a_idx, a_val, b_ptr, b_idx, b_val, ra, rb, out):
    """Numba kernel: cosine of sparse L2-normalized rows ra[k] of A and rb[k] of B (CSR), into out[k]."""
    for k in prange(ra.shape[0]):
        p, pe = a_ptr[ra[k]], a_ptr[ra[k] + 1]
        q, qe = b_ptr[rb[k]], b_ptr[rb[k] + 1]
        s = 0.0
        while p < pe and q < qe:
            if a_idx[p] == b_idx[q]:
                s += a_val[p] * b_val[q]
                p += 1
                q += 1
            elif a_idx[p] < b_idx[q]:
                p += 1
            else:
                q += 1
        out[k] = s


def pair_cosine(a, b, ra, rb):
    """Cosine of L2-normalised CSR rows a[ra[k]] . b[rb[k]] (indices must be sorted)."""
    out = np.zeros(len(ra), dtype=np.float32)
    if len(ra):
        _pair_dot(a.indptr, a.indices, a.data, b.indptr, b.indices, b.data,
                  ra.astype(np.int64), rb.astype(np.int64), out)
    return out


@njit(parallel=True, cache=True)
def _house_match(a, b, ra, rb, out):
    """Numba kernel: out[k] = True if rows ra[k] of a and rb[k] of b share a house number (0 = padding)."""
    for k in prange(ra.shape[0]):
        hit = False
        for i in range(a.shape[1]):
            x = a[ra[k], i]
            if x == 0:
                break
            for j in range(b.shape[1]):
                y = b[rb[k], j]
                if y == 0:
                    break
                if x == y:
                    hit = True
        out[k] = hit


def house_matrix(df):
    """(rows, N_HOUSE_MATCH) uint64 hashes of the first house numbers, 0 = none."""
    n = pl.col("house_nums").list.len()
    return df.select([
        pl.when(n > i).then(pl.col("house_nums").list.get(i, null_on_oob=True).hash(HASH_SEED))
          .otherwise(0).cast(pl.UInt64).alias(f"h{i}")
        for i in range(N_HOUSE_MATCH)
    ]).to_numpy()


def phonetic_names(names):
    """Double-metaphone code string per (legal-suffix-stripped) name, e.g. 'TNMK TRTN'."""
    out = []
    for name in names:
        codes = []
        for w in name.split():
            if len(w) >= 2 and w.isalpha():
                c = _PHONETIC.get(w)
                if c is None:
                    c = _PHONETIC[w] = doublemetaphone(w)[0]
                if c:
                    codes.append(c)
        out.append(" ".join(codes))
    return np.array(out, dtype=object)


class Side:
    """Per-row inputs of the ranking score for one side (S1s or a chunk of records)."""

    def __init__(self, df, name_vec, addr_vec):
        """TF-IDF matrices, field presence flags, phonetic codes and house numbers of one side."""
        self.name = name_vec.transform(df["name_norm"].to_list())
        self.addr = addr_vec.transform(df["addr_norm"].to_list())
        self.name.sort_indices()
        self.addr.sort_indices()
        self.has_name = (df["name_norm"].str.len_bytes() > 0).to_numpy()
        self.has_addr = (df["addr_norm"].str.len_bytes() > 0).to_numpy()
        self.nonlatin = df["name_nonlatin"].to_numpy()
        self.phonetic = phonetic_names(df["name_norm"].to_list())
        self.house = house_matrix(df)


def score_pairs(r, s, ri, si, params):
    """Ranking score and its components for pairs (record r[ri], S1 s[si])."""
    name_tfidf = pair_cosine(r.name, s.name, ri, si)
    addr_sim = pair_cosine(r.addr, s.addr, ri, si)

    ph_sim = np.full(len(ri), np.nan, dtype=np.float32)
    use_ph = r.nonlatin[ri] | s.nonlatin[si]
    if use_ph.any():
        rp, sp = r.phonetic[ri[use_ph]], s.phonetic[si[use_ph]]
        sims = process.cpdist(rp, sp, scorer=fuzz.token_sort_ratio, workers=-1).astype(np.float32) / 100
        sims[(rp == "") | (sp == "")] = 0
        ph_sim[use_ph] = sims
    name_sim = np.where(use_ph, np.fmax(name_tfidf, ph_sim), name_tfidf)

    both_name = r.has_name[ri] & s.has_name[si]
    both_addr = r.has_addr[ri] & s.has_addr[si]
    num = params.w_name * name_sim * both_name + params.w_addr * addr_sim * both_addr
    den = params.w_name * both_name + params.w_addr * both_addr
    house = np.zeros(len(ri), dtype=np.bool_)
    if len(ri):
        _house_match(r.house, s.house, ri, si, house)
    score = np.where(den > 0, num / np.maximum(den, 1e-9), 0) + params.w_house * house
    return {
        "name_sim": name_tfidf,
        "ph_sim": ph_sim,
        "addr_sim": np.where(both_addr, addr_sim, np.nan).astype(np.float32),
        "house_match": house,
        "score": score.astype(np.float32),
    }


@njit(parallel=True, cache=True)
def _accumulate(r_ptr, r_kid, k_ptr, p_s1, p_w, n_s1, cap, nt, out_s1, out_score, out_nk, out_n, out_hits):
    """Per record: sum key weights per S1 in a dense accumulator, keep the `cap` best
    (score desc, s1_idx asc). Records are split into one contiguous block per thread."""
    n_rec = len(r_ptr) - 1
    block = (n_rec + nt - 1) // nt
    for t in prange(nt):
        acc = np.zeros(n_s1, np.float32)
        cnt = np.zeros(n_s1, np.int32)
        touched = np.empty(n_s1, np.int32)
        for r in range(t * block, min((t + 1) * block, n_rec)):
            m = 0
            for j in range(r_ptr[r], r_ptr[r + 1]):
                k = r_kid[j]
                for q in range(k_ptr[k], k_ptr[k + 1]):
                    s = p_s1[q]
                    if cnt[s] == 0:
                        touched[m] = s
                        m += 1
                    acc[s] += p_w[q]
                    cnt[s] += 1
            out_hits[r] = m
            ts = np.sort(touched[:m])
            order = np.argsort(-acc[ts], kind="mergesort")
            kk = min(m, cap)
            for i in range(kk):
                s = ts[order[i]]
                out_s1[r, i] = s
                out_score[r, i] = acc[s]
                out_nk[r, i] = cnt[s]
            out_n[r] = kk
            for i in range(m):
                acc[touched[i]] = 0
                cnt[touched[i]] = 0


def key_hits(lk, index, lo, hi, pre_cap):
    """(ridx, s1_idx, key_score, n_keys) of the pre_cap best S1s by key score for records
    lo..hi, plus the number of S1s sharing any key with each record."""
    n_rec = hi - lo + 1
    rk = lk.filter(pl.col("ridx").is_between(lo, hi)).join(index.keys, on="khash").select("ridx", "kid").sort("ridx")
    r_ptr = np.concatenate([[0], np.cumsum(np.bincount(rk["ridx"].to_numpy() - lo, minlength=n_rec))]).astype(np.int64)
    out_s1 = np.zeros((n_rec, pre_cap), np.int32)
    out_score = np.zeros((n_rec, pre_cap), np.float32)
    out_nk = np.zeros((n_rec, pre_cap), np.int32)
    out_n = np.zeros(n_rec, np.int64)
    out_hits = np.zeros(n_rec, np.int64)
    _accumulate(r_ptr, rk["kid"].to_numpy().astype(np.int64), index.k_ptr, index.p_s1, index.p_w, index.n_s1,
                pre_cap, numba.get_num_threads(), out_s1, out_score, out_nk, out_n, out_hits)
    mask = np.arange(pre_cap)[None, :] < out_n[:, None]
    hits = pl.DataFrame({
        "ridx": np.repeat(np.arange(lo, hi + 1, dtype=np.uint32), out_n),
        "s1_idx": out_s1[mask].astype(np.uint32),
        "key_score": out_score[mask],
        "n_keys": out_nk[mask].astype(np.uint32),
    })
    return hits, int(out_hits.sum())


def _with_record_context(ranked, top_m):
    """Keep the top_m S1s per record, adding record-level context:
    rec_gap   score minus the best score among the record's other scored S1s (0 if none)
    rec_n     number of S1s scored for the record"""
    top2 = ranked.filter(pl.col("rank") <= 2).group_by("ridx").agg(
        s_1=pl.col("score").max(), s_2=pl.when(pl.len() > 1).then(pl.col("score").min()))
    rec_n = ranked.group_by("ridx").agg(rec_n=pl.len().cast(pl.UInt32))
    top2 = top2.join(rec_n, on="ridx")
    return ranked.filter(pl.col("rank") <= top_m).join(top2, on="ridx").with_columns(
        rec_gap=pl.col("score") - pl.when(pl.col("rank") == 1).then(pl.col("s_2").fill_null(0))
                                     .otherwise(pl.col("s_1"))).drop("s_1", "s_2")


def _rank_within(df, col):
    """Rank S1s within each record by `col` (descending, ties by s1_idx) into column `rank`."""
    return (df.sort(["ridx", col, "s1_idx"], descending=[False, True, False])
              .with_columns(rank=(pl.int_range(pl.len()).over("ridx") + 1).cast(pl.UInt16)))


# --------------------------------------------------------------------------------------
# Per-country driver
# --------------------------------------------------------------------------------------

def _miss_reasons(missed, s1, rec, index, params):
    """Why a true pair never shared a usable key: not_selected / only_oversized / no_shared_key."""
    if len(missed) == 0:
        return missed.with_columns(miss_reason=pl.lit(None, pl.String))
    r_items = build_items(rec.join(missed.select("ridx").unique(), on="ridx", how="semi"), "ridx",
                          params.n_house_s1)
    s_items = build_items(s1.join(missed.select("s1_idx").unique(), on="s1_idx", how="semi"), "s1_idx",
                          params.n_house_s1)
    shared = (missed.join(r_items, on="ridx").join(s_items, on=["s1_idx", "khash"])
                    .join(index.sizes, on="khash", how="left")
                    .group_by("ridx", "s1_idx").agg(
                        usable=((pl.col("size") <= index.cap) & (pl.col("kind") != KIND["hn"])).any()))
    return missed.join(shared, on=["ridx", "s1_idx"], how="left").with_columns(
        miss_reason=pl.when(pl.col("usable").is_null()).then(pl.lit("no_shared_key"))
                      .when(pl.col("usable")).then(pl.lit("not_selected"))
                      .otherwise(pl.lit("only_oversized"))).drop("usable")


def block_country(s1, rec, params, truth=None, log=print):
    """Return (pairs, stats, truth_status) for one country.

    truth: optional DataFrame (s1_id, cand_id). Used only for diagnostics: truth_status
    records for each true pair the furthest stage it reached and why it was lost.
    """
    stats = {"n_s1": len(s1), "n_rec": len(rec)}
    if len(s1) == 0 or len(rec) == 0:
        return None, stats, None

    t = time.time()
    index = S1Index(s1, params)
    stats.update(index.stats)
    stats["t_index"] = time.time() - t
    log(f"    index: {index.stats['postings']:,} postings; single keys {index.stats['single_keys']:,} "
        f"({index.stats['single_keys_oversized']:,} oversized, {index.stats['single_postings_dropped_by_cap']:,} "
        f"postings only reachable via combos); combo keys {index.stats['combo_keys']:,} "
        f"({index.stats['combo_keys_oversized']:,} oversized) ({time.time() - t:.0f}s)")

    t = time.time()
    tfidf = dict(analyzer="char_wb", ngram_range=(3, 3), dtype=np.float32, sublinear_tf=True)
    name_vec = TfidfVectorizer(**tfidf).fit(s1["name_norm"].to_list())
    addr_vec = TfidfVectorizer(**tfidf).fit(s1["addr_norm"].to_list())
    s1_side = Side(s1, name_vec, addr_vec)
    stats["t_tfidf"] = time.time() - t
    log(f"    tfidf: name vocab {len(name_vec.vocabulary_):,}, addr vocab {len(addr_vec.vocabulary_):,} "
        f"({time.time() - t:.0f}s)")

    truth_idx = None
    if truth is not None:
        truth_idx = (truth.join(s1.select("s1_idx", s1_id="entity_id"), on="s1_id")
                          .join(rec.select("ridx", cand_id="entity_id"), on="cand_id")
                          .select("ridx", "s1_idx"))
    reached = {"keys": [], "precap": [], "gather": []}

    def note(stage, df):
        """Record which true pairs are still present at `stage` (train recall diagnostics only)."""
        if truth_idx is not None:
            reached[stage].append(df.select("ridx", "s1_idx").join(truth_idx, on=["ridx", "s1_idx"], how="semi"))

    counts = {"rec_with_key": 0, "rec_only_combo": 0, "key_hits": 0, "hits_after_precap": 0}
    out, t = [], time.time()
    for b in range(0, len(rec), params.rec_batch):
        batch = rec.slice(b, params.rec_batch)
        lk = record_lookups(build_items(batch, "ridx", params.n_house_rec), index, params)
        kinds = lk.group_by("ridx").agg(single=(pl.col("kind") < KIND["hn"]).any())
        counts["rec_with_key"] += len(kinds)
        counts["rec_only_combo"] += int((~kinds["single"]).sum())

        if truth_idx is not None:   # true pairs sharing a looked-up usable key
            note("keys", lk.join(truth_idx, on="ridx").join(index.postings, on=["khash", "s1_idx"], how="semi")
                           .select("ridx", "s1_idx").unique())
        step = max(1, params.expansion_budget // params.pre_cap)
        b_hi = min(b + params.rec_batch, len(rec)) - 1
        for lo in range(b, b_hi + 1, step):
            hi = min(lo + step - 1, b_hi)
            hits, n_hits = key_hits(lk, index, lo, hi, params.pre_cap)
            counts["key_hits"] += n_hits
            counts["hits_after_precap"] += len(hits)
            note("precap", hits)

            r_side = Side(rec.slice(lo, hi - lo + 1), name_vec, addr_vec)
            ri = hits["ridx"].to_numpy().astype(np.int64) - lo
            si = hits["s1_idx"].to_numpy().astype(np.int64)
            scores = score_pairs(r_side, s1_side, ri, si, params)
            ranked = _rank_within(hits.with_columns(**{k: pl.Series(v) for k, v in scores.items()}), "score")
            note("gather", ranked.filter(pl.col("rank") <= params.gather_k))
            out.append(_with_record_context(ranked, params.top_m))
        log(f"    records {min(b + params.rec_batch, len(rec)):,}/{len(rec):,}: {time.time() - t:.0f}s, "
            f"peak mem {peak_mem_gb():.2f} GB")

    stats.update(counts)
    stats["t_records"] = time.time() - t
    log(f"    records with a usable key: {counts['rec_with_key']:,}/{len(rec):,} "
        f"({counts['rec_only_combo']:,} only via combo keys); key hits {counts['key_hits']:,}, "
        f"after pre-cap {counts['hits_after_precap']:,}")

    pairs = pl.concat(out)
    pairs = pairs.select(
        s1_id=s1["entity_id"].gather(pairs["s1_idx"]),
        cand_id=rec["entity_id"].gather(pairs["ridx"]),
        rank=pl.col("rank").cast(pl.UInt8), score="score", name_sim="name_sim", ph_sim="ph_sim",
        addr_sim="addr_sim", house_match="house_match", key_score="key_score",
        n_keys=pl.col("n_keys").cast(pl.UInt8), rec_gap="rec_gap", rec_n="rec_n",
    )
    stats["pairs"] = len(pairs)

    truth_status = None
    if truth_idx is not None:
        status = truth_idx
        for stage, parts in reached.items():
            seen = pl.concat(parts).unique() if parts else truth_idx.clear()
            status = status.join(seen.with_columns(pl.lit(True).alias(f"in_{stage}")),
                                 on=["ridx", "s1_idx"], how="left")
        status = status.with_columns(pl.col("in_keys", "in_precap", "in_gather").fill_null(False))
        missed = _miss_reasons(status.filter(~pl.col("in_keys")), s1, rec, index, params)
        status = pl.concat([status.filter(pl.col("in_keys")).with_columns(miss_reason=pl.lit(None, pl.String)),
                            missed])
        found = pl.concat(out).select("ridx", "s1_idx", "rank")
        truth_status = (status.join(found, on=["ridx", "s1_idx"], how="left")
                              .join(s1.select("s1_idx", s1_id="entity_id"), on="s1_idx")
                              .join(rec.select("ridx", cand_id="entity_id"), on="ridx")
                              .select("s1_id", "cand_id", "in_keys", "in_precap", "in_gather", "rank",
                                      "miss_reason"))
    return pairs, stats, truth_status


def run_blocking(split, out_dir, params=None, s1_ids=None, truth=None, log=print, countries=None):
    """Block every country of `split`; write one parquet per country.

    Returns (per-country stats, truth_status DataFrame or None).
    s1_ids: optional DataFrame(entity_id) restricting the S1 pool (e.g. a held-out split).
    """
    params = params or BlockingParams()
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"blocking {split} with {asdict(params)}")
    all_stats, statuses = {}, []
    for country in countries or list_countries(split):
        t0 = time.time()
        s1 = load_s1(split, country, s1_ids)
        rec = load_records(split, country)
        log(f"  [{country}] S1={len(s1):,} records={len(rec):,}")
        pairs, stats, status = block_country(s1, rec, params, truth=truth, log=log)
        if pairs is not None:
            pairs.with_columns(country=pl.lit(country)).write_parquet(
                out_dir / f"pairs_{country_slug(country)}.parquet", compression="zstd")
        if status is not None:
            statuses.append(status.with_columns(country=pl.lit(country)))
        stats["seconds"] = round(time.time() - t0, 1)
        all_stats[country] = stats
        log(f"  [{country}] done: {stats.get('pairs', 0):,} pairs in {stats['seconds']:.0f}s, "
            f"peak mem {peak_mem_gb():.2f} GB")
        del s1, rec, pairs
    return all_stats, (pl.concat(statuses) if statuses else None)


def s1_subset(split, n):
    """Random n S1 ids of `split` (fixed seed, saved for reuse); None = all S1s."""
    ids = pl.read_parquet(config.norm_path(split, "S1"), columns=["entity_id"])["entity_id"].sort()
    if not n or n >= len(ids):
        return None
    path = config.subset_path(split, n)
    if path.exists():
        return pl.read_parquet(path)
    sub = pl.DataFrame({"entity_id": ids.sample(n, seed=config.SEED)}).sort("entity_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.write_parquet(path)
    return sub


def main():
    """Block every country of a split and write one pairs parquet per country (+ stats JSON)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=config.SPLITS, default="test")
    ap.add_argument("--sample", type=int, default=0, help="use the normalize.py --sample N cache")
    ap.add_argument("--countries", nargs="+", help="block only these countries (default: all)")
    ap.add_argument("--s1_subset", type=int, default=0,
                    help="experiments only: index a random N S1s (records stay the full pool); 0 = all")
    for f, v in vars(BlockingParams()).items():
        ap.add_argument(f"--{f}", type=type(v), default=v)
    args = ap.parse_args()
    params = BlockingParams(**{f: getattr(args, f) for f in vars(BlockingParams())})
    config.use_sample(args.sample)

    t0 = time.time()
    done = config.blocking_done(args.split)
    done.unlink(missing_ok=True)
    stats, _ = run_blocking(args.split, config.blocking_dir(args.split), params,
                            s1_ids=s1_subset(args.split, args.s1_subset), countries=args.countries)
    if not args.countries and not args.s1_subset:
        done.write_text(json.dumps({"params": asdict(params), "stats": stats}, indent=1, default=str))
    print(f"runtime: {time.time() - t0:.0f}s, peak memory: {peak_mem_gb():.2f} GB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
