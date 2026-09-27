"""Normalise raw source TSVs into cached parquet files.

Pipeline position: stage 1 of the CPU pipeline (run_all.py step `normalize`); every later stage reads its parquet
cache.

Output columns (one parquet per split/source, see config.norm_path):
    entity_id, country
    name_full   normalised name, all tokens
    name_norm   name with legal-form and web tokens removed (falls back to name_full)
    name_legal  canonical legal forms found in the name, sorted, space-joined
    tokens      name comparison keys: name_norm tokens with letter/digit confusions folded
    addr_norm   normalised address, abbreviations expanded
    postal      5-6 digit code that ends an address component, else ""
    house_nums  other numeric tokens (leading zeros stripped), in order
    hs_key      "<house number>|<first street word>" blocking key, else ""

No branching on country: every rule is applied to every record.

Usage (from student_resource/):
    python code/business_entity_resolution/src/normalize.py --splits train
"""

import argparse
import re
import sys
import time
import zlib
from multiprocessing import Pool

import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from anyascii import anyascii

import config

BATCH_ROWS = 100_000

# --------------------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------------------

# Legal forms -> canonical form. Includes anyascii output of common Indic spellings.
LEGAL = {
    "llc": "llc", "pllc": "pllc", "llp": "llp", "elelpi": "llp", "lllp": "lllp", "lp": "lp", "plc": "plc", "pc": "pc",
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cie": "co",
    "ltd": "ltd", "limited": "ltd", "limitet": "ltd", "limirrd": "ltd", "li": "ltd",
    "pvt": "pvt", "private": "pvt", "praivet": "pvt", "praibhet": "pvt", "piraivet": "pvt", "praivrr": "pvt", "pra": "pvt", "prvt": "pvt",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci", "sa": "sa", "snc": "snc",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
    "pte": "pte", "pty": "pty",
}
WEB = {"www", "com", "net", "org", "http", "https", "in", "co.in"}

# Address abbreviations -> expansion. One table for every country.
ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard",
    "cir": "circle", "crcl": "circle", "ln": "lane", "dr": "drive", "drv": "drive", "ct": "court",
    "pl": "place", "pkwy": "parkway", "pky": "parkway", "hwy": "highway", "trl": "trail",
    "sq": "square", "mt": "mount", "ft": "fort", "cres": "crescent", "terr": "terrace",
    "expy": "expressway", "fwy": "freeway", "jct": "junction", "rte": "route", "rt": "route",
    "r": "rue", "imp": "impasse", "all": "allee", "chem": "chemin", "fbg": "faubourg",
    "apt": "apartment", "apts": "apartments", "appts": "apartments", "bldg": "building",
    "fl": "floor", "flr": "floor", "ste": "suite", "nr": "near", "opp": "opposite",
    "ngr": "nagar", "dist": "district", "tq": "taluk", "ndeg": "no",
}
STREET_WORDS = {
    "street", "road", "avenue", "boulevard", "circle", "lane", "drive", "court", "place", "parkway",
    "highway", "trail", "square", "terrace", "way", "route", "rue", "impasse", "allee", "chemin",
    "quai", "cours", "marg", "path",
}
# Words skipped when choosing the street word for hs_key ("1 rue de la paix" -> 1|paix).
HS_SKIP = STREET_WORDS | {"de", "du", "des", "la", "le", "les", "l", "d", "of", "the", "and", "no", "saint"}

# --------------------------------------------------------------------------------------
# Regexes
# --------------------------------------------------------------------------------------

PLACEHOLDER = re.compile(r"(?i)<\s*null\s*>|\bnull\b|\bn/a\b")
# A whole field equal to one of these (case-insensitive, stripped) is empty.
JUNK_FIELD = {"none", "nan", "na", "n/a", "n.a.", "null", "<null>", "nil", "-", "--", ".", "?", "#", "##",
              "0", "unknown", "not available"}
NONLATIN = re.compile("[Ͱ-᷿Ⰰ-￿]")   # scripts other than Latin (punctuation excluded)
POSSESSIVE = re.compile(r"'s\b")
SAINT = re.compile(r"\bst\s*\.?\s*-\s*")          # "St.-Herblain", "st-nazaire" -> saint
SAINTE = re.compile(r"\bste\s*\.?\s*-\s*")
NON_ALNUM = re.compile(r"[^a-z0-9]+")
HOUSE_NUM = re.compile(r"^\d+[a-z]?$")
ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")
POSTAL = re.compile(r"^\d{5,6}$")
HAS_DIGIT = re.compile(r"\d")
HAS_ALPHA = re.compile(r"[a-z]")
MIXED_FOLD = str.maketrans({"0": "o", "1": "i", "5": "s", "8": "b"})


def base_clean(text):
    """Placeholder removal, transliteration, lowercase, &->and, saint handling."""
    if not text or text.strip().lower() in JUNK_FIELD:
        return ""
    text = PLACEHOLDER.sub(" ", text)
    if not text.isascii():
        text = anyascii(text)
    text = text.lower().replace("&", " and ")
    text = POSSESSIVE.sub("s", text)
    text = SAINT.sub("saint ", text)
    text = SAINTE.sub("sainte ", text)
    return text


def merge_single_letters(tokens):
    """Join runs of single letters: 'l l c' -> 'llc', 'm s' -> 'ms'."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


def tokenize(text):
    """Split on non-alphanumerics and merge runs of single letters ('a b c' -> 'abc')."""
    return merge_single_letters(NON_ALNUM.sub(" ", text).split())


def fold_confusions(tok):
    """Comparison key only: l/1/i collapse to i; in mixed tokens 0->o, 5->s, 8->b too."""
    if HAS_DIGIT.search(tok) and HAS_ALPHA.search(tok):
        tok = tok.translate(MIXED_FOLD)
    return tok.replace("l", "i")


def normalize_name(raw):
    """Normalize a business name.

    Returns: (all tokens, core tokens without legal forms / web words, sorted legal forms, blocking
    keys = core tokens with confusable characters folded, deduplicated).
    """
    toks = tokenize(base_clean(raw))
    legal = sorted({LEGAL[t] for t in toks if t in LEGAL})
    core = [t for t in toks if t not in LEGAL and t not in WEB] or toks
    keys = list(dict.fromkeys(fold_confusions(t) for t in core))
    return " ".join(toks), " ".join(core), " ".join(legal), keys


def find_postal(seg_toks, seg_index):
    """Index of a postal code within one address component, or None.

    Most 5-6 digit numbers in this data are (zero-padded) house numbers, so a code is
    only accepted where house numbers do not appear: ending a component
    ("mumbai 400001", "44600") or leading a later, street-free component ("44600 saint nazaire").
    """
    if POSTAL.match(seg_toks[-1]) and (len(seg_toks) == 1 or not HAS_DIGIT.search(seg_toks[0])):
        return len(seg_toks) - 1
    # "st" is ambiguous (street / saint), so it does not count as a street word here.
    if (seg_index > 0 and len(seg_toks) > 1 and POSTAL.match(seg_toks[0])
            and not any(ADDR_ABBR.get(t, t) in STREET_WORDS for t in seg_toks if t != "st")):
        return 0
    return None


def normalize_address(raw):
    """Normalize an address.

    Returns: (normalized text without the postal code, postal code, house numbers, hs_key =
    'number|first distinctive word after it').
    """
    toks, postal = [], ""
    for seg_index, seg in enumerate(base_clean(raw).split(",")):
        seg_toks = tokenize(seg)
        if not seg_toks:
            continue
        if not postal:
            i = find_postal(seg_toks, seg_index)
            if i is not None:
                postal = seg_toks.pop(i)
        toks.extend(ADDR_ABBR.get(t, t) for t in seg_toks)

    # hs_key: first distinctive word following a number, paired with the closest preceding
    # number. "88 olive circle" -> 88|olive, "flat no 1 101 anuska towers" -> 101|anuska.
    house_nums, hs_key, last_num = [], "", None
    for t in toks:
        if HOUSE_NUM.match(t) and not ORDINAL.match(t):
            n = t.lstrip("0") or "0"
            house_nums.append(n)
            if not hs_key:
                last_num = n
        elif (last_num and not hs_key and len(t) >= 2 and t not in HS_SKIP
              and (ORDINAL.match(t) or not HAS_DIGIT.search(t))):   # "7222 6th avenue" -> 7222|6th
            hs_key = f"{last_num}|{t}"
    return " ".join(toks), postal, house_nums, hs_key


# --------------------------------------------------------------------------------------
# Parallel batch processing
# --------------------------------------------------------------------------------------

SCHEMA = pa.schema([
    ("entity_id", pa.string()), ("country", pa.string()),
    ("name_full", pa.string()), ("name_norm", pa.string()), ("name_legal", pa.string()), ("name_nonlatin", pa.bool_()),
    ("tokens", pa.list_(pa.string())),
    ("addr_norm", pa.string()), ("postal", pa.string()),
    ("house_nums", pa.list_(pa.string())), ("hs_key", pa.string()),
])


def process_batch(args):
    """Normalize one batch (ids, countries, names, addresses) into a pyarrow table of SCHEMA."""
    ids, countries, names, addrs = args
    cols = {k: [] for k in SCHEMA.names}
    cols["entity_id"] = ids
    cols["country"] = [c.strip() for c in countries]
    for name, addr in zip(names, addrs):
        full, core, legal, keys = normalize_name(name)
        a, postal, nums, hs = normalize_address(addr)
        cols["name_full"].append(full)
        cols["name_norm"].append(core)
        cols["name_legal"].append(legal)
        cols["name_nonlatin"].append(bool(NONLATIN.search(name)))
        cols["tokens"].append(keys)
        cols["addr_norm"].append(a)
        cols["postal"].append(postal)
        cols["house_nums"].append(nums)
        cols["hs_key"].append(hs)
    return pa.table(cols, schema=SCHEMA)


def iter_batches(path, keep=None):
    """Stream the TSV as lists of strings; no quoting, empty stays empty (not null).

    keep: optional predicate on entity_id (used by --sample).
    """
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=32 << 20),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in ("entity_id", "business_name", "business_address", "country")},
            strings_can_be_null=False, quoted_strings_can_be_null=False,
        ),
    )
    for rb in reader:
        d = rb.to_pydict()
        if keep is not None:
            rows = [i for i, eid in enumerate(d["entity_id"]) if keep(eid)]
            d = {k: [v[i] for i in rows] for k, v in d.items()}
        n = len(d["entity_id"])
        for s in range(0, n, BATCH_ROWS):
            e = s + BATCH_ROWS
            yield (d["entity_id"][s:e], d["country"][s:e], d["business_name"][s:e], d["business_address"][s:e])


class SampleFilter:
    """Keep sampled S1s, all of their true matches, and the same fraction of other records."""

    def __init__(self, n):
        """Draw n ground-truth S1s (config.SEED); keep their matches and the same fraction of others."""
        gt = pl.read_csv(config.gt_path(), separator="\t", quote_char=None, infer_schema=False).fill_null("")  # empty ID list stays ""
        self.frac = min(n / len(gt), 1.0)
        gt = gt.sample(n=min(n, len(gt)), seed=config.SEED)
        self.s1 = set(gt["source1_entity_id"])
        self.matched = set(gt["matched_entity_ids"].str.split(",").explode(empty_as_null=True).drop_nulls()) - {""}

    def for_source(self, src):
        """Predicate entity_id -> keep for a source (hash-based, so it is deterministic)."""
        if src == "S1":
            return self.s1.__contains__
        cutoff = int(self.frac * 2 ** 32)
        return lambda eid: eid in self.matched or zlib.crc32(eid.encode()) < cutoff


def normalize_file(split, src, pool, keep=None):
    """Stream one raw TSV through the worker pool into its parquet (atomic rename when done)."""
    src_path, out_path = config.source_path(split, src), config.norm_path(split, src)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    t0, n = time.time(), 0
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as w:
        for tbl in pool.imap(process_batch, iter_batches(src_path, keep)):
            w.write_table(tbl)
            n += tbl.num_rows
    tmp.replace(out_path)
    print(f"  {split} {src}: {n:,} rows -> {out_path.name} in {time.time() - t0:.0f}s", flush=True)


def main():
    """Normalize the requested splits / sources (skipping existing parquets unless --force)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=list(config.SPLITS), choices=config.SPLITS)
    ap.add_argument("--sources", nargs="+", default=list(config.SOURCES), choices=config.SOURCES)
    ap.add_argument("--force", action="store_true", help="rebuild even if the parquet exists")
    ap.add_argument("--sample", type=int, default=0,
                    help="smoke test: N train S1s + their matches + same fraction of other records "
                         "(train only, written to a separate cache)")
    args = ap.parse_args()

    sample = None
    if args.sample:
        config.use_sample(args.sample)
        args.splits = ["train"]
        sample = SampleFilter(args.sample)
        print(f"sample: {len(sample.s1):,} S1, {len(sample.matched):,} matched records, "
              f"other records kept at {sample.frac:.4f} -> {config.CACHE_DIR}")

    with Pool(config.WORKERS) as pool:
        for split in args.splits:
            for src in args.sources:
                if config.norm_path(split, src).exists() and not args.force:
                    print(f"  {split} {src}: cached, skipping (use --force to rebuild)")
                    continue
                normalize_file(split, src, pool, sample.for_source(src) if sample else None)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
