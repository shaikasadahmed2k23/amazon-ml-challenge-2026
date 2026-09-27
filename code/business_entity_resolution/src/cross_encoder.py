"""Second scorer: a fine-tuned multilingual cross-encoder on the SAME candidate pairs, blended with LightGBM.

Pipeline position: GPU stage after the CPU pipeline (CE v1; CE v2 with --from_ce); its ce_B / ce_test scores feed
stack.py.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 117.7M parameters, BERT
architecture), fine-tuned as a binary pair classifier (classification head on [CLS], BCE loss). Only the
pretrained weights are downloaded; no external data or lookups.

Input per pair: "<raw name> | <raw address>" of the S1 and of the record (raw text, native script kept),
as a tokenizer text pair, truncated to --max_len tokens.

Stages (each checkpointed under <ce_dir>, rerun skips finished ones):
  pairs       the matcher's train universe (same group sample as train_matcher.py), split by S1 id into
              fold A (70%, cross-encoder training, subsampled to --n_train pairs) and fold B (30%, blending);
              fold B keeps the LightGBM OOF probability and the pre-filter flag
  texts       raw "name | address" for every id the pairs need (train + test TSVs)
  train       1 epoch, fp16, fixed seed, all visible GPUs (DataParallel)       -> <ce_dir>/model/
  score_b     cross-encoder probability for every fold-B pair (sharded, resumable)
  score_test  cross-encoder probability for every test candidate pair = the pre-filter keep set of the
              cached run (sharded, resumable)
  blend       on fold B: p = w * p_lgbm + (1 - w) * p_ce for w in 0..1, many-to-one, best threshold for macro
              F0.5 at distractor weight 2 (and w = 1 reported); LightGBM alone and CE alone on the same S1s.
              Test: the chosen blend -> <out>/matching_results.tsv, <out>/candidate_pairs.tsv (the cached
              run's keep set, i.e. identical to its candidate file).

Usage (from student_resource/, with the cache of the run whose candidates are kept):
    python code/business_entity_resolution/src/cross_encoder.py --ce_dir cache_ce --out output_ce
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv

import config
from memory import peak_mem_gb

MODEL_ID = config.CE_MODEL_ID
STAGES = ("pairs", "texts", "train", "score_b", "score_test", "blend")
FOLD_A = 0.70
SHARD = 500_000
W_GRID = np.round(np.arange(0.0, 1.0001, 0.1), 2)


def log(msg):
    """Print a time-stamped message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# pairs
# --------------------------------------------------------------------------------------

def stage_pairs(args, ce):
    """Split the matcher's OOF universe by S1 into fold A (70%, CE training) and fold B (30%, held out).

    Writes pairs_A (n_train sampled with --seed), pairs_B (all fold-B pairs, with p_lgbm OOF and the
    pre-filter flag) and universe_B. With --from_ce, fold B must equal the first CE's exactly.
    """
    from train_matcher import load_universe
    n_pairs = pl.scan_parquet(config.features_path("train")).select(pl.len()).collect().item()
    universe = load_universe(n_pairs, config.TRAIN_MAX_PAIRS)          # same universe as the matcher's OOF
    # fold split by S1 with a fixed seed: every cross-encoder (v1, v2 --from_ce) and stack.py see the same fold B,
    # whose p_lgbm is out-of-fold, so the stacking never trains on in-sample scores
    rng = np.random.default_rng(config.SEED + 7)
    universe = universe.with_columns(fold=pl.Series(np.where(rng.random(len(universe)) < FOLD_A, "A", "B")))
    oof = pl.read_parquet(config.model_dir() / "oof.parquet").select("s1_id", "cand_id", "label", p_lgbm="prob")
    oof = oof.join(universe.select("s1_id", "fold"), on="s1_id")
    kp = config.prefilter_keep_path("train")
    if config.PREFILTER and kp.exists():
        oof = oof.join(pl.read_parquet(kp).with_columns(pf=pl.lit(True)), on=["s1_id", "cand_id"], how="left") \
                 .with_columns(pl.col("pf").fill_null(False))
    else:
        oof = oof.with_columns(pf=pl.lit(True))
    a = oof.filter(pl.col("fold") == "A")
    n = min(args.n_train, len(a))
    a = a.sample(n, seed=args.seed, shuffle=True)
    b = oof.filter(pl.col("fold") == "B")
    if args.max_b:
        keep_s1 = b.select("s1_id").unique().sort("s1_id").head(max(1, args.max_b // 4))
        b = b.join(keep_s1, on="s1_id")
        universe = pl.concat([universe.filter(pl.col("fold") == "A"),
                              universe.join(keep_s1, on="s1_id", how="semi")])
    a.select("s1_id", "cand_id", "label").write_parquet(ce / "pairs_A.parquet")
    b.write_parquet(ce / "pairs_B.parquet")
    universe.filter(pl.col("fold") == "B").select("s1_id", "country", "n_true").write_parquet(ce / "universe_B.parquet")
    log(f"pairs: fold A {n:,} training pairs of {len(oof.filter(pl.col('fold') == 'A')):,} (positives "
        f"{a['label'].mean():.3f}, seed {args.seed}); fold B {len(b):,} pairs over {(universe['fold'] == 'B').sum():,} S1s")
    if args.from_ce:   # second cross-encoder: fold B must be exactly the first one's (same S1s, same pairs)
        keys = ["s1_id", "cand_id"]
        v1 = pl.read_parquet(Path(args.from_ce) / "pairs_B.parquet").select(keys)
        mine = b.select(keys)
        same = len(v1) == len(mine) and len(v1.join(mine, on=keys, how="anti")) == 0
        log(f"fold B vs {args.from_ce}: {len(mine):,} vs {len(v1):,} pairs, identical: {same}")
        assert same, "fold B differs from the first cross-encoder's: the stacking would mix splits"
        v1a = pl.read_parquet(Path(args.from_ce) / "pairs_A.parquet").select("s1_id").unique()
        assert len(v1a.join(mine.select("s1_id").unique(), on="s1_id", how="semi")) == 0, "fold A/B overlap"


# --------------------------------------------------------------------------------------
# texts
# --------------------------------------------------------------------------------------

def test_pairs(args):
    """The test pairs to score: the pre-filter keep set (first --max_test pairs in smoke tests)."""
    keep = config.prefilter_keep_path("test")
    if keep.exists():
        p = pl.read_parquet(keep).select("s1_id", "cand_id")
    else:
        p = pl.read_parquet(config.features_path("test"), columns=["s1_id", "cand_id"])
        log(f"!!! no pre-filter keep set at {keep}: using every test feature pair")
    if args.max_test:
        p = p.sort(["s1_id", "cand_id"]).head(args.max_test)
    return p


def read_texts(split, src, ids):
    """Raw 'name | address' text of the given ids from one source TSV (streamed with pyarrow)."""
    reader = pacsv.open_csv(
        config.source_path(split, src), read_options=pacsv.ReadOptions(block_size=32 << 20),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in ("entity_id", "business_name", "business_address", "country")},
            strings_can_be_null=False, quoted_strings_can_be_null=False))
    df = pl.concat([pl.from_arrow(rb).select("entity_id", "business_name", "business_address") for rb in reader])
    df = df.join(pl.DataFrame({"entity_id": ids}).unique(), on="entity_id", how="semi")   # one hash join per file
    return df.select("entity_id", text=pl.col("business_name").str.strip_chars() + " | "
                     + pl.col("business_address").str.strip_chars())


def stage_texts(args, ce):
    """Write texts_train / texts_test (entity_id -> text) for every id in the pairs; with --from_ce
    the first CE's texts_test is reused when it covers the test pairs.
    """
    a, b = pl.read_parquet(ce / "pairs_A.parquet"), pl.read_parquet(ce / "pairs_B.parquet")
    t = test_pairs(args)
    todo = [("train", [a, b]), ("test", [t])]
    v1_test = Path(args.from_ce) / "texts_test.parquet" if args.from_ce else None
    if v1_test is not None and v1_test.exists():
        v1t = pl.read_parquet(v1_test)
        need = pl.concat([t.select(entity_id="s1_id"), t.select(entity_id="cand_id")]).unique()
        if len(need.join(v1t, on="entity_id", how="anti")) == 0:
            v1t.write_parquet(ce / "texts_test.parquet")
            log(f"texts test: reused {v1_test} ({len(v1t):,} ids)")
            todo = todo[:1]
    for split, frames in todo:
        s1_ids = pl.concat([f.select("s1_id") for f in frames])["s1_id"].unique()
        rec_ids = pl.concat([f.select("cand_id") for f in frames])["cand_id"].unique()
        parts = [read_texts(split, "S1", s1_ids)]
        for src in ("S2", "S3"):
            parts.append(read_texts(split, src, rec_ids.filter(rec_ids.str.starts_with(src))))
        texts = pl.concat(parts).unique("entity_id")
        texts.write_parquet(ce / f"texts_{split}.parquet")
        log(f"texts {split}: {len(texts):,} ids (needed {len(s1_ids) + len(rec_ids):,})")


def with_texts(pairs, texts):
    """Attach the S1 text (t1) and record text (t2) to each pair ('' when missing)."""
    return (pairs.join(texts.rename({"entity_id": "s1_id", "text": "t1"}), on="s1_id", how="left")
                 .join(texts.rename({"entity_id": "cand_id", "text": "t2"}), on="cand_id", how="left")
                 .with_columns(pl.col("t1", "t2").fill_null("")))


# --------------------------------------------------------------------------------------
# model helpers
# --------------------------------------------------------------------------------------

def devices():
    """(torch device, number of GPUs): cuda when available, else cpu."""
    import torch
    n = torch.cuda.device_count()
    return (torch.device("cuda") if n else torch.device("cpu")), n


class Collate:
    """DataLoader collate function: tokenizes (t1, t2) pairs of a batch of indices (+ float labels)."""
    def __init__(self, tok, t1, t2, max_len, y=None):
        """Store the tokenizer, the two text lists, max_len and optional labels."""
        self.tok, self.t1, self.t2, self.max_len, self.y = tok, t1, t2, max_len, y

    def __call__(self, idx):
        """Tokenize the pairs at `idx` (padded, truncated to max_len) and add labels if present."""
        import torch
        enc = self.tok([self.t1[i] for i in idx], [self.t2[i] for i in idx], truncation="longest_first",
                       max_length=self.max_len, padding=True, return_tensors="pt")
        if self.y is not None:
            enc["labels"] = torch.tensor([self.y[i] for i in idx], dtype=torch.float32)
        return enc


def amp_forward_module(model, enabled):
    """Wrap `model` so forward returns fp16-autocast logits (works under DataParallel)."""
    import torch

    class AmpWrap(torch.nn.Module):          # autocast inside forward so DataParallel replicas use it too
        """Module returning the squeezed logits of the wrapped model under autocast."""
        def __init__(self, m):
            """Keep the wrapped model."""
            super().__init__()
            self.m = m

        def forward(self, **kw):
            """Logits (one per pair) of the wrapped model, fp16 autocast when enabled."""
            with torch.autocast("cuda", dtype=torch.float16, enabled=enabled):
                return self.m(**kw).logits.squeeze(-1)

    return AmpWrap(model)


# --------------------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------------------

def stage_train(args, ce):
    """Fine-tune the cross-encoder on fold-A pairs: 1 epoch, BCE on one logit, AdamW + linear warmup,
    fp16, all GPUs; saves the model and DONE.json to <ce_dir>/model.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev, n_gpu = devices()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(dev)
    pairs = with_texts(pl.read_parquet(ce / "pairs_A.parquet"), pl.read_parquet(ce / "texts_train.parquet"))
    t1, t2, y = pairs["t1"].to_list(), pairs["t2"].to_list(), pairs["label"].cast(pl.Float32).to_list()
    loader = DataLoader(range(len(pairs)), batch_size=args.train_bs, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=Collate(tok, t1, t2, args.max_len, y), num_workers=args.workers,
                        persistent_workers=args.workers > 0)
    fwd = amp_forward_module(model, n_gpu > 0)
    if n_gpu > 1:
        fwd = torch.nn.DataParallel(fwd)
    steps = len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=n_gpu > 0)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, run_loss = time.time(), 0.0
    log(f"train: {len(pairs):,} pairs, {steps:,} steps of {args.train_bs}, max_len {args.max_len}, "
        f"{n_gpu} GPU(s), lr {args.lr}")
    for step, batch in enumerate(loader, 1):
        labels = batch.pop("labels").to(dev)
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        loss = lossf(fwd(**batch).float(), labels)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run_loss += loss.item()
        if step % args.log_every == 0 or step == steps:
            el = time.time() - t0
            log(f"  step {step:,}/{steps:,}: loss {run_loss / min(step, args.log_every):.4f}, "
                f"{step * args.train_bs / el:,.0f} pairs/s, eta {el / step * (steps - step) / 60:.1f} min")
            run_loss = 0.0
    out = ce / "model"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "DONE.json").write_text(json.dumps({"base": args.model, "pairs": len(pairs), "max_len": args.max_len,
                                               "lr": args.lr, "batch": args.train_bs, "seed": args.seed,
                                               "seconds": time.time() - t0}))
    log(f"train done in {(time.time() - t0) / 60:.1f} min -> {out}")


# --------------------------------------------------------------------------------------
# scoring (sharded, resumable)
# --------------------------------------------------------------------------------------

def score_pairs(args, ce, pairs, texts, name):
    """Score pairs with the fine-tuned model in resumable 500k-pair shards (length-sorted, fixed order).

    Writes <ce_dir>/scores_<name>/part*.parquet and the merged ce_<name>.parquet (s1_id, cand_id, p_ce).
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev, n_gpu = devices()
    shard_dir = ce / f"scores_{name}"
    shard_dir.mkdir(exist_ok=True)
    df = with_texts(pairs.select("s1_id", "cand_id"), texts)
    # sort by text length (fixed order: resumable shards, little padding per batch)
    df = df.with_columns(_len=pl.col("t1").str.len_chars() + pl.col("t2").str.len_chars()) \
           .sort(["_len", "s1_id", "cand_id"])
    n_shards = math.ceil(len(df) / SHARD)
    todo = [k for k in range(n_shards) if not (shard_dir / f"part{k:04d}.parquet").exists()]
    log(f"score {name}: {len(df):,} pairs in {n_shards} shards, {len(todo)} to do, {n_gpu} GPU(s)")
    if todo:
        tok = AutoTokenizer.from_pretrained(ce / "model")
        model = AutoModelForSequenceClassification.from_pretrained(ce / "model").to(dev).eval()
        if n_gpu:
            model = model.half()
        fwd = amp_forward_module(model, False)
        if n_gpu > 1:
            fwd = torch.nn.DataParallel(fwd)
        t0, done = time.time(), 0
        for k in todo:
            part = df.slice(k * SHARD, SHARD)
            loader = DataLoader(range(len(part)), batch_size=args.score_bs, shuffle=False,
                                collate_fn=Collate(tok, part["t1"].to_list(), part["t2"].to_list(), args.max_len),
                                num_workers=args.workers)
            probs = []
            with torch.inference_mode():
                for batch in loader:
                    batch = {k2: v.to(dev, non_blocking=True) for k2, v in batch.items()}
                    probs.append(torch.sigmoid(fwd(**batch).float()).cpu().numpy())
            tmp = shard_dir / f"part{k:04d}.tmp"
            part.select("s1_id", "cand_id").with_columns(p_ce=pl.Series(np.concatenate(probs))).write_parquet(tmp)
            tmp.replace(shard_dir / f"part{k:04d}.parquet")
            done += len(part)
            el = time.time() - t0
            log(f"  shard {k + 1}/{n_shards}: {done / el:,.0f} pairs/s, eta "
                f"{el / done * (sum(min(SHARD, len(df) - j * SHARD) for j in todo) - done) / 60:.1f} min")
    out = pl.read_parquet(str(shard_dir / "part*.parquet"))
    assert len(out) == len(df), f"{name}: {len(out)} scores for {len(df)} pairs"
    out.write_parquet(ce / f"ce_{name}.parquet")
    log(f"score {name} done -> {ce / f'ce_{name}.parquet'}")


def stage_score_b(args, ce):
    """Score every fold-B pair (-> ce_B.parquet)."""
    score_pairs(args, ce, pl.read_parquet(ce / "pairs_B.parquet"), pl.read_parquet(ce / "texts_train.parquet"), "B")


def stage_score_test(args, ce):
    """Score every test candidate (-> ce_test.parquet)."""
    score_pairs(args, ce, test_pairs(args), pl.read_parquet(ce / "texts_test.parquet"), "test")


# --------------------------------------------------------------------------------------
# blend
# --------------------------------------------------------------------------------------

def evaluate_blend(b, universe):
    """Fold-B macro F0.5 (w = 2 and w = 1, best thresholds) of w * p_lgbm + (1 - w) * p_ce per weight."""
    from train_matcher import many_to_one
    from tune_threshold import best_threshold
    n_true = universe["n_true"].to_numpy().astype(float)
    rows = []
    for w in W_GRID:
        p = b.with_columns(prob=pl.when("pf").then(w * pl.col("p_lgbm") + (1 - w) * pl.col("p_ce")).otherwise(0.0))
        k = many_to_one(p)
        code, label, prob = k["code"].to_numpy(), k["label"].to_numpy().astype(float), k["prob"].to_numpy()
        t2, f2, _ = best_threshold(code, prob, label, n_true, 2.0)
        t1, f1, _ = best_threshold(code, prob, label, n_true, 1.0)
        rows.append({"w_lgbm": float(w), "thr_w2": t2, "F0.5_w2": f2, "thr_w1": t1, "F0.5_w1": f1})
    return pl.DataFrame(rows)


def stage_blend(args, ce):
    """Print the fold-B blend table and write the blend submission (config weight / threshold, or
    the fold-B best with --refit) to --out.
    """
    from predict import all_s1_ids, summarize, write_lists
    from train_matcher import many_to_one
    universe = pl.read_parquet(ce / "universe_B.parquet").sort("s1_id").with_row_index("code")
    b = (pl.read_parquet(ce / "pairs_B.parquet").join(pl.read_parquet(ce / "ce_B.parquet"), on=["s1_id", "cand_id"])
           .join(universe.select("s1_id", "code"), on="s1_id"))
    table = evaluate_blend(b, universe)
    pl.Config.set_float_precision(5)
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(200)
    log(f"=== Fold B ({len(universe):,} S1s, {len(b):,} pairs; pre-filtered pairs rejected): "
        f"blend p = w_lgbm * p_lgbm + (1 - w_lgbm) * p_ce; w_lgbm = 1 is LightGBM alone, 0 the cross-encoder alone ===")
    print(table)
    best = table.sort(["F0.5_w2", "w_lgbm"], descending=[True, True]).row(0, named=True)
    lgbm = table.filter(pl.col("w_lgbm") == 1.0).row(0, named=True)
    ce_only = table.filter(pl.col("w_lgbm") == 0.0).row(0, named=True)
    w, thr = (args.w, args.threshold) if args.w is not None else (best["w_lgbm"], best["thr_w2"])
    if args.w is not None and args.threshold is None:
        thr = table.filter(pl.col("w_lgbm") == args.w).row(0, named=True)["thr_w2"]
    log(f"LightGBM alone: w=2 {lgbm['F0.5_w2']:.5f} @ {lgbm['thr_w2']:.2f}, w=1 {lgbm['F0.5_w1']:.5f}; "
        f"CE alone: w=2 {ce_only['F0.5_w2']:.5f} @ {ce_only['thr_w2']:.2f}, w=1 {ce_only['F0.5_w1']:.5f}; "
        f"best blend w_lgbm={best['w_lgbm']:.1f}: w=2 {best['F0.5_w2']:.5f} @ {best['thr_w2']:.2f} "
        f"(gain {best['F0.5_w2'] - lgbm['F0.5_w2']:+.5f}), w=1 {best['F0.5_w1']:.5f}")
    gain = best["F0.5_w2"] - lgbm["F0.5_w2"]
    log(f"VERDICT: blend {'BEATS' if gain >= 1e-4 else 'does NOT beat'} LightGBM alone on fold B "
        f"({gain:+.5f} macro F0.5 at w=2)" + ("" if gain >= 1e-4 else " -> keep the run-4 package"))
    log(f"chosen: w_lgbm={w:.2f}, threshold {thr:.2f}" + (" (pinned)" if args.w is not None else " (fold-B best at w=2)"))
    (ce / "blend.json").write_text(json.dumps({"table": table.to_dicts(), "w_lgbm": w, "threshold": thr,
                                               "lgbm": lgbm, "ce": ce_only, "best": best}, indent=2))
    if args.no_test:
        return
    t = test_pairs(args)
    probs_path = config.CACHE_DIR / "predict" / f"test_probs{'_pf' if config.prefilter_keep_path('test').exists() else ''}.parquet"
    lg = pl.read_parquet(probs_path, columns=["s1_id", "cand_id", "prob"]).rename({"prob": "p_lgbm"})
    t = t.join(lg, on=["s1_id", "cand_id"], how="left").join(pl.read_parquet(ce / "ce_test.parquet"),
                                                               on=["s1_id", "cand_id"], how="left")
    assert t["p_lgbm"].null_count() == 0 and t["p_ce"].null_count() == 0, "test scores do not cover the candidates"
    t = t.with_columns(prob=w * pl.col("p_lgbm") + (1 - w) * pl.col("p_ce"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s1_ids = all_s1_ids("test")
    if args.max_test:
        s1_ids = s1_ids.join(t.select("s1_id").unique(), on="s1_id", how="semi")
    write_lists(s1_ids, t, "candidate_entity_ids", out / "candidate_pairs.tsv")
    k = many_to_one(t)
    sel = k.filter(pl.col("prob") >= thr)
    res = write_lists(s1_ids, sel, "matched_entity_ids", out / "matching_results.tsv")
    base = k.filter(pl.col("p_lgbm") >= config.FINAL_THRESHOLD)
    s1_country = pl.read_parquet(config.norm_path("test", "S1"), columns=["entity_id", "country"]).rename(
        {"entity_id": "s1_id"})
    summarize(res, s1_country, f"output_ce matching_results.tsv: blend w_lgbm={w:.2f} @ {thr:.2f}: {len(sel):,} "
                               f"matches of {len(t):,} candidate pairs (LightGBM @ {config.FINAL_THRESHOLD}: {len(base):,}; "
                               f"added {len(sel.join(base, on=['s1_id', 'cand_id'], how='anti')):,}, removed "
                               f"{len(base.join(sel, on=['s1_id', 'cand_id'], how='anti')):,})")
    t.select("s1_id", rec_id="cand_id", p_lgbm="p_lgbm", p_ce="p_ce").write_parquet(
        out / "test_pair_probs_ce.parquet", compression="zstd")
    log(f"-> {out}")


# --------------------------------------------------------------------------------------

def main():
    """Parse args and run the requested stages in order (each skipped when its output exists)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce_dir", default=str(config.REPO_ROOT / "cache_ce"))
    ap.add_argument("--out", default=str(config.REPO_ROOT / "output_ce"))
    ap.add_argument("--steps", nargs="+", choices=STAGES, default=list(STAGES))
    ap.add_argument("--sample", type=int, default=0, help="use the normalize.py --sample N cache (smoke test)")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--n_train", type=int, default=config.CE_N_TRAIN)
    ap.add_argument("--max_len", type=int, default=config.CE_MAX_LEN)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train_bs", type=int, default=128)
    ap.add_argument("--score_bs", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--max_b", type=int, default=0, help="smoke test: about this many fold-B pairs")
    ap.add_argument("--max_test", type=int, default=0, help="smoke test: only this many test pairs")
    ap.add_argument("--w", type=float, default=config.CE_W_LGBM, help="LightGBM blend weight (config.CE_W_LGBM)")
    ap.add_argument("--threshold", type=float, default=config.CE_THRESHOLD,
                    help="blend threshold (config.CE_THRESHOLD)")
    ap.add_argument("--refit", action="store_true", help="choose weight and threshold on fold B instead")
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=config.SEED, help="fold-A sampling + training seed")
    ap.add_argument("--from_ce", help="second cross-encoder: the first one's ce_dir (fold B must match; test texts "
                                      "are reused)")
    args = ap.parse_args()
    if args.refit:
        args.w = args.threshold = None
    config.use_sample(args.sample)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ce = Path(args.ce_dir)
    ce.mkdir(parents=True, exist_ok=True)
    done = {"pairs": ce / "pairs_B.parquet", "texts": ce / "texts_test.parquet", "train": ce / "model" / "DONE.json",
            "score_b": ce / "ce_B.parquet", "score_test": ce / "ce_test.parquet", "blend": None}
    t_all = time.time()
    log(f"cross-encoder: model {args.model}; cache {config.CACHE_DIR}; ce_dir {ce}; out {args.out}")
    for st in STAGES:
        if st not in args.steps:
            continue
        if done[st] is not None and done[st].exists() and not args.force:
            log(f"--- {st}: checkpoint {done[st].name} found, skipping")
            continue
        t = time.time()
        log(f">>> {st}")
        globals()[f"stage_{st}"](args, ce)
        log(f"<<< {st} {(time.time() - t) / 60:.1f} min, peak mem {peak_mem_gb():.2f} GB")
    log(f"all stages {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
