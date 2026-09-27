# Decisions log

Small choices made without stopping to ask, with the reason. Newest at the bottom.

## Blocking (final)

- **M = 1** (`top_m`): each S2/S3 record keeps only its best-scoring S1. Sample recall 0.990 at M=1
  vs 0.994 at M=2 for twice the candidates; the matcher only has to accept/reject.
- **`pre_cap` = 150** (was 1000, then 300): sample top-1 recall 0.9902 vs 0.9903 at 300, and the
  pair scoring + ranking that it bounds is about half of the per-record blocking time.
  Configurable (`--pre_cap`).
- **Record-level context columns added to blocking output**: `rec_gap` (score minus the record's best
  *other* S1 score, 0 if it has none) and `rec_n` (S1s scored for the record). These need the full
  ranked list, which only exists inside blocking.
- **Train is blocked exactly like test: full train S1 index x full train S2/S3 pool.** (An earlier
  300k-S1 subset was dropped: records whose true S1 was not indexed attached to wrong S1s, which
  shifted `s1_n_cands` and `rec_gap` away from test conditions, and `rec_gap` carries most of the
  model's gain.) Any subsampling for training happens after features, by S1 group (below).
- **Dependencies pinned** in `requirements.txt`: Metaphone 0.6 (BSD, checked in the wheel's LICENSE:
  BSD-style, Collins/Leong/Somerville/Dornseif/McGreggor), numba 0.67.0 (BSD), llvmlite 0.49.0
  (BSD-2-Clause / Apache-2.0 with LLVM exception), lightgbm 4.7.0 (MIT).

- **Key-hit aggregation moved to a numba accumulator** (per record: add key weights into a dense
  per-S1 array, keep the `pre_cap` best by score desc / s1_idx asc). Same result as the polars
  join + group_by + sort (identical hit counts on the France probe; sample top-1 recall 0.9903 vs
  0.9904 before), 3-4x faster per record: the hash group_by over tens of millions of
  (record, S1) groups was the bottleneck at full scale.
- **`expansion_budget` now means (record, S1) pairs scored per chunk** (records per chunk =
  budget / pre_cap). Default 20M; the notebook no longer raises it (bigger chunks measured slower).
- `blocking.py --countries` restricts a run to some countries (used for the local France-only
  end-to-end check; the real run blocks every country).

## Scale probe

- `probe_scale.py` runs each country in its own subprocess so the peak memory is per country, and warms
  up numba on a tiny slice first (JIT compilation after a code edit otherwise shows up as ~100 s of
  "per-record" time).
- Full-scale peak is estimated as probe peak + (in-memory size of the probed records x records in
  country / records probed), since the probe only loads the slice.

## Matcher

- **Features use `name_norm`** (legal suffixes and web tokens stripped, transliterated) for the fuzzy
  set; the full name enters only via `nf_tsort` and the `legal_conflict` flag (both sides carry a legal
  form and they differ).
- **Phonetic similarity is computed for every pair** in the features (blocking computes it only for
  non-Latin names); NaN when either side has no codable word.
- **Fuzzy scores are missing when either side is empty** (LightGBM routes missing values), instead
  of 0. They are stored as **null, not NaN**, in the feature parquet so polars aggregations skip them
  (a NaN made column means NaN and was invisible to `null_count`); `to_numpy()` hands LightGBM NaN.
- **`rank` is dropped at M=1** (it is constant). features.py drops the M-dependent columns when the
  blocking pairs all have rank 1; train uses whatever features are present; predict asserts the test
  features contain every feature the model was trained on.
- **Extra features beyond the brief** (cheap and plausibly useful): `n_extra_s1` (S1 name words
  missing from the record), name lengths, `hs_match`, `postal_match`, `s1_gap` (pair score minus the
  S1's best other candidate), blocking `score` and `rec_n`.
- **Country is not a feature** (France unseen); it is kept in the feature file only for reporting.
- **LightGBM**: lr 0.05, 127 leaves, min 100 samples per leaf, 0.8 feature/bagging fraction, L2 1.0,
  up to 3000 rounds with early stopping (100) on the validation fold. Using the validation fold for
  early stopping makes OOF very slightly optimistic; with millions of rows the effect is negligible.
- **Prediction averages the 5 fold models** instead of refitting on all data (no extra training
  run, and the thresholds were tuned on exactly these models' OOF outputs).
- **Thresholds**: global grid 0.02..0.98 (step 0.01) on OOF macro F0.5; per-source (S2, S3) by
  coordinate ascent starting from the global optimum. Per-source is used only if it beats global by
  more than 1e-4.
- **Training set = random S1 groups up to `TRAIN_MAX_PAIRS` = 5M pairs** (`--max_pairs`, 0 = all). Full
  train blocking gives ~10M pairs (one per record at M=1). S1s are sampled uniformly (seed 42),
  including S1s without candidates, and every pair of a chosen S1 is kept, so per-S1 features and the
  per-S1 metric are exact for the chosen S1s.
- **Metric universe**: every chosen train S1, including S1s with no candidates, so blocking misses and
  singletons count exactly as on the leaderboard.
- **LightGBM learning rate 0.1** (was 0.05) to halve the rounds at full scale; early stopping picks
  the round count.
- **Submission**: every S1 id is read from the raw `test_source1.tsv`, so an S1 dropped anywhere
  upstream still gets a row (empty list). Lists are sorted and comma-joined, no quoting.
- **Validator location**: `run_all.py` looks for `utils/validate_submission.py` in `$BER_VALIDATOR`,
  next to the repo, then next to the dataset directory (the Kaggle dataset layout), and skips with a
  warning if none is found.
- **Weighted negative downsampling for training** (`--max_train_neg`, default 5M negatives per fold):
  LightGBM time scales with rows. Every
  positive is kept, negatives are sampled at frac = 5M / n_neg and weighted 1/frac; early stopping,
  OOF predictions and threshold tuning always use every validation pair. Cost measured on the sample
  when forcing frac = 0.32: OOF macro F0.5 0.9863 -> 0.9855. With the 5M-pair S1-group sample
  (most records have a true S1 in train) it should rarely trigger; it is a safety net.

## Verification done locally

- Full test normalisation (11.7M rows, ~13 min, streaming).
- France-only full test blocking (16 min, 3.8 GB peak) -> features (53 s) -> predict with the sample
  model -> `utils/validate_submission.py`: PASS on all 1,732,544 S1 rows. India / US full test
  blocking does not fit in the local 8 GB (a watchdog killed the India probe at 0.5 GB free), so their
  scale probe runs on Kaggle (`probe` step in run_kaggle.ipynb).

## Kaggle run (Save & Run All)

- `run_all.py` owns checkpointing, logging and the summary so the notebook stays thin: each stage
  is skipped when its output exists and is newer than its inputs' checkpoints (normalised parquet ->
  `blocking/<split>/_DONE.json` -> `features/<split>.parquet` -> `model/meta.json` -> output TSVs).
  Outputs are written to a temp name and renamed on completion, so a crash never leaves a false
  checkpoint. `--force` reruns everything.
- Everything is appended to `/kaggle/working/run.log`; the summary (OOF macro F0.5, thresholds, fold
  AUCs, output sizes, validator verdict) is printed at the end even when a stage fails.
- The runtime estimate printed at the start uses throughput measured on the 4-core laptop
  (`RATE` in run_all.py) and the row counts, with a 1.5x margin for a slower CPU.

## Submission package

- `make_submission_zip.py` adds only the required files (both TSVs, `src/*.py`, README, requirements,
  documentation), so caches / parquet / `__pycache__` / data can never slip in. `run_kaggle.ipynb` and
  this file stay in the repository, not the zip ("exactly the required structure"). The build refuses
  unfilled `{{...}}` placeholders unless `--allow-placeholders`.
- ~~Known issue: literal address "None"~~ **retracted**: an audit of all six files (top-50 full name and
  address strings per source + a junk-value scan) found no literal "None". The "None" seen in the
  error-analysis printout was Python printing an EMPTY (null) raw field. Whole-field junk in the data is
  only empty addresses (130-176k per S2/S3 file) and a few names equal to "na"/"nan"/"null" (<= 60 per file).

## Run 2 (sample-validated; the final package comes from run 2)

- **Whole-field placeholders** (`JUNK_FIELD` in normalize.py): a name or address that is, after strip +
  lower-case, one of none / nan / na / n/a / n.a. / null / <null> / nil / - / -- / . / ? / # / ## / 0 /
  unknown / not available becomes empty. Embedded placeholders were already removed. Effect on the sample:
  none measurable (rare in the data), kept as a safety net for test.
- **Error analysis** (`src/error_analysis.py`) buckets OOF FPs/FNs by pattern and computes the oracle F0.5
  gain of fixing each bucket. Run-1 sample: FP 1,106 (575 records with no true S1 = near-duplicate
  businesses; 531 records whose true S1 is another entity), FN 2,339, blocking misses 1,731. Largest
  buckets: generic names, empty record address, house-number conflicts, extra words.
- **New features, kept**:
  - distinctive words: IDF-weighted name-word Jaccard (IDF from the country's S1 names; unseen words get
    the max IDF), IDF mass and max IDF of the record's extra words / the S1's missing words,
    double-metaphone extra codes on each side (catches "tredars" = traders in transliterated names),
    no-space name ratio ("earnosethroat.com").
  - house numbers: first number equal, conflict (both have numbers, none shared), numeric distance of the
    first numbers (11304 vs 11305), each side's numbers missing from the other.
- **New features, dropped**: S1 counts sharing the S1's / record's normalised name (aimed at empty-address
  records). Ablation: removing them changed OOF F0.5 by +0.0002 (noise level), so they were removed.
- **Ablation on the sample** (OOF macro F0.5): all run-2 features 0.98912; without distinctive words
  0.98815 (-0.0010); without house features 0.98802 (-0.0011); without ambiguity counts 0.98933;
  without all new features 0.98610 (= run 1). The two kept groups are complementary.
- **Result** (sample, final run-2 feature set): OOF macro F0.5 0.9860 -> 0.9893, precision
  0.9935 -> 0.9958, recall 0.9765 -> 0.9784, singleton accuracy 0.969 -> 0.986; FP 1,106 -> 709,
  FN 2,339 -> 2,005. Blocking unchanged (recall 0.9902 at M=1, 8.0 candidates per held-out S1).
- `run_all.py`: the blocking checkpoint now also has to be newer than the normalised parquet (so a
  normalisation change re-runs blocking). `train_matcher.py --drop_features/--out_subdir` for ablations.
- Kaggle run 1 (baseline leaderboard) uses the previously uploaded code and is unaffected.

## Run 3 (threshold for test density; reuses run 2's cache)

- **Test/train density shift** (`analyze_test_outputs.py` on run-2 outputs): test has ~5.8 S2/S3 records
  per S1 vs 4.7 in train, in every country, while predicted matches per S1 (3.2-3.3) and zero-match S1s
  (5.6-6.3%) mirror train truth (3.46, 5.6%). France does not deviate: its accepted pairs are the cleanest
  (2% differing first house number vs 12% US / 18.5% India; train accepted pairs 13.5% / 22%). The extra
  test records are most likely distractors, i.e. ~2x distractors per S1, which hurts precision everywhere.
- **Distractor-weighted threshold tuning** (`tune_threshold.py`): kept OOF negatives count w times in the
  per-S1 precision, P = TP / (TP + w*FP); recall unchanged; a singleton scores 1 only with nothing kept.
  Best global threshold per w in {1, 1.5, 2} (sample: 0.76 / 0.82 / 0.82). Global threshold only: per-source
  thresholds gained < 1e-4 in run 2. Chosen density for the tuned variant: w = 2.0 (the measured ratio).
- **Variants** for the public leaderboard: 0.69 (run-2 OOF optimum), 0.75, 0.80 and the w = 2 optimum
  (0.85 if it coincides with one of those): `output/variants/matching_results_t{thr}.tsv`, all validated.
  4 variants fit the 5-submissions/day budget with 1 in reserve.
- **Final threshold**: `config.FINAL_THRESHOLD` (None until the leaderboard check; then set and committed so
  the package reproduces the submitted file). `predict.py --threshold` overrides it.
- **Rule C (empty address + exact unique name) skipped** unless it improves OOF at w = 2: on the sample the
  292 pairs it newly accepts are 67.8% true and macro F0.5 at w = 2 drops 0.98776 -> 0.98754. The full-run
  verdict is in `model/threshold.json` / run.log.
- **Probabilities exported**: `output/test_pair_probs.parquet` (s1_id, rec_id, prob for every test
  candidate), `output/oof_pair_probs.parquet` (+ label, source, rule C flag) and `output/oof_universe.parquet`
  (s1_id, country, n_true), so thresholds can be re-tuned and new variants written locally without a Kaggle run.
  Test probabilities are cached in `cache/predict/`, so a threshold-only re-run re-writes TSVs only.
- **Resume**: the notebook copies `cache/` from an attached earlier notebook output (timestamps preserved)
  and runs with `--trust-cache` (a checkpoint only has to exist). Safe only when the code of the skipped
  stages is unchanged, which holds for run 3 (normalize/blocking/features/train identical to run 2).
- In sample mode `tune_threshold.py` exports into the cache, never into `output/`.

## Run 4 (candidate pre-filter; reuses run 3's cache)

- **Leaderboard so far**: run 1 0.956, run 2 0.963 (threshold 0.69), run 3 0.963 (threshold 0.79, w = 2). The
  threshold curve is flat, so the threshold is no longer the lever. `config.FINAL_THRESHOLD = 0.79` (the w = 2
  OOF optimum of run 3, same LB score as 0.69 and more robust to test's higher distractor density).
- **Goal**: a smaller candidate set per S1 (the organisers rank smaller candidate sets higher) at near-zero F0.5
  cost. Rule: `candidate_pairs.tsv` must be exactly the set the matcher runs inference on, so pruning is a stage
  BEFORE the matcher (`prefilter.py`), never "drop what the matcher rejected".
- **Stage 1** = LightGBM (63 leaves, 300 rounds, 5-fold GroupKFold by S1 on the matcher's train universe) on
  19 inputs: blocking outputs (score, name_sim, ph_sim, addr_sim, key_score, n_keys, rec_gap, rec_n, source)
  plus cheap equality flags from the features stage (house_match, hn_first_eq, hn_conflict, hn_first_dist,
  hn_s1_missing, hn_rec_missing, postal_match, legal_conflict, n_exact, hs_match). No fuzzy/IDF string score.
  - **No S1-context features** (s1_n_cands, s1_rank, s1_gap): they scale with the candidates-per-S1 density,
    ~2x higher on test. Without them the sample lost only ~4% of the pruning (4.04 vs 3.88 candidates/S1 at
    the same loss), and stage 1 becomes independent of that density.
  - **Cheap flags added** after a France test check (below): blocking signals alone separate the hard test
    distractors poorly (rejected France pairs had median p1 0.64 while the matcher gave them 0.0014). At an
    equal number of dropped run-3 matches, the flags prune clearly more (France, sample-trained stage 1:
    ~0.03% of matches dropped -> 5.24 vs 5.42 candidates/S1).
- **Cutoff** = largest p1 cutoff whose OOF macro F0.5 (w = 2, threshold 0.79, pruned pairs = rejected) is at
  most **0.0001** below no pruning (`prefilter.MAX_LOSS`), chosen on the full OOF in the Kaggle run and printed
  as a trade-off table (avg / p95 candidates per S1, % true pairs lost, % matcher-accepted pairs lost, F0.5).
  0.0001 rather than the 0.0005 target: on the sample the curve has a knee there (8.08 -> 3.74 candidates/S1
  for -0.00009; the next 0.0004 buys only 3.74 -> 3.49), and test drops ~2.5x more accepted pairs than OOF
  at the same cutoff (hard distractors), so the OOF budget is kept small. The cutoff is chosen on train OOF
  only; calibrating it on the matcher's test decisions would reintroduce "drop what the matcher rejected".
  `config.PREFILTER_CUTOFF` pins it afterwards so the package reproduces the submitted candidate file.
- **Matcher not retrained**: it scores the surviving pairs with features computed on the full blocking list
  (S1-context features included), so the pruned OOF score is exactly what the pipeline does. `tune` counts
  pruned OOF pairs as rejected (re-tuned thresholds and variants include the pre-filter).
- **Sample is not representative of the pruning size**: the sample's negatives are mostly easy (records whose
  true S1 is not in the 50k sample), so 8.08 -> 3.74 candidates/S1 there overstates it. Test France with the
  sample-trained stage 1 at the sample cutoff 0.015: **5.53 -> 5.21 candidates/S1 (-6%)**, 433 of 843,091 run-3
  matches (at 0.79) dropped, 429 S1s affected, worst-case loss 0.00018 on France (1 - macro F0.5 of the kept vs
  run-3 matches, i.e. assuming every dropped match was correct). The full-OOF cutoff and the full stage-1 model
  will differ; run 4 prints the same diagnostic for all countries (from run 3's cached test probabilities,
  diagnostic only).
- **Local France check**: `blocking.py --split test --countries France` locally reproduces run 3's France
  candidates (1,434,089 of 1,434,108 pairs identical; the rest are float ties); features for France in 61 s.
- **Resume semantics**: with `--trust-cache` a checkpoint now becomes stale when one of its inputs was rebuilt
  during the same run (so `tune` reruns after the new `prefilter`). Test probabilities of a pre-filtered run are
  cached as `predict/test_probs_pf.parquet`, never confused with run 3's unfiltered `test_probs.parquet`.

## Run 5 (pairwise-only matcher + ensemble; reuses run 4's cache)

- **Leaderboard**: run 2, 3 and 4 all 0.963 (thresholds 0.69 / 0.79, with and without the pre-filter); top
  teams 0.991. Run 4 is the safe fallback (`run4/`).
- **Recall hypothesis checked on run 3's outputs** (test 74% vs ~58% of records assigned):
  - true matches per S1 in train are 3.457 (US) / 3.468 (India), i.e. a generator constant; test has ~1.1 more
    records per S1 (5.8 vs 4.66 candidates/S1 in OOF), most simply explained by extra distractors;
  - we predict 3.20 matches/S1 on test at 0.79 vs 3.29 on OOF (OOF recall 94.8%), so at most ~3% extra recall
    is missing on test, not 20-25%;
  - test probabilities are more polarised, not stuck mid-band: 34% of test pairs < 0.01 (OOF 24%), 6.7% in
    0.05-0.79 (OOF 3.1%), 55.7% >= 0.79 (OOF 70.7%); even t = 0.15 assigns only 59-64% of records;
  - France test pairs with prob < 0.001 have high address similarity (0.78) but 84% conflicting house numbers
    (P(match | house-number conflict) = 6.6% in train): designed sibling-branch distractors, not missed matches;
  - OOF ceiling (perfect matcher on our candidates) 0.9878 < 0.991: blocking recall (M = 1) also caps us.
  Probe files from run 3's probabilities: `run3/probe/matching_results_t{0.50,0.30,0.15}.tsv`
  (`probe_thresholds.py`), for an LB check of the recall hypothesis at no pipeline cost.
- **Pairwise-only model** (`train_pw`): same LightGBM, universe and GroupKFold, without
  `config.CONTEXT_FEATURES` = rec_gap, rec_n, s1_n_cands, s1_rank, s1_gap, key_score, n_keys. key_score and
  n_keys are dropped too: they come from blocking-key frequencies, which change with how crowded a
  neighbourhood is. `score` (a pair similarity) is kept.
- **Evaluation** (`models.py`): macro F0.5 at w = 1 and w = 2 per model, pruned pairs = rejected, and on the
  top quartile of S1s by candidates per S1 as a proxy for test's density (the w = 2 weighting only inflates
  false positives, it does not move the context features). Sample (50k S1s): base 0.98768 / pairwise 0.98379 /
  ensemble 0.98737 at w = 2; dense quartile 0.98495 / 0.97951 / 0.98490. So on the sample the context features
  help, dense neighbourhoods included; the full-OOF numbers come from the Kaggle run.
- **Variants** (max 4): pairwise and ensemble at their w = 2 OOF-best threshold, and both at that - 0.15
  (the recall probe for each). predict.py's base variants are off (`config.BASE_VARIANTS = False`);
  `matching_results.tsv` stays base @ 0.79 (= run 4) unless `config.FINAL_MODEL` says otherwise.
- **Same candidate file for every model**: all score exactly the pre-filter keep set, so
  `candidate_pairs.tsv` is shared and every variant is a subset of it.

## Final submission (run 4 package)

- **Public leaderboard, threshold curve** (same model): 0.30 → 0.957, 0.69 → 0.963, 0.79 → 0.963 (0.963 also
  with the pre-filter, run 4), 0.90 → 0.961. It peaks at the OOF-chosen 0.79 (distractor-weighted, w = 2), so the
  threshold stays 0.79 and the run-4 package is final (`config.PREFILTER_CUTOFF = 0.025` pinned).
- **Pairwise-only model and ensemble (run 5): built but NOT used.** It was never run on Kaggle or submitted: on
  the sample it scored below the base model (w = 2: 0.98379 vs 0.98768; dense quartile 0.97951 vs 0.98495), and
  the leaderboard probes showed test is precision-limited, not recall-limited. `train_pw` / `models` are
  optional steps, removed from `run_all.py`'s default steps, so reproducing the package runs exactly
  normalize → blocking → features → train → prefilter → tune → predict → validate (~4.7 h on Kaggle).

## Final submission: cross-encoder blend (run CE, public LB 0.972)

- **Why:** test is precision-limited (threshold curve 0.30 → 0.957, 0.79 → 0.963, 0.90 → 0.961) and the errors are
  look-alike distractors (sibling names, nearby house numbers, transliterations) that handcrafted string features
  separate poorly.
- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, Apache-2.0, 117,654,272 parameters
  (checked on the Hugging Face model card), fine-tuned as a binary pair classifier on raw "name | address" text
  (native script kept), max_len 96, 1 epoch, fp16, seed 42, on 1.2M pairs from fold A (70% of the matcher's OOF
  S1s). No external data; only the pretrained weights are downloaded.
- **Same candidates:** it scores exactly the run-4 pre-filter keep set; `candidate_pairs.tsv` is byte-identical
  to run 4's.
- **Fold B** (30% of S1s, 321,861 S1s, 1,501,064 pairs; LightGBM probabilities are OOF): macro F0.5 at w = 2 —
  LightGBM alone 0.97695 @ 0.79, CE alone 0.97512 @ 0.78, blend 0.6·LGBM + 0.4·CE 0.98087 @ 0.69 (+0.0039);
  at w = 1: 0.97871 / 0.97766 / 0.98185.
- **Leaderboard:** 0.972 vs 0.963 (+0.009, more than fold B's +0.0039: the CE helps most on test's denser
  look-alikes). Test: 5,620,486 matches (LightGBM @ 0.79: 5,543,011; 131,229 added, 53,754 removed).
- **Runtime:** 86 min on Kaggle GPU T4 x2 (train 39, test scoring 37). Pinned in `config.CE_W_LGBM` /
  `CE_THRESHOLD`; `ML_submission_ce.zip` is the final package, `ML_submission.zip` (run 4) the fallback.

### Blend weight chosen by fold B + public LB (final: 0.4 @ 0.73, LB 0.973177)

- Fold B (w = 2) is flat over w_lgbm 0.4-0.7 (0.98054-0.98087), so the three best-scoring weights were probed on the
  public LB, each at its fold-B-best threshold at w = 2: 0.6 @ 0.69 → 0.972, 0.5 @ 0.69 → 0.973,
  **0.4 @ 0.73 → 0.973177**. More cross-encoder weight helps on test, consistent with its larger test gain.
- Final: `config.CE_W_LGBM = 0.4`, `CE_THRESHOLD = 0.73`; 5,643,029 matches (vs LightGBM alone at 0.79: 173,931 added,
  73,913 removed). The 0.6 package is kept as `ML_submission_ce_w0.6.zip`, run 4 as `ML_submission.zip`.

## Final submission: stacking (stack_ctx @ 0.78, public LB 0.977)

- **Why:** the blend uses one global weight; a second-level model can weigh LightGBM and the cross-encoder per pair
  type (empty address, house-number conflicts, ...) and see how a pair ranks among its S1's candidates.
- **No leakage:** trained only on fold B, whose p_lgbm is the matcher's OOF and whose p_ce comes from the
  cross-encoder trained on fold A; 5-fold GroupKFold by S1 inside fold B.
- **Fold B OOF** (321,861 S1s, 1,501,064 pairs), macro F0.5 w = 2 / w = 1: LightGBM 0.97695 / 0.97871, CE 0.97512 /
  0.97766, blend 0.4/0.6 0.98054 / 0.98155, stack 0.98311 / 0.98369, **stack_ctx 0.98342 / 0.98396** (+0.00288 over
  the blend at w = 2). Top gain: blend 91%, p_lgbm 3%, p_ce 2%, rec_gap 0.6%.
- **Leaderboard:** 0.977 (history: LightGBM 0.963 → blend 0.6 @ 0.69 0.972 → blend 0.4 @ 0.73 0.973177 → stack_ctx
  0.977). Test: 5,730,770 matches (vs the blend file: 111,739 added, 23,998 removed).
- **Bug on the first Kaggle run:** the fold-B feature join was checked through nulls of the first feature, which is
  legitimately null for empty names; now checked with an indicator column (100% matched).
- **Runtime:** 19 min on Kaggle (training 10 min, test scoring 7 min). Pinned: `config.STACK_VARIANT`,
  `STACK_THRESHOLD`; `CE_W_LGBM` stays 0.4 (it defines the blend feature). Backups: `ML_submission_ce_w0.4.zip`,
  `ML_submission_ce_w0.6.zip`, `ML_submission_run4.zip`.

## Final submission: stacking with two cross-encoders (stack_ctx2 @ 0.83, public LB 0.978359)

- **CE v2:** same model (Apache-2.0, 117.7M), same fold A/B split enforced by `cross_encoder.py --from_ce` (fold B
  identical pair for pair, no fold A/B overlap), 2.2M fold-A pairs, max_len 128, seed 2026 (~2.3 h on 2×T4; 2.5M
  pairs would have exceeded the 2.5 h budget).
- **stack_ctx2** (`stack.py --ce2_dir`): stack_ctx + p_ce2, blend2, mean(CE, CE2), blend12 and CE2's S1 context.
  Fold B at w = 2: 0.98401 @ 0.83 vs stack_ctx 0.98342 (+0.00059; bar set before the run: +0.0005). Public LB
  0.978359 (+0.0014 over 0.977). Pinned: `config.STACK_VARIANT = "stack_ctx2"`, `STACK_THRESHOLD = 0.83`;
  `STACK_V1_THRESHOLD = 0.78` keeps the previous model reproducible.
- **Not adopted** (public LB): stack_ctx2 @ 0.75 (its w = 1 threshold) 0.978254; 5-seed bagged stack_ctx2
  (`stack3.py`, lr 0.1) 0.978289. A third cross-encoder (`--ce3_dir`, multilingual-e5-small; XLM-R base needs
  ~4-5 h on 2×T4) was built but not run before the deadline.
- Backups: `ML_submission_stack_v1.zip` (stack_ctx, 0.977), `ML_submission_ce_w0.4.zip` (0.973177),
  `ML_submission_ce_w0.6.zip` (0.972), `ML_submission_run4.zip` (0.963).
