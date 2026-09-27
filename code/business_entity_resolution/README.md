# Business entity resolution: reproduction guide

For every Source 1 (S1) entity, the pipeline finds all matching Source 2/3 records, following
**normalize → reverse blocking → features → candidate pre-filter → LightGBM matcher → thresholds tuned for macro F0.5**,
then **two fine-tuned multilingual cross-encoders** (CE v1, CE v2) score the same candidates, and a **second-level
LightGBM (stacking, `stack_ctx2`)** combines all scores with the pair features (final submission, public LB 0.978359).
A 2-page overview is `ML_Approach_Summary.md` / `.pdf` at the package root. It uses only the
provided data: no external APIs, lookups or geocoding. The methodology is in `Documentation_template.md` at the
package root, and design decisions with their measurements are in `DECISIONS.md` (repository copy).

## Requirements

- Python 3.11 or newer (developed on 3.14, run on Kaggle's Python), with 4 or more CPU cores.
- **RAM:** at least 16 GB for the full run (Kaggle's 30 GB is plenty). Blocking holds one country's full S1 index
  in memory at a time. An 8 GB machine can run the `--sample` smoke test and France-only blocking, but not India
  or US.
- **Disk:** about 5 GB for caches.
- **Dependencies:** `pip install -r requirements.txt`. Versions are pinned, and every package is MIT, BSD, ISC or
  Apache-2.0 licensed.

## Data layout

```
<DATA_DIR>/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
<DATA_DIR>/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

Paths are set with environment variables or `run_all.py` flags:

| variable | flag | default |
|---|---|---|
| `BER_DATA_DIR` | `--data-dir` | `<student_resource>/dataset` |
| `BER_CACHE_DIR` | `--cache-dir` | `<student_resource>/cache` |
| `BER_OUTPUT_DIR` | `--output-dir` | `<student_resource>/output` |
| `BER_WORKERS` | | normalisation processes; default = CPU count |
| `BER_VALIDATOR` | | path to `utils/validate_submission.py`; otherwise searched next to the repo and the dataset |

## One command (end to end)

Run from `student_resource/`:

```bash
python code/business_entity_resolution/src/run_all.py --log run.log
```

This runs the default stages in order. Each stage is a separate process, and each writes a checkpoint that is
reused on a rerun, but only if it is newer than its inputs. `--force` reruns everything.

| # | stage | script | output (under the cache dir unless noted) | est. time* |
|---|---|---|---|---|
| 1 | normalize | `normalize.py` | `normalized/{train,test}_S{1,2,3}.parquet` | ~27 min |
| 2 | blocking | `blocking.py --split train`, then `--split test` | `blocking/<split>/pairs_<country>.parquet`, `_DONE.json` | ~3.1 h |
| 3 | features | `features.py --split train`, then `--split test` | `features/<split>.parquet` | ~20 min |
| 4 | train | `train_matcher.py` | `model/fold{0..4}.txt`, `model/meta.json` (thresholds, OOF scores), `model/oof.parquet` | ~30 min |
| 5 | prefilter | `prefilter.py` | `model/prefilter/` (stage-1 folds, `prefilter.json` with the OOF trade-off and cutoff), `prefilter/{train,test}_keep.parquet`, `<output>/oof_prefilter.parquet`, `<output>/test_prefilter.parquet` | ~30 min |
| 6 | tune | `tune_threshold.py` | `model/threshold.json` (distractor-weighted thresholds, variants), `<output>/oof_pair_probs.parquet`, `<output>/oof_universe.parquet` | ~3 min |
| 7 | predict | `predict.py [--threshold T]` | **`<output>/matching_results.tsv`, `<output>/candidate_pairs.tsv`** (= the pre-filtered set the matcher scored), `<output>/variants/matching_results_t*.tsv`, `<output>/test_pair_probs.parquet` | ~75 min |
| 8 | validate | `utils/validate_submission.py` | printed verdict for the main file and every variant | ~5 min |

\*Estimated from throughput measured on a 4-core laptop. **Total about 6 h estimated; measured on a Kaggle CPU
session about 4.7 h**, and up to about 9 h on a slower CPU. These 8 stages are exactly what produced the
submitted package (run 4). `run_all.py` prints its own estimate at the start of `run.log`.

Two optional stages:
- `--steps probe` measures the full-scale blocking cost on test: the full S1 index against a 20k-record slice
  per country.
- `--steps eval` measures blocking recall on a 10% held-out slice of train S1s.

Blocking settings can be passed straight through, e.g. `run_all.py --pre_cap 100`. The defaults are M = 1,
`pre_cap` 150, `block_cap` 2000 and `n_rare` 2 (see `BlockingParams` in `blocking.py`).

At the end, `run.log` holds a **SUMMARY**: OOF macro F0.5, the chosen threshold(s), fold AUCs, output file sizes
and the validator verdict.

**Threshold of the submission.** `matching_results.tsv` uses `config.FINAL_THRESHOLD` (chosen from the
distractor-weighted OOF tuning plus a public-leaderboard check of the variants; see the documentation, section 4.3).
`predict.py --threshold T` overrides it; `None` falls back to the OOF-optimal threshold.

**Candidate pre-filter.** `candidate_pairs.tsv` must be exactly the set the matcher scores, so pruning happens
before the matcher: `prefilter.py` trains a small stage-1 LightGBM on blocking signals plus cheap equality flags
(house number, postal code, legal form, exact name), picks the largest cutoff that loses at most 0.0001 OOF macro
F0.5 (w = 2, at `FINAL_THRESHOLD`), and writes the kept test pairs; `predict.py` scores only those.
`config.PREFILTER = False` turns it off; `config.PREFILTER_CUTOFF` pins the cutoff of a submitted run.
`simulate_prefilter_test.py` replays a stage-1 model on the test outputs of an earlier run (diagnostic only).

## Local smoke test (8 GB machine)

```bash
python code/business_entity_resolution/src/run_all.py --sample 50000 --steps normalize blocking features train
```

This uses 50,000 train S1s, all of their true matches, and a proportional slice of other records. It is written
to `cache/sample_50000/`, takes about 10 minutes, and prints the blocking and OOF matcher report. Test-side stages
are skipped in sample mode.

## Kaggle (full run, unattended)

1. Zip the folder `code/business_entity_resolution` itself, so the archive's top-level entry is
   `business_entity_resolution/`. Upload it as a Kaggle dataset.
2. Create a notebook from `run_kaggle.ipynb` and attach two inputs: the challenge data and the code dataset.
   Turn **Internet ON**; it is needed only for `pip install`.
3. Click **Save Version → Save & Run All**.
   - The notebook finds the data and code automatically, copies the code to `/kaggle/working`, installs the
     requirements, and runs `normalize → probe → blocking → features → train → prefilter → tune → predict →
     validate`.
   - Outputs land in `/kaggle/working/output/{matching_results.tsv, candidate_pairs.tsv}`.
   - Checkpoints go to `/kaggle/working/cache/` and the progress log to `/kaggle/working/run.log`.
   - The last cell prints the summary.
4. Download `output/` and `run.log` from the saved version's Output tab.

**Resume from an earlier run (new late stages, threshold or predict-only changes).** In the new notebook version, click
**Add Input → Notebook Output Files** (or "Your Work" → Notebooks) and select the earlier version of this notebook.
The notebook copies its `cache/` into `/kaggle/working/cache` and runs with `--trust-cache`, so normalize,
blocking, features and train are skipped. A stage whose inputs were rebuilt during the run (e.g. `tune` after a
new `prefilter`) still reruns. Do this only when the code of the skipped stages is unchanged.

## Reproducing the final submission: 4 stages

| # | stage | command / notebook | hardware, time | output |
|---|---|---|---|---|
| 1 | CPU pipeline: normalize → blocking → features → train → prefilter → tune → predict → validate | `src/run_all.py` / `run_kaggle.ipynb` | CPU, ~4.7 h | cache (features, OOF, pre-filter keep sets, test p_lgbm), `output/candidate_pairs.tsv` |
| 2 | CE v1: pairs → texts → train → score_b → score_test → blend | `src/cross_encoder.py` / `run_kaggle_ce.ipynb` | GPU T4 x2, ~86 min | `cache_ce/` (fold A/B split, CE model, fold-B and test scores) |
| 3 | CE v2: same model, same fold B (`--from_ce`), 2.2M pairs, max_len 128, seed 2026 | `src/cross_encoder.py --from_ce` / `run_kaggle_ce2.ipynb` | GPU T4 x2, ~2.3 h | `cache_ce2/` |
| 4 | stacking `stack_ctx2`: fold-B second-level LightGBM → test | `src/stack.py --ce2_dir` / `run_kaggle_stack2.ipynb` | CPU, ~35 min | **`output_stack2/matching_results.tsv`** (final), `candidate_pairs.tsv` (= stage 1's) |

Run from `student_resource/` after stage 1:

```bash
python code/business_entity_resolution/src/cross_encoder.py --ce_dir cache_ce --out output_ce
python code/business_entity_resolution/src/cross_encoder.py --ce_dir cache_ce2 --out output_ce2 --from_ce cache_ce \
    --n_train 2200000 --max_len 128 --seed 2026 --refit
python code/business_entity_resolution/src/stack.py --ce_dir cache_ce --ce2_dir cache_ce2 --out output_stack2 \
    --stack_dir stack2_models
```

The candidate set is fixed after stage 1: stages 2-4 only rescore it, so `candidate_pairs.tsv` never changes.
The submitted `output/matching_results.tsv` is `output_stack2/matching_results.tsv`.

## Stage 2: cross-encoder CE v1 (GPU, ~86 min)

- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 117.7M parameters), downloaded
  from the Hugging Face Hub (weights only) and fine-tuned on train pairs. Settings live in `config.py` (`CE_*`).
- Needs the cache of stage 1 (features, `model/oof.parquet`, pre-filter keep sets, `predict/test_probs_pf.parquet`)
  and a CUDA GPU (CPU only works for a smoke test).

Stages (checkpointed in `cache_ce/`, a rerun skips finished ones):

| stage | what | Kaggle T4 x2 |
|---|---|---|
| pairs | matcher's OOF universe split by S1 id (fixed seed): fold A 70% (1.2M training pairs sampled), fold B 30% (1.5M pairs, held out for blending / stacking) | 0.6 min |
| texts | raw `name \| address` (native script kept) for every id needed | 0.8 min |
| train | 1 epoch, fp16, seed 42, max_len 96, batch 128, lr 5e-5, both GPUs → `cache_ce/model/` | 39 min |
| score_b | CE probability of every fold-B pair | 7 min |
| score_test | CE probability of all 7,726,561 test candidates (500k-pair shards, resumable) | 37 min |
| blend | fold-B table (w_lgbm = 0..1, distractor weight 1 and 2); test: 0.4·p_lgbm + 0.6·p_ce ≥ 0.73 → `output_ce/` (LB 0.973177; informational) | 1.3 min |

**On Kaggle:** new notebook from `run_kaggle_ce.ipynb`; Accelerator **GPU T4 x2**, Internet **ON** (pip + model
download); inputs: the challenge data, the code dataset and the stage-1 notebook output. Save & Run All. The notebook
reinstalls polars 1.44.2 cleanly, checks that `output_ce/candidate_pairs.tsv` is byte-identical to stage 1's and runs
the validator. To resume after a crash, attach the notebook's own output: `cache_ce/` is copied back.

## Stage 3: cross-encoder CE v2 (GPU, ~2.3 h)

Same model and the **same fold A/B split** as CE v1, with a different fold-A sample (2.2M pairs, seed 2026) and
max_len 128. `--from_ce cache_ce` makes the pairs stage fail unless fold B is identical to v1's, pair for pair, and
reuses v1's test texts. On Kaggle: `run_kaggle_ce2.ipynb`, GPU T4 x2, Internet ON; inputs: data, code, the stage-1
output and the stage-2 output (`cache_ce/`).

## Stage 4: stacking `stack_ctx2` (final submission, ~35 min)

- Fold B (321,861 S1s, 1.5M pairs; p_lgbm out-of-fold, both CE scores held out), 5-fold GroupKFold by S1 inside it.
- Trains `stack_ctx` (51 matcher features + p_lgbm + p_ce + blend + S1 context: the LB 0.977 model, as reference)
  and `stack_ctx2` (+ p_ce2, blend2, mean(CE, CE2), blend12 and CE2's S1 context; 70 features). LightGBM lr 0.05,
  63 leaves, early stopping 100 (cap 1,500).
- Prints the fold-B table and `VERDICT CE v2`; scores all test candidates with the 5 fold models of `stack_ctx2`,
  many-to-one, match if p ≥ `config.STACK_THRESHOLD` = 0.83 (its fold-B best at distractor weight 2: 0.98401 vs
  0.98342 for `stack_ctx`). `--refit` re-chooses the threshold on fold B.
- Writes `output_stack2/matching_results.tsv`, `candidate_pairs.tsv`, `variants/` (w = 1 threshold), `stack.json`,
  `test_pair_probs_stack.parquet`.
- On Kaggle: `run_kaggle_stack2.ipynb` (CPU session, Internet ON for pip); inputs: data, code, stage-1, stage-2 and
  stage-3 outputs. It checks that `candidate_pairs.tsv` is byte-identical to stage 1's and validates every output.

Without `--ce2_dir`, `stack.py` reproduces the previous submission (`stack_ctx` @ `config.STACK_V1_THRESHOLD` = 0.78,
LB 0.977; `run_kaggle_stack.ipynb`).

**Experiments not in the submission** (code kept for reference): `stack3.py` (5-seed bagging of `stack_ctx2`, LB
0.978289), `stack_ctx2` at its w = 1 threshold 0.75 (LB 0.978254), `stack.py --ce3_dir` with a third cross-encoder
(`run_kaggle_ce3.ipynb`, `run_kaggle_stack4.ipynb`; not run). The experiment notebooks (`run_kaggle_ce3`,
`_stack3`, `_stack4`) are in the repository only, not in the submission package.

## Filling in the documentation and building the submission

Run from `student_resource/`, with `output/` and `run.log` downloaded from Kaggle:

```bash
python code/business_entity_resolution/src/report_numbers.py --log run.log --output-dir output \
    --test-dir dataset/test --fill Documentation_template.md
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
python code/business_entity_resolution/src/make_submission_zip.py ML
```

- `report_numbers.py` prints and fills the `{{...}}` numbers: OOF F0.5, precision, recall, threshold,
  candidates per test S1 (avg / p95 / % with any) and the share of test S1s with matches, by country.
  - **Fill by hand:** team members, date and leaderboard score. It lists whatever is still unfilled.
- Final package (`stack_ctx2`): copy `output_stack2/matching_results.tsv` and stage 1's `candidate_pairs.tsv` to
  `output/`, then `make_submission_zip.py ML`.
- `make_submission_zip.py ML` writes `ML_submission.zip` with exactly `output/` (2 TSVs),
  `code/business_entity_resolution/{src, README.md, requirements.txt, run_kaggle{,_ce,_ce2,_stack,_stack2}.ipynb}`,
  `Documentation_template.md` and `ML_Approach_Summary.{md,pdf}`.
  - It refuses to build if a file is missing, a TSV header is wrong, or placeholders remain.

## Source files

| file | role |
|---|---|
| `src/run_all.py` | entry point: stages, checkpoints, log, runtime estimate, summary |
| `src/config.py` | paths, seeds (42), training sample size |
| `src/normalize.py` | streaming normalisation (anyascii transliteration, legal forms, abbreviations, house numbers) |
| `src/blocking.py` | reverse blocking: keys, combination keys, numba accumulator, TF-IDF/phonetic/house-number ranking |
| `src/features.py` | pairwise features |
| `src/train_matcher.py` | LightGBM, 5-fold GroupKFold by S1, exact macro-F0.5 threshold search |
| `src/cross_encoder.py` | stages 2-3: fine-tuned multilingual cross-encoder on the run-4 candidates (CE v1; CE v2 with `--from_ce`), fold-B blend with LightGBM |
| `src/stack.py` | stage 4 (final, `--ce2_dir`): second-level LightGBM on fold B over matcher features + all probabilities, `output_stack2/` |
| `src/stack3.py` | experiment: seed-bagged `stack_ctx2` (LB 0.978289, not submitted) |
| `src/models.py` | unused experiment (run 5): base vs pairwise-only vs ensemble on OOF (+ dense-quartile proxy) |
| `src/probe_thresholds.py` | low-threshold LB probe files from an earlier run's cached test probabilities |
| `src/prefilter.py` | candidate pre-filter: stage-1 model, OOF trade-off table, cutoff, test keep set |
| `src/tune_threshold.py` | distractor-weighted OOF threshold tuning, rule C check, OOF probability export |
| `src/predict.py` | scores the pre-filtered candidates, writes the submission TSVs (every test S1 gets a row) |
| `src/simulate_prefilter_test.py` | replays the pre-filter on an earlier run's test outputs (sanity check) |
| `src/analyze_test_outputs.py` | per-country test output checks against train ground-truth structure |
| `src/eval_blocking.py` | blocking recall by stage on held-out train S1s |
| `src/probe_scale.py` | full-scale blocking cost probe |
| `src/report_numbers.py` | numbers for the documentation |
| `src/make_submission_zip.py` | submission zip |
| `src/eda.py` | exploratory analysis (`reports/eda_report.txt`) |
| `src/error_analysis.py` | OOF error buckets with the oracle gain of fixing each |
| `src/memory.py` | peak-memory helper |

**Run 5 (optional, unused experiment: pairwise-only model + ensemble).** Not in the default steps and not part
of the submitted package; run it with `--steps train_pw models`. `train_pw` trains the matcher again without
`config.CONTEXT_FEATURES` (rec_gap, rec_n, s1_n_cands, s1_rank, s1_gap, key_score, n_keys: their distribution
depends on how many candidates surround an S1 / a record). `models` compares base, pairwise and their mean on
OOF (w = 1, w = 2, and the top quartile of candidates per S1), scores the pre-filtered test pairs with the
pairwise model (base probabilities come from predict's cache) and writes up to 4 variants
`output/variants/matching_results_{pairwise,ensemble}_t{thr}.tsv` (OOF-best at w = 2, and that - 0.15).
`config.FINAL_MODEL` picks the model of `matching_results.tsv` ("base" by default), `config.MODEL_THRESHOLDS`
the threshold per model (None = OOF-best at w = 2).
