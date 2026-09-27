# Business Entity Resolution: Approach Summary

**Team:** ML (Shaik Asad Ahmed). **Final submission:** `stack_ctx2` @ 0.83, **public LB 0.978359**.

## 1. Problem and metric

Each Source 1 (S1) business must be matched to every S2/S3 record of the same real business. The score is macro
F0.5 per S1, which weights precision 4× more than recall. A singleton S1 (no true match) scores 1 only if nothing is
predicted for it, else 0. Test holds 1.73M S1s and 9.97M S2/S3 records in three countries, including France,
which is absent from train. Country is treated as an open set, with no country-specific code.

## 2. Approach overview

```
raw TSVs ─► normalize      names: legal forms, confusable characters, phonetic codes; addresses: postal, house numbers
         ─► reverse blocking (M = 1)   each S2/S3 record looks up its keys in a per-country S1 inverted index
                           and keeps its single best S1 (TF-IDF + phonetic + house-number score)
         ─► pre-filter     stage-1 LightGBM drops hopeless pairs (p1 < 0.025) → 7.73M candidates = candidate_pairs.tsv
         ─► LightGBM matcher      51 pair features, 5-fold GroupKFold by S1 → p_lgbm
         ─► cross-encoders CE v1 + CE v2   multilingual MiniLM-L12 fine-tuned on raw "name | address" pairs → p_ce, p_ce2
         ─► stacking (stack_ctx2)   2nd-level LightGBM on the held-out fold → many-to-one → threshold 0.83
```

- **Many-to-one:** in the train truth every record belongs to at most one S1. So each record keeps only its best
  S1, both when blocking (M = 1) and before thresholding.
- **Stacking inputs (70 features):**
  - the 51 matcher features;
  - p_lgbm, p_ce, p_ce2;
  - four blends: 0.4·p_lgbm + 0.6·p_ce, the same with p_ce2, the mean of p_ce and p_ce2, and the blend with that mean;
  - S1 context for each probability: its rank among the S1's candidates, the S1's maximum, the gap to that
    maximum, and the number of candidates with p ≥ 0.5.

## 3. Models used

| model | role | licence | size / training |
|---|---|---|---|
| LightGBM 4.7.0 (stage 1 and matcher) | pre-filter; pair matcher | MIT | from scratch, 5-fold GroupKFold by S1 |
| paraphrase-multilingual-MiniLM-L12-v2, CE v1 | cross-encoder | Apache-2.0 | 117.7M params; 1.2M fold-A pairs, 96 tokens, seed 42, 1 epoch fp16 |
| same model, CE v2 | 2nd cross-encoder (diversity) | Apache-2.0 | 2.2M fold-A pairs, 128 tokens, seed 2026; identical fold B (checked) |
| LightGBM, stacking | combines everything | MIT | fold B only; 5-fold GroupKFold by S1; learning rate 0.05, 63 leaves |

No external data, APIs or geocoding were used. Supporting libraries: PyTorch (BSD-3-Clause), transformers
(Apache-2.0), polars and scikit-learn.

## 4. Key experiments (public leaderboard)

| submission | LB | what it taught us |
|---|---|---|
| run 1: LightGBM matcher, baseline features | 0.956 | the pipeline works end to end |
| run 4: final features + pre-filter, threshold 0.79 | 0.963 | features (record / S1 context, house-number and legal-form conflicts, run 2) gave +0.007; the pre-filter cost nothing; thresholds 0.69 and 0.79 tie |
| probe: threshold 0.30 | 0.957 | test is precision-limited: extra recall buys nothing |
| probe: threshold 0.90 | 0.961 | the curve peaks at the OOF choice made with w = 2 (§6) |
| blend 0.6·LightGBM + 0.4·CE @ 0.69 | 0.972 | a text model reading both records jointly rejects look-alike distractors |
| blend 0.4·LightGBM + 0.6·CE @ 0.73 | 0.973177 | fold B was flat for weights 0.4–0.7; the leaderboard preferred more CE |
| stack_ctx @ 0.78 | 0.977 | a non-linear, per-pair-type combination with S1 context beats one global weight |
| **stack_ctx2 @ 0.83 (final)** | **0.978359** | a second CE (other seed, sample and max_len) adds diversity: +0.0006 on fold B |
| stack_ctx2 @ 0.75 (w = 1 threshold) | 0.978254 | flat between 0.75 and 0.83; the w = 2 choice holds |
| stack_ctx2, 5 seeds bagged | 0.978289 | variance of the stacker is not the bottleneck |

## 5. Candidate set

- **Size:** 7,726,561 test pairs, **4.46 candidates per S1** (p95 8). That is 0.78 per record, since the
  pre-filter removes 22% of the M = 1 proposals (5.74 → 4.46 per S1).
- **Reduction ratio:** 0.9999989 against the 6.72 × 10^12 same-country S1 × record pairs.
- **Scaling:** no S1 × record comparison is made. Keys shared by more than 2,000 S1s are used only inside
  combination keys, and each record fully scores at most 150 S1s, so cost is linear in the number of records.
  Test blocking takes 44 min on a Kaggle CPU session, with peak memory 7.1 GB.
- **Recall:** 99.0% of true pairs at M = 1 on a held-out train sample, and 96.3% on the full-train OOF.
- **Recall ceiling:** a perfect matcher on these candidates would reach macro F0.5 0.9878. The pre-filter loses
  0.1% of true pairs, costing 0.0001 macro F0.5.

## 6. Validation

- **Matcher:** 5-fold GroupKFold by S1, so all candidates of an S1 fall in the same fold. Thresholds are tuned
  on the exact OOF macro F0.5, singleton rule included.
- **Distractor weighting:** test has about 5.8 records per S1 against 4.7 in train. Each false positive is
  therefore counted twice (w = 2, i.e. P = TP / (TP + 2·FP)) when choosing thresholds, and w = 1 is reported.
- **Cross-encoders and stacking:** the matcher's OOF S1s are split 70/30 by S1 with a fixed seed.
  - The cross-encoders train on fold A.
  - The stacker is evaluated only on fold B, where p_lgbm is out-of-fold and the CE scores are held out. It uses
    an inner 5-fold GroupKFold by S1.
  - CE v2 refuses to run unless its fold B equals v1's, pair for pair.
- **Selection:** every model is compared at its own fold-B-best threshold (w = 2). A candidate was submitted only
  if it beat the incumbent by a set margin: 0.0005, or 0.0003 for the seed-bagging test.

**Fold B results** (macro F0.5, w = 2):

| LightGBM | CE | blend | stack_ctx | stack_ctx2 |
|---|---|---|---|---|
| 0.97695 | 0.97512 | 0.98054 | 0.98342 | **0.98401** |

**OOF ↔ LB agreement:** each fold-B gain was confirmed on the leaderboard, in the same order and larger:

| step | fold B | LB |
|---|---|---|
| blend over LightGBM | +0.0036 | +0.010 |
| stack_ctx over the blend | +0.0029 | +0.004 |
| stack_ctx2 over stack_ctx | +0.0006 | +0.0014 |

The absolute OOF → LB gap (LightGBM 0.978 → 0.963) comes from test's denser distractors. This is why precision
gains transfer more than proportionally.

## 7. Conclusion and limitations

The largest lever was structural: reverse blocking with many noise-robust keys, M = 1 many-to-one, and features
that tell the matcher how decisive the record's choice was. The remaining gains came from differently built
scorers, the cross-encoders, combined by a stacker that is out-of-fold by construction.
The final file has 5,727,307 matches, and 94.0% of S1s get at least one match (France 94.2%, India 93.9%, US 94.1%).

**Limitations:**
- **M = 1 blocking ceiling:** 3.7% of true pairs never reach the matcher (true S1 not ranked first, empty
  addresses, only over-common keys). That caps macro F0.5 at about 0.988, so further gains are in candidate
  generation.
- **Compute:**
  - CPU pipeline: about 4.7 h on Kaggle;
  - CE v1: 86 min on 2×T4 GPUs; CE v2: about 2.3 h;
  - stacking: about 35 min.
  - A larger cross-encoder (e.g. XLM-R base, about 4× the cost) did not fit the time budget, because scoring
    all 7.73M test pairs dominates.
- **Test-only country:** France has no training labels, so its quality is inferred from train and from output
  structure.
