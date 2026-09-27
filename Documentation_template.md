# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** ML  
**Team Members:** Shaik Asad Ahmed  
**Submission Date:** 27 Sep 2026

> Full-train and test numbers come from the Kaggle runs (run 2: training; run 4: pre-filter, tuning and the submitted
> outputs) and were filled by `src/report_numbers.py`
> (see Appendix A). All other numbers were measured on a 50,000-S1 train sample, which is labelled wherever it is used.

---

## 1. Executive Summary

The pipeline has two stages: **reverse blocking**, then a **LightGBM pair classifier** whose threshold is tuned
for the competition metric.
- **Blocking:** every Source 2/3 record looks up its most likely Source 1 entities through many cheap, typo- and
  transliteration-robust keys. The candidates are ranked with a TF-IDF, phonetic and house-number score, and each
  record keeps only its best S1 (M = 1).
- **Pre-filter:** a small stage-1 model on blocking signals and cheap equality flags drops clearly hopeless
  candidates before the matcher runs, at no more than 0.0001 OOF macro F0.5. The submitted candidate set is
  exactly what the matcher scores.
- **Matching:** the classifier sees fuzzy, phonetic and structural features of each (S1, record) pair, plus
  record-level and S1-level context. Its decision threshold is tuned directly on out-of-fold **macro F0.5**, with
  the singleton rule implemented exactly.
- **Second scorer:** a fine-tuned multilingual cross-encoder (MiniLM-L12, Apache-2.0, 117.7M
  parameters) reads each candidate pair's raw name and address and is blended with LightGBM
  (0.4·LightGBM + 0.6·cross-encoder, threshold 0.73) on the same candidate set.
- **Stacking (final submission):** a second-level LightGBM trained on the cross-encoders' held-out fold combines
  the matcher features, the LightGBM probability and **two cross-encoder probabilities** (CE v1 and CE v2: same
  model, different sample, length and seed) with their S1-level context: +0.0029 (one CE) and +0.0035 (two CEs)
  over the blend on that fold.
- **Results:** blocking keeps **99.0%** of true pairs at M = 1 (sample). The full-train out-of-fold macro F0.5 of
  LightGBM is **0.9788** (sample: 0.9893). On the held-out fold B the blend adds +0.0039 at w = 2 and the stacking
  another +0.0035 (`stack_ctx2`). Public leaderboard: **0.978359** with `stack_ctx2` (stack with one CE 0.977,
  blend 0.973177, LightGBM alone 0.963). A 2-page overview is in `ML_Approach_Summary.md` / `.pdf`.
- **France:** it is unseen in train and handled without any country-specific code path.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA (`src/eda.py`, report in `code/business_entity_resolution/reports/eda_report.txt`) established:

| Finding | Consequence for the design |
|---|---|
| Train: 2.21M S1 (US 1.32M, India 0.88M), 10.3M S2+S3 records; test: 1.73M S1 (**France 259k**, India 810k, US 663k), 9.97M records | Everything streams country by country; the code must handle a country it has never seen |
| 5.58% of S1 are singletons; match lists peak at 3–4 records (max 11); 7.64M of 10.3M train records have a true S1 | Singletons are worth 1.0 each only if predicted empty, so precision and rejection matter |
| **No S2/S3 record matches more than one S1** | Matching is many-to-one: each record should go to at most one S1, which motivates reverse blocking with M = 1 |
| Every true match shares its S1's country | Blocking runs within a country; the country label is used only to partition the data |
| Address empty in 2–4% of S2/S3 records (0% of S1) | Scores must degrade gracefully when a field is missing |
| Placeholders ("null", "N/A", "NULL", "<NULL>", "##") inside fields; a few whole-field "na"/"nan"/"null" names | Stripped during normalisation; whole-field junk becomes empty |
| Native scripts (Devanagari, Tamil, …) in S2/S3 names and in state names | Transliteration (anyascii) plus phonetic comparison |
| Postal codes are practically absent; most 5–6 digit numbers are zero-padded house numbers ("003808") | No reliance on postal codes; house numbers are normalised and used as a key and a feature |
| Typos: OCR-like confusions (l/1/i, 0/o, 5/s, 8/b), letter swaps, doubled letters, dropped words, appended words ("… Enterprises") | Confusion folding, character n-grams, token-set and phonetic similarities |
| France (test only): legal forms SARL/SAS/EURL/SCI, "R."/"AVE"/"BD", "St.-Herblain", accents | Generic legal-form and abbreviation tables that include French forms and apply to every record |

### 2.2 Solution Strategy

**Approach Type:** blocking + classifier (reverse blocking → LightGBM → many-to-one → threshold for macro F0.5).  
**Core Innovation:**
- *Reverse* blocking with M = 1 exploits the many-to-one structure: each record chooses its S1, instead of each S1
  choosing records. This makes candidate sets small and precise.
- *Combination keys* recover pairs whose single keys are too common to index.
- A numba accumulator makes full-index blocking feasible on a 4-core machine.
- Thresholds are tuned on the exact leaderboard metric, including singletons and S1s that have no candidates.

Pipeline (`src/run_all.py`):

```
normalize ──► blocking (train + test, same code) ──► features ──► train (5-fold GroupKFold) ──► predict ──► validate
```

---

## 3. Candidate Generation (Blocking)

### 3.1 Normalisation (`src/normalize.py`)

Every rule is applied to every record. **There is no branching on country.**
- **Cleaning:** a whole field that is a placeholder (none, nan, na, n/a, null, <null>, nil, -, ., ?, #, ##, 0,
  unknown, not available; case-insensitive, after stripping) becomes empty, and embedded placeholders are
  removed. The list was checked against the 50 most frequent full name and address strings of every source
  file. If the text is not ASCII, it is transliterated with **anyascii** (ISC;
  chosen over unidecode for its licence and wider script coverage). Then: lower-case, `&` → `and`, possessive
  `'s` removed, `St.-`/`Ste-` → `saint`/`sainte` (French place names), non-alphanumerics → space, and runs of
  single letters merged (`l l c` → `llc`, `m/s` → `ms`).
- **Names:**
  - `name_full` keeps all tokens.
  - `name_norm` removes legal forms and web tokens.
  - `name_legal` holds the canonical legal forms found, e.g. `private limited` → `ltd pvt`, `S.A.R.L.` → `sarl`.
    The legal table covers US, Indian, French and other European forms, plus common transliteration outputs of
    Indic spellings (`praivet`, `praibhet`, `piraivet` → pvt; `limitet`, `limirrd` → ltd; `elelpi` → llp).
  - `tokens` are the comparison keys: `name_norm` tokens with confusion folding (`l`/`1` → `i` everywhere; in
    mixed letter–digit tokens also `0` → `o`, `5` → `s`, `8` → `b`).
  - `name_nonlatin` flags a raw name that contains non-Latin script. Accented Latin does not count.
- **Addresses:**
  - `addr_norm` expands one shared abbreviation table: US (st, rd, ave, blvd, …), French (r → rue, bd →
    boulevard, imp → impasse, all → allée, chem → chemin, fbg → faubourg) and Indian (ngr → nagar, opp →
    opposite, nr → near, tq → taluk).
  - `house_nums` holds the numeric tokens with leading zeros stripped (`No.73` → `73`, `003808` → `3808`).
  - `hs_key` pairs a house number with the first distinctive word after it, skipping street types and function
    words: `1 rue de la Paix` → `1|paix`, `7222 6th Avenue` → `7222|6th`.
  - `postal` is a generic 5–6 digit code, accepted only in positions where house numbers do not occur. It is
    rarely present.
- **Processing:** files are streamed with pyarrow in 100k-row batches across a multiprocessing pool. Test takes
  about 13 minutes for 11.7M rows.

### 3.2 Reverse blocking design (`src/blocking.py`)

Blocking runs per country, taken from the data as an open set. It indexes **every** S1 of the country and queries
it with **every** S2/S3 record. Train is blocked exactly like test, so the matcher sees the same candidate
distributions it will face at test time.

**Key types:**

| Kind | Key | Purpose |
|---|---|---|
| `p` | postal code | rare, but exact when present |
| `h` | house number + first street word (`hs_key`) | strongest address key |
| `s` | name signature (sorted folded name tokens) | word reordering |
| `n` | folded name token (legal forms stripped) | typos / OCR confusions |
| `a` | address token (no digits, street types or function words) | locality / street names |
| `f` | double-metaphone code of each name word | transliteration and spelling variants |
| `hna` | house number × address token (combination) | precise address match even when both parts are common |
| `na` | oversized name token × oversized address token (combination) | recovers pairs whose only shared keys are too common |

**Caps and lookups:**
- A key whose S1 block has more than `block_cap` = 2000 S1s is **never looked up alone**. It contributes only
  inside a combination key, which must itself be under the cap. Pairs lost this way are counted (the `only_oversized`
  miss reason).
- Each record looks up its `p`/`h`/`s` keys, its **2 rarest** `n`, `a` and `f` keys (rarity = S1 block size;
  tokens unseen in S1 are ignored), house number × its 4 rarest address tokens, and oversized name × oversized
  address tokens.
- Keys are weighted by IDF, `ln(N_S1 / block size) + 0.1`.

**Pre-cut:** the S1s sharing any key are summed by key weight in a numba accumulator. For each record this adds
into a dense per-S1 array, then keeps the **150** best by key score (`pre_cap`). This replaced a polars
join + group_by and gave identical results 3–4× faster.

**Ranking score** for the ≤ 150 survivors:
- `name` = char 3-gram TF-IDF cosine (`char_wb`, sublinear tf; the vocabulary is fit per country on S1). When
  either name was originally non-Latin, `name` = max(TF-IDF, phonetic), where phonetic = token-sort ratio of the
  double-metaphone codes of the legal-stripped name words. For example, `tainmik tiretin` vs `dynamic trading`
  scores 0.95.
- `addr` = char 3-gram TF-IDF cosine.
- `score` = 0.4·name + 0.6·addr, averaged over the fields present on both sides (an empty field hands its weight
  to the other), **+ 0.1** if the two share an exact normalised house number.

**M = 1:** each record keeps only its best-scoring S1. Because a record belongs to at most one S1, the best-scoring
S1 is the only one it can be matched to. M = 2 doubles the candidates for +0.35 points of recall (table below),
and those extra candidates would be rejected by the many-to-one rule anyway. Blocking also emits two record-level
context columns for the matcher: `rec_gap` (score minus the record's best *other* S1 score) and `rec_n` (S1s scored).

### 3.3 Blocking metrics (50k-S1 train sample)

The sample is 50,000 train S1s, all of their true matches, and a proportional 2.3% of other records. Settings:
index = all sample S1s (test-like), 4,941 held-out S1s (10%), 17,102 true pairs. Script: `src/eval_blocking.py`.

**Recall by stage** (fraction of true pairs surviving each stage):

| | keys | pre-cut 150 | top-1 (M=1) | top-2 (M=2) |
|---|---|---|---|---|
| ALL | 0.9993 | 0.9986 | **0.9902** | 0.9937 |

Key-stage misses (fraction of true pairs): 0.0001 share no key at all, 0.0005 share only oversized keys, 0.0001
share a usable key that was not selected.

**Candidates per S1 and recall by country:**

| country | M | recall | S1s with all matches covered | avg cands / S1 | p95 cands / S1 |
|---|---|---|---|---|---|
| ALL | 1 | 0.9902 | 96.8% | 8.0 | 17 |
| ALL | 2 | 0.9937 | 98.0% | 16.1 | 35 |
| India | 1 | 0.9874 | 96.0% | 7.9 | 16 |
| US | 1 | 0.9921 | 97.4% | 8.1 | 17 |

**Cap-45 simulation.** Sample blocks are about 44× smaller than full-scale blocks, so cap 2000 is almost never
reached on the sample. Re-running the sample with `block_cap` = 45 simulates full-scale cap pressure (these runs
used pre-cut 1000):
- M = 1 recall 0.9861 overall (India 0.9819, US 0.9891); M = 2 0.9887.
- Key-stage recall 0.9908; 0.91% of true pairs are lost because they share only oversized keys.
- About 3,400 records per country are reachable only through combination keys.

This is the expected full-scale cost of the cap, and why the combination keys exist.

**Pre-cut and `n_rare` choice** (sample, M = 1 recall; local blocking time):

| pre-cut | 300 | **150** | 100 | 150 + `n_rare` = 1 |
|---|---|---|---|---|
| recall | 0.9903 | **0.9902** | 0.9898 | 0.9870 |
| blocking time | 216 s | 182 s | 156 s | 126 s |

### 3.4 Candidate pre-filter (`src/prefilter.py`)

After blocking, test has about 5.7 candidates per S1, but the matcher keeps only about 3.3. The submitted
`candidate_pairs.tsv` must be exactly the set the matcher scores. Pruning is therefore a separate stage that runs
**before** the matcher, not a filter on its decisions.

- **Stage-1 model:** a small LightGBM with 19 inputs; the matcher has 50.
  - Blocking outputs: combined score, name / phonetic / address TF-IDF similarity, key score, number of shared
    keys, the record's gap to its second-best S1, the record's number of S1 hits, and the source.
  - Cheap equality flags: house-number match or conflict, postal code, legal form, exact name, house-number +
    street key.
  - No fuzzy or IDF string scores are used. No per-S1 context is used either, because its scale follows the
    number of candidates per S1, which is about twice as high on test as on train.
- **Training:** 5-fold GroupKFold by S1 on the matcher's train universe. This gives out-of-fold stage-1
  probabilities p1.
- **Cutoff:** a pair stays a candidate when p1 ≥ cutoff.
  - For each cutoff, the trade-off is measured on OOF, with pruned pairs counted as rejected.
  - The chosen cutoff is the largest one that costs at most 0.0001 macro F0.5 at w = 2 and the final threshold.
  - Full train: cutoff 0.025. OOF candidates per S1 go from 4.66 to 3.94 (p95
    8 → 7).
  - 0.104% of true pairs are lost, and 0.036% of the pairs the matcher accepts.
  - OOF macro F0.5 (w = 2) goes from 0.97701 to 0.97691.
- **The matcher is not retrained.** It scores the surviving pairs with features computed on the full blocking
  list, so the pruned OOF score is exactly what the pipeline produces.
- **Test check (France):**
  - Local France test blocking reproduces the Kaggle candidates: 1,434,089 of 1,434,108 pairs are identical.
  - The sample-trained stage 1 cuts France from 5.53 to 5.21 candidates per S1.
  - It drops 433 of the 843,091 run-3 matches (0.05%).
  - Most of the remaining rejected test candidates are hard distractors. Only the full matcher separates them
    (median p1 0.64 against a matcher probability of 0.0014), which limits how far cheap signals can prune.

**Full test set** (after the pre-filter; this is the submitted `candidate_pairs.tsv`):
- 7,726,561 candidate pairs.
- On average 4.46 candidates per S1 (p95 8).
- 97.8% of S1s have at least one candidate.

Per country: France 4.26, India 4.69, US 4.25 candidates
per S1.

**How true matches are protected:**
- Many independent key types, each robust to a different noise type (typos, reordering, transliteration, spelling,
  address-only).
- Oversized keys still contribute through combination keys.
- The pre-cut removes 0.02% of true pairs at 300 and 0.07% at 150.
- Recall is measured at every stage, with a miss reason for each lost pair, which drove each design change: key
  recall rose from 0.895 (India, v1) to 0.999.

### 3.5 Scalability & candidate-set size

**Why blocking scales.** No S1 × record comparison is ever made:
- Per country, an inverted index maps each blocking key (name tokens, phonetic codes, house+street, postal and
  combination keys) to the S1s carrying it. Each S2/S3 record looks up only its own keys.
- Keys hitting more than 2,000 S1s (block cap) do not enumerate their block. They contribute only through
  combination keys.
- Each record accumulates key scores, fully scores only its 150 best S1s (pre-cap), and keeps its single best
  (M = 1).
- Cost is linear in the number of records: about 1,640–2,020 key hits and at most 150 scored pairs per record at
  full test scale.

Kaggle CPU session, full test (1,732,544 S1 × 9,969,589 S2/S3 records):
- Test blocking: 44 min, peak memory 7.1 GB. Train blocking: 47 min.
- Features: 5 min per split; peak 9.0 GB, the pipeline maximum.
- Pre-filter: 17 min. Matcher inference: 57 min.
- Cross-encoder (GPU T4 x2, same candidates): 86 min, of which 37 min scoring the 7.73M test pairs.
- Stacking (same candidates): 19 min with one CE, of which 7 min scoring the test pairs; about 35 min with two
  CEs (final). CE v2: about 2.3 h on GPU T4 x2.
- The whole pipeline runs in about 4.7 h.

**Candidate-set size on test** (submitted `candidate_pairs.tsv`, after the pre-filter):

| country | S1 | S2+S3 records | candidate pairs | avg / p95 candidates per S1 | S1 with 0 candidates | all same-country S1 × record pairs | reduction ratio |
|---|---|---|---|---|---|---|---|
| France | 259,452 | 1,434,993 | 1,105,682 | 4.26 / 8 | 2.67% | 3.72 × 10^11 | 0.9999970 |
| India | 809,986 | 4,717,565 | 3,799,665 | 4.69 / 8 | 1.72% | 3.82 × 10^12 | 0.9999990 |
| US | 663,106 | 3,817,031 | 2,821,214 | 4.25 / 8 | 2.64% | 2.53 × 10^12 | 0.9999989 |
| **ALL** | 1,732,544 | 9,969,589 | **7,726,561** | **4.46 / 8** | 2.21% | 6.72 × 10^12 | **0.9999989** |

Reduction ratio = 1 − candidate pairs / (same-country S1 × (S2 + S3) pairs). That is 0.78 candidates per record:
fewer than one, because the pre-filter removes 22% of the M = 1 proposals.

**M = 1 and the pre-filter.**
- *M = 1 is valid:* S1 is deduplicated, and in the train ground truth every S2/S3 record belongs to at most one
  S1 (7,638,365 matched records, all distinct). A record therefore needs only one S1 proposal. It loses its
  match only when its true S1 is not ranked first.
- *Pre-filter:* the stage-1 model (§3.4, cutoff p1 ≥ 0.025) prunes test candidates from 5.74 to 4.46 per S1
  (France 5.53 → 4.26, India 5.81 → 4.69, US 5.75 → 4.25). On OOF it loses 0.104% of true pairs and 0.0001
  macro F0.5.
- *Recall ceiling (full-train OOF):* the M = 1 candidates contain 96.3% of true pairs. A perfect matcher on
  them would reach macro F0.5 **0.9878** (India 0.9846, US 0.9900), against 0.9788 for our matcher.

---

## 4. Matching Model

### 4.1 Features (`src/features.py`)

There is one row per blocking candidate. Missing values (a field empty on either side) are left missing, so
LightGBM learns a separate branch for them instead of reading them as zero similarity.

| Group | Features | Rationale |
|---|---|---|
| **Name** (legal-stripped `name_norm`) | rapidfuzz `ratio`, `partial_ratio`, `token_set_ratio`, `token_sort_ratio`, Jaro-Winkler; token-sort on the full name (with legal forms); char 3-gram TF-IDF cosine; double-metaphone phonetic similarity; exact match; non-Latin flag; words in the record name not in the S1 name and vice versa; name lengths; legal-form conflict (both have a legal form and they differ) | Each captures one noise type: character typos (ratio, JW), dropped or appended words (partial, token-set, extra-word counts), reordering (token-sort), transliteration (phonetic, non-Latin flag). A legal-form conflict (e.g. `llc` vs `inc`) is evidence of a different business |
| **Address** | the same fuzzy set + TF-IDF cosine; exact house-number match; house-number overlap count; token Jaccard; house+street key match; postal match; record-address-empty flag; S1 and record address lengths | The house number is the most discriminative address token; lengths and the empty flag tell the model how much evidence the address can carry |
| **Blocking** | ranking score, key IDF score, number of shared keys, `rec_gap` (score minus the record's best other S1), `rec_n` (S1s scored for the record); `rank` only when M > 1 | `rec_gap` measures how unambiguous the record's choice is. A large gap means one clear S1, which is exactly what separates matches from near-misses |
| **S1 context** | number of candidates of this S1; rank of this pair's score within the S1's candidates; gap to the S1's best other candidate | An S1 attracting many records is either a large true cluster or a "magnet" for generic records |
| **Source** | 0 = S2, 1 = S3 | The sources have different noise profiles |
| **Distinctive words** (added after error analysis) | IDF-weighted name-word Jaccard (IDF from the country's S1 names); IDF mass and max IDF of the record's extra words and of the S1's missing words; double-metaphone codes present on only one side; no-space name ratio | Separates "Perfect Products" from "Perfect Products **Traders**" (a different business, even when transliterated as "tredars") and "Premier Constructions" from "Sunrise Construction" at one address, where character similarity is high. The no-space ratio handles domain-style names ("earnosethroat.com") |
| **House-number conflict** (added after error analysis) | first house number equal; conflict flag (both have numbers, none shared); numeric distance of the first numbers; each side's numbers missing from the other | Neighbouring or sibling units ("stall no 5" vs "no g 10", "11304" vs "11305 Earls Estate St") are different businesses; small distances can also be typos, so the model gets the distance itself |

**Country is deliberately not a feature**, since France is unseen in training.

Feature importance by gain, summed over folds on the sample:

| rank | feature | gain |
|---|---|---|
| 1 | `rec_gap` | 78.7% |
| 2 | `score` | 9.7% |
| 3 | `key_score` | 2.2% |
| 4 | `nf_tsort` | 0.8% |
| 5 | `n_ph_extra_rec` (phonetic extra words) | 0.8% |
| 6 | `hn_first_dist` (house-number distance) | 0.6% |

Full run top 5: hn_s1_missing 35.0%, rec_gap 12.3%, nf_tsort 8.7%, hn_rec_missing 8.3%, hn_first_dist 4.4%.

### 4.2 Model and validation (`src/train_matcher.py`)

- **Model type:** LightGBM binary classifier (MIT licence). Settings: learning rate 0.1, 127 leaves, min 100
  samples per leaf, 0.8 feature and bagging fraction, L2 = 1.0, up to 3000 rounds with early stopping (100 rounds)
  on the validation fold, seed 42.
- **Size:** 5 models, each around 150 trees with 127 leaves. That is under 10⁵ leaf values, orders of magnitude
  below the 8B-parameter limit. No pretrained model is used by the matcher (the cross-encoder of §4.5 is).
- **Training data:** the candidate pairs from full train blocking (full S1 index × full S2/S3 pool, identical to
  test). The label is whether the pair is in the ground truth. When there are more than 5M pairs, random **S1
  groups** are sampled (all pairs of each chosen S1, including chosen S1s without candidates), so per-S1 features
  and the per-S1 metric stay exact. An earlier version indexed a 300k-S1 subset. It was dropped because records
  whose true S1 was not indexed attached to wrong S1s, which shifted `rec_gap` and `s1_n_cands` away from test
  conditions.
- **Validation:** 5-fold `GroupKFold` by S1 id (shuffled, seed 42), so no S1 appears in both training and
  validation of a fold. The out-of-fold (OOF) probabilities are used for threshold tuning and every reported
  score. At prediction time the 5 fold models are averaged.
- **Negative downsampling:** a safety net. If a fold has more than 5M negatives, they are sampled and re-weighted
  by 1/frac, which keeps probabilities calibrated. Validation and OOF always use every pair. Forcing it on the
  sample (frac 0.32) cost 0.0008 F0.5, so with the 5M-pair group sample it rarely triggers.

### 4.3 Post-processing and threshold selection

1. **Many-to-one:** each record is kept only for its highest-probability S1. This is automatic at M = 1, and the
   logic is kept for M > 1.
2. **Threshold:** a pair is kept when its probability is at least the threshold. The threshold maximises the
   **exact macro F0.5 over every S1 in the evaluation universe**:
   - an S1 with no true matches scores 1 if nothing is predicted, else 0;
   - otherwise F0.5 = 1.25·P·R / (0.25·P + R), which is 0 if nothing correct is predicted;
   - S1s with no candidates are included, so blocking misses cost exactly what they cost on the leaderboard.

   Search: a global grid 0.02–0.98 (step 0.01), then per-source S2/S3 thresholds by coordinate ascent starting
   from the global optimum. Per-source thresholds are used only if they gain more than 1e-4.
   - Sample (run 2): global 0.76 → 0.98933; per-source (0.71 / 0.76) → 0.98938, so global was used.
   - Full train, OOF-optimal: **0.69 (global)**.
3. **Distractor-aware threshold** (`src/tune_threshold.py`). Test has about 5.8 S2/S3 records per S1 against 4.7
   in train, while true matches per S1 look the same (predicted 3.2–3.3 per S1 vs 3.46 in train). A test S1
   therefore meets about twice as many distractor records, so the OOF-optimal threshold is too permissive for
   test. To simulate test density, every kept OOF negative counts as *w* false positives in its S1's precision:
   P = TP / (TP + w·FP). Recall is unchanged, and a singleton still scores 1 only if nothing is kept. The best
   global threshold is found for w ∈ {1, 1.5, 2}:

   | w | best threshold | OOF macro F0.5 at w |
   |---|---|---|
   | 1.0 | 0.71 | 0.9786 |
   | 1.5 | 0.73 | 0.9777 |
   | 2.0 | 0.79 | 0.9769 |

   The sample gave 0.76 / 0.82 / 0.82.
4. **Public-leaderboard check:** submissions at thresholds 0.69, 0.75 and 0.80, plus the w = 2 optimum (0.85 if
   that coincides with one of them), were scored on the public leaderboard: 0.69 → 0.963 (run 2), 0.79 → 0.963 (run 3), 0.79 with the pre-filter → 0.963 (run 4), and probes at 0.30 → 0.957 and 0.90 → 0.961:

   | threshold | 0.30 | 0.69 | 0.79 | 0.90 |
   |---|---|---|---|---|
   | public LB F0.5 | 0.957 | 0.963 | **0.963** | 0.961 |

   The threshold curve peaks at the OOF-chosen 0.79 on the public LB (0.30 → 0.957, 0.90 → 0.961), confirming
   the distractor-weighted OOF tuning. The LightGBM stage uses **0.79** (the final submission then combines it with the cross-encoder, §4.5–4.6), recorded as `config.FINAL_THRESHOLD`, so `run_all.py` / `predict.py` reproduce it.
   `predict.py --threshold T` overrides it.
5. **Empty-address rule, rejected:** force-accept an empty-address record whose legal-stripped name equals the S1's
   and is unique among the country's S1 names. On the sample, the pairs it newly accepts are only 68% true, and
   macro F0.5 at w = 2 falls from 0.98776 to 0.98754. Full run: rule off.

### 4.4 France and open-set countries

- **Country as a partition only:** it is read from the data, and the pipeline loops over whatever countries are
  present. It is never a feature, a filter or a one-hot variable, and there are no country-specific branches.
- **Generic tables include French forms:** the legal forms (SARL, SAS, SASU, EURL, SCI, SA, SNC, "& Cie"), street
  abbreviations (R. → rue, BD → boulevard, AVE, IMP, ALL, CHEM, FBG), `St.-`/`Ste-` → saint/sainte, and accent
  folding via anyascii. The house+street key skips French articles (`de`, `la`, `du`, `des`, …): `1 rue de la paix`
  → `1|paix`.
- **Per-country fitting:** TF-IDF vocabularies and key IDFs are fit on each country's own S1, so French n-grams get
  French statistics without any labels.
- **Complete output:** every test S1 id is read from the raw `test_source1.tsv`, so every entity, France included,
  has a row (empty when nothing is kept).
- **Verified locally:** full France test blocking (1.43M records, 16 min) → features → predict → the official
  validator returned PASS.

### 4.5 Cross-encoder second scorer and blend (`src/cross_encoder.py`)

The remaining test errors are look-alike distractors: sibling entities ("X Holding" vs "X"), nearby house numbers
and transliterated names. Handcrafted similarities separate them poorly, so a fine-tuned transformer reads both
records jointly and scores **the same run-4 candidate pairs**. Blocking, the pre-filter and `candidate_pairs.tsv`
are unchanged (byte-identical to run 4's file).

- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Hugging Face), **Apache-2.0**,
  **117.7M parameters** (117,654,272; 12-layer BERT, multilingual vocabulary covering Devanagari and other
  scripts). Only the pretrained weights are downloaded; no external data or lookups. A one-output classification
  head on [CLS] is trained with binary cross-entropy.
- **Input:** a text pair, S1 `"<raw name> | <raw address>"` and record `"<raw name> | <raw address>"` (raw text,
  native script kept), truncated to 96 tokens.
- **Training data, no leakage:** the S1s of the matcher's OOF universe are split by S1 id into fold A (70%) and
  fold B (30%). The cross-encoder trains on 1.2M pairs sampled from fold A (our blocking candidates, so hard
  negatives come naturally), 1 epoch, fp16, learning rate 5e-5, batch 128, seed 42, on 2 GPUs. Fold B is only
  used for the blend, and its LightGBM probabilities are out-of-fold.
- **Blend:** p = w·p_LightGBM + (1 − w)·p_CE, pairs removed by the pre-filter count as rejected, each record keeps
  only its best S1 (many-to-one), and a pair is matched if p ≥ threshold. w and the threshold maximise macro F0.5 on
  fold B with the distractor weighting of §4.3 (w = 2); w = 1 is reported too. The final weight was chosen by
  fold-B OOF and the public leaderboard together (below).

**Fold B** (321,861 S1s, 1,501,064 pairs):

| w_lgbm | threshold (w = 2) | macro F0.5, w = 2 | threshold (w = 1) | macro F0.5, w = 1 |
|---|---|---|---|---|
| 0.0 (CE alone) | 0.78 | 0.97512 | 0.72 | 0.97766 |
| 0.2 | 0.80 | 0.97936 | 0.80 | 0.98021 |
| **0.4 (submitted)** | **0.73** | 0.98054 | 0.66 | 0.98155 |
| 0.5 | 0.69 | 0.98075 | 0.63 | 0.98171 |
| 0.6 (fold-B best) | 0.69 | **0.98087** | 0.62 | **0.98185** |
| 0.7 | 0.71 | 0.98078 | 0.70 | 0.98154 |
| 0.8 | 0.80 | 0.97991 | 0.80 | 0.98043 |
| 1.0 (LightGBM alone) | 0.79 | 0.97695 | 0.71 | 0.97871 |

- The blend beats both scorers: **+0.0039** macro F0.5 at w = 2 over LightGBM alone (+0.0031 at w = 1). The two
  models make different errors.
- **Weight choice, fold-B OOF + public leaderboard:** fold B is flat between w_lgbm 0.4 and 0.7 (within 0.0004 at
  w = 2), so the best weights were each submitted at their fold-B-best threshold (w = 2):

  | w_lgbm @ threshold | fold B, w = 2 | public LB F0.5 |
  |---|---|---|
  | 1.0 @ 0.79 (LightGBM alone, run 4) | 0.97695 | 0.963 |
  | 0.6 @ 0.69 | 0.98087 | 0.972 |
  | 0.5 @ 0.69 | 0.98075 | 0.973 |
  | **0.4 @ 0.73 (submitted)** | 0.98054 | **0.973177** |

  More cross-encoder weight helps on test. This blend (0.4·LightGBM + 0.6·cross-encoder at 0.73) was the
  submission until the stacking of §4.6, which uses it as an input feature.
- **Public leaderboard: 0.973177** (LightGBM alone 0.963, **+0.010**). The gain on test is more than twice the
  fold-B gain, consistent with the cross-encoder fixing exactly the denser look-alike distractors of test.
- **Test output:** 5,643,029 matches (LightGBM alone at 0.79: 5,543,011; 173,931 added, 73,913 removed). Matches
  per S1: France 3.30, India 3.21, US 3.30; 93.9% of S1s receive at least one match.
- **Runtime** (Kaggle, GPU T4 x2): 86 min in total — pairs and texts 1.4 min, training 39 min (1.2M pairs), fold-B
  scoring 7 min (1.5M pairs), test scoring 37 min (7.73M pairs, fp16), blend and output 1.3 min.
- **Reproduction:** `config.CE_W_LGBM = 0.4`, `config.CE_THRESHOLD = 0.73`; `python src/cross_encoder.py`
  after the CPU pipeline, or `run_kaggle_ce.ipynb` (README).

### 4.6 Stacking (`src/stack.py`), previous submission (LB 0.977)

The linear blend applies one global weight. A second-level LightGBM can weigh the two scorers differently per kind
of pair (empty address, house-number conflict, native-script name, ...) and use how a pair ranks among its S1's
candidates.

- **Training data, no leakage:** fold B only (321,861 S1s, 1,501,064 pairs, 1,268,281 surviving the pre-filter;
  pruned pairs count as rejected). Its p_LightGBM is the matcher's out-of-fold probability and its p_CE comes from
  the cross-encoder trained on fold A. Out-of-fold predictions inside fold B come from a 5-fold GroupKFold by S1.
- **Two variants:**
  - `stack`: the 51 matcher features + p_LightGBM + p_CE + the blend 0.4·p_LightGBM + 0.6·p_CE;
  - `stack_ctx`: the same + 8 S1-level context features: for each probability, its rank among the S1's
    candidates, the S1's maximum, the gap to that maximum and the number of candidates ≥ 0.5, computed over the same
    candidate set as on test.
- **Model:** LightGBM, learning rate 0.05, 63 leaves, min 100 samples per leaf, 0.8 feature and bagging fraction,
  L2 = 1, early stopping after 100 rounds (cap 1,500): 198–236 rounds (`stack`), 350–437 (`stack_ctx`).
- **Test:** average of the 5 fold models on all 7,726,561 candidates, many-to-one, match if p ≥ threshold (the
  fold-B best at distractor weight 2).

**Fold B, out-of-fold** (each row at its own best threshold):

| model | threshold (w = 2) | macro F0.5, w = 2 | threshold (w = 1) | macro F0.5, w = 1 |
|---|---|---|---|---|
| LightGBM alone | 0.79 | 0.97695 | 0.71 | 0.97871 |
| cross-encoder alone | 0.78 | 0.97512 | 0.72 | 0.97766 |
| blend 0.4 / 0.6 (§4.5) | 0.73 | 0.98054 | 0.66 | 0.98155 |
| stack | 0.78 | 0.98311 | 0.71 | 0.98369 |
| **stack_ctx (submitted)** | **0.78** | **0.98342** | 0.71 | **0.98396** |

- **Verdict:** `stack_ctx` beats the blend by **+0.00288** at w = 2 (+0.00241 at w = 1), and the S1 context adds
  +0.0003 over `stack`. Choosing between the two variants on the same fold B is a mild selection effect; the public
  leaderboard confirms the gain.
- **Top gain:** blend 91.3%, p_LightGBM 3.4%, p_CE 2.3%, `rec_gap` 0.6%, `s1_n_cands` 0.2%, then the context gaps.
  The model mostly recalibrates the blend per pair type rather than replacing it.
- **Leaderboard history:**

  | submission | public LB F0.5 |
  |---|---|
  | LightGBM matcher alone (run 4, threshold 0.79) | 0.963 |
  | blend 0.6·LightGBM + 0.4·CE @ 0.69 | 0.972 |
  | blend 0.4·LightGBM + 0.6·CE @ 0.73 | 0.973177 |
  | **stack_ctx @ 0.78 (final)** | **0.977** |

- **Test output:** 5,730,770 matches (blend file: 5,643,029, 111,739 added and 23,998 removed; LightGBM alone:
  5,543,011, 238,505 added and 50,746 removed). Matches per S1: France 3.32, India 3.27, US 3.34; 94.0% of S1s
  receive at least one match (France 94.2%, India 93.8%, US 94.1%).
- **Runtime:** 19 min on a Kaggle session (fold-B training of both variants 10 min, test scoring 7 min).
- **Reproduction:** `config.STACK_VARIANT = "stack_ctx"`, `config.STACK_THRESHOLD = 0.78`, `config.CE_W_LGBM = 0.4`
  (it defines the blend feature); `python src/stack.py` after the cross-encoder stage, or `run_kaggle_stack.ipynb`.

### 4.7 Second cross-encoder and stacking v2 (`stack.py --ce2_dir`) — final submission

A second cross-encoder adds diversity at no risk to the validation design: same model and licence, but a different
fold-A sample, sequence length and seed, trained on **the same fold A/B split** (`cross_encoder.py --from_ce` stops
unless its fold B equals CE v1's pair for pair), so fold B stays held out for both cross-encoders.

- **CE v2:** `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 117.7M), 2.2M fold-A pairs (CE v1: 1.2M),
  max_len 128 (CE v1: 96; only ~2% of pairs exceed 96 tokens), seed 2026 (CE v1: 42), 1 epoch fp16 on 2×T4
  (~2.3 h). It scores the same fold-B pairs and all 7,726,561 test candidates.
- **stack_ctx2:** the `stack_ctx` inputs + p_CE2, blend2 = 0.4·p_LightGBM + 0.6·p_CE2, mean(p_CE, p_CE2),
  blend12 = 0.4·p_LightGBM + 0.6·mean, and CE2's four S1-context features: 70 features. Same fold B, same
  5-fold GroupKFold by S1, same LightGBM settings as §4.6; `stack_ctx` is retrained in the same run as reference.
- **Fold B (w = 2):** `stack_ctx2` **0.98401** @ 0.83 vs `stack_ctx` 0.98342 (+0.00059, above the +0.0005 bar set
  before the run).
- **Test output:** 5,727,307 matches (vs the `stack_ctx` file: 20,976 added, 24,439 removed); 94.0% of S1s get at
  least one match (France 94.2%, India 93.9%, US 94.1%), 3.30 / 3.28 / 3.34 matches per S1.
- **Checks after the final choice** (public LB, not adopted):
  - `stack_ctx2` at its w = 1 threshold 0.75: 0.978254 (flat between 0.75 and 0.83; the w = 2 choice holds);
  - 5-seed bagging of `stack_ctx2` (`src/stack3.py`): 0.978289 (the stacker's variance is not the bottleneck).
- **Reproduction:** `config.STACK_VARIANT = "stack_ctx2"`, `config.STACK_THRESHOLD = 0.83`, `config.CE_W_LGBM = 0.4`;
  `cross_encoder.py --from_ce cache_ce --n_train 2200000 --max_len 128 --seed 2026`, then
  `stack.py --ce2_dir cache_ce2` (`run_kaggle_ce2.ipynb`, `run_kaggle_stack2.ipynb`; README stages 3-4).

**Public leaderboard history:**

| submission | public LB F0.5 |
|---|---|
| run 1: LightGBM matcher, baseline features | 0.956 |
| run 2-4: final features (+ pre-filter in run 4), threshold 0.69 / 0.79 | 0.963 |
| threshold probes 0.30 / 0.90 (run 4 model) | 0.957 / 0.961 |
| blend 0.6·LightGBM + 0.4·CE @ 0.69 | 0.972 |
| blend 0.4·LightGBM + 0.6·CE @ 0.73 | 0.973177 |
| stack_ctx @ 0.78 | 0.977 |
| **stack_ctx2 @ 0.83 (final)** | **0.978359** |
| stack_ctx2 @ 0.75 | 0.978254 |
| stack_ctx2, 5 seeds bagged | 0.978289 |

---

## 5. Results & Error Analysis

**Full train, OOF** (run-2 model, trained once and reused by runs 3–4), at its OOF-optimal threshold 0.69 (global),
before the pre-filter:

| country | macro F0.5 | precision | recall | singleton accuracy | blocking ceiling F0.5 |
|---|---|---|---|---|---|
| ALL | **0.9788** | 0.9948 | 0.9512 | 0.9824 | 0.9878 |
| India | 0.9738 | 0.9933 | 0.9428 | 0.9766 | 0.9846 |
| US | 0.9821 | 0.9959 | 0.9568 | 0.9863 | 0.9900 |

- Submitted configuration (threshold 0.79, pre-filter p1 ≥ 0.025): OOF macro F0.5 at w = 1 India 0.9733,
  US 0.9818; at w = 2, 0.9769 overall.
- OOF AUC: 0.99925.
- Test predictions, LightGBM alone (run 4): 5,543,011 matched pairs; 93.8% of S1s receive at least one match
  (France 94.2%, India 93.5%, US 94.0%).
- Test predictions, blend (§4.5): 5,643,029 matched pairs; 93.9% of S1s receive at least one match
  (France 94.4%, India 93.7%, US 94.0%).
- Test predictions, final stacking (§4.6): 5,730,770 matched pairs; 94.0% of S1s receive at least one match
  (France 94.2%, India 93.8%, US 94.1%).
- Leaderboard F0.5: **0.977** (final submission: stacking, §4.6); cross-encoder blend 0.4 / 0.6 at 0.73: 0.973177 (§4.5; 0.6 / 0.4 at 0.69: 0.972); LightGBM alone (run 4: threshold
  0.79, pre-filter p1 ≥ 0.025) 0.963.

**Sample (50k S1), OOF: run 1 (baseline features) vs run 2 (final features):**

| country | run | threshold | macro F0.5 | precision | recall | singleton accuracy | blocking ceiling F0.5 |
|---|---|---|---|---|---|---|---|
| ALL | 1 | 0.74 | 0.9860 | 0.9935 | 0.9765 | 0.9693 | 0.9969 |
| ALL | **2** | 0.76 | **0.9893** | **0.9958** | **0.9784** | **0.9863** | 0.9969 |
| India | 1 | 0.74 | 0.9830 | 0.9929 | 0.9708 | 0.9592 | 0.9962 |
| India | **2** | 0.76 | **0.9870** | **0.9955** | **0.9730** | **0.9819** | 0.9962 |
| US | 1 | 0.74 | 0.9880 | 0.9939 | 0.9804 | 0.9760 | 0.9974 |
| US | **2** | 0.76 | **0.9909** | **0.9961** | **0.9821** | **0.9892** | 0.9974 |

Blocking is identical in both runs: 0.9902 recall at M = 1 and 8.0 candidates per held-out S1. Run-2 OOF AUC is
0.9998 on every fold. The sample is positive-heavy (42% of candidates are true pairs), so full-train numbers are
the reference.

**Error analysis** (`src/error_analysis.py`, sample OOF at the chosen threshold). Each error is tagged with every
pattern it fits, so buckets overlap. "Gain" is the macro-F0.5 increase if every error of that bucket were
fixed, an upper bound used to rank fixes.

| bucket | FP run 1 | FP run 2 | FN run 1 | FN run 2 | share of all pairs | gain if FP fixed (run 2) | gain if FN fixed (run 2) |
|---|---|---|---|---|---|---|---|
| all errors | 1,106 | **709** | 2,339 | **2,005** | 100% | +0.0036 | +0.0040 |
| generic name (≤ 1 non-generic word) | 604 | 418 | 1,446 | 1,302 | 49% | +0.0021 | +0.0025 |
| record address empty | 273 | 256 | 928 | 878 | 3.8% | +0.0012 | +0.0019 |
| house-number conflict | 359 | **128** | 460 | 464 | 39% | +0.0007 | +0.0009 |
| extra words in record name only | 189 | **72** | 363 | 258 | 7.1% | +0.0004 | +0.0006 |
| same address, different name | 133 | 104 | 172 | 143 | 3.9% | +0.0005 | +0.0002 |
| name looks like noise (address fine) | 137 | 115 | 269 | 226 | 4.3% | +0.0006 | +0.0004 |
| legal-form conflict | 112 | 66 | 232 | 173 | 17% | +0.0003 | +0.0004 |
| non-Latin record name | 77 | 50 | 233 | 200 | 7.3% | +0.0002 | +0.0005 |
| ambiguous (`rec_gap` < 0.05) | 38 | 34 | 650 | 613 | 43% | +0.0001 | +0.0013 |
| domain-style name / initials only | 39 | 37 | 77 | 73 | 4.3% | +0.0003 | +0.0002 |

False positives split into two kinds (run 1 → run 2):
- records with **no true S1**, i.e. near-duplicate businesses such as "Perfect Products Pvt Ltd" vs "Parphekt
  Prodakts **Tredars** Praibhet Limited": 575 → 252;
- records whose **true S1 is a different entity**: 531 → 457.

The run-2 distinctive-word and house-number features target exactly these (ablation in Appendix B).

**Remaining false positives:**
- *Same address, different business:* "Clear Electronics, 11209 Chase Drive" vs "Lucas Electronics LLC, #11209
  Chase Dr". The shared address and category word outweigh the differing distinctive word.
- *Name-only records:* "Lucky Diner" with an empty address, matched on name alone.
- *Same name at a different address:* "Supreme Network Corporation, 8412 Raymond Ave" vs "Supreme Network, 1427
  Arizona Ave". These are chains or near-homonyms.

**Remaining false negatives:**
- *Name replaced by noise, address intact:* "West Express Associated LLP, 19 Cameron St" vs "Lyrapyra, 19 CMAERON
  ST".
- *Native-script names with a thin address:* "Lakshmi Infotech Private Limited, Plot No.12, …" vs "लक्ष्मी इंफोटेक
  प्राइवेट लिमिटेड, DOOR NO 12, MUMBAI".
- *Records with no address and a generic name:* "Universal Tech Private Limited" vs "Smt Universal Tech Private
  Ltd" (empty address).
- *Ambiguous records:* the record's two best S1s score almost the same (`rec_gap` < 0.05); 613 false negatives.
- *Blocking misses:* about 1% of true pairs at top-1 (1,731 on the sample). These are mostly empty-address records
  with generic names, and pairs sharing only over-common keys; the matcher cannot recover them at M = 1.

**Known limitations:**
- *Empty-address records* (about 3% of records) are decided on the name alone and remain the largest bucket per
  pair. Name-ambiguity counts (how many S1s share the name) were tried and did not help (Appendix B).
- *Cap pressure at full scale:* the cap-45 simulation loses about 0.9% of true pairs to oversized keys. The
  combination keys recover most, but not all.
- *M = 1:* when a record's true S1 is ranked second (about 0.35% of pairs), it cannot be matched.
- *Blocking caps the score:* on the full-train OOF, the M = 1 candidates hold 96.3% of true pairs, so a perfect
  matcher would reach only about 0.988 macro F0.5 (§3.5). The best public scores (0.991) are above that ceiling,
  so the remaining gap is in candidate generation more than in the classifier.
- *Train → test gap:* OOF 0.978 vs public leaderboard 0.963. Test has about 5.8 records per S1 against 4.7 in
  train, i.e. more hard distractors (same street, conflicting house numbers). Lower thresholds hurt on the
  leaderboard (0.30 → 0.957), so test is precision-limited, not recall-limited. Thresholds 0.69 and 0.79 score the
  same (0.963). The cross-encoder blend and the stacking narrow the gap to 0.977 by rejecting look-alike distractors.

---

## 6. Conclusion

Reverse blocking with many independent noise-robust keys, combination keys for common tokens and an M = 1
many-to-one choice gives 99% pair recall with about one candidate per record. A compact LightGBM on fuzzy,
phonetic and context features, with a threshold tuned on the exact macro F0.5 (singletons included), turns that
into a precise matcher.

The largest single lever was structural rather than model capacity. Letting each record choose its S1, and
feeding the matcher how decisive that choice was (`rec_gap`), carries most of the signal. It also mattered to
block train exactly like test, so this signal keeps its test-time distribution.

The last gain came from a second, differently built scorer. A small multilingual cross-encoder fine-tuned on our
own blocking candidates reads the two raw records jointly; blended with LightGBM on the same candidate set it lifts
the public leaderboard from 0.963 to 0.973, more than its held-out fold-B gain (+0.0039), because test's errors are
exactly the look-alike distractors it separates best. A second-level LightGBM stacked on the held-out fold, over
the pair features, both probabilities and their S1-level context, adds +0.0029 there and lifts the public
leaderboard to 0.977; a second cross-encoder (other seed, sample and length, same held-out fold) in the stack adds
+0.0006 on fold B and gives the final **0.978359**.

---

## 7. Licences

| Component | Licence | Use |
|---|---|---|
| LightGBM 4.7.0 | MIT | matcher model (trained from scratch; no pretrained weights) |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (117.7M parameters) | Apache-2.0 | pretrained weights, fine-tuned twice as cross-encoders CE v1 / CE v2 (§4.5, §4.7) |
| PyTorch | BSD-3-Clause | cross-encoder training and inference |
| transformers, tokenizers (Hugging Face) | Apache-2.0 | model loading, tokenization |
| anyascii 0.3.3 | ISC | transliteration |
| Metaphone 0.6 | BSD | double metaphone |
| rapidfuzz 3.14.6 | MIT | fuzzy string similarities |
| scikit-learn 1.8.0 | BSD-3-Clause | TF-IDF, GroupKFold, AUC |
| numba 0.67.0 | BSD-2-Clause | JIT kernels (sparse dot, accumulator, house match) |
| llvmlite 0.49.0 | BSD-2-Clause / Apache-2.0 with LLVM exception | numba backend |
| polars 1.44.2 | MIT | data processing |
| pyarrow 23.0.1 | Apache-2.0 | streaming TSV / parquet |
| numpy 2.4.4, scipy 1.17.1, pandas 3.0.2 | BSD-3-Clause | numerics |
| psutil 7.2.2 | BSD-3-Clause | memory reporting |

Two ML models are used, both within the rules: the LightGBM classifier we train (MIT, under 10⁵ leaf values) and
the fine-tuned cross-encoder `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 117.7M
parameters, far below 8B; licence and size checked on its Hugging Face model card). Only its pretrained weights
are downloaded; it is fine-tuned on the provided train data only.

## 8. Fair Play

- **Only the provided data** in `dataset/` is used. There are no external APIs, web requests, business registries,
  geocoding, address-normalisation services, gazetteers, or data augmentation from the internet.
- **All vocabularies are hand-written from EDA of the provided files.** This covers the legal-form and
  street-abbreviation tables and the placeholder list.
- **Internet is used only to `pip install` the pinned open-source packages.**
- **No test labels exist or are used.** Test data only passes through the same unsupervised normalisation,
  blocking and feature code. Per-country TF-IDF and IDF statistics are fit on each split's own S1 text, without
  labels.
- Results are fully reproducible from `code/business_entity_resolution/`, with fixed seeds (42).

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (see its `README.md` for exact commands):

| File | Role |
|---|---|
| `src/config.py` | paths (`BER_DATA_DIR`, `BER_CACHE_DIR`, `BER_OUTPUT_DIR`), seeds, training sample size |
| `src/normalize.py` | raw TSV → normalised parquet (streaming, multiprocess) |
| `src/blocking.py` | reverse blocking (keys, index, numba accumulator, ranking) → candidate pairs |
| `src/features.py` | pairwise features |
| `src/train_matcher.py` | LightGBM, GroupKFold, OOF metric, threshold search |
| `src/predict.py` | test scoring → `output/matching_results.tsv`, `output/candidate_pairs.tsv` |
| `src/run_all.py` | **entry point**: all stages with checkpoints, log, runtime estimate, summary |
| `src/eval_blocking.py`, `src/probe_scale.py` | blocking recall evaluation; full-scale cost probe |
| `src/eda.py` | exploratory analysis |
| `src/error_analysis.py` | buckets OOF errors by pattern, with the oracle gain of fixing each bucket |
| `src/prefilter.py` | candidate pre-filter (stage-1 model on blocking signals + cheap flags), OOF trade-off, test keep set |
| `src/simulate_prefilter_test.py` | replays the pre-filter on an earlier run's test outputs (sanity check) |
| `src/tune_threshold.py` | distractor-weighted OOF threshold tuning, empty-address rule check, OOF probability export |
| `src/analyze_test_outputs.py` | per-country test output checks against train ground-truth structure |
| `src/models.py`, `src/probe_thresholds.py` | unused experiment (pairwise-only matcher + ensemble, built but not used); low/high-threshold LB probes from cached test probabilities |
| `src/report_numbers.py` | fills the `{{...}}` numbers of this document from `run.log` and the outputs |
| `src/cross_encoder.py` | cross-encoder fine-tuning, fold-B blend, test scoring → `output_ce/` (stage 2) |
| `src/stack.py` | second-level LightGBM on fold B, test scoring; `--ce2_dir` → `output_stack2/` (final submission) |
| `src/stack3.py` | experiment: seed-bagged `stack_ctx2` (LB 0.978289, not submitted) |
| `src/make_submission_zip.py` | builds `<team>_submission.zip` |
| `run_kaggle.ipynb` | unattended Kaggle run (Save & Run All) |
| `run_kaggle_ce.ipynb` | cross-encoder stage on a Kaggle GPU T4 x2 (repository copy; steps in README) |
| `run_kaggle_ce2.ipynb` | CE v2 on a Kaggle GPU T4 x2 (same fold B as CE v1) |
| `run_kaggle_stack.ipynb`, `run_kaggle_stack2.ipynb` | stacking with one CE (LB 0.977) / two CEs (final) on Kaggle |
| `DECISIONS.md` | log of design decisions with measurements |

Reproduce: `python code/business_entity_resolution/src/run_all.py`. The from-scratch run (run 2:
normalize, probe, blocking, features, train, predict) took 4.8 h on a Kaggle CPU session. The pre-filter and
tuning stages added in runs 3–4 take about 17 min and cut prediction from 86 to 57 min, so the CPU pipeline
runs in about 4.7 h. The cross-encoder stages (`src/cross_encoder.py`) then take 86 min (CE v1) and about 2.3 h
(CE v2) on a Kaggle GPU T4 x2, and the final stacking (`src/stack.py --ce2_dir`) about 35 min.

### B. Additional Results

- Pairwise-only matcher (built but not used): the same LightGBM without the density/context features (rec_gap,
  rec_n, s1_n_cands, s1_rank, s1_gap, key_score, n_keys), plus its average with the base model, to test whether
  test's denser neighbourhoods shift those features. Sample OOF macro F0.5 at w = 2: base 0.98768, pairwise
  0.98379, ensemble 0.98737; on the densest quartile of S1s 0.98495 / 0.97951 / 0.98490. It was not submitted:
  the context features help even in dense neighbourhoods, and the leaderboard probes showed test is
  precision-limited.
- Run-2 feature ablation (sample, OOF macro F0.5; all run-2 features 0.98912):
  - without the distinctive-word features: 0.98815 (−0.0010);
  - without the house-number features: 0.98802 (−0.0011);
  - without the name-ambiguity counts: 0.98933 (+0.0002, so they were dropped);
  - without all new features (run 1 plus the placeholder fix): 0.98610.
  - The final run-2 model (without the ambiguity counts) scores 0.98933, +0.0033 over run 1.
- Blocking v1 → v2 (sample, M = 1 recall):
  - v1: India 0.864, US 0.954, overall 0.917.
  - v2 keys: overall 0.987.
  - v2 with the improved ranking score (empty-field reweighting, house-number bonus, phonetic names): overall
    0.990, India 0.987, US 0.992.
- Full-scale blocking cost (probe, full France test index, 20k records):
  - 2,019 key hits per record, 148 pairs scored per record, 5.1 s per 10k records.
  - The accumulator rewrite cut per-record time 3–4×.
- Full-test probe for all countries: France: 2,019 key hits and 148 pairs scored per record, 2.3 s/10k records, est. 6 min; India: 1,687 key hits and 144 pairs scored per record, 2.7 s/10k records, est. 23 min; US: 1,637 key hits and 148 pairs scored per record, 2.2 s/10k records, est. 15 min.
