# Amazon ML Challenge 2026 — Business Entity Resolution

Match each Source 1 (deduplicated reference) record to all Source 2 / Source 3 records
that refer to the same real-world business. See README.md for the full problem statement.

## Hard rules

- **No external APIs, lookups, or geocoding — ever.** Use only the provided data in `dataset/`.
  No web requests, no business registries, no address-normalisation services, no internet data augmentation.
- **Any ML model must be MIT or Apache-2.0 licensed and <= 8B parameters.** Check the license before adding a model.
- **Country is an open set.** Test contains `France`, which is absent from train. Never hard-code,
  filter, or one-hot to `{US, India}`; every test S1 entity (France included) must appear in the output.

## Metric

- Macro F0.5 (beta = 0.5, precision-weighted), computed per Source 1 entity and averaged over all of them.
- Singletons (no true matches) score 1.0 if predicted empty, 0.0 if anything is predicted.
- Score locally on a held-out split of train; there is no test ground truth.

## Data

- `dataset/train/train_source{1,2,3}.tsv`, `dataset/train/train_ground_truth.tsv`, `dataset/test/test_source{1,2,3}.tsv`.
- All files are TSV. Always read with `sep="\t", keep_default_na=False` (and `dtype=str`).
- Files are large (~2.5 GB total, millions of rows) and the machine has ~8 GB RAM: stream in chunks
  or load one file at a time.

## Outputs

- `output/matching_results.tsv` (scored) and `output/candidate_pairs.tsv` (blocking set fed to the model;
  matches must be a subset of candidates).
- Validate before submitting:
  `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`

## Code

- All source lives in `code/business_entity_resolution/src/`.
- Environment: Windows / PowerShell. Use `python`, not `python3`.
