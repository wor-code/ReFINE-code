# ReFINE

ReFINE (**Reinforced Feedback-guided Iterative Negative-space Evolution**) is a feedback-guided framework for ligand-based virtual screening. It iteratively retrains a classifier, scores the negative/candidate pool, removes high-confidence low-information negatives, and updates the effective negative training distribution until the model approaches chance-level separability.

This repository contains the cleaned code used for the ReFINE manuscript.

## Repository layout

```text
refine/
  data.py                  # table loading/writing and feature-column selection
  metrics.py               # classification metrics
  model.py                 # parameterized ReFINE Random Forest workflow
  signatures.py            # Signaturizer feature generation helper
scripts/
  run_refine_rf.py         # main ReFINE RF training workflow
  generate_signatures.py   # 3200-dimensional bioactivity/signature features
  predict_with_negative_subset.py
                           # baseline / retained / removed negative-set prediction
examples/
  run_refine_rf.sh         # example command
requirements.txt
```

## Installation

Create an environment with Python 3.9+ and install the dependencies:

```bash
pip install -r requirements.txt
```

`signaturizer` is only required for generating bioactivity signature features. If you already have precomputed feature tables, the ReFINE training workflow does not need to run Signaturizer.

## Input data format

The cleaned scripts accept `.csv`, `.csv.gz`, and `.parquet` files.

For ReFINE RF training, provide:

- a positive/active training table
- a negative/candidate table
- shared feature columns in both tables
- optional metadata columns in the negative table, such as `smile`, `smiles`, `hit_id`, `inchikey`, and `label`

For the bioactivity representation used in the main experiments, feature columns are typically named `0` to `3199`. MACCS features can be selected by prefix, for example `MACCS_`.

Large benchmark and screening data are not included in this repository. Place them under `data/` or provide absolute paths in the commands below.

## Generate Signaturizer features

```bash
python scripts/generate_signatures.py \
  --input data/new_screening_dt_wash.csv \
  --output data/new_screening_dt_sign.csv \
  --model-dir /path/to/signaturizer/models \
  --smiles-col smiles.smiles
```

The model directory should contain the local Signaturizer model folders `A1` to `E5`.

## Run ReFINE with Random Forest

For the 3200-dimensional bioactivity representation:

```bash
python scripts/run_refine_rf.py \
  --positives data/VDR_trainset1.csv \
  --negatives data/VDR_chemdiv_test1.parquet \
  --out-dir outputs/VDR_refine_rf \
  --feature-start 0 \
  --feature-end 3200 \
  --keep-frac 0.8 \
  --repeats 100 \
  --auc-stop 0.5
```

For MACCS features:

```bash
python scripts/run_refine_rf.py \
  --positives data/ESR1_trainset1_MACCS.csv \
  --negatives data/ESR1_chemdiv_test1_MACCS.parquet \
  --out-dir outputs/ESR1_refine_maccs \
  --feature-prefix MACCS_ \
  --keep-frac 0.8 \
  --repeats 100
```

Main outputs:

- `cv_res.csv`: five-fold CV metrics for each iteration and repeat
- `Iter_<n>.csv`: retained negative/candidate molecules after iteration `n`, with `proba` and `iter`

The manuscript default uses `keep-frac = 0.8`, equivalent to removing 20% of scored negatives per iteration.

## Predict using a selected negative subset

This script reproduces the baseline-style prediction workflow used to compare models trained on retained or removed negative subsets.

```bash
python scripts/predict_with_negative_subset.py \
  --positives data/VDR_trainset1.csv \
  --negative-subset outputs/VDR_refine_rf/Iter_9.csv \
  --screening data/VDR_chemdiv_test1.csv.gz \
  --out-prefix outputs/VDR_iter10_retained \
  --feature-start 0 \
  --feature-end 3200 \
  --repeats 100
```

Outputs:

- `<out-prefix>_prediction.csv.gz`: screening predictions for each repeat
- `<out-prefix>_cv.csv`: CV performance for each repeat

## Citation

If you use this code, please cite the ReFINE manuscript:

> ReFINE: Feedback-Guided Framework for Optimizing Negative Supervision in Virtual Screening.
