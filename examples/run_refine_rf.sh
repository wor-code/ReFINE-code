#!/usr/bin/env bash
set -euo pipefail

python scripts/run_refine_rf.py \
  --positives data/VDR_trainset1.csv \
  --negatives data/VDR_chemdiv_test1.parquet \
  --out-dir outputs/VDR_refine_rf \
  --feature-start 0 \
  --feature-end 3200 \
  --keep-frac 0.8 \
  --repeats 100 \
  --auc-stop 0.5

