#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refine.data import infer_feature_columns, read_table, write_table
from refine.metrics import classification_metrics
from refine.model import ReFINEConfig, predict_proba_in_batches


def parse_cols(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RF on a selected negative subset and predict a screening table.")
    parser.add_argument("--positives", required=True)
    parser.add_argument("--negative-subset", required=True, help="Table containing retained or removed negatives.")
    parser.add_argument("--screening", required=True, help="Full table to score.")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--feature-cols")
    parser.add_argument("--feature-prefix")
    parser.add_argument("--feature-start", type=int)
    parser.add_argument("--feature-end", type=int)
    parser.add_argument("--metadata-cols", default="smile,smiles,hit_id,inchikey,label")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    positives = read_table(args.positives)
    negatives = read_table(args.negative_subset)
    screening = read_table(args.screening)
    feature_cols = infer_feature_columns(
        positives,
        explicit=parse_cols(args.feature_cols),
        prefix=args.feature_prefix,
        start=args.feature_start,
        end=args.feature_end,
    )

    config = ReFINEConfig(random_state=args.random_state, n_estimators=args.n_estimators, max_depth=args.max_depth, n_jobs=args.n_jobs)
    rng = np.random.default_rng(args.random_state)
    pos_x = positives[feature_cols].reset_index(drop=True)
    neg_x = negatives[feature_cols]
    screen_x = screening[feature_cols].to_numpy()

    predictions = pd.DataFrame(index=screening.index)
    metrics_rows = []
    for repeat in range(args.repeats):
        sampled_idx = rng.choice(neg_x.index.to_numpy(), size=len(pos_x), replace=False)
        x = pd.concat([pos_x, neg_x.loc[sampled_idx].reset_index(drop=True)], axis=0).to_numpy()
        y = np.repeat([1, 0], len(pos_x))

        y_score = np.array([])
        y_true = np.array([])
        kf = KFold(n_splits=5, shuffle=True, random_state=args.random_state + repeat)
        for train_index, test_index in kf.split(x):
            model = RandomForestClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                n_jobs=config.n_jobs,
                random_state=args.random_state,
            )
            model.fit(x[train_index], y[train_index])
            y_score = np.hstack([y_score, model.predict_proba(x[test_index])[:, 1]])
            y_true = np.hstack([y_true, y[test_index]])

        row = classification_metrics(y_true, y_score)
        row["repeat"] = repeat
        metrics_rows.append(row)

        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            n_jobs=config.n_jobs,
            random_state=args.random_state,
        )
        model.fit(x, y)
        predictions[f"repeat_{repeat}"] = predict_proba_in_batches(model, screen_x, config.batch_size)

    metadata_cols = [col for col in parse_cols(args.metadata_cols) or [] if col in screening.columns]
    out_pred = pd.concat([screening[metadata_cols].reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    prefix = Path(args.out_prefix)
    write_table(out_pred, prefix.with_name(prefix.name + "_prediction.csv.gz"), index=False)
    write_table(pd.DataFrame(metrics_rows), prefix.with_name(prefix.name + "_cv.csv"), index=False)


if __name__ == "__main__":
    main()
