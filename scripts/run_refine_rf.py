#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refine.data import infer_feature_columns, read_table, write_table
from refine.model import ReFINEConfig, run_refine_rf


def parse_feature_cols(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReFINE with a Random Forest classifier.")
    parser.add_argument("--positives", required=True, help="CSV/CSV.GZ/Parquet table containing active training molecules.")
    parser.add_argument("--negatives", required=True, help="CSV/CSV.GZ/Parquet table containing candidate/negative molecules.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--feature-cols", help="Comma-separated feature columns.")
    parser.add_argument("--feature-prefix", help="Select feature columns by prefix, e.g. MACCS_.")
    parser.add_argument("--feature-start", type=int, help="First numeric feature column, inclusive.")
    parser.add_argument("--feature-end", type=int, help="Last numeric feature column, exclusive.")
    parser.add_argument("--metadata-cols", default="smile,smiles,hit_id,inchikey,label", help="Comma-separated metadata columns to keep.")
    parser.add_argument("--keep-frac", type=float, default=0.8)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--auc-stop", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    positives = read_table(args.positives)
    negatives = read_table(args.negatives)
    feature_cols = infer_feature_columns(
        positives,
        explicit=parse_feature_cols(args.feature_cols),
        prefix=args.feature_prefix,
        start=args.feature_start,
        end=args.feature_end,
    )

    missing = [col for col in feature_cols if col not in negatives.columns]
    if missing:
        raise ValueError(f"Negative table is missing feature columns: {missing[:5]}")

    metadata_cols = [col for col in parse_feature_cols(args.metadata_cols) or [] if col in negatives.columns]
    metadata = negatives[metadata_cols] if metadata_cols else None

    config = ReFINEConfig(
        keep_frac=args.keep_frac,
        repeats=args.repeats,
        max_iter=args.max_iter,
        auc_stop=args.auc_stop,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
    )
    retained_by_iter, cv = run_refine_rf(positives[feature_cols], negatives[feature_cols], config, metadata=metadata)

    for i, retained in enumerate(retained_by_iter):
        write_table(retained, out_dir / f"Iter_{i}.csv", index=False)
    write_table(cv, out_dir / "cv_res.csv", index=False)


if __name__ == "__main__":
    main()
