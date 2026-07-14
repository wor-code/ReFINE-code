from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffixes = "".join(path.suffixes)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(path.suffixes)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=index)
    elif suffixes.endswith(".csv.gz"):
        df.to_csv(path, index=index, compression="gzip")
    elif path.suffix == ".csv":
        df.to_csv(path, index=index)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def infer_feature_columns(
    df: pd.DataFrame,
    explicit: Iterable[str] | None = None,
    prefix: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> list[str]:
    if explicit:
        return list(explicit)
    if prefix is not None:
        cols = [c for c in df.columns if str(c).startswith(prefix)]
        if not cols:
            raise ValueError(f"No feature columns found with prefix {prefix!r}")
        return cols
    if start is not None or end is not None:
        if start is None or end is None:
            raise ValueError("Both start and end must be provided for numeric feature columns.")
        cols = [str(i) for i in range(start, end)]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing numeric feature columns: {missing[:5]}")
        return cols

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    excluded = {"label", "iter", "proba", "v"}
    cols = [c for c in numeric_cols if c not in excluded]
    if not cols:
        raise ValueError("No numeric feature columns found. Provide --feature-cols or --feature-prefix.")
    return cols

