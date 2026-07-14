from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import read_table, write_table


def generate_signaturizer_features(
    input_path: str | Path,
    output_path: str | Path,
    model_dir: str | Path,
    smiles_col: str = "smiles",
    h5_path: str | Path | None = None,
) -> None:
    import h5py
    from signaturizer import Signaturizer

    input_df = read_table(input_path)
    if smiles_col not in input_df.columns:
        raise ValueError(f"Column {smiles_col!r} not found in {input_path}")

    model_dir = Path(model_dir)
    folders = [f"{letter}{i}" for letter in "ABCDE" for i in range(1, 6)]
    model_paths = [str(model_dir / folder) for folder in folders if (model_dir / folder).is_dir()]
    if not model_paths:
        raise ValueError(f"No Signaturizer model folders found under {model_dir}")

    h5_path = Path(h5_path) if h5_path is not None else Path(output_path).with_suffix(".h5")
    sign = Signaturizer(model_name=model_paths, local=True, verbose=True)
    sign.predict(input_df[smiles_col], str(h5_path))

    with h5py.File(h5_path, "r") as handle:
        features = pd.DataFrame(handle["signature"][:])
    output = pd.concat([input_df.reset_index(drop=True), features], axis=1)
    write_table(output, output_path, index=False)
