#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refine.signatures import generate_signaturizer_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3200-dimensional Signaturizer features.")
    parser.add_argument("--input", required=True, help="Input molecule table.")
    parser.add_argument("--output", required=True, help="Output table with appended signature columns.")
    parser.add_argument("--model-dir", required=True, help="Directory containing A1-E5 Signaturizer model folders.")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column name.")
    parser.add_argument("--h5", help="Optional intermediate HDF5 output path.")
    args = parser.parse_args()

    generate_signaturizer_features(args.input, args.output, args.model_dir, args.smiles_col, args.h5)


if __name__ == "__main__":
    main()
