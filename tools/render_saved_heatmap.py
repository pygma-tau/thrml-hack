#!/usr/bin/env python3
"""Render energy heatmaps from saved THRML sampling runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from thrml_momentum_swap_search import (
    build_program_and_spaces,
    score_unique_designs,
    build_energy_heatmap,
    save_energy_heatmap,
    DEFAULT_HEATMAP_DIR,
)


def _load_npz_samples(npz_paths: Iterable[Path]) -> List[np.ndarray]:
    aggregated: List[np.ndarray] | None = None
    for npz_path in sorted(npz_paths):
        with np.load(npz_path) as data:
            var_keys = sorted((k for k in data.files if k.startswith("var")), key=lambda k: int(k[3:]))
            if not var_keys:
                continue
            flattened = [data[key].reshape(-1, 1) for key in var_keys]
            if aggregated is None:
                aggregated = flattened
            else:
                for idx, arr in enumerate(flattened):
                    aggregated[idx] = np.concatenate([aggregated[idx], arr], axis=0)
    if aggregated is None:
        raise ValueError("No var* entries found in provided NPZ files.")
    return [arr.astype(np.int32) for arr in aggregated]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a heatmap from saved THRML sampling outputs.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing <prefix>_samples_rank*.npz files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="thrml_momentum_swap",
        help="Filename prefix used during sampling (default: thrml_momentum_swap).",
    )
    parser.add_argument("--var-x", type=str, required=True, help="Variable name for x-axis (e.g., L).")
    parser.add_argument("--var-y", type=str, required=True, help="Variable name for y-axis (e.g., DeltaC).")
    parser.add_argument(
        "--stat",
        choices=["min", "mean"],
        default="min",
        help="Energy statistic aggregated into each heatmap cell (default: min).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Optional extra output path; a copy always goes into {DEFAULT_HEATMAP_DIR}.",
    )
    args = parser.parse_args()

    npz_paths = sorted(args.data_dir.glob(f"{args.prefix}_samples_rank*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No NPZ files matching {args.prefix}_samples_rank*.npz found in {args.data_dir}")

    samples = _load_npz_samples(npz_paths)
    _, _, _, _, spaces, oracle = build_program_and_spaces()
    scored = score_unique_designs(samples, oracle)
    if not scored:
        raise RuntimeError("No designs found inside the saved samples.")

    grid, _, vals_x, vals_y = build_energy_heatmap(scored, spaces, args.var_x, args.var_y, reduction=args.stat)
    heatmap_name = f"{args.prefix}_{args.var_x}_{args.var_y}_saved.png"
    repo_target = DEFAULT_HEATMAP_DIR / heatmap_name
    outputs: List[Path] = [repo_target]
    if args.out is not None:
        outputs.insert(0, args.out)

    unique_outputs: List[Path] = []
    for path in outputs:
        resolved = path.resolve()
        if all(resolved != existing.resolve() for existing in unique_outputs):
            unique_outputs.append(path)

    save_energy_heatmap(grid, vals_x, vals_y, args.var_x, args.var_y, unique_outputs, args.stat)
    print("Saved heatmap to:")
    for path in unique_outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
