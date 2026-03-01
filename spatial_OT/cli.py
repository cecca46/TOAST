from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .pipeline import align_anndata, align_csv


def _parse_feature_cols(raw: Optional[str]):
    if raw is None or raw.strip() == "":
        return None
    return [col.strip() for col in raw.split(",") if col.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TOAST CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    align_parser = subparsers.add_parser("align", help="Align two spatial slices")

    input_group = align_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--source-h5ad", type=str, help="Source AnnData .h5ad path")
    input_group.add_argument("--source-csv", type=str, help="Source CSV path")

    align_parser.add_argument("--target-h5ad", type=str, help="Target AnnData .h5ad path")
    align_parser.add_argument("--target-csv", type=str, help="Target CSV path")

    align_parser.add_argument("--x-col", type=str, default="x", help="CSV x-coordinate column")
    align_parser.add_argument("--y-col", type=str, default=None, help="CSV y-coordinate column")
    align_parser.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help="Comma-separated CSV feature columns (default: infer from all non-coordinate and non-label columns)",
    )
    align_parser.add_argument("--label-col", type=str, default=None, help="CSV label column (required for CSV mode)")

    align_parser.add_argument("--spatial-key", type=str, default="spatial", help="AnnData spatial key in obsm")
    align_parser.add_argument("--label-key", type=str, default=None, help="AnnData label key in obs (required for AnnData mode)")
    align_parser.add_argument("--embedding-key", type=str, default=None, help="AnnData embedding key in obsm")
    align_parser.add_argument(
        "--gene-join",
        type=str,
        default="intersection",
        choices=["intersection", "none"],
        help="How to harmonize AnnData genes when embedding_key is not provided",
    )

    align_parser.add_argument("--k", type=int, default=10, help="k for spatial kNN graph")
    align_parser.add_argument("--alpha", type=float, default=0.5, help="FGW/TOAST alpha")
    align_parser.add_argument("--epsilon", type=float, default=0.1, help="Sinkhorn regularization")
    align_parser.add_argument("--n-comps", type=int, default=50, help="PCA components for joint embedding")
    align_parser.add_argument("--tol", type=float, default=1e-9, help="Transport convergence tolerance")
    align_parser.add_argument("--max-iter", type=int, default=1000, help="Max transport iterations")
    align_parser.add_argument(
        "--use-spatial-terms",
        action="store_true",
        default=False,
        help="Enable TOAST spatial coherence and neighborhood terms",
    )

    align_parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    return parser


def run_align(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common_kwargs = dict(
        alpha=args.alpha,
        epsilon=args.epsilon,
        k=args.k,
        n_comps=args.n_comps,
        use_spatial_terms=args.use_spatial_terms,
        tol=args.tol,
        max_iter=args.max_iter,
    )

    if args.source_h5ad is not None:
        if not args.target_h5ad:
            raise ValueError("When using --source-h5ad, provide --target-h5ad.")
        if not args.label_key:
            raise ValueError("When using --source-h5ad, provide --label-key.")
        import scanpy as sc

        source_adata = sc.read_h5ad(args.source_h5ad)
        target_adata = sc.read_h5ad(args.target_h5ad)
        result = align_anndata(
            source_adata=source_adata,
            target_adata=target_adata,
            spatial_key=args.spatial_key,
            label_key=args.label_key,
            embedding_key=args.embedding_key,
            gene_join=args.gene_join,
            **common_kwargs,
        )
    else:
        if not args.target_csv:
            raise ValueError("When using --source-csv, provide --target-csv.")
        if not args.label_col:
            raise ValueError("When using --source-csv, provide --label-col.")

        result = align_csv(
            source_csv=args.source_csv,
            target_csv=args.target_csv,
            x_col=args.x_col,
            y_col=args.y_col,
            label_col=args.label_col,
            feature_cols=_parse_feature_cols(args.feature_cols),
            **common_kwargs,
        )

    np.save(output_dir / "transport.npy", result.transport)
    np.savetxt(output_dir / "transport.csv", result.transport, delimiter=",")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(result.metrics, fp, indent=2)

    if result.metrics:
        print("accuracy_max_prob:", result.metrics.get("accuracy_max_prob"))
        print("JSD:", result.metrics.get("JSD"))

    print(f"Saved transport to: {output_dir / 'transport.npy'}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "align":
        run_align(args)


if __name__ == "__main__":
    main()
