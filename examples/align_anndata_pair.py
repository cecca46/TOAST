import scanpy as sc

from spatial_OT import align_anndata


def main():
    source = sc.read_h5ad("source.h5ad")
    target = sc.read_h5ad("target.h5ad")

    result = align_anndata(
        source_adata=source,
        target_adata=target,
        spatial_key="spatial",
        label_key="cell_type",
        embedding_key=None,
        gene_join="intersection",
        alpha=0.5,
        epsilon=0.1,
        k=10,
        n_comps=50,
        use_spatial_terms=True,
    )

    print("Transport shape:", result.transport.shape)
    print("accuracy_max_prob:", result.metrics.get("accuracy_max_prob"))
    print("JSD:", result.metrics.get("JSD"))


if __name__ == "__main__":
    main()
