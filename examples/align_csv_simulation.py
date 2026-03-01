from spatial_OT import align_csv


def main():
    result = align_csv(
        source_csv="data/simulations/2D_sim_t1.csv",
        target_csv="data/simulations/2D_sim_t2.csv",
        x_col="x",
        y_col="y",
        label_col="cell_type",
        alpha=0.5,
        epsilon=0.1,
        k=10,
        n_comps=8,
        use_spatial_terms=True,
    )

    print("Transport shape:", result.transport.shape)
    print("mapping_accuracy:", result.metrics.get("mapping_accuracy"))
    print("JSD:", result.metrics.get("JSD"))


if __name__ == "__main__":
    main()
