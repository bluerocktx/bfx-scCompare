from __future__ import annotations

import argparse
import os
from collections import Counter
from warnings import warn

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.metrics import r2_score

from scCompare import plots
from scCompare.helpers import (
    _reformat_adata_for_export,
    assign_class_to_cells,
    assign_clusters_to_cells,
    calc_assigned_cluster_fracs,
    calc_frac_misclassified_cells,
    calc_frac_unmapped_cells,
    calc_frequencies,
    calc_silhouette_score,
    calc_simple_match_similarity,
    derive_statistical_group_cutoff,
    generate_bulk_sig,
    get_gene_presence_in_cluster,
    get_marker_genes_per_cluster,
)
from scCompare.logger import InternalLogger


def enforce_grouping_key(adata: AnnData, grouping_key: str):
    """Validate the grouping key exists in the adata object.

    Args:
        adata: The object to QC.
        grouping_key: They key with the ground truth groupings.

    Raises:
        KeyError: if the `grouping_key` is not in `adata.obs`.
    """
    run_params = InternalLogger()

    if grouping_key not in adata.obs:
        error_text = (
            f"The grouping key `{grouping_key}` was not found in `obs`. `grouping_key` "
            "needs to be a column in obs that defines the grouping for each cell."
        )
        raise KeyError(error_text)

    if grouping_key not in adata.uns:
        warning_text = (
            f"{grouping_key} not found in `uns`. Metadata about the clustering in "
            f"{grouping_key} can not be saved."
        )
        warn(warning_text)
        run_params.write_warning(warning_text)


def qc_adata_map(
    adata: AnnData, grouping_key: str, color_map: str = "RdYlBu_r"
) -> AnnData:
    """Ensure the object can be used as the mapping object in `sc_compare`.

    Checks that proper preprocessing has been done and the appropriate structures are
    defined.

    Args:
        adata: The object to QC.
        grouping_key: The key defining the ground truth groupings.
        color_map (optional): The color map to use. Default = "RdYlBu_r".

    Returns:
        The adata object, with the grouping key colors added if they did not already
            exist.

    Raises:
        KeyError: if the `grouping_key` is not in `adata.obs`.
    """
    enforce_grouping_key(adata, grouping_key)
    adata.obs["control_vs_experimental"] = "control"

    grouping_key_counts = adata.obs[grouping_key].value_counts()
    insuff_keys = grouping_key_counts.loc[grouping_key_counts < 2].index.tolist()
    if len(insuff_keys) != 0:
        do_something = 1
        
    if "highly_variable" not in adata.var:
        error_text = (
            "`highly_variable` not found in adata_map.var. "
            "`scanpy.pp.highly_variable_genes` needs to be run on this object before "
            "using scCompare."
        )
        raise KeyError(error_text)

    if f"{grouping_key}_colors" not in adata.uns:
        sc.pl.umap(
            adata, color=[grouping_key], color_map=color_map, use_raw=False, show=True
        )  # need this to generate cluster_key colors column

    adata.uns["asgd_cluster_colors"] = adata.uns[f"{grouping_key}_colors"]

    return adata


def qc_adata_test(adata: AnnData) -> AnnData:
    """Ensure the object can be used as the test object in `sc_compare`.

    Args:
        adata: The object to QC.

    Returns:
        The prepared object.
    """

    adata.obs["control_vs_experimental"] = "testing"

    return adata


def sc_compare(
    adata_test: AnnData | str,
    adata_map: AnnData | str,
    map_cluster_key: str,
    test_cluster_key: str | None = None,
    outdir: str = "./scCompare_output",
    n_mad_floor: float = 5,
    n_mad: float = 0,
    make_plots: bool = True,
    show_plots: bool = True,
) -> AnnData:
    """Run the scCompare pipeline.

    This function takes a "mapping" and "test" adata objects. The mapping object must
    have a set of gropuings defined in the `obs`. The similarity of the test object to
    the mapping object is then assessed by finding how the distribution of cells
    assigned to each grouping compare between the two datasets.

    Args:
        adata_test: Test dataset to compare to the mapping dataset.
        adata_map: Mapping dataset that the test dataset will be mapped onto.
        outdir: Path to output directory.
        map_cluster_key: adata.obs column name containing cluster IDs for the mapping
            dataset.
        test_cluster_key (optional): adata.obs column name containing cluster IDs for
            the test dataset. Defaults to same as `map_cluster_key`.
        n_mad_floor (optional): lowest MAD to be used before it is automatically
            calculated. Default = 5.
        n_mad (optional): Number of MADs to use for cutoff calculation. If set to 0,
            will be statistically calculated by finding the knee. Default = 0.
        make_plots (optional): Make the plots?. Default = `False`.

    Returns:
        adata_test object with additional annotations provided by scCompare pipeline.
    """

    if test_cluster_key is None:
        test_cluster_key = map_cluster_key

    # Accommodate reading from CLI
    if type(adata_test) is str:
        adata_test = sc.read_h5ad(adata_test)
    if type(adata_map) is str:
        adata_map = sc.read_h5ad(adata_map)

    print("Running QC on adata objects...", end="")
    adata_test = qc_adata_test(adata_test)
    adata_map = qc_adata_map(adata_map, map_cluster_key)
    print("Done!")
    print("adata objects pass QC!")

    # Make plot output directory
    plot_outdir = os.path.join(outdir, "plots/")
    if not os.path.exists(plot_outdir):
        os.makedirs(plot_outdir, exist_ok=True)

    # TODO: Do I need this in order to keep adata_map intact?
    adata_map = adata_map.copy()

    run_params = InternalLogger()
    run_params.write_log(["call_local_variables"], locals())

    sc.set_figure_params(
        scanpy=True,
        dpi=200,
        dpi_save=250,
        frameon=True,
        vector_friendly=False,
        fontsize=14,
        figsize=None,
        color_map=None,
        format="pdf",
        facecolor=None,
        transparent=True,
    )

    print("Generating bulk signatures...", end="")
    bulk_sig = generate_bulk_sig(adata_map, cluster_key=map_cluster_key)
    print("Done!")
    run_params.write_log(["bulk_sig"], bulk_sig)

    print("Assigning clusters to mapping cells...", end="")
    adata_map = assign_clusters_to_cells(
        adata_map, bulk_sig=bulk_sig, subset_unique=True
    )
    print("Done!")

    print("Deriving statistical cutoff...", end="")
    stat_group_cutoff = derive_statistical_group_cutoff(
        adata_map,
        cluster_key=map_cluster_key,
        n_mad_floor=n_mad_floor,
        n_mad=n_mad,
        show_plot=show_plots,
    )
    print("Done!")
    run_params.write_log(["stat_group_cutoff"], stat_group_cutoff)

    print("Assigning class to mapping cells...", end="")
    adata_map = assign_class_to_cells(adata_map, stat_group_cutoff=stat_group_cutoff)
    print("Done!")

    map_postthresh_grp_counts = adata_map.obs['canon_label_asgd'].value_counts()
    insufficient_postthresh_cells = map_postthresh_grp_counts.loc[map_postthresh_grp_counts < 2].index.tolist()

    if len(insufficient_postthresh_cells) != 0:
        print(f"Categories with less than 2 cells passing thresholding will be excluded from process: {insufficient_postthresh_cells}")
        adata_map = adata_map[~adata_map.obs['canon_label_asgd'].isin(insufficient_postthresh_cells)]
        bulk_sig.drop(insufficient_postthresh_cells,axis=1,inplace=True)
        for k in insufficient_postthresh_cells:
            del stat_group_cutoff[k]
    # PLOTS AND METRICS FOR MAPPING DATASET

    print("Calculating mapping metrics...", end="")
    frac_misclass_cells = calc_frac_misclassified_cells(adata_map, key1=map_cluster_key)
    run_params.write_log(
        ["adata_map_fraction_misclassified_cells"], frac_misclass_cells
    )

    frac_unmapped_cells = calc_frac_unmapped_cells(adata_map)
    run_params.write_log(["adata_map_fraction_unmapped_cells"], frac_unmapped_cells)
    print("Done!")

    if make_plots:
        thisplot = plots.plot_mapping_correlation(
            bulk_sig,
            title="Mapping correlation plot",
            save=os.path.join(plot_outdir, "mapping_correlation.png"),
            show=show_plots,
        )
        run_params.write_log(["plots", "mapping_correlation"], thisplot)

    freqs = Counter(adata_map.obs["canon_label_asgd"])
    map_freqs = pd.DataFrame(freqs, index=[0]) / len(adata_map.obs)

    sc.tl.rank_genes_groups(adata_map, "canon_label_asgd", use_raw=False)
    sc.tl.filter_rank_genes_groups(
        adata_map,
        key=None,
        groupby="canon_label_asgd",
        use_raw=False,
        key_added="rank_genes_groups_filtered",
        min_in_group_fraction=0.1,
        min_fold_change=2,
        max_out_group_fraction=0.9,
    )

    # METRICS FOR MAPPING DATA
    marker_genes_per_cluster_df = get_marker_genes_per_cluster(adata_map)
    run_params.write_log(["marker_genes_per_cluster_df"], marker_genes_per_cluster_df)

    bin_genes_df_map = get_gene_presence_in_cluster(adata_map)
    run_params.write_log(["bin_genes_df_map"], bin_genes_df_map)

    # Main workflow

    adata_test = adata_test.copy()
    adata_test = assign_clusters_to_cells(adata_test, bulk_sig)
    adata_test.uns["asgd_cluster_colors"] = adata_map.uns[f"{map_cluster_key}_colors"]
    adata_test = assign_class_to_cells(adata_test, stat_group_cutoff)

    print("Calculating cluster frequencies...", end="")
    asgd_clust_freqs = calc_frequencies(adata_test, "canon_label_asgd", return_as="df")
    run_params.write_log(["assigned_cluster_frequencies"], asgd_clust_freqs)
    print("Done!")

    fraction_mapped_cells = 1 - np.sum(
        adata_test.obs["canon_label_asgd"] == "unmapped"
    ) / len(adata_test.obs)
    run_params.write_log(["fraction_mapped_cells"], fraction_mapped_cells)

    asgd_clust_freqs.index = [1]
    df_fracs = pd.concat((map_freqs, asgd_clust_freqs)).fillna(0).T.sort_values(by=0)
    r2 = r2_score(df_fracs[0].values, df_fracs[1].values)

    # This removes cells classified to a class that has only 1 cell classified as it.
    # This is because scanpy.rank_genes_groups will throw an error if only one class

    fkeys = list(freqs.keys())
    problem_cluster = []
    for i in range(len(fkeys)):
        if freqs[fkeys[i]] == 1:
            problem_cluster.append(fkeys[i])
    if len(problem_cluster) > 0:
        adata_test = adata_test[
            adata_test.obs["canon_label_asgd"].values != problem_cluster[0], :
        ]

    print("fraction mapped cells = " + str(fraction_mapped_cells))
    run_params.write_log(["fraction_mapped_calls"], fraction_mapped_cells)
    print("r2 score = " + str(r2))
    run_params.write_log(["r2_score"], r2)

    sc.tl.rank_genes_groups(adata_test, "canon_label_asgd", use_raw=False)
    sc.tl.filter_rank_genes_groups(
        adata_test,
        key=None,
        groupby="canon_label_asgd",
        use_raw=False,
        key_added="rank_genes_groups_filtered",
        min_in_group_fraction=0.1,
        min_fold_change=2,
        max_out_group_fraction=0.9,
    )
    df_genes = pd.DataFrame(adata_test.uns["rank_genes_groups_filtered"]["names"])

    bin_genes_df_test = get_gene_presence_in_cluster(adata_test)
    run_params.write_log(["adata_test_gene_presence_in_cluster"], bin_genes_df_test)

    asgd_clust_fracts = calc_assigned_cluster_fracs(adata_test, adata_map)
    run_params.write_log(["assigned_cluster_fractions"], asgd_clust_fracts)

    match_similarity_res = calc_simple_match_similarity(
        bin_genes_df_map, bin_genes_df_test
    )
    run_params.write_log(["simple_match_similarity"], match_similarity_res)

    combined_cluster_metrics = pd.concat(
        (asgd_clust_fracts, match_similarity_res), axis=1
    )
    run_params.write_log(["combined_cluster_metrics"], combined_cluster_metrics)

    mapping_frac_mapped = 1 - calc_frac_unmapped_cells(adata_map)
    df_frac_mapped_compare = pd.DataFrame(
        [mapping_frac_mapped, fraction_mapped_cells],
        index=["Map", "Test"],
        columns=["Fraction of Cells Mapped"],
    )
    run_params.write_log(["df_frac_mapped_compare"], df_frac_mapped_compare)

    df_sil_score_compare = pd.DataFrame(
        [calc_silhouette_score(adata_test), calc_silhouette_score(adata_map)],
        index=["Map", "Test"],
        columns=["Silhouette Score"],
    )
    run_params.write_log(["silhouette_score_comparison"], df_sil_score_compare)

    if make_plots:
        print("Generating plots")

        thisplot = plots.plot_cluster_and_assignments_umap(
            adata_map,
            title="Original and Assigned Annotations (mapping data)",
            cluster_key=map_cluster_key,
            save=os.path.join(plot_outdir, "mapping_clusters_and_assignments_umap.png"),
            show=show_plots,
        )
        run_params.write_log(["plots", "clusters_and_assignments_umap"], thisplot)

        thisplot = plots.plot_bulk_sig_heatmap(
            bulk_sig,
            title="Bulk signature heatmap",
            save=os.path.join(plot_outdir, "bulk_sig_heatmap.png"),
            show=show_plots,
        )
        run_params.write_log(["plots", "bulk_sig_heatmap"], thisplot)

        thisplot = plots.plot_asgd_pearson_violin(
            adata_map,
            canon_label_asgd_key=map_cluster_key,
            title="Assigned Pearson violin (mapping data)",
            save=os.path.join(plot_outdir, "asgd_pearson_violin.png"),
            show=show_plots,
        )
        run_params.write_log(["plots", "assigned_pearson_violin"], thisplot)

        thisplot = plots.plot_canon_assigned_labels_umap(
            adata_map,
            title="Original and final assigned labels UMAP plot (mapping data)",
            cluster_key=map_cluster_key,
            save=os.path.join(plot_outdir, "map_canon_label_assignments_umap.png"),
            show=show_plots,
        )
        run_params.write_log(["plots", "map_canon_assigned_labels"], thisplot)

        thisplot = plots.plot_grouped_umaps(
            adata_test,
            title="Raw and final assigned labels UMAP plot (test data)",
            keys=["asgd_cluster", "canon_label_asgd"],
            save=os.path.join(plot_outdir, "test_label_assignments_umap.png"),
            show=show_plots,
        )
        run_params.write_log(["plots", "test_assigned_labels"], thisplot)

        thisplot = plots.plot_map_vs_test_pearson_violin(
            adata_test,
            adata_map,
            title="Map vs test Pearson violin plot",
            save=os.path.join(plot_outdir, "map_vs_test_pearson_violin.png"),
            show=show_plots,
        )
        run_params.write_log(["plots", "map_vs_test_pearson_violin"], thisplot)

        plots.plot_map_vs_test_cluster_fractions(
            adata_test,
            adata_map,
            title="Map vs test cluster fractions",
            save=os.path.join(plot_outdir, "map_vs_test_cluster_fractions.png"),
            show=show_plots,
        )
        run_params.write_log(["plots", "map_vs_test_cluster_fractions"], thisplot)

        print("Done generating plots!")

    print("Writing logs...", end="")
    adata_map = _reformat_adata_for_export(adata_map)
    run_params.write_log(["adata_map"], adata_map)
    print("Done!")

    adata_test.uns["scCompare"] = run_params.output_log()
    adata_test = _reformat_adata_for_export(adata_test)
    print("scCompare Complete!")

    return adata_test


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scCompare",
        description="Compare 2 single-cell RNAseq datasets",
        epilog="Link to github",
    )

    parser.add_argument("test_data", help="Path to h5ad file as test dataset")
    parser.add_argument("map_data", help="Path to h5ad file as map dataset")
    parser.add_argument(
        "--outdir", help="Path to output directory", default="./scCompare_output"
    )
    parser.add_argument(
        "--map_cluster_key",
        help="adata.obs column name containing cluster IDs for mapping dataset",
        default="leiden",
    )
    parser.add_argument(
        "--test_cluster_key",
        help="adata.obs column name containing cluster IDs for test dataset",
        default="leiden",
    )
    parser.add_argument(
        "--n_mad_floor",
        help="Lowest MAD to be used before it is automatically calculated",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--n_mad",
        help="Number of MADs to use for cutoff calculation",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--color_map", help="Color map to use for plots", default="RdYlBu_r"
    )
    parser.add_argument("--make_plots", help="Make the plots?", default=True, type=bool)

    args = parser.parse_args()

    return args


def write_scCompare_adata(
    adata: AnnData, outpath: str = "./scCompare_output/adata_out.h5ad"
):
    """Helper to write scCompare adata object to disk

    :params: adata: adata object that has run scCompare
    """

    out = _reformat_adata_for_export(adata)
    out.write_h5ad(outpath)


if __name__ == "__main__":
    args = _parse_arguments()
    adata_out = sc_compare(
        adata_test=args.test_data,
        adata_map=args.map_data,
        outdir=args.outdir,
        map_cluster_key=args.map_cluster_key,
        test_cluster_key=args.test_cluster_key,
        n_mad_floor=args.n_mad_floor,
        n_mad=args.n_mad,
        color_map=args.color_map,
        make_plots=args.make_plots,
    )

    adata_out = _reformat_adata_for_export(adata_out)

    print("Writing adata object...", end="")
    adata_out.write(os.path.join(args.outdir, "adata_out.h5ad"))
    print("Done!")
