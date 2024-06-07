from __future__ import annotations

from collections import Counter

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from kneed import KneeLocator
from scipy import stats

from scCompare.logger import InternalLogger


def _reformat_adata_for_export(adata: anndata.AnnData) -> anndata.AnnData:
    """Reformat scCompare-produced adata object to prepare for export. This involves
    removing very large elements and cleaning NaN values.

    Args:
        adata: Object for reformatting.

    Returns
        Reformatted adata object.
    """

    keys = ["rank_genes_groups", "rank_genes_groups_filtered"]

    for k in keys:
        if k in adata.uns:
            names = adata.uns[k]["names"]
            newnames = []
            for rec in names:
                newnames.append(tuple("nan" if pd.isna(x) else x for x in rec))

            adata.uns[k]["names"] = np.rec.array(newnames)

    if "scCompare" in adata.uns:
        if "call_local_variables" in adata.uns["scCompare"]:
            if "run_params" in adata.uns["scCompare"]["call_local_variables"]:
                del adata.uns["scCompare"]["call_local_variables"]["run_params"]

        if "plots" in adata.uns["scCompare"]:
            del adata.uns["scCompare"]["plots"]

        if "adata_map" in adata.uns["scCompare"]:
            adata.uns["scCompare"]["adata_map"] = _reformat_adata_for_export(
                adata.uns["scCompare"]["adata_map"]
            )

    return adata


def calc_silhouette_score(
    adata: anndata.AnnData, key: str = "canon_label_asgd"
) -> float:
    """Calculate the silhouette score from an anndata object.

    Args:
        adata: Object from which to calculate the silhouette score.
        key: Key in adata.obs which assigns the mapping.

    Returns:
        The silhouette score.
    """

    adata_not_unmapped = adata[adata.obs[key] != "unmapped"]
    sil_score = sklearn.metrics.silhouette_score(
        adata_not_unmapped.obsm["X_umap"], adata_not_unmapped.obs[key]
    )

    return sil_score


def calc_simple_match_similarity(
    bin_genes_df_map: pd.DataFrame, bin_genes_df_test: pd.DataFrame
) -> pd.DataFrame:
    """Calculate a match similarity score between the mapping and testing genes in each
    cluster. The input dataframes should be generated from
    `get_gene_presence_in_cluster`.

    Args:
        bin_genes_df_map: Mapping dataset df.
        bin_genes_df_test: Testing dataset df.

    Returns:
        A single-column dataframe ("Sample Match Simarity") with results (one row per
        column in input dataframes).
    """

    def match_similarity(df, col_A, col_B):
        A_0 = set(df.index[df[col_A] == 0])
        B_0 = set(df.index[df[col_B] == 0])

        A_1 = set(df.index[df[col_A] == 1])
        B_1 = set(df.index[df[col_B] == 1])

        match_0 = len(A_0.intersection(B_0))
        match_1 = len(A_1.intersection(B_1))

        N = len(df.index)

        score = (match_0 + match_1) / N
        return score

    cols2test = [x for x in bin_genes_df_test.columns if x in bin_genes_df_map.columns]
    bin_genes_df_map.columns = ["map_" + x for x in bin_genes_df_map.columns]
    bin_genes_df_test.columns = ["test_" + x for x in bin_genes_df_test.columns]

    df_bin_genes = pd.concat((bin_genes_df_map, bin_genes_df_test), axis=1).fillna(0)

    ms_out = [
        match_similarity(df_bin_genes, "map_" + x, "test_" + x) for x in cols2test
    ]

    match_similarity_res = pd.DataFrame(
        ms_out, index=cols2test, columns=["Simple Match Similarity"]
    )

    return match_similarity_res


def calc_assigned_cluster_fracs(
    adata_test: anndata.AnnData,
    adata_map: anndata.AnnData,
    key: str = "canon_label_asgd",
) -> pd.DataFrame:
    """Calculate assigned cluster fractions for both mapping and testing datasets.

    Args:
        adata_test: Test dataset.
        adata_map: Mapping dataset.
        key: Column name from `adata.obs` to read cluster identities.

    Returns:
        Dataframe of the fraction of of cells assigned to each cluster per dataset.
    """

    test_freq = Counter(adata_test.obs[key])
    df_test_fracs = pd.DataFrame(
        test_freq, index=["Testing Cluster Fractions"]
    ).T / sum(test_freq.values())

    map_freq = Counter(adata_map.obs[key])
    df_map_fracs = pd.DataFrame(map_freq, index=["Mapping Cluster Fractions"]).T / sum(
        map_freq.values()
    )

    asgd_clust_fracts = pd.concat((df_map_fracs, df_test_fracs), axis=1).sort_values(
        by="Mapping Cluster Fractions", ascending=False
    )

    return asgd_clust_fracts


def get_gene_presence_in_cluster(
    adata: anndata.AnnData, uns_key: str = "rank_genes_groups_filtered"
) -> pd.DataFrame:
    """Get a dataframe representing if a gene is present in a cluster.

    Args:
        cluster_genes_for_bin: Dataframe of differentially expressed genes per cluster.

    Returns:
        Dataframe indicating each gene's presence as a differentially expressed gene in
            each cluster. 1 indicates it is a differentially expressed gene in that
            cluster. 0 indicates it is not.
    """

    cluster_genes_for_bin = pd.DataFrame(adata.uns[uns_key]["names"])
    all_bin_genes = {x for x in cluster_genes_for_bin.values.flatten() if x == x}

    bin_genes_dict = dict()
    for gene in all_bin_genes:
        gene_present_in_col = [
            int(gene in cluster_genes_for_bin[x].values)
            for x in cluster_genes_for_bin.columns
        ]
        bin_genes_dict.update({gene: gene_present_in_col})

    bin_genes_df = pd.DataFrame(bin_genes_dict).T
    bin_genes_df.columns = cluster_genes_for_bin.columns

    return bin_genes_df


def get_marker_genes_per_cluster(
    adata: anndata.AnnData, uns_key: str = "rank_genes_groups_filtered"
) -> pd.DataFrame:
    """Return a dataframe of marker genes from each cluster.

    Args:
        adata: AnnData object after running `sc.tl.rank_genes_groups` or
            `sc.tl.rank_genes_groups_filtered`.

    Returns:
        Dataframe of marker genes for each cluster.
    """

    cluster_genes = pd.DataFrame(adata.uns[uns_key]["names"]).stack()
    cluster_names = cluster_genes.index.get_level_values(1)

    df = pd.DataFrame([cluster_names, cluster_genes]).T.sort_values(0)
    df.rename(columns={0: "cluster", 1: "gene"}, inplace=True)

    return df


def rename_adata_obs_values(
    adata: anndata.AnnData, key: str, mapping_dict: dict
) -> list:
    """Rename the values of an adata.obs column based on mapping_dict. Keeps values that
    have no remapping key.

    Args:
        adata: Object to be renamed.
        key: Name of adata.obs column.
        mapping_dict: Mapping to use when renaming with key=original name and value=new
            name.

    Returns:
        Renamed values.
    """

    out = [mapping_dict[x] if x in mapping_dict else x for x in adata.obs[key]]

    return out


def calc_frequencies(
    adata: anndata.AnnData, key: str, return_as: str = "dict"
) -> dict | pd.DataFrame:
    """Calculate frequencies of items in adata.obs[key].

    Args:
        adata: Object from which to calcualte frequencies.
        key: adata.obs column name for calculating frequencies.
        return_as (optional): Indicates return type. Can be either "dict" or "df".
            Default = "dict".

    Returns:
        Either a dictionary or DataFrame of frequencies, depending on `return_as`.
    """

    freqs = Counter(adata.obs[key])
    out = {k: v / len(adata.obs[key]) for k, v in freqs.items()}

    if return_as == "dict":
        return out
    elif return_as == "df":
        return pd.DataFrame(out, index=[0])


def calc_frac_unmapped_cells(
    adata: anndata.AnnData, key: str = "canon_label_asgd"
) -> float:
    """Calculate the fraction of unmapped cells resulting from an scCompare
    classification.

    Args:
      adata: Object from which to calculate fractions.
      key: adata.obs column name that contains the classifications.

    Returns:
      Fraction of cells that are unmapped.
    """

    freqs = Counter(adata.obs[key])
    out = freqs.get("unmapped", 0) / len(adata.obs)

    return out


def calc_frac_misclassified_cells(
    adata: anndata.AnnData, key1: str, key2: str = "canon_label_asgd"
) -> float:
    """Calculate the number of misclassified cells.

    Finds the fraction difference between `key1` and `key2` in adata.obs.

    Args:
        adata: Object to assess.
        key1: First column name from adata.obs.
        key2: Second column name from adata.obs.

    Returns:
        Proportion of cells annotated in `key1` that don't match in `key2`.
    """

    n_misclassified = len(adata.obs[adata.obs[key1] != adata.obs[key2]])

    return n_misclassified / len(adata.obs)


def assign_class_to_cells(
    adata: anndata.AnnData, stat_group_cutoff: float, outkey: str = "canon_label_asgd"
) -> anndata.AnnData:
    """Assigns a class to cells based on `stat_group_cutoff`.

    If the Pearson score for a cell's assignment exceeds `stat_group_cutoff`, that cell
    keeps that assignment. Otherwise, it is classified as "unmapped".

    Args:
        adata: Object to classify.
        stat_group_cutoff: Pearson score below which to label cell "unmatched".
        outkey: `adata.obs` key to store the output.

    Returns:
        Same `adata` object with added `adata.obs[outkey]`.
    """

    out = [
        y if x > stat_group_cutoff[y] else "unmapped"
        for x, y in zip(adata.obs["asgd_pearson"], adata.obs["asgd_cluster"])
    ]
    adata.obs[outkey] = out

    return adata


def derive_aggregate_metric_map(adata: anndata.AnnData, stat_cutoff: float) -> float:
    pearson_avg = adata.obs["asgd_pearson"].sum() / len(adata.obs)
    pearson_met = 1 - (np.sum(adata.obs["asgd_pearson"] < stat_cutoff) / len(adata.obs))
    agg_met_map = pearson_avg * pearson_met

    return agg_met_map


def derive_statistical_cutoff(adata: anndata.AnnData, n_mads: float = 3):
    """Returns the statistical cutoff generated from the mapping dataset.

    Args:
        adata: Mapping object.
        n_mads: The number of median absolute deviations away from the median to
            define the cutoff.
    Returns:
        Statistical cutoff.
    """

    lower = adata.obs["asgd_pearson"][
        adata.obs["asgd_pearson"] < adata.obs["asgd_pearson"].median()
    ]
    scale = 1 / (
        adata.obs["asgd_pearson"].std()
        / stats.median_abs_deviation(adata.obs["asgd_pearson"], scale=1)
    )
    stat_cutoff = adata.obs[
        "asgd_pearson"
    ].median() - n_mads * stats.median_abs_deviation(lower, scale=scale)

    return stat_cutoff


def assign_clusters_to_cells(
    adata: anndata.AnnData, bulk_sig: pd.DataFrame, subset_unique: bool = False
) -> anndata.AnnData:
    """Assigns a cluster and a Pearson score to each cell based on the bulk signatures.

    Args:
        adata: Object to assign clusters to.
        bulk_sig: DataFrame of bulk signatures.
        subset_unique (optional): Whether or not to subset highly variable genes.
            Default = `False`.

    Returns:
        Returns the same `adata` object with added `asgd_pearson` and `asgd_cluster`
        columns to `adata.obs`
    """

    # Here, each cell's expression of its unique genes is correlated to the bulk
    # signature of those genes that came from the mapping dataset
    df = pd.DataFrame(
        adata.X.todense(), columns=adata.var.index, index=adata.obs.index
    )[bulk_sig.index].T

    bulk_cols = list(bulk_sig.columns)
    corr_cluster = df.corrwith(bulk_sig[bulk_cols[0]])
    corr_cluster = pd.DataFrame(corr_cluster, columns=[bulk_cols[0]])
    for i in range(1, len(bulk_sig.columns)):
        temp = df.corrwith(bulk_sig[bulk_cols[i]])
        temp = pd.DataFrame(temp, columns=[bulk_cols[i]])
        corr_cluster = pd.concat([corr_cluster, temp], axis=1)

    unique_genes = list(
        adata.var["highly_variable"][adata.var["highly_variable"]].index
    )
    if subset_unique:
        adata_temp = adata.copy()[:, unique_genes]
    else:
        adata_temp = adata.copy()

    # For each cell, find which cluster it most correlates with and assign the value to
    # asgd_pearson and the cluster to asgd_cluster
    asgd_cluster = []
    asgd_pearson = []
    for i in range(len(adata_temp.obs.index)):
        asgd_cluster.append(corr_cluster.iloc[i].sort_values().index[-1])
        asgd_pearson.append(corr_cluster.iloc[i].sort_values()[-1])

    adata.obs["asgd_cluster"] = asgd_cluster
    adata.obs["asgd_pearson"] = asgd_pearson

    return adata


def generate_bulk_sig(
    adata: anndata.AnnData, cluster_key: str = "leiden"
) -> pd.DataFrame:
    """Generate bulk signatures by cluster.

    Args:
        adata: Mapping object.
        cluster_key: The adata.obs column that contains the cluster identity.

    Returns:
        DataFrame of bulk signatures by cluster.
    """

    unique_genes = list(
        adata.var["highly_variable"][adata.var["highly_variable"]].index
    )
    adata = adata[:, unique_genes]
    i = 0
    obs_clust = np.unique(adata.obs[cluster_key])
    x_mean = np.array(
        adata[adata.obs[cluster_key] == obs_clust[i]].X.mean(axis=0)
    ).squeeze()
    bulk_sig = pd.DataFrame(x_mean, index=adata.var.index, columns=[obs_clust[i]])
    for i in range(1, len(obs_clust)):
        x_mean = adata[adata.obs[cluster_key] == obs_clust[i]].X.mean(axis=0)
        bulk_sig[obs_clust[i]] = pd.DataFrame(
            np.array(x_mean).squeeze(), index=adata.var.index, columns=[obs_clust[i]]
        )

    return bulk_sig


def derive_statistical_group_cutoff(
    adata_map: anndata.AnnData,
    cluster_key: str,
    use_fisher: bool = False,
    n_stdev: float = 3,
    n_mad_floor: float | None = 5,
    n_mad: float | None = None,
    show_plot: bool = True,
) -> dict[str, float]:
    """Derive the statistical cutoff for each applied group in the mapping dataset.

    Args:
        adata_map: Mapping dataset.
        cluster_key: `adata.obs` key to cluster by.
        use_fisher (optional): Whether or not to use a fisher-transformed correlation to
            derive a threshold for cutting off correlations. Default = `False`.
        n_stdev (optional): Standard deviation cutoff if using fisher transformation.
            Ignored if `use_fisher` = `False`. Default = 3.
        n_mad_floor (optional): Automatically calculate MAD, but can't be lower than
            `n_mad_floor`. If set to `None`, no lower bound. Default = 5.
        n_mad (optional): Use exactly this many MADs to calculate statistical cutoffs.
            If `None`, use `n_mad_floor` and automatically calculate MAD instead.
            Default = `None`.

    Returns:
        A mapping of cluster names to cutoff values (one for each cluster).
    """
    if not use_fisher:
        return _derive_statistical_group_cutoff_n_mad(
            adata_map, cluster_key, n_mad_floor, n_mad, show_plot
        )

    return _derive_statistical_group_cutoff_fisher(adata_map, cluster_key, n_stdev)


def _derive_statistical_group_cutoff_fisher(
    adata_map: anndata.AnnData,
    cluster_key: str,
    n_stdev: float,
) -> dict[str, float]:
    print(
        "Using Z-transformed Pearson Correlation stdev for mapping threshold (stdev: "
        f"{n_stdev})"
    )
    clusters = adata_map.obs[cluster_key].unique()
    stat_group_cutoff = {}
    for cluster in clusters:
        group = adata_map.obs["asgd_pearson"][adata_map.obs[cluster_key] == cluster]
        transform = np.arctanh(group)
        stat_group_cutoff[cluster] = np.tanh(
            transform.mean() - n_stdev * transform.std()
        )

    canon_label_asgd = []
    for i in range(len(adata_map.obs)):
        if (
            adata_map.obs["asgd_pearson"][i]
            > stat_group_cutoff[adata_map.obs["asgd_cluster"][i]]
        ):
            canon_label_asgd.append(adata_map.obs["asgd_cluster"][i])
        else:
            canon_label_asgd.append("unmapped")
    adata_map.obs["canon_label_asgd"] = canon_label_asgd

    return stat_group_cutoff


def _derive_statistical_group_cutoff_n_mad(
    adata_map: anndata.AnnData,
    cluster_key: str,
    n_mad_floor: float | None = 5,
    n_mad: float | None = None,
    show_plot: bool = True,
) -> dict[str, float]:
    run_params = InternalLogger()

    mads = np.linspace(0, 10, num=50)
    misclass = []
    for i in range(len(mads)):
        leiden_clusters = np.unique(adata_map.obs[cluster_key])
        stat_group_cutoff = {}
        n_mads = mads[i]
        for i in range(len(leiden_clusters)):
            group = adata_map.obs["asgd_pearson"][
                adata_map.obs[cluster_key] == leiden_clusters[i]
            ]
            lower = group[group < group.median()]
            scale = 1 / (group.std() / stats.median_abs_deviation(group, scale=1))
            stat_cutoff = group.median() - n_mads * stats.median_abs_deviation(
                lower, scale=scale
            )
            stat_group_cutoff[leiden_clusters[i]] = stat_cutoff

        canon_label_asgd = []
        for i in range(len(adata_map.obs)):
            if (
                adata_map.obs["asgd_pearson"][i]
                > stat_group_cutoff[adata_map.obs["asgd_cluster"][i]]
            ):
                canon_label_asgd.append(adata_map.obs["asgd_cluster"][i])
            else:
                canon_label_asgd.append("unmapped")
        adata_map.obs["canon_label_asgd"] = canon_label_asgd
        gt = list(adata_map.obs[cluster_key].values)
        gd = list(adata_map.obs["canon_label_asgd"].values)
        num_mis_clas = 0
        for i in range(len(gt)):
            if gt[i] != gd[i]:
                num_mis_clas = num_mis_clas + 1
        misclass.append(num_mis_clas / len(gt))

    leiden_clusters = np.unique(adata_map.obs[cluster_key])
    stat_group_cutoff = {}

    if not n_mad:
        kneedle = KneeLocator(
            mads, misclass, S=1.0, curve="convex", direction="decreasing"
        )
        print(f"Knee Locator Result for Number of MADs Selection: {kneedle.knee}...")
        run_params.write_log(["stat_cutoff_knee_value"], kneedle.knee)

        if (n_mad_floor is not None) and (kneedle.knee < n_mad_floor):
            print(
                f"Knee Locator Result under MAD floor, using {n_mad_floor} MADs "
                "instead..."
            )
            n_mads = n_mad_floor
        else:
            n_mads = kneedle.knee

    if show_plot:
        plt.plot(mads, misclass)
        plt.title("Elbow Plot for Statistical Cutoff Threshold")
        plt.axvline(x=n_mads, color="r")
        plt.show()

    for i in range(len(leiden_clusters)):
        group = adata_map.obs["asgd_pearson"][
            adata_map.obs[cluster_key] == leiden_clusters[i]
        ]
        lower = group[group < group.median()]
        scale = 1 / (group.std() / stats.median_abs_deviation(group, scale=1))
        stat_cutoff = group.median() - n_mads * stats.median_abs_deviation(
            lower, scale=scale
        )  #
        stat_group_cutoff[leiden_clusters[i]] = stat_cutoff

    return stat_group_cutoff
