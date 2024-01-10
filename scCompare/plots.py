from __future__ import annotations

import scanpy as sc
from anndata import AnnData

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from scCompare import helpers


def plot_clusters_and_batch(
    adata: AnnData,
    cluster_key: str,
    batch_key: str,
    title: str | None = None,
    save: bool | str = False,
    show: bool = True,
    **scanpy_kwargs,
) -> list[Axes]:
    """Create UMAP plots of clusters and batches side by side.

    A convenience wrapper around
    `scanpy.pl.umap <https://scanpy.readthedocs.io/en/stable/generated/scanpy.pl.umap.html>`_.  # noqa: E501

    Args:
        adata: The object from which to create the plots.
        cluster_key: The `adata.obs` key in which the clusters are encoded.
        batch_key: The `adata.obs` key in which the batches are encoded.
        title (optional): The title to display. Default = `None`.
        save (optional): The location at which to save the plot, or a falsey value. If
            falsey, do not save the plot. Default = `False`.
        show (optional): Whether or not to show the plot. Default = `True`.
        **scanpy_kwargs: Keyword arguments to be passed to
            `scanpy.pl.umap <https://scanpy.readthedocs.io/en/stable/generated/scanpy.pl.umap.html>`_.  # noqa: E501

    Returns:
        The UMAP plot axes.
    """
    return plot_grouped_umaps(
        adata=adata,
        keys=[cluster_key, batch_key],
        title=title,
        save=save,
        show=show,
        **scanpy_kwargs,
    )


def plot_mapping_correlation(
    bulk_sig: pd.DataFrame,
    title: str | None = None,
    save: bool | str = False,
    show: bool = True,
    **kwargs,
) -> sns.matrix.ClusterGrid:
    """Plot correlation of bulk signatures between groupings.

    Produces a clustermap showing correlation values and hierarchical clustering the
    results. Takes the input of `helpers.generate_bulk_sigs`.

    Args:
        bulk_sig: A pre-calculated dataframe of bulk-signatures by grouping.
        title (optional): A title for the plot. Default = `None`.
        save (optional): A path to save the plot to, or a falsey value. If falsey, does
            not save the plot. Default = `False`.
        show (optional): Whether or not to show the plot. Default = `True`.
        **kwargs: Additional command line arguments to pass to
            `sns.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.  # noqa: E501

    Returns:
        The clustermap plot.
    """
    default_kwargs = {
        "cmap": "RdYlBu_r",
        "vmin": 0,
        "cbar_kws": {"label": "correlation"},
    }
    out = sns.clustermap(bulk_sig.corr(), **{**default_kwargs, **kwargs})

    if title:
        out.fig.suptitle(title, y=1)

    if save:
        out.savefig(save, bbox_inches="tight")
    if show:
        plt.show()

    return out


def plot_cluster_and_assignments_umap(
    adata: AnnData,
    title: str | None = None,
    cluster_key: str = "leiden",
    asgd_cluster_key: str = "asgd_cluster",
    asgd_pearson_key: str = "asgd_pearson",
    save: bool | str = False,
    show: bool = True,
    **scanpy_kwargs,
) -> list[Axes]:
    """UMAP plot of clusters, assigned groupings, and gropuing correlation side-by-side.

    A convenience wrapper around
    `scanpy.pl.umap <https://scanpy.readthedocs.io/en/stable/generated/scanpy.pl.umap.html>`_.  # noqa: E501

    Args:
        adata: The object from which to create the plots.
        cluster_key: The `adata.obs` key in which the clusters are encoded.
        asgd_cluster_key: The `adata.obs` key in which the assigned groupings are
            encoded.
        asgd_pearson_key: The `adata.obs` key in which the correlations to the assigned
            groupings are encoded.
        title (optional): The title to display. Default = `None`.
        save (optional): The location at which to save the plot, or a falsey value. If
            falsey, do not save the plot. Default = `False`.
        show (optional): Whether or not to show the plot. Default = `True`.
        **scanpy_kwargs: Keyword arguments to be passed to
            `scanpy.pl.umap <https://scanpy.readthedocs.io/en/stable/generated/scanpy.pl.umap.html>`_.  # noqa: E501
    Returns:
        The UMAP plot axes.
    """
    return plot_grouped_umaps(
        adata=adata,
        keys=[cluster_key, asgd_cluster_key, asgd_pearson_key],
        title=title,
        save=save,
        show=show,
        **scanpy_kwargs,
    )


def plot_bulk_sig_heatmap(
    bulk_sig: pd.DataFrame,
    title: str | None = None,
    save: bool | str = False,
    show: bool = True,
    **sns_kwargs,
) -> sns.matrix.ClusterGrid:
    """A heatmap of the bulk signature.

    Takes the output of `helpers.generate_bulk_sig`.

    Args:
        bulk_sig: A DataFrame with a bulk signature per grouping.
        title (optional): The title to display. Default = `None`.
        save (optional): The location at which to save the plot, or a falsey value. If
            falsey, do not save the plot. Default = `False`.
        show (optional): Whether or not to show the plot. Default = `True`.
        **sns_kwargs: Additional key word arguments to pass to
            `sns.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.  # noqa: E501

    Returns:
        A clustermap plot.
    """
    sns_kwargs_final = {
        "xticklabels": 1,
        "yticklabels": False,
        "cmap": "viridis",
        "robust": True,
    }
    sns_kwargs_final.update(sns_kwargs)
    out = sns.clustermap(bulk_sig, **sns_kwargs_final)

    if title:
        out.fig.suptitle(title, y=1.05)

    if save:
        plt.savefig(save, bbox_inches="tight")
    if show:
        plt.show()

    return out


def plot_asgd_pearson_violin(
    adata: AnnData,
    canon_label_asgd_key: str = "canon_label_asgd",
    asgd_pearson_key: str = "asgd_pearson",
    title: str | None = None,
    show: bool = True,
    save: bool | str = False,
    **scanpy_kwargs,
) -> Axes:
    """Show correlation of expression patterns to assigned grouping bulk signature.

    Creates a violin plot showing distribution of correlation values for each grouping.
    Can be run after running `helpers.assign_clusters_to_cells`.

    Args:
        adata: Object for which assignments and assignment correlations have been
            generated.
        canon_label_asgd_key (optional): The `adata.obs` column with the assigned
            groupings. Default = "canon_label_asgd".
        asgd_pearson_key (optional): The `adata.obs` column with the correlation to
            assigned grouping bulk signature. Default = "asgd_pearson".
        title (optional): The title to display. Default = `None`.
        save (optional): The location at which to save the plot, or a falsey value. If
            falsey, do not save the plot. Default = `False`.
        show (optional): Whether or not to show the plot. Default = `True`.
        **scanpy_kwargs: Keyword arguments to pass to
            `sc.pl.violin <https://scanpy.readthedocs.io/en/stable/generated/scanpy.pl.violin.html>`_.  # noqa: E501

    Returns:
        The violin plot.
    """
    scanpy_kwargs_final = {"rotation": 90, "use_raw": False}
    scanpy_kwargs_final.update(scanpy_kwargs)
    ax = plt.axes()

    sc.pl.violin(
        adata,
        [asgd_pearson_key],
        groupby=canon_label_asgd_key,
        ax=ax,
        show=False,
        **scanpy_kwargs_final,
    )

    if title:
        ax.set_title(title)

    if save:
        ax.get_figure().savefig(save, bbox_inches="tight")
    if show:
        ax.get_figure().show()

    return ax


def plot_canon_assigned_labels_umap(
    adata: AnnData,
    title: str | None = None,
    cluster_key: str = "leiden",
    canon_label_asgd_key: str = "canon_label_asgd",
    save: bool | str = False,
    show: bool = True,
    **scanpy_kwargs,
) -> list[Axes]:
    """UMAP plot of clusters and assigned groupings side-by-side.

    A convenience wrapper around
    `scanpy.pl.umap <https://scanpy.readthedocs.io/en/stable/generated/scanpy.pl.umap.html>`_.

    Args:
        adata: The object from which to create the plots.
        cluster_key: The `adata.obs` key in which the clusters are encoded.
        canon_asgd_cluster_key: The `adata.obs` key in which the assigned groupings are
            encoded.
        title (optional): The title to display. Default = `None`.
        save (optional): The location at which to save the plot, or a falsey value. If
            falsey, do not save the plot. Default = `False`.
        show (optional): Whether or not to show the plot. Default = `True`.
        **scanpy_kwargs: Keyword arguments to be passed to
            `scanpy.pl.umap <https://scanpy.readthedocs.io/en/stable/generated/scanpy.pl.umap.html>`_.  # noqa: E501
    Returns:
        The UMAP plot axes.
    """
    return plot_grouped_umaps(
        adata=adata,
        keys=[cluster_key, canon_label_asgd_key],
        title=title,
        save=save,
        show=show,
        **scanpy_kwargs,
    )


def plot_grouped_umaps(
    adata: AnnData,
    keys: list[str],
    title: str | None = None,
    save: bool | str = False,
    show: bool = True,
    **scanpy_kwargs,
) -> list[Axes]:
    """UMAP plots colored by an arbitrary number of facets, side-by-side.

    A convenience wrapper around
    `scanpy.pl.umap <https://scanpy.readthedocs.io/en/stable/generated/scanpy.pl.umap.html>`_.  # noqa: E501

    Args:
        adata: The object from which to create the plots.
        keys: The columns in `adata.obs` to color by.
        title (optional): The title to display. Default = `None`.
        save (optional): The location at which to save the plot, or a falsey value. If
            falsey, do not save the plot. Default = `False`.
        show (optional): Whether or not to show the plot. Default = `True`.
        **scanpy_kwargs: Keyword arguments to be passed to
            `scanpy.pl.umap <https://scanpy.readthedocs.io/en/stable/generated/scanpy.pl.umap.html>`_.  # noqa: E501
    Returns:
        The UMAP plot axes.
    """
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

    default_kwgs = {
        "use_raw": False,
        "wspace": 0.25,
        "color_map": "RdYlBu_r",
    }
    default_kwgs.update(scanpy_kwargs)

    out = sc.pl.umap(adata, color=keys, show=False, return_fig=True, **default_kwgs)

    if title:
        out.suptitle(title, y=1.05)

    if save:
        plt.savefig(save, bbox_inches="tight")
    if show:
        plt.show()

    return out


def plot_map_vs_test_pearson_violin(
    adata_test: AnnData,
    adata_map: AnnData,
    canon_label_asgd_key: str = "canon_label_asgd",
    asgd_pearson_key: str = "asgd_pearson",
    x_label_rotation: float = 90,
    legend_loc: str | tuple[float, float] = "lower left",
    title: str | None = None,
    save: bool | str = False,
    show: str = True,
    **sns_kwargs,
) -> Axes:
    """Compare bulk signature correlation disributions between test and mapping.

    Creates a violin plot showing distribution of correlation values for each grouping,
    colored separately for the test and mapping datasets. Can be run after running
    `helpers.assign_clusters_to_cells` on the test and mapping objects.

    Args:
        adata_test, adata_map: Objects for which assignments and assignment correlations
            have been generated.
        canon_label_asgd_key (optional): The `adata.obs` column with the assigned
            groupings in both `adata` objects. Default = "canon_label_asgd".
        asgd_pearson_key (optional): The `adata.obs` column with the correlation to
            assigned grouping bulk signature in both `adata` objects. Default =
            "asgd_pearson".
        x_label_rotation: Rotation of x labels in degrees. Default = 90.
        legend_loc: Legend loc to be passed to `ax.legend`. Default = "lower left".
        title (optional): The title to display. Default = `None`.
        save (optional): The location at which to save the plot, or a falsey value. If
            falsey, do not save the plot. Default = `False`.
        show (optional): Whether or not to show the plot. Default = `True`.
        **sns_kwargs: Keyword arguments to pass to
            `seaborn.violinplot <https://seaborn.pydata.org/generated/seaborn.violinplot.html>`_.  # noqa: E501

    Returns:
        The violin plot.
    """
    final_sns_kwargs = {
        "hue_key": "control_vs_experimental",
        "palette": "muted",
        "split": True,
    }
    final_sns_kwargs.update(sns_kwargs)
    df_vp = pd.concat((adata_map.obs, adata_test.obs))
    plt.figure(figsize=(5, 5))
    ax = plt.axes()
    out = sns.violinplot(
        x=canon_label_asgd_key, y=asgd_pearson_key, data=df_vp, ax=ax, **sns_kwargs
    )
    plt.xticks(rotation=x_label_rotation)
    out.legend(loc=legend_loc)

    if title:
        ax.set_title(title)

    if save:
        plt.savefig(save, bbox_inches="tight")
    if show:
        plt.show()

    return out


def plot_map_vs_test_cluster_fractions(
    adata_test: AnnData,
    adata_map: AnnData,
    title: str | None = None,
    map_name: str = "Map",
    test_name: str = "Test",
    canon_label_asgd_key: str = "canon_label_asgd",
    save: bool | str = False,
    show: bool = True,
    log: bool = False,
    marker: str | Path | MarkerStyle = "o",
) -> Axes:
    """Show relationship between grouping assignments in test and mapping datasets.

    Produces a scatterplot of proportion of cells assigned to each grouping in the test
    vs mapping datasets. Plots an x=y line for reference as well.

    Args:
        adata_test, adata_map: The test and mapping dataset objects.
        title (optional): The title to put on the plot. Default = `None`.
        map_name, test_name (optional): The title to label each dataset with. Defaults =
            "Map", "Test.
        canon_label_asgd_key (optional): The `adata_<test|map>.obs` column in which
            the assigned groupings are labeled. Default = "canon_label_asgd".
        save (optional): A path to save the plot to, or a falsey value. If falsey, does
            not save the plot. Default = `False`.
        show (optional): Whether or not to show the plot. Default = `True`.
        log (optional): Whether or not to plot both axes on a log scale. Default =
            `False`.
        marker (optional): The marker to use for scatterplot dots. Default = "o".

    Return:
        The scatterplot axes.
    """
    sc.set_figure_params(
        scanpy=True,
        dpi=200,
        dpi_save=250,
        frameon=True,
        vector_friendly=True,
        fontsize=10,
        figsize=None,
        color_map=None,
        format="pdf",
        facecolor=None,
        transparent=False,
        ipython_format="png2x",
    )

    map_freqs = helpers.calc_frequencies(
        adata_map, canon_label_asgd_key, return_as="df"
    )
    asgd_clust_freqs = helpers.calc_frequencies(
        adata_test, canon_label_asgd_key, return_as="df"
    )
    asgd_clust_freqs.index = [1]
    df_fracs = pd.concat((map_freqs, asgd_clust_freqs))

    xlabel = f"{map_name} Cluster Fractions"
    ylabel = f"{test_name} Cluster Fractions"
    if log:
        df_fracs.loc[0] = np.log(df_fracs.loc[0])
        df_fracs.loc[1] = np.log(df_fracs.loc[1])
        xlabel = f"Log({xlabel})"
        ylabel = f"Log({ylabel})"
        xmin = np.min(np.min(df_fracs))
    else:
        df_fracs = df_fracs.fillna(0)
        xmin = 0

    df_fracs = df_fracs.T.sort_values(by=0, ascending=False)

    canon_label_asgd_key_colors = canon_label_asgd_key + "_colors"

    x = np.linspace(xmin, np.max(np.max(df_fracs)), 10)
    plt.figure()
    plt.plot(x, x, "k", label="x=y")

    if (canon_label_asgd_key in adata_map.obs) and (
        canon_label_asgd_key_colors in adata_map.uns
    ):
        color_labels = (
            adata_map.obs[canon_label_asgd_key]
            .drop_duplicates()
            .sort_values()
            .values.tolist()
        )
        color_dict = dict(zip(color_labels, adata_map.uns[canon_label_asgd_key_colors]))
        color_dict["unmapped"] = "#050500"

        for i, idx in enumerate(df_fracs.index):
            plt.plot(
                df_fracs[0].values[i],
                df_fracs[1].values[i],
                marker=marker,
                color=color_dict[idx],
                label=idx,
            )
    else:
        for i, idx in enumerate(df_fracs.index):
            plt.plot(
                df_fracs[0].values[i], df_fracs[1].values[i], marker=marker, label=idx
            )

    ax = plt.gca()
    ax.legend(bbox_to_anchor=(1.05, 1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title:
        ax.set_title(title)

    if save:
        plt.savefig(save, bbox_inches="tight")

    if show:
        plt.show()

    return ax
