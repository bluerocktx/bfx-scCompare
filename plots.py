from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import helpers


def plot_clusters_and_batch(
    adata: AnnData,
    cluster_key: str,
    batch_key: str,
    title: str | None = None,
    save: bool = False,
    show: bool = True,
    **scanpy_kwargs,
) -> list[Axes]:
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
):
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


def plot_confusion_matrix(
    adata: AnnData, bulk_sig: pd.DataFrame, save: bool | str = False, show: bool = True
) -> ConfusionMatrixDisplay:
    sc.set_figure_params(
        scanpy=True,
        dpi=200,
        dpi_save=250,
        frameon=True,
        vector_friendly=False,
        fontsize=4,
        figsize=None,
        color_map=None,
        format="pdf",
        facecolor=None,
        transparent=True,
    )

    if "highly_variable" not in adata.var:
        error_text = (
            '"highly_variable" not found in adata object. Did you run '
            "`scanpy.pp.highly_variable`?"
        )
        raise KeyError(error_text)

    unique_genes = list(
        adata.var["highly_variable"][adata.var["highly_variable"]].index
    )
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

    sc.set_figure_params(
        scanpy=True,
        dpi=200,
        dpi_save=250,
        frameon=True,
        vector_friendly=False,
        fontsize=4,
        figsize=None,
        color_map=None,
        format="pdf",
        facecolor=None,
        transparent=True,
    )

    corr_cluster_T = corr_cluster.T
    prediction = []
    true_value = []
    for i in range(len(corr_cluster_T.columns)):
        prediction.append(
            corr_cluster_T[corr_cluster_T.columns[i]]
            .sort_values(ascending=False)
            .index[0]
        )
        true_value.append(
            adata[:, unique_genes].obs["leiden"][corr_cluster_T.columns[i]]
        )

    cm = confusion_matrix(true_value, prediction, labels=None, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)

    out = disp.plot(cmap="viridis")

    if show:
        plt.show()
    if save:
        plt.savefig(save, bbox_inches="tight")

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
    xticklabels: str | bool | list | int = 1,
    yticklabels: str | bool | list | int = False,
    cmap: str = "viridis",
    robust: bool = True,
    save: bool | str = False,
    show: bool = True,
) -> sns.matrix.ClusterGrid:
    out = sns.clustermap(
        bulk_sig,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap=cmap,
        robust=robust,
    )

    if title:
        out.fig.suptitle(title, y=1.05)

    if save:
        plt.savefig(save, bbox_inches="tight")
    if show:
        plt.show()

    return out


def plot_asgd_pearson_violin(
    adata: AnnData,
    canon_label_asgd_key: str,
    title: str | None = None,
    asgd_pearson_key: str = "asgd_pearson",
    use_raw: bool = False,
    show: bool = True,
    save: bool | str = False,
    **scanpy_kwargs,
) -> Axes:
    scanpy_kwargs_final = {
        "rotation": 90
    }
    scanpy_kwargs_final.update(scanpy_kwargs)
    ax = plt.axes()

    sc.pl.violin(
        adata,
        [asgd_pearson_key],
        groupby=canon_label_asgd_key,
        use_raw=use_raw,
        return_fig=True,
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
    title: str | None = None,
    canon_label_asgd_key: str = "canon_label_asgd",
    asgd_pearson_key: str = "asgd_pearson",
    hue_key: str = "control_vs_experimental",
    palette: str | list | dict = "muted",
    split: bool = True,
    x_label_rotation: float = 90,
    legend_loc: str | tuple[float, float] = "lower left",
    save: bool | str = False,
    show: str = True,
) -> Axes:
    df_vp = pd.concat((adata_map.obs, adata_test.obs))
    plt.figure(figsize=(5, 5))
    ax = plt.axes()
    out = sns.violinplot(
        x=canon_label_asgd_key,
        y=asgd_pearson_key,
        hue=hue_key,
        data=df_vp,
        palette=palette,
        split=split,
        rotation=x_label_rotation,
        ax=ax,
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
    plt.plot(x, x, "k",label="x=y")

    if (canon_label_asgd_key in adata_map.obs) and (canon_label_asgd_key_colors in adata_map.uns):
        color_labels = adata_map.obs[canon_label_asgd_key].drop_duplicates().sort_values().values.tolist()
        color_dict = dict(zip(color_labels, adata_map.uns[canon_label_asgd_key_colors]))
        color_dict['unmapped'] = "#050500"

        for i, idx in enumerate(df_fracs.index):
            plt.plot(df_fracs[0].values[i], df_fracs[1].values[i], marker=marker,color=color_dict[idx],label=idx)
    else:
        for i, idx in enumerate(df_fracs.index):
            plt.plot(df_fracs[0].values[i], df_fracs[1].values[i], marker=marker,label=idx)

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
