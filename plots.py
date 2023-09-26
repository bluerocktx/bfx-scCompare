from __future__ import annotations

from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path

import helpers


def plot_clusters_and_batch(
    adata: AnnData,
    cluster_key: str = "leiden",
    batch_key: str = "batch_id",
    color_map: str = "RdYlBu_r",
    use_raw: bool = False,
    wspace: float = 0.25,
    save: bool = False,
    show: bool = True,
) -> list[Axes]:
    sc.set_figure_params(
        scanpy=True,
        dpi=100,
        dpi_save=250,
        frameon=True,
        vector_friendly=False,
        fontsize=14,
        figsize=None,
        color_map=None,
        format="pdf",
        facecolor=None,
        transparent=True,
    )  # , ipython_format='png2x')

    out = sc.pl.umap(
        adata,
        color=[cluster_key, batch_key],
        color_map=color_map,
        use_raw=use_raw,
        wspace=wspace,
        return_fig=True,
    )

    if save:
        plt.savefig(save, bbox_inches="tight")
    if show:
        plt.show()

    return out


def plot_mapping_correlation(
    bulk_sig: pd.DataFrame,
    cmap: str = "RdYlBu_r",
    vmin: float = 0,
    save: bool | str = False,
    show: bool = True,
) -> sns.matrix.ClusterGrid:
    out = sns.clustermap(bulk_sig.corr(), cmap=cmap, vmin=vmin)

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
    )  # , ipython_format='png2x')

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
    cluster_key: str = "leiden",
    ass_cluster_key: str = "ass_cluster",
    ass_pearson_key: str = "ass_pearson",
    color_map: str = "RdYlBu_r",
    use_raw: bool = False,
    wspace: float = 0.25,
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
    )  # , ipython_format='png2x')
    out = sc.pl.umap(
        adata,
        color=[cluster_key, ass_cluster_key, ass_pearson_key],
        color_map=color_map,
        use_raw=use_raw,
        wspace=wspace,
        return_fig=True,
        **scanpy_kwargs,
    )

    if save:
        plt.savefig(save, bbox_inches="tight")
    if show:
        plt.show()

    return out


def plot_bulk_sig_heatmap(
    bulk_sig: pd.DataFrame,
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

    if save:
        plt.savefig(save, bbox_inches="tight")
    if show:
        plt.show()

    return out


def plot_ass_pearson_violin(
    adata: AnnData,
    ass_pearson_key: str = "ass_pearson",
    cluster_key: str = "leiden",
    use_raw: bool = False,
    show: bool = True,
    save: bool | str = False,
    **scanpy_kwargs,
) -> Axes:
    out = sc.pl.violin(
        adata,
        [ass_pearson_key],
        groupby=cluster_key,
        use_raw=use_raw,
        return_fig=True,
        **scanpy_kwargs,
    )

    if save:
        plt.savefig(save, bbox_inches="tight")
    if show:
        plt.show()

    return out


def plot_canon_assigned_labels_umap(
    adata: AnnData,
    cluster_key: str = "leiden",
    canon_label_ass_key: str = "canon_label_ass",
    color_map: str = "RdYlBu_r",
    use_raw: bool = False,
    wspace: float = 0.25,
    save: bool | str = False,
    show: bool = True,
    **scanpy_kwargs,
) -> list[Axes]:
    out = sc.pl.umap(
        adata,
        color=[cluster_key, canon_label_ass_key],
        color_map=color_map,
        use_raw=use_raw,
        wspace=wspace,
        return_fig=True,
        **scanpy_kwargs,
    )  # ,save='adata_map_MSK_phenotypes.png')#,legend_loc='on data')

    if save:
        plt.savefig(save, bbox_inches="tight")
    if show:
        plt.show()

    return out


def plot_map_vs_test_pearson_violin(
    adata_test: AnnData,
    adata_map: AnnData,
    canon_label_ass_key: str = "canon_label_ass",
    ass_pearson_key: str = "ass_pearson",
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
    out = sns.violinplot(
        x=canon_label_ass_key,
        y=ass_pearson_key,
        hue=hue_key,
        data=df_vp,
        palette=palette,
        split=split,
        rotation=x_label_rotation,
    )
    plt.xticks(rotation=x_label_rotation)
    out.legend(loc=legend_loc)

    if save:
        plt.savefig(save, bbox_inches="tight")
    if show:
        plt.show()

    return out


def plot_map_vs_test_cluster_fractions(
    adata_test: AnnData,
    adata_map: AnnData,
    canon_label_ass_key: str = "canon_label_ass",
    save: bool | str = False,
    show: bool = True,
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

    freqs = Counter(adata_map.obs["canon_label_ass"])
    map_freqs = pd.DataFrame(freqs, index=[0]) / len(adata_map.obs)

    ass_clust_freqs = helpers.calc_frequencies(
        adata_test, canon_label_ass_key, return_as="df"
    )
    ass_clust_freqs.index = [1]

    df_fracs = pd.concat((map_freqs, ass_clust_freqs)).fillna(0).T.sort_values(by=0)

    x = np.linspace(0, np.max(np.max(df_fracs)), 10)
    plt.figure()
    plt.plot(x, x, "k")
    for i in range(len(df_fracs.index)):
        plt.plot(
            df_fracs[0].values[i], df_fracs[1].values[i], marker=marker
        )  # color=colordict[df_fracs.index[i]]

    ax = plt.gca()
    ax.legend(["x=y"] + list(df_fracs.index), bbox_to_anchor=(1.05, 1))
    plt.xlabel("Map Cluster Fractions")
    plt.ylabel("Test Cluster Fractions")

    if save:
        plt.savefig(save, bbox_inches="tight")

    if show:
        plt.show()

    return ax
