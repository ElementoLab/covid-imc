from typing import Sequence, Dict, List, Any
import json

import numpy as np
import pandas as pd
from anndata import AnnData  # type: ignore[import]
import scanpy as sc  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import seaborn as sns  # type: ignore[import]

from imc import Project, IMCSample
from imc.types import DataFrame, Path
from imc.graphics import rasterize_scanpy

from src.utils import z_score_by_column


# prj: Project
# colors: Dict[str, pd.Series[float]]
# figkws: Dict[str, Any]
# output_dir: Path
# prefix: Path
# roi_attributes: DataFrame
# or...
# from src.config import *
# output_dir: Path


def process_single_cell(
    quant: DataFrame,
    samples: Sequence["IMCSample"] = None,
    attributes: DataFrame = None,
    output_h5ad_file: Path = None,
    use_zscore: bool = False,
    cell_filtering_string_query: str = "total > 0",
    resolutions: Sequence[float] = (0.5, 1.0),
) -> AnnData:
    ids = [
        "roi",
        "sample",
    ]
    sample_names = [
        s.name for s in (samples if samples is not None else prj.samples)
    ]
    attributes = roi_attributes if attributes is None else attributes
    quant = quant.loc[quant["sample"].isin(sample_names)]

    total = np.log1p(quant.loc[:, quant.columns.str.contains(r"\(")].sum(1))

    # # Z-score per image
    obs = None
    if use_zscore:
        obs = quant.copy()
        quant = z_score_by_column(quant, "roi")

    quant["total"] = total.values

    quant["index"] = range(quant.shape[0])
    quant = quant.query(cell_filtering_string_query).drop("total", 1)
    if obs is not None:
        obs = obs.iloc[quant["index"], :]
    quant = quant.drop("index", axis=1)

    obsn = (
        quant[["roi", "sample"]]
        .merge(attributes, left_on="roi", right_index=True)
        .rename_axis(index="cell")
        .reset_index(drop=True)
    ).assign(obj_id=quant.index.tolist())
    obs = pd.concat([obsn, obs], axis=1) if obs is not None else obsn

    ann = AnnData(
        quant.reset_index(drop=True).drop(ids, axis=1, errors="ignore"), obs=obs
    )
    ann.raw = ann
    if not use_zscore:
        sc.pp.log1p(ann)
        sc.pp.scale(ann)
    sc.pp.pca(ann)
    # sc.pl.pca(ann, color=ann.var_names)
    sc.pp.neighbors(ann, n_neighbors=3)
    sc.tl.umap(ann)
    for res in resolutions:
        sc.tl.leiden(ann, key_added=f"cluster_{res}", resolution=res)
    for c in ann.obs.columns.to_series().filter(like="cluster"):
        ann.obs[c] = (ann.obs[c].astype(int) + 1).astype(str)
    if output_h5ad_file is not None:
        ann.write_h5ad(output_h5ad_file)
    return ann


def plot_umap(ann: AnnData) -> None:
    chs = ann.var_names.tolist()

    clust = ann.obs.columns.to_series().filter(like="cluster").tolist()
    clins = ["disease", "phenotypes"]
    cats = clins + clust
    for ch in chs:
        fig = sc.pl.umap(
            ann,
            color=ch,
            use_raw=False,
            show=False,
            return_fig=True,
            palette=colors[ch].tolist() if ch in clins else None,
        )
        rasterize_scanpy(fig)
        fig.savefig(output_dir / f"clustering.{prefix}umap.{ch}.pdf", **figkws)
        plt.close(fig)

    fig = sc.pl.umap(
        ann,
        color=chs + cats,  #  + ["cluster"],
        use_raw=False,
        show=False,
        return_fig=True,
    )
    rasterize_scanpy(fig)
    fig.savefig(output_dir / f"clustering.{prefix}umap.pdf", **figkws)


def plot_umap_with_labeled_clusters(ann):
    for cluster_str in ann.obs.columns.to_series().filter(like="cluster"):
        plot_prefix = output_dir / f"clustering.{prefix}umap.{cluster_str}"
        # With labels
        new_labels = json.load(open("metadata/cluster_names.json"))[
            f"{prefix};{cluster_str}"
        ]
        ann2 = ann[ann.obs[cluster_str].isin(new_labels.keys()), :]
        new_labels = (
            pd.Series(new_labels).str.extract(r"\d - (.*)\(")[0].to_dict()
        )
        ann2.obs[cluster_str + "_agg"] = ann2.obs[cluster_str].replace(
            new_labels
        )
        fig = sc.pl.umap(
            ann2,
            color=cluster_str,
            use_raw=True,
            palette="tab20",
            show=False,
            return_fig=True,
            s=4,
            alpha=0.1,
        )
        rasterize_scanpy(fig)
        # add centroids
        centroids = (
            pd.DataFrame(ann2.obsm["X_umap"])
            .groupby(ann2.obs[cluster_str + "_agg"].values)
            .median()
        )
        ax = fig.axes[0]
        for name, pos in centroids.iterrows():
            ax.text(pos[0], pos[1], s=name, ha="center")

        fig.savefig(
            plot_prefix + "_agg.pdf", **figkws,
        )
        plt.close(fig)

        # Aggregate by metacluster
        new_labels = json.load(open("metadata/cluster_names.json"))[
            f"{prefix};{cluster_str}"
        ]
        new_labels = {
            k: re.findall(r"\d+ - (.*)\(", v)[0] for k, v in new_labels.items()
        }
        ann2.obs[cluster_str + "_pretty"] = ann2.obs[cluster_str].replace(
            new_labels
        )
        fig = sc.pl.umap(
            ann2,
            color=cluster_str + "_pretty",
            use_raw=True,
            palette="tab20b",
            show=False,
            return_fig=True,
            s=4,
            alpha=0.1,
        )
        rasterize_scanpy(fig)
        # add centroids
        centroids = (
            pd.DataFrame(ann2.obsm["X_umap"])
            .groupby(ann2.obs[cluster_str + "_pretty"].values)
            .median()
        )
        ax = fig.axes[0]

        for name, pos in centroids.iterrows():
            ax.text(pos[0], pos[1], s=name.strip(), ha="center")
        fig.savefig(
            plot_prefix + "_agglabels.pdf", **figkws,
        )
        plt.close(fig)


def plot_cluster_heatmaps(ann: AnnData, min_cells_per_cluster: int = 50):
    for cluster_str in ann.obs.columns.to_series().filter(like="cluster"):
        plot_prefix = output_dir / f"clustering.{prefix}mean_per_{cluster_str}"
        cluster_means = quant.groupby(ann.obs[cluster_str].values).mean()
        cluster_means = ann.to_df().groupby(ann.obs[cluster_str]).mean()
        cluster_counts = (
            ann.obs[cluster_str]
            .value_counts()
            .sort_values()
            .rename("Cells per cluster")
        )
        clusters = cluster_counts[
            cluster_counts >= min_cells_per_cluster
        ].index  # .drop(["8", "24", "28"])
        cluster_means = cluster_means.loc[clusters]

        disease_counts = (
            pd.get_dummies(ann.obs["disease"])
            .groupby(ann.obs[cluster_str])
            .sum()
        )
        phenotypes_counts = (
            pd.get_dummies(ann.obs["phenotypes"])
            .groupby(ann.obs[cluster_str])
            .sum()
        )
        # phenotypes_counts.loc[:, 'total'] = phenotypes_counts.sum(1)
        row_colors = phenotypes_counts  # .join(cluster_counts)
        # row_colors = (
        #     phenotypes_counts / phenotypes_counts.sum()
        # )  # .join(cluster_counts)

        kwargs = dict(
            robust=True,
            xticklabels=True,
            yticklabels=True,
            metric="correlation",
            # standard_scale=0,
            row_colors=row_colors,
            colors_ratio=(0.01, 0.03),
            dendrogram_ratio=0.05,
            cbar_kws=dict(label="Z-score"),
        )
        grid = sns.clustermap(
            cluster_means.sort_index(),
            z_score=1,
            center=0,
            cmap="RdBu_r",
            row_cluster=False,
            **kwargs,
        )
        grid.savefig(
            plot_prefix + ".sorted.svg", **figkws,
        )
        grid = sns.clustermap(cluster_means, **kwargs)
        grid.savefig(
            output_dir / f"clustering.{prefix}mean_per_{cluster_str}.svg",
            **figkws,
        )

        grid = sns.clustermap(cluster_means, center=0, cmap="RdBu_r", **kwargs)
        grid.savefig(
            plot_prefix + ".zscore.svg", **figkws,
        )
        # make barplot of cell type abundances
        ccp = cluster_counts.loc[clusters].iloc[
            grid.dendrogram_row.reordered_ind
        ]
        ccp.index = ccp.index.astype(
            str
        )  # remove categorical for correct order
        fig, ax = plt.subplots(1, 1, figsize=(2, 10))
        sns.barplot(ccp, ccp.index, orient="horiz", color="grey", ax=ax)
        fig.savefig(
            plot_prefix + ".zscore.filtered.cell_abundance.barplot.svg",
            **figkws,
        )

        # Now separately with/without morphological markers
        ppp = cluster_means.loc[
            :, ~cluster_means.columns.str.contains("DNA|Histone")
        ]
        ppp = ppp.loc[:, ppp.columns.str.contains(r"\(")]
        kwargs.update(dict(metric="correlation"))
        grid1 = sns.clustermap(
            ppp, center=0, cmap="RdBu_r", figsize=(8, 10), **kwargs
        )
        grid1.savefig(
            plot_prefix + ".zscore.no_structural.svg", **figkws,
        )
        # make barplot of cell type abundances
        ccp = cluster_counts.loc[clusters].iloc[
            grid1.dendrogram_row.reordered_ind
        ]
        ccp.index = ccp.index.astype(
            str
        )  # remove categorical for correct order
        fig, ax = plt.subplots(1, 1, figsize=(2, 10))
        sns.barplot(ccp, ccp.index, orient="horiz", color="grey", ax=ax)
        fig.savefig(
            plot_prefix + ".zscore.no_structural.barplot.svg", **figkws,
        )

        ppp = cluster_means.loc[
            :,
            cluster_means.columns.str.contains("DNA|Histone")
            | ~cluster_means.columns.str.contains(r"\("),
        ]
        for cmap in ["PRGn_r", "PuOr_r", "PiYG_r"]:
            grid2 = sns.clustermap(
                ppp,
                center=0,
                cmap=cmap,
                figsize=(3, 10),
                row_linkage=grid1.dendrogram_row.linkage,
                **kwargs,
            )
            grid2.savefig(
                plot_prefix + f".zscore.filtered_only_structural.{cmap}.svg",
                **figkws,
            )


def plot_cluster_heatmaps_with_labeled_clusters(
    ann: AnnData, min_cells_per_cluster: int = 50
):
    for cluster_str in ann.obs.columns.to_series().filter(like="cluster"):
        plot_prefix = (
            output_dir / f"clustering.{prefix}mean_per_{cluster_str}.zscore."
        )
        cluster_means = quant.groupby(ann.obs[cluster_str].values).mean()
        cluster_means = ann.to_df().groupby(ann.obs[cluster_str]).mean()
        cluster_counts = (
            ann.obs[cluster_str]
            .value_counts()
            .sort_values()
            .rename("Cells per cluster")
        )
        clusters = cluster_counts[
            cluster_counts >= min_cells_per_cluster
        ].index  # .drop(["8", "24", "28"])

        disease_counts = (
            pd.get_dummies(ann.obs["disease"])
            .groupby(ann.obs[cluster_str])
            .sum()
        )
        phenotypes_counts = (
            pd.get_dummies(ann.obs["phenotypes"])
            .groupby(ann.obs[cluster_str])
            .sum()
        )
        # phenotypes_counts.loc[:, 'total'] = phenotypes_counts.sum(1)
        row_colors = phenotypes_counts  # .join(cluster_counts)
        kwargs = dict(
            robust=True,
            xticklabels=True,
            yticklabels=True,
            metric="correlation",
            # standard_scale=0,
            row_colors=row_colors,
            colors_ratio=(0.01, 0.03),
            dendrogram_ratio=0.05,
            cbar_kws=dict(label="Z-score"),
        )
        # With labels, no weird clusters
        new_labels = json.load(open("metadata/cluster_names.json"))[
            f"{prefix};{cluster_str}"
        ]
        p = cluster_means.loc[clusters]
        p.index = p.index.to_series().replace(new_labels)
        pp = p.loc[~p.index.str.contains(r"\?"), :]
        ppp = pp.loc[:, ~pp.columns.str.contains("DNA|Histone")]
        ppp = ppp.loc[:, ppp.columns.str.contains(r"\(")]
        kwargs.update(dict(row_colors=row_colors, metric="correlation"))
        grid1 = sns.clustermap(
            ppp, center=0, cmap="RdBu_r", figsize=(18, 10), **kwargs
        )
        grid1.savefig(
            plot_prefix + "_no_structural.svg", **figkws,
        )
        ppp = pp.loc[
            :,
            pp.columns.str.contains("DNA|Histone")
            | ~pp.columns.str.contains(r"\("),
        ]
        for cmap in ["PRGn_r", "PuOr_r", "PiYG_r"]:
            grid2 = sns.clustermap(
                ppp,
                center=0,
                cmap=cmap,
                figsize=(3, 10),
                row_linkage=grid1.dendrogram_row.linkage,
                **kwargs,
            )
            grid2.savefig(
                output_dir / plot_prefix + "_only_structural.{cmap}.svg",
                **figkws,
            )
        grid3 = sns.clustermap(
            ppp,
            center=0,
            cmap="RdBu_r",
            figsize=(18, 10),
            row_linkage=grid1.dendrogram_row.linkage,
            **kwargs,
        )
        grid3.savefig(
            plot_prefix + ".svg", **figkws,
        )
        # make barplot of cell type abundances
        cluster_counts.index = cluster_counts.index.to_series().replace(
            new_labels
        )
        ccp = cluster_counts.loc[ppp.index].iloc[
            grid3.dendrogram_row.reordered_ind
        ]
        fig, ax = plt.subplots(1, 1, figsize=(2, 10))
        sns.barplot(ccp, ccp.index, orient="horiz", color="grey", ax=ax)
        fig.savefig(
            plot_prefix + ".cell_abundance.barplot.svg", **figkws,
        )

        # Bubble plots with abundance as fraction of cluster or phenotype
        phenotypes_counts.columns = phenotypes_counts.columns.astype(str)
        phenotypes_counts.index = phenotypes_counts.index.to_series().replace(
            new_labels
        )
        # # normalize to total cells per condition
        x = phenotypes_counts.loc[ppp.index].iloc[
            grid3.dendrogram_row.reordered_ind[::-1]
        ]

        # # normalize by row
        x1 = (x / x.sum()) * 100
        x1 = x1.reset_index().melt(
            id_vars=cluster_str, value_name="percent_disease"
        )
        # # normalize by col
        x2 = (x.T / x.sum(1)).T * 100
        x2 = x2.reset_index().melt(
            id_vars=cluster_str, value_name="percent_cluster"
        )

        x = x1.join(x2[["percent_cluster"]])

        grid = sns.relplot(
            data=x,
            x="variable",
            y=cluster_str,
            size="percent_cluster",
            size_norm=(10, 50),
            hue="percent_disease",
            palette="Reds",
            aspect=0.3,
        )
        ax = grid.axes[0][0]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        grid.fig.savefig(
            plot_prefix + ".cell_abundance.bubbles.svg", **figkws,
        )

        #

        #

        # Now aggregated
        plot_prefix = (
            output_dir
            / f"clustering.{prefix}mean_per_{cluster_str}.zscore.renamed.aggregated"
        )
        pagg = p.groupby(p.index.str.extract(r"\d+ - (.*) \(")[0].values).mean()
        pagg = pagg.loc[
            :,
            pagg.columns.str.contains(r"\(")
            & ~pagg.columns.str.contains(r"DNA|Histone"),
        ]

        cluster_countsagg = cluster_counts.groupby(
            cluster_counts.index.str.extract(r"\d+ - (.*) \(")[0].values
        ).sum()

        kwargs.update(
            dict(row_colors=np.log10(cluster_countsagg), metric="correlation")
        )
        grid = sns.clustermap(
            pagg, z_score=0, center=0, cmap="RdBu_r", figsize=(8, 5), **kwargs
        )
        grid.savefig(
            plot_prefix + ".svg", **figkws,
        )
        ccp = cluster_countsagg.loc[pagg.index].iloc[
            grid.dendrogram_row.reordered_ind
        ]
        fig, ax = plt.subplots(1, 1, figsize=(2, 10))
        sns.barplot(ccp, ccp.index, orient="horiz", ax=ax)
        fig.savefig(
            plot_prefix + ".cell_abundance.barplot.svg", **figkws,
        )

        # Bubble plots with abundance as fraction of cluster or phenotype
        phenotypes_countsagg = phenotypes_counts.groupby(
            phenotypes_counts.index.str.extract(r"\d+ - (.*) \(")[0].values
        ).mean()
        # # normalize to total cells per condition
        x = phenotypes_countsagg.loc[pagg.index].iloc[
            grid.dendrogram_row.reordered_ind[::-1]
        ]
        x.index.name = cluster_str

        # # normalize by row
        x1 = (x / x.sum()) * 100
        x1 = x1.reset_index().melt(
            id_vars=cluster_str, value_name="percent_disease"
        )
        # # normalize by col
        x2 = (x.T / x.sum(1)).T * 100
        x2 = x2.reset_index().melt(
            id_vars=cluster_str, value_name="percent_cluster"
        )

        x = x1.join(x2[["percent_cluster"]])

        grid = sns.relplot(
            data=x,
            x="variable",
            y=cluster_str,
            size="percent_cluster",
            size_norm=(10, 50),
            hue="percent_disease",
            palette="Reds",
            aspect=0.35,
        )
        ax = grid.axes[0][0]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        grid.fig.savefig(
            plot_prefix + ".cell_abundance.bubbles.svg", **figkws,
        )


def plot_cluster_illustrations(ann):
    for cluster_str in ann.obs.columns.to_series().filter(like="cluster"):
        r = ann.obs[["sample", "roi", "obj_id", cluster_str]]
        r.index = r.index.astype(int)
        r = r.sort_index()
        prj._clusters = None
        prj.set_clusters(
            r.set_index(["sample", "roi", "obj_id"])[cluster_str]
            .rename("cluster")
            .astype(int),
            write_to_disk=False,
            samples=samples,
        )
        # # Heatmaps
        roi_counts = r.assign(count=1).pivot_table(
            index="roi",
            columns=cluster_str,
            values="count",
            aggfunc=sum,
            fill_value=0,
        )
        roi_areas = pd.Series({r.name: r.area for r in prj.rois}, name="area")
        sample_counts = r.assign(count=1).pivot_table(
            index="sample",
            columns=cluster_str,
            values="count",
            aggfunc=sum,
            fill_value=0,
        )
        sample_areas = pd.Series(
            {s.name: sum([r.area for r in s]) for s in prj}, name="area"
        )
        cluster_counts = (
            ann.obs[cluster_str]
            .value_counts()
            .sort_values()
            .rename("Cells per cluster")
        )
        clusters = cluster_counts[
            cluster_counts >= min_cells_per_cluster
        ].index  # .drop(["8", "24", "28"])

        # Illustrate clusters
        freqs = roi_counts.T / roi_counts.sum(1)
        freqs.index = freqs.index.astype(int)
        for cluster in freqs.index:
            cluster_s = (
                cluster.replace("/", "--")
                if isinstance(cluster, str)
                else cluster
            )
            fig_file = (
                output_dir
                / f"roi_with_highest_cluster.{prefix}{cluster_str}.{cluster_s}.svg"
            )
            # # get roi with highest frequency
            rois = [
                r
                for r in prj.rois
                if r.name
                in freqs.loc[cluster][freqs.loc[cluster] > 0]
                .sort_values()
                .tail(3)
                .index
            ]

            if isinstance(cluster, str):
                markers = cluster.split("(")[-1].split(")")[0].split(",")
                markers = [
                    x.replace("+", "")
                    .replace("-", "")
                    .replace("dim", "")
                    .strip()
                    for x in markers
                ]
                markers = [
                    x + "("
                    for x in markers
                    if x != "" and x in rois[0].channel_names.values
                ]
            else:
                markers = []

            n = len(markers) + 2
            fig, axes = plt.subplots(3, n, figsize=(4 * n, 4 * 3))
            for axs, roi in zip(axes, rois):
                roi.plot_cell_types(ax=axs[0])
                # plot_cell_types(roi, ax=axs[0])
                roi.plot_cell_type(cluster, ax=axs[1])
                if markers:
                    roi.plot_channels(markers, axes=axs[2:])
            fig.savefig(fig_file, **figkws)
            plt.close(fig)
