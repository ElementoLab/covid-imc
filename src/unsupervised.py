#!/usr/bin/env python

"""
Investigate relationships between Samples/ROIs and channels in an unsupervised way.
"""

import sys, re, json

import numpy as np
import pandas as pd
import parmap
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from anndata import AnnData
import scanpy as sc
import pingouin as pg

from seaborn_extensions import swarmboxenplot

from src.config import (
    prj,
    set_prj_clusters,
    roi_attributes,
    colors,
    results_dir,
    figkws,
)

output_dir = results_dir / "unsupervised"
output_dir.mkdir()


def main():
    # overview()

    sample_cell_type_dimres()
    plot_sample_cell_type_dimres()
    pca_association()

    # sample_interaction_dimres()


def overview():
    summary_file = qc_dir / prj.name + ".channel_summary.csv"
    if not summary_file.exists():
        summary, fig = prj.channel_summary(channel_exclude=channels_exclude)
        summary.to_csv(summary_file)
    summary = pd.read_csv(summary_file, index_col=0)

    fig = prj.channel_correlation(channel_exclude=channels_exclude)
    fig.savefig(
        output_dir / prj.name + ".channel_summary.per_roi.clustermap.svg",
        **figkws,
    )

    # Cell density
    density_file = results_dir / "cell_density_per_roi.csv"
    if not density_file.exists():
        cell_density = pd.Series(
            [r.cells_per_area_unit() for r in prj.rois],
            index=roi_names,
            name="cell_density",
        ).rename_axis("roi")
        cell_density.to_csv(density_file)
    cell_density = pd.read_csv(density_file, index_col=0, squeeze=True)
    for grouping in ["disease", "phenotypes"]:
        fig = swarmboxenplot(
            data=roi_attributes.join(cell_density),
            x=grouping,
            y="cell_density",
            ax=None,
        )
        fig.savefig(
            output_dir
            / f"cell_density.per_roi.by_{grouping}.swarmboxenplot.svg",
            dpi=300,
            bbox_inches="tight",
        )

    # TODO: recalculate with only (non-)immune cells

    channel_means = summary.mean(1).rename("channel_mean").rename_axis("roi")
    kwargs = dict(
        xticklabels=True,
        yticklabels=True,
        center=0,
        cmap="RdBu_r",
        z_score=0,
        cbar_kws=dict(label="Mean counts (Z-score)"),
        rasterized=True,
        metric="correlation",
        robust=True,
        dendrogram_ratio=(0.05, 0.05),
    )

    grid = sns.clustermap(
        summary,
        row_colors=channel_means,
        col_colors=roi_attributes.join(cell_density),
        **kwargs,
    )
    grid.ax_heatmap.set_xticklabels(
        grid.ax_heatmap.get_xticklabels(), fontsize=4
    )
    grid.fig.savefig(
        output_dir / "channel_summary.per_roi.clustermap.svg", **figkws
    )

    sample_summary = summary.T.groupby(
        summary.columns.str.split("-").to_series().apply(pd.Series)[0].values
    ).mean()
    grid = sns.clustermap(
        sample_summary.T,
        row_colors=channel_means,
        col_colors=sample_attributes,
        figsize=(5, 10),
        **kwargs,
    )
    # grid.ax_heatmap.set_xticklabels(grid.ax_heatmap.get_xticklabels(), fontsize=4)
    grid.fig.savefig(
        output_dir / "channel_summary.per_sample.clustermap.svg",
        **figkws,
    )


def sample_cell_type_dimres():
    """
    Unsupervised dimentionality reduction
    with cell type abundance per ROI.
    """
    plot_prefix = output_dir / f"cell_type_abundance."

    set_prj_clusters(aggregated=False)
    counts = (
        prj.clusters.to_frame()
        .assign(count=1)
        .pivot_table(
            index="roi",
            columns="cluster",
            values="count",
            aggfunc=sum,
            fill_value=0,
        )
    )

    # aggregate just to illustrate abundance
    agg_counts = (
        counts.T.groupby(counts.columns.str.extract(r"\d+ - (.*)\(")[0].values)
        .sum()
        .T
    )
    agg_counts = (agg_counts / agg_counts.sum()) * 1e4
    agg_counts = agg_counts.loc[~agg_counts.index.str.contains(r"\?")]

    ann = AnnData(counts, obs=roi_attributes.loc[counts.index].join(agg_counts))
    sc.pp.normalize_total(ann)
    sc.pp.scale(ann)
    sc.pp.pca(ann)
    sc.pp.neighbors(ann)
    for res in [0.5, 1.0]:
        sc.tl.leiden(ann, resolution=res, key_added=f"cluster_{res}")
    sc.tl.umap(ann)
    sc.tl.diffmap(ann, n_comps=3)

    # Save anndata and PCA matrices
    sc.write(output_dir / "cell_type_abundance.h5ad", ann)
    pcs = pd.DataFrame(ann.obsm["X_pca"], index=ann.obs.index)
    pcs.to_csv(output_dir / "pcs.csv")
    loadings = pd.DataFrame(ann.varm["PCs"], index=ann.var.index) * 20
    loadings.to_csv(output_dir / "loadings.csv")

    factors = ann.obs.columns[
        ann.obs.columns.str.startswith("cluster")
    ].tolist()
    factors += agg_counts.columns.tolist()
    factors += ["disease", "phenotypes", "acquisition_id"]

    fig = sc.pl.pca(ann, color=factors, return_fig=True, norm=LogNorm())
    fig.savefig(plot_prefix + "PCA.svg", **figkws)
    fig = sc.pl.umap(ann, color=factors, return_fig=True, norm=LogNorm())
    fig.savefig(plot_prefix + "UMAP.svg", **figkws)
    fig = sc.pl.diffmap(ann, color=factors, return_fig=True, norm=LogNorm())
    fig.savefig(plot_prefix + "diffmap.svg", **figkws)

    for grouping in ["disease", "phenotypes", "acquisition_id"]:
        axes = sc.pl.correlation_matrix(
            ann,
            grouping,
            show_correlation_numbers=True,
            show=False,
            cmap="RdBu_r",
        )
        axes[0].figure.savefig(
            plot_prefix + f"pairwise_correlation.by_{grouping}.svg",
            **figkws,
        )

        sc.tl.rank_genes_groups(ann, grouping)
        sc.pl.rank_genes_groups(ann, show=False)
        ax = plt.gca()
        ax.figure.savefig(
            plot_prefix + f"differential_features.by_{grouping}.list.svg",
            **figkws,
        )

        ax = sc.pl.rank_genes_groups_heatmap(
            ann,
            vmin=-1.5,
            vmax=1.5,
            show_gene_labels=True,
            show=False,
            n_genes=3,
        )["heatmap_ax"]
        ax.figure.savefig(
            plot_prefix + f"differential_features.by_{grouping}.heatmap.svg",
            **figkws,
        )
        ax = sc.pl.rank_genes_groups_heatmap(
            ann, vmin=-1.5, vmax=1.5, show_gene_labels=False, show=False
        )["heatmap_ax"]
        ax.figure.savefig(
            plot_prefix
            + f"differential_features.by_{grouping}.heatmap.nolabels.svg",
            **figkws,
        )

        ax = sc.pl.rank_genes_groups_matrixplot(
            ann, vmin=-1.5, vmax=1.5, show=False
        )["mainplot_ax"]
        ax.figure.savefig(
            plot_prefix + f"differential_features.by_{grouping}.matrixplot.svg",
            **figkws,
        )
        plt.close("all")


def plot_sample_cell_type_dimres() -> None:
    """
    Plot samples and loadings in PCA latent space.
    """
    pcs = pd.read_csv(output_dir / "pcs.csv", index_col=0)
    pcs.columns = pcs.columns.astype(int)
    loadings = pd.read_csv(output_dir / "loadings.csv", index_col=0)
    loadings.columns = loadings.columns.astype(int)

    bigpalette = (
        sns.color_palette("Set1")
        + sns.color_palette("Set2")
        + sns.color_palette("Set3")
    )

    for grouping in ["disease", "phenotypes", "acquisition_id", "sample"]:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        cats = roi_attributes[grouping].drop_duplicates().sort_values()
        # # roi positions
        for i, level in enumerate(cats):
            xy = pcs.loc[roi_attributes.query(f"{grouping} == '{level}'").index]
            c = colors.get(grouping, bigpalette)[i]
            ax.scatter(xy[0], xy[1], color=c)
            ax.scatter(
                xy[0].mean(),
                xy[1].mean(),
                marker="s",
                s=50,
                edgecolor="black",
                color=c,
                label=level,
            )
        # # plot loadings on top
        for cluster, (x, y) in loadings.iloc[:, [0, 1]].iterrows():
            if "?" in cluster:
                continue
            ax.arrow(0, 0, x, y)
            ax.text(
                x,
                y,
                s="\n(".join(cluster.split(" (")),
                ha="center",
                va="bottom" if y > 0 else "top",
            )
        ax.legend()
        fig.savefig(
            output_dir
            / f"cell_type_abundance.PCA_annotated_loadings.by_{grouping}.svg",
            **figkws,
        )

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        cats = roi_attributes[grouping].drop_duplicates().sort_values()
        # # roi positions
        for i, level in enumerate(cats):
            xy = pcs.loc[roi_attributes.query(f"{grouping} == '{level}'").index]
            c = colors.get(grouping, bigpalette)[i]
            q = ax.scatter(xy[0], xy[1], color=c)
            ax.scatter(
                xy[0].mean(),
                xy[1].mean(),
                marker="s",
                s=50,
                edgecolor="black",
                color=c,
                label=level,
            )
        # # plot loadings on top
        for cluster, (x, y) in loadings.iloc[:, [0, 1]].iterrows():
            if "?" in cluster:
                continue
            ax.arrow(0, 0, x, y)
            ax.text(
                x,
                y,
                s=cluster.split(" (")[0].split(" - ")[1],
                ha="center",
                va="bottom" if y > 0 else "top",
            )
        ax.legend()
        fig.savefig(
            output_dir
            / f"cell_type_abundance.PCA_annotated_loadings.by_{grouping}.less_labels.svg",
            **figkws,
        )

    # get most representative ROIs across the latent vectors
    # let's do 5, 2, 5, 2, clockwise starting in top
    xy = pcs[[0, 1]]
    xy - xy.mean() / xy.std()

    n = 12
    step = 1 / 10
    xw = xy[0].abs().max()
    yw = xy[1].abs().max()

    f = n // 2
    xdir = 1
    ydir = 1
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    for i in range(n):
        if i % f == 0:
            xdir *= -1
        if (i * 2) % f == 0:
            ydir *= -1
        if i == 0:
            pos = np.array([0, 1])
        else:
            pos = pos + (xdir * step * xw, ydir * step * yw)
        ax.arrow(0, 0, *pos)
    # fig.savefig(
    #     output_dir
    #     / f"cell_type_abundance.PCA.representative_images_around.svg",
    #     **figkws,
    # )


def pca_association():
    """
    Test the association of clinical factors associated with PCA latent space.
    """

    def shuffle_count(_, pcs, cont_roi):
        shuff = cont_roi.sample(frac=1)
        shuff.index = cont_roi.index
        return (
            pcs.join(shuff)
            .corr()
            .reindex(index=pcs.columns, columns=meta_roi.columns)
            .T.dropna()
        )

    n = 1_000_000  # number of randomizations
    pcs = pd.read_csv(output_dir / "pcs.csv", index_col=0)
    pcs.columns = pcs.columns.astype(int)
    meta = pd.read_parquet(metadata_dir / "clinical_annotation.pq")
    variables = json.load(
        open(metadata_dir / "variables.class_to_variable.json")
    )

    subsets = [
        "demographics",
        "clinical",
        "temporal",
        "symptoms",
        "pathology",
        "lab",
    ]
    subvars = [x for y in [variables[c] for c in subsets] for x in y]
    meta = meta[subvars + ["sample_name"]]

    meta_roi = (
        meta.set_index("sample_name")  # .join(scores)
        .join(roi_attributes.reset_index().set_index("sample")[["roi"]])
        .reset_index()
        .set_index("roi")
    )
    cont_vars = meta.columns[
        list(map(lambda x: x.name.lower() in ["float64", "int64"], meta.dtypes))
    ].tolist()
    cont_roi = meta_roi.loc[:, cont_vars]  #  + ['clinical_score']
    corrs_roi = (
        pcs.join(cont_roi)
        .corr()
        .reindex(index=pcs.columns, columns=meta_roi.columns)
        .T.dropna()
    )

    randomized_file = output_dir / "pca_associations.randomized.pq"
    if not randomized_file.exists():
        _shuffled_corrs = list()
        _shuffled_corrs = parmap.map(
            shuffle_count, range(n), pcs, cont_roi, pm_pbar=True
        )

        shuffled_corrs = pd.concat(_shuffled_corrs)
        shuffled_corrs.to_parquet(randomized_file)

    shuffled_corrs = pd.read_parquet(randomized_file)
    shuffled_corrs.columns = shuffled_corrs.columns.astype(int)
    shuffled_corrs_mean = shuffled_corrs.groupby(level=0).mean()
    shuffled_corrs_std = shuffled_corrs.groupby(level=0).std()
    p = 2 * (
        1
        - np.asarray(
            [
                scipy.stats.norm(
                    loc=shuffled_corrs_mean.loc[attr, col],
                    scale=shuffled_corrs_std.loc[attr, col],
                ).cdf(abs(corrs_roi.loc[attr, col]))
                for attr in corrs_roi.index
                for col in corrs_roi.columns
            ]
        ).reshape(corrs_roi.index.shape[0], corrs_roi.columns.shape[0])
    )

    pvals = pd.DataFrame(p, index=corrs_roi.index, columns=corrs_roi.columns)
    qvals = pd.DataFrame(
        pg.multicomp(pvals.values, method="fdr_bh")[1],
        index=corrs_roi.index,
        columns=corrs_roi.columns,
    )

    fig, axes = plt.subplots(1, 4, figsize=(6 * 4, 4))
    sns.heatmap(
        corrs_roi,
        ax=axes[0],
        square=True,
        cmap="RdBu_r",
        center=0,
        cbar_kws=dict(label="Pearson correlation"),
    )
    v = (-np.log10(pvals)).values.max()
    sns.heatmap(
        -np.log10(pvals),
        ax=axes[1],
        vmax=v,
        square=True,
        cbar_kws=dict(label="-np.log10(p)"),
    )
    sns.heatmap(
        -np.log10(qvals),
        ax=axes[2],
        vmax=v,
        square=True,
        cbar_kws=dict(label="-np.log10(FDR(p))"),
    )
    sns.heatmap(
        (corrs_roi > 0).astype(int).replace(0, -1) * -np.log10(qvals),
        ax=axes[3],
        square=True,
        cmap="RdBu_r",
        center=0,
        cbar_kws=dict(label="Signed(-np.log10(p))"),
    )
    for ax in axes:
        ax.set(xlabel="PC", ylabel="Factor")
    fig.savefig(output_dir / "pca_associations.heatmap.svg", **figkws)

    corrs_roi = corrs_roi.sort_values(0)
    pvals = pvals.reindex(corrs_roi.index)
    qvals = qvals.reindex(corrs_roi.index)
    rank = corrs_roi[0].rank(ascending=True)

    qsum = (1 - qvals).sum(1).sort_values()
    qrank = qsum.rank(ascending=False)
    v = corrs_roi[0].abs().max()
    v += v / 10.0

    fig, axes = plt.subplots(1, 3, figsize=(3 * 4, 2))
    axes[0].scatter(
        qrank, qsum, s=5 + (7.5 ** (qsum / 10)), c=qsum, cmap="inferno"
    )
    for i in qsum.index:
        axes[0].text(qrank.loc[i], qsum.loc[i], s=i, rotation=90, va="top")
    axes[0].set(xlabel="Rank", ylabel="Sum of associations")

    axes[1].axhline(0, linestyle="--", color="grey")
    axes[1].scatter(
        rank,
        corrs_roi[0],
        s=5 + (50 ** (1 - pvals[0])),
        c=corrs_roi[0],
        vmin=-v,
        vmax=v,
        cmap="RdBu_r",
    )
    for i, v_ in corrs_roi.iterrows():
        axes[1].text(
            rank.loc[i],
            corrs_roi.loc[i, 0],
            s=i,
            rotation=90,
            va="bottom" if v_[0] < 0 else "top",
        )
    axes[1].set(xlabel="Rank", ylabel="Pearson correlation")

    axes[2].axvline(0, linestyle="--", color="grey")
    axes[2].scatter(
        corrs_roi[0],
        -np.log10(pvals[0]),
        c=corrs_roi[0],
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
    )
    for i, v_ in corrs_roi.iterrows():
        axes[2].text(
            v_[0],
            -np.log10(pvals.loc[i, 0]),
            s=i,
            ha="left" if v_[0] < 0 else "right",
        )
    axes[2].set(
        xlabel="Pearson correlation", ylabel="-log10(p-value)", xlim=(-v, v)
    )
    fig.savefig(output_dir / "pca_associations.scatter_volcano.svg", **figkws)

    # pcs_sample = pcs.join(roi_attributes[["sample"]]).groupby("sample").mean()
    #

    # Project a weighted kernel density back into the PCA space
    pcs_pheno = pcs[[0, 1]].join(roi_attributes[["phenotypes"]])
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    cs = colors.get("phenotypes")
    for i, pheno in enumerate(pcs_pheno["phenotypes"].cat.categories):
        p = pcs_pheno.query(f"phenotypes == '{pheno}'")
        ax.scatter(*p[[0, 1]].T.values, s=2, c=cs[i])
    fig.savefig(output_dir / "pca.svg", **figkws)

    # xmin, ymin, xmax, ymax = pcs[[0, 1]].apply([np.min, np.max]).values.flatten()
    (xmin, xmax), (ymin, ymax) = (ax.get_xlim(), ax.get_ylim())

    c = len(cont_vars)
    fig, axes = plt.subplots(c, 3, figsize=(8 * 3, 4 * c))
    for i, clinvar in enumerate(cont_vars):
        axes[i][0].set(ylabel=clinvar)
        m1, m2, d = (
            pcs[[0, 1]].join(meta_roi[clinvar]).dropna().astype(float).T.values
        )

        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:200j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([m1, m2])
        kernel1 = scipy.stats.gaussian_kde(values, weights=d)
        kernel2 = scipy.stats.gaussian_kde(values)
        Z1 = np.reshape(kernel1(positions).T, X.shape)
        Z2 = np.reshape(kernel2(positions).T, X.shape)

        v = pd.DataFrame(Z1 - Z2).abs().stack().max()
        v += v * 0.1
        for ax, z, cmap, kw in zip(
            axes[i],
            [Z1, Z2, Z1 - Z2],
            [plt.cm.gist_stern_r, plt.cm.gist_stern_r, plt.cm.PuOr_r],
            [{}, {}, {"vmin": -v, "vmax": v}],
        ):
            r = ax.imshow(
                np.rot90(z),
                cmap=cmap,
                extent=[xmin, xmax, ymin, ymax],
                rasterized=True,
                **kw,
            )
            ax.axhline(0, linestyle="--", color="grey")
            ax.axvline(0, linestyle="--", color="grey")
            for i, pheno in enumerate(pcs_pheno["phenotypes"].cat.categories):
                p = pcs_pheno.query(f"phenotypes == '{pheno}'")
                ax.scatter(*p[[0, 1]].T.values, s=2, c=cs[i])
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))
            fig.colorbar(r, ax=ax)
            ax.set_aspect("auto")
    fig.savefig(output_dir / "pca_associations.weighted_kde.svg", **figkws)


def sample_interaction_dimres() -> None:
    """
    Unsupervised dimentionality reduction
    only with cell-type cell-type interactions
    instead of cell type abundance.
    """
    prefix = "roi_zscored.filtered."
    cluster_str = "cluster_1.0"
    new_labels = json.load(open("metadata/cluster_names.json"))[
        f"{prefix};{cluster_str}"
    ]
    # easier cluster names and no weird clusters
    new_labels = {
        k: re.findall(r"(\d+ - .*) \(", v)[0]
        for k, v in new_labels.items()
        if "?" not in v
    }
    interaction_file = (
        results_dir
        / "interaction"
        / f"pairwise_cell_type_interaction.{prefix}.{cluster_str}.per_roi.pq"
    )
    norm_freqs = pd.read_parquet(interaction_file)
    # # remove self interactions
    # norm_freqs = norm_freqs.loc[norm_freqs["A"] != norm_freqs["B"]]

    # label clusters wit proper name
    norm_freqs["A_label"] = norm_freqs["A"].astype(str).replace(new_labels)
    norm_freqs["B_label"] = norm_freqs["B"].astype(str).replace(new_labels)
    norm_freqs = norm_freqs.loc[
        (norm_freqs["A"].isin(new_labels)) & (norm_freqs["B"].isin(new_labels))
    ]

    # add label
    norm_freqs["label"] = norm_freqs["A_label"] + " -> " + norm_freqs["B_label"]

    # TODO: drop duplicates
    norm_freqs["abs_value"] = norm_freqs["value"].abs()
    norm_freqs = norm_freqs.drop_duplicates(subset=["abs_value", "roi"])

    f = norm_freqs.pivot_table(
        index="roi", columns="label", values="value", fill_value=0
    )
    f = f.T.dropna().T

    grid = sns.clustermap(
        f,
        center=0,
        cmap="RdBu_r",
        metric="correlation",
        xticklabels=False,
        yticklabels=False,
        row_colors=roi_attributes.reindex(f.index),
        z_score=0,
        rasterized=True,
        figsize=(6, 4),
        dendrogram_ratio=0.05,
        vmin=-2.5,
        vmax=2.5,  # robust=True,
        cbar_kws=dict(label="Strength of interaction"),
    )
    grid.ax_heatmap.set(xlabel="Celular interactions (pairs)", ylabel="ROIs")
    grid.fig.savefig(
        results_dir
        / "interaction"
        / "interactions.allpairwise_vs_rois.filtered.clustermap.svg",
        **figkws,
    )

    s = f.abs().sum().sort_values()
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    sns.histplot(s, bins=100, ax=ax)
    ax.set(xlabel="Sum of absolute interactions", ylabel="Frequency")
    fig.savefig(
        results_dir
        / "interaction"
        / "interactions.allpairwise_vs_rois.filtered.histplot.svg",
        **figkws,
    )

    # f = f.loc[:, s[s > 60].index]
    # grid = sns.clustermap(
    #     f,
    #     center=0,
    #     cmap="RdBu_r",
    #     metric="correlation",
    #     robust=True,
    #     xticklabels=False,
    #     yticklabels=False,
    #     row_colors=roi_attributes.reindex(f.index),
    #     z_score=0
    # )

    # f = f.loc[f.index.str.contains("COVID")]
    ann = AnnData(f, obs=roi_attributes.reindex(f.index))
    sc.pp.scale(ann)
    sc.pp.pca(ann)
    sc.pp.neighbors(ann)
    for res in [0.5, 1.0]:
        sc.tl.leiden(ann, resolution=res, key_added=f"cluster_{res}")
    sc.tl.umap(ann)
    sc.tl.diffmap(ann, n_comps=3)

    factors = ann.obs.columns[
        ann.obs.columns.str.startswith("cluster")
    ].tolist()
    factors += ["disease", "phenotypes", "acquisition_id"]

    sc.pl.pca(ann, color=factors, components=["1,2", "2,3", "3,4", "4,5"], s=50)
    sc.pl.umap(ann, color=factors)
    sc.pl.diffmap(ann, color=factors)

    pcs = pd.DataFrame(ann.obsm["X_pca"], index=ann.obs.index)
    loadings = pd.DataFrame(ann.varm["PCs"], index=ann.var.index) * 20

    l = loadings[0].sort_values()
    l.index[0].split(" -> ")
    l.index[-1].split(" -> ")

    sc.tl.rank_genes_groups(
        ann, "phenotypes", n_genes=-1, method="t-test_overestim_var"
    )
    sc.pl.rank_genes_groups(ann)

    metrics = ["scores", "pvals", "pvals_adj", "logfoldchanges"]
    _stats = list()
    for group in ann.obs["phenotypes"].unique():
        df = pd.DataFrame(
            [ann.uns["rank_genes_groups"][metric][group] for metric in metrics],
            columns=ann.uns["rank_genes_groups"]["names"][group],
            index=metrics,
        ).T.assign(group=group)
        _stats.append(df)
    stats = pd.concat(_stats)
    stats = pd.concat(
        [
            stats,
            (
                stats.index.str.split(" -> ")
                .to_series()
                .apply(pd.Series)
                .rename(columns={0: "A_label", 1: "B_label"})
                .set_index(stats.index)
            ),
        ],
        axis=1,
    )
    stats.index.name = "interaction"
    stats["self"] = stats["A_label"] == stats["B_label"]
    stats.to_csv(
        results_dir / "interaction" / "differential_testing.by_phenotype.csv"
    )

    set_prj_clusters(aggregated=False)
    set_prj_clusters(aggregated=True)

    for group in ann.obs["phenotypes"].unique():
        gs = stats.query(f"~self & (group == '{group}')")[
            "scores"
        ].sort_values()
        mini = gs.index[0]
        maxi = gs.index[-1]

        minroi = (
            roi_attributes.join(f[mini])
            .query(f"phenotypes == '{group}'")[mini]
            .sort_values()
            .idxmin()
        )
        maxroi = (
            roi_attributes.join(f[maxi])
            .query(f"phenotypes == '{group}'")[maxi]
            .sort_values()
            .idxmax()
        )

        minroi = [r for r in prj.rois if r.name == minroi][0]
        maxroi = [r for r in prj.rois if r.name == maxroi][0]

        minroi.plot_channels(["Vimentin", "CD14"])
        minroi.plot_cell_types()

        maxroi.plot_channels(["AlphaSMA", "Collagen", "CD68"])
        maxroi.plot_cell_types()

        maxroi.plot_channels(["Keratin", "SARS"])
        maxroi.plot_cell_types()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
