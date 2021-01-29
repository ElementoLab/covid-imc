#!/usr/bin/env python

"""
Investigate relationships between Samples/ROIs and channels in an unsupervised way.
"""

import sys, re, json
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import parmap
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from anndata import AnnData
import scanpy as sc
import pingouin as pg

from seaborn_extensions import swarmboxenplot, clustermap

from src.config import (
    prj,
    set_prj_clusters,
    roi_attributes,
    colors,
    metadata_dir,
    results_dir,
    figkws,
)

args: Namespace


output_dir = results_dir / "unsupervised"
output_dir.mkdir()


def main() -> int:
    global args
    args = get_cli_parser().parse_args()

    # overview()

    sample_cell_type_dimres()
    plot_sample_cell_type_dimres()
    pca_association()

    # sample_interaction_dimres()

    return 0


def get_cli_parser():
    parser = ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    return parser


def overview():
    summary_file = qc_dir / prj.name + ".channel_summary.csv"
    if not summary_file.exists() or args.overwrite:
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
    if not density_file.exists() or args.overwrite:
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

    grid = clustermap(
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
    grid = clustermap(
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


def sample_cell_type_dimres() -> None:
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
    ann.raw = ann
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


def pca_association() -> None:
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
    n = 100_000
    pcs = pd.read_csv(output_dir / "pcs.csv", index_col=0)
    pcs.columns = pcs.columns.astype(int) + 1
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
    subvars = [
        x
        for y in [variables[c] for c in subsets]
        for x in y
        if (not x.endswith("_text"))
        and (meta[x].dtype != object)
        and (x not in ["disease", "phenotypes"])
    ]
    meta = meta[subvars + ["sample_name"]]

    meta_roi = (
        meta.set_index("sample_name")  # .join(scores)
        .join(
            roi_attributes[["sample"]]
            .reset_index()
            .set_index("sample")[["roi"]]
        )
        .reset_index(drop=True)
        .set_index("roi")
    )

    # Get
    # # For continuous variables
    cont_vars = meta.columns[
        list(map(lambda x: x.name.lower() in ["float64", "int64"], meta.dtypes))
    ].tolist()
    cont_roi = meta_roi.loc[:, cont_vars]

    # # For categoricals
    cat_vars = meta.columns[
        list(
            map(
                lambda x: x.name.lower() in ["category", "bool", "boolean"],
                meta.dtypes,
            )
        )
    ].tolist()
    cat_roi = meta_roi.loc[:, cat_vars]
    cat_roi = cat_roi.loc[:, cat_roi.nunique() > 1]

    # # # convert categoricals
    cat_roi = pd.DataFrame(
        {
            x: cat_roi[x].astype(float)
            if cat_roi[x].dtype.name in ["bool", "boolean"]
            else cat_roi[x].cat.codes
            for x in cat_roi.columns
        }
    ).replace(-1, np.nan)

    #
    corrs_roi = (
        pcs.join(cont_roi)
        .join(cat_roi)
        .corr()
        .reindex(index=pcs.columns, columns=meta_roi.columns)
        .T.dropna()
    )

    # Get pvalues based on randomization
    randomized_cont_file = (
        output_dir / "pca_associations.randomized.continuous.pq"
    )
    randomized_cat_file = (
        output_dir / "pca_associations.randomized.categorical.pq"
    )
    if not randomized_cat_file.exists() or args.overwrite:
        _shuffled_cont_corrs = parmap.map(
            shuffle_count, range(n), pcs, cont_roi, pm_pbar=True
        )
        shuffled_corr_cont = pd.concat(_shuffled_cont_corrs)
        shuffled_corr_cont.columns = shuffled_corr_cont.columns.astype(str)
        shuffled_corr_cont.to_parquet(randomized_cont_file)

        _shuffled_cat_corrs = parmap.map(
            shuffle_count, range(n), pcs, cat_roi, pm_pbar=True
        )
        shuffled_corr_cat = pd.concat(_shuffled_cat_corrs)
        shuffled_corr_cat.columns = shuffled_corr_cat.columns.astype(str)
        shuffled_corr_cat.to_parquet(randomized_cat_file)

    shuffled_corr_cont = pd.read_parquet(randomized_cont_file)
    shuffled_corr_cat = pd.read_parquet(randomized_cat_file)

    shuffled_corrs = pd.concat([shuffled_corr_cont, shuffled_corr_cat])
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
        ).reshape(corrs_roi.shape[0], corrs_roi.shape[1])
    )

    pvals = pd.DataFrame(p, index=corrs_roi.index, columns=corrs_roi.columns)
    qvals = pd.DataFrame(
        pg.multicomp(pvals.values, method="fdr_bh")[1],
        index=corrs_roi.index,
        columns=corrs_roi.columns,
    )
    pvals.to_csv(output_dir / "pca_associations.pvals.csv")
    qvals.to_csv(output_dir / "pca_associations.qvals.csv")

    pvals = pd.read_csv(output_dir / "pca_associations.pvals.csv", index_col=0)
    pvals.columns = pvals.columns.astype(int)
    qvals = pd.read_csv(output_dir / "pca_associations.qvals.csv", index_col=0)
    qvals.columns = qvals.columns.astype(int)

    # # Get pvalues based on regression
    # import statsmodels.api as sm
    # import statsmodels.formula.api as smf

    # # Fix data types for statsmodels compatibility
    # X, Y = cont_roi.copy(), pcs.copy()
    # for col in X.columns[X.dtypes == "Int64"]:
    #     X[col] = X[col].astype(float)
    # Y.columns = "PC" + Y.columns.astype(str)
    # dat = X.join(Y)
    # cov_str = " + ".join(X.columns)
    # attributes = [
    #     "fvalue",
    #     "f_pvalue",
    #     "rsquared",
    #     "rsquared_adj",
    #     "aic",
    #     "bic",
    #     "llf",
    #     "mse_model",
    #     "mse_resid",
    # ]
    # _coefs = list()
    # _pvals = list()
    # for var in Y.columns:
    #     model = smf.ols(f"{var} ~ {cov_str}", data=dat.fillna(0)).fit()
    #     _coefs.append(model.params.to_frame(var))
    #     _pvals.append(model.pvalues.to_frame(var))
    # coefs = pd.concat(_coefs, axis=1).drop("Intercept")
    # pvals = pd.concat(_pvals, axis=1).drop("Intercept")

    # See associations across first 20 PCs
    for pc in range(1, 5):
        corrs_p = corrs_roi.sort_values(pc).iloc[:, range(20)]
        pvals_p = pvals.reindex(corrs_p.index).iloc[:, range(20)]
        qvals_p = qvals.reindex(corrs_p.index).iloc[:, range(20)]
        annot = qvals_p < 1e-5  # .replace({False: "", True: "*"})

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(7.5 * 4, 15),
        )
        sns.heatmap(
            corrs_p,
            ax=axes[0],
            square=True,
            annot=annot,
            cmap="RdBu_r",
            center=0,
            yticklabels=True,
            cbar_kws=dict(label="Pearson correlation"),
        )
        lpv = log_pvalues(pvals_p)
        v = (lpv).values.max()
        sns.heatmap(
            lpv,
            ax=axes[1],
            vmax=v,
            square=True,
            yticklabels=True,
            cbar_kws=dict(label="-log10(p)"),
        )
        lqv = log_pvalues(qvals_p)
        sns.heatmap(
            lqv,
            ax=axes[2],
            vmax=v,
            square=True,
            yticklabels=True,
            cbar_kws=dict(label="-log10(FDR(p))"),
        )
        sns.heatmap(
            (corrs_p > 0).astype(int).replace(0, -1) * lqv,
            ax=axes[3],
            square=True,
            annot=annot,
            cmap="RdBu_r",
            center=0,
            yticklabels=True,
            cbar_kws=dict(label="Signed(-log10(p))"),
        )
        for ax in [axes[0], axes[3]]:
            for i, c in enumerate(ax.get_children()):
                if isinstance(c, matplotlib.text.Text):
                    if c.get_text() == "0":
                        c.set_visible(False)
                        # ax.get_children().pop(i)
                    elif c.get_text() == "1":
                        c.set_text("*")
        for ax in axes:
            ax.set(xlabel="PC", ylabel="Factor")
        fig.savefig(
            output_dir / f"pca_associations.heatmap.sorted_by_PC_{pc}.svg",
            **figkws,
        )

    # See how variables relate to each other
    kws = dict(
        center=0,
        cmap="RdBu_r",
        config="abs",
        robust=False,
        xticklabels=False,
        cbar_kws=dict(label="Pearson correlation"),
        figsize=(15, 10),
    )
    grid = clustermap(corrs_roi.T.corr(), **kws)
    grid.fig.savefig(
        output_dir / f"pca_associations.correlation_of_vars_in_coefs.svg",
        **figkws,
    )

    c = (corrs_roi > 0).astype(int).replace(0, -1) * log_pvalues(pvals)
    grid = clustermap(c.T.corr(), **kws)
    grid.fig.savefig(
        output_dir
        / f"pca_associations.correlation_of_vars_in_signed_pvalues.svg",
        **figkws,
    )

    # Volcano plots and rank vs value plots
    for pc in range(1, 5):
        corrs_roi = corrs_roi.sort_values(pc)
        pvals = pvals.reindex(corrs_roi.index)
        qvals = qvals.reindex(corrs_roi.index)
        rank = corrs_roi[pc].rank(ascending=True)

        qsum = (1 - qvals).sum(1).sort_values()
        qrank = qsum.rank(ascending=False)
        v = corrs_roi[pc].abs().max()
        v += v / 10.0

        fig, axes = plt.subplots(1, 3, figsize=(3 * 4, 2))
        axes[0].scatter(
            qrank, qsum, s=5 + (7.5 ** (qsum / 10)), c=qsum, cmap="inferno"
        )
        for i in qsum.tail(20).index:
            axes[0].text(qrank.loc[i], qsum.loc[i], s=i, rotation=90, va="top")
        axes[0].set(xlabel="Rank", ylabel="Sum of associations")

        axes[1].axhline(0, linestyle="--", color="grey")
        axes[1].scatter(
            rank,
            corrs_roi[pc],
            s=5 + (50 ** (1 - pvals[pc])),
            c=corrs_roi[pc],
            vmin=-v,
            vmax=v,
            cmap="RdBu_r",
        )
        for i in corrs_roi[pc].abs().sort_values().tail(20).index:
            axes[1].text(
                rank.loc[i],
                corrs_roi.loc[i, pc],
                s=i,
                rotation=90,
                va="bottom" if corrs_roi.loc[i, pc] < 0 else "top",
            )
        axes[1].set(xlabel="Rank", ylabel="Pearson correlation")

        axes[2].axvline(0, linestyle="--", color="grey")
        lpv = log_pvalues(pvals[pc])
        axes[2].scatter(
            corrs_roi[pc],
            lpv,
            c=corrs_roi[pc],
            cmap="RdBu_r",
            vmin=-v,
            vmax=v,
        )
        for i in corrs_roi[pc].abs().sort_values().tail(20).index:
            axes[2].text(
                corrs_roi.loc[i, pc],
                lpv.loc[i],
                s=i,
                ha="left" if corrs_roi.loc[i, pc] < 0 else "right",
            )
        axes[2].set(
            xlabel="Pearson correlation", ylabel="-log10(p-value)", xlim=(-v, v)
        )
        fig.savefig(
            output_dir / f"pca_associations.PC_{pc}.scatter_volcano.svg",
            **figkws,
        )

    # pcs_sample = pcs.join(roi_attributes[["sample"]]).groupby("sample").mean()
    #

    # Project a weighted kernel density back into the PCA space
    for pc in range(1, 5):
        pcs_pheno = pcs[[pc, pc + 1]].join(roi_attributes[["phenotypes"]])
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        cs = colors.get("phenotypes")
        for i, pheno in enumerate(pcs_pheno["phenotypes"].cat.categories):
            p = pcs_pheno.query(f"phenotypes == '{pheno}'")
            ax.scatter(*p[[pc, pc + 1]].T.values, s=2, color=cs[i])
            ax.set(xlabel=f"PC {pc}", ylabel=f"PC {pc + 1}")
        fig.savefig(output_dir / f"pca.PC_{pc}.svg", **figkws)

        # xmin, ymin, xmax, ymax = pcs[[pc, pc + 1]].apply([np.min, np.max]).values.flatten()
        (xmin, xmax), (ymin, ymax) = (ax.get_xlim(), ax.get_ylim())

        # top_vars = corrs_roi.index
        top_vars = corrs_roi[pc].abs().sort_values().tail(8).index
        top_vars = corrs_roi[pc][top_vars].sort_values().index

        c = len(top_vars)
        fig, axes = plt.subplots(c, 3, figsize=(8 * 3, 4 * c))
        for i, clinvar in enumerate(top_vars):
            axes[i][0].set(ylabel=clinvar)
            m1, m2, d = (
                pcs[[pc, pc + 1]]
                .join(meta_roi[clinvar])
                .dropna()
                .astype(float)
                .T.values
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
                for i, pheno in enumerate(
                    pcs_pheno["phenotypes"].cat.categories
                ):
                    p = pcs_pheno.query(f"phenotypes == '{pheno}'")
                    ax.scatter(*p[[pc, pc + 1]].T.values, s=2, color=cs[i])
                ax.set_xlim((xmin, xmax))
                ax.set_ylim((ymin, ymax))
                fig.colorbar(r, ax=ax)
                ax.set_aspect("auto")
        fig.savefig(
            output_dir / f"pca_associations.PC_{pc}.weighted_kde.top_vars.svg",
            **figkws,
        )


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

    grid = clustermap(
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
    # grid = clustermap(
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


def cell_type_association() -> None:
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
    n = 100_000

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
    counts = counts.loc[:, ~counts.columns.str.contains("?", regex=False)]

    # aggregate just to illustrate abundance
    agg_counts = (
        counts.T.groupby(counts.columns.str.extract(r"\d+ - (.*)\(")[0].values)
        .sum()
        .T
    )
    agg_counts = (agg_counts / agg_counts.sum()) * 1e4

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
    subvars = [
        x
        for y in [variables[c] for c in subsets]
        for x in y
        if (not x.endswith("_text"))
        and (meta[x].dtype != object)
        and (x not in ["disease", "phenotypes"])
    ]
    meta = meta[subvars + ["sample_name"]]

    meta_roi = (
        meta.set_index("sample_name")  # .join(scores)
        .join(
            roi_attributes[["sample"]]
            .reset_index()
            .set_index("sample")[["roi"]]
        )
        .reset_index(drop=True)
        .set_index("roi")
    )

    # Get
    # # For continuous variables
    cont_vars = meta.columns[
        list(map(lambda x: x.name.lower() in ["float64", "int64"], meta.dtypes))
    ].tolist()
    cont_roi = meta_roi.loc[:, cont_vars]

    # # For categoricals
    cat_vars = meta.columns[
        list(
            map(
                lambda x: x.name.lower() in ["category", "bool", "boolean"],
                meta.dtypes,
            )
        )
    ].tolist()
    cat_roi = meta_roi.loc[:, cat_vars]
    cat_roi = cat_roi.loc[:, cat_roi.nunique() > 1]

    # # # convert categoricals
    cat_roi = pd.DataFrame(
        {
            x: cat_roi[x].astype(float)
            if cat_roi[x].dtype.name in ["bool", "boolean"]
            else cat_roi[x].cat.codes
            for x in cat_roi.columns
        }
    ).replace(-1, np.nan)

    #
    corrs_roi = (
        agg_counts.join(cont_roi)
        .join(cat_roi)
        .corr()
        .reindex(index=agg_counts.columns, columns=meta_roi.columns)
        .T.dropna()
    )

    # Get pvalues based on randomization
    randomized_cont_file = (
        output_dir / "cell_type_associations.randomized.continuous.pq"
    )
    randomized_cat_file = (
        output_dir / "cell_type_associations.randomized.categorical.pq"
    )
    if not randomized_cat_file.exists() or args.overwrite:
        _shuffled_cont_corrs = parmap.map(
            shuffle_count, range(n), agg_counts, cont_roi, pm_pbar=True
        )
        shuffled_corr_cont = pd.concat(_shuffled_cont_corrs)
        shuffled_corr_cont.columns = shuffled_corr_cont.columns.astype(str)
        shuffled_corr_cont.to_parquet(randomized_cont_file)

        _shuffled_cat_corrs = parmap.map(
            shuffle_count, range(n), agg_counts, cat_roi, pm_pbar=True
        )
        shuffled_corr_cat = pd.concat(_shuffled_cat_corrs)
        shuffled_corr_cat.columns = shuffled_corr_cat.columns.astype(str)
        shuffled_corr_cat.to_parquet(randomized_cat_file)

    shuffled_corr_cont = pd.read_parquet(randomized_cont_file)
    shuffled_corr_cat = pd.read_parquet(randomized_cat_file)

    shuffled_corrs = pd.concat([shuffled_corr_cont, shuffled_corr_cat])
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
        ).reshape(corrs_roi.shape[0], corrs_roi.shape[1])
    )

    pvals = pd.DataFrame(p, index=corrs_roi.index, columns=corrs_roi.columns)
    qvals = pd.DataFrame(
        pg.multicomp(pvals.values, method="fdr_bh")[1],
        index=corrs_roi.index,
        columns=corrs_roi.columns,
    )
    pvals.to_csv(output_dir / "cell_type_associations.pvals.csv")
    qvals.to_csv(output_dir / "cell_type_associations.qvals.csv")

    # # Get pvalues based on regression
    # import statsmodels.api as sm
    # import statsmodels.formula.api as smf

    # # Fix data types for statsmodels compatibility
    # X, Y = cont_roi.copy(), pcs.copy()
    # for col in X.columns[X.dtypes == "Int64"]:
    #     X[col] = X[col].astype(float)
    # Y.columns = "PC" + Y.columns.astype(str)
    # dat = X.join(Y)
    # cov_str = " + ".join(X.columns)
    # attributes = [
    #     "fvalue",
    #     "f_pvalue",
    #     "rsquared",
    #     "rsquared_adj",
    #     "aic",
    #     "bic",
    #     "llf",
    #     "mse_model",
    #     "mse_resid",
    # ]
    # _coefs = list()
    # _pvals = list()
    # for var in Y.columns:
    #     model = smf.ols(f"{var} ~ {cov_str}", data=dat.fillna(0)).fit()
    #     _coefs.append(model.params.to_frame(var))
    #     _pvals.append(model.pvalues.to_frame(var))
    # coefs = pd.concat(_coefs, axis=1).drop("Intercept")
    # pvals = pd.concat(_pvals, axis=1).drop("Intercept")

    corrs_p = corrs_roi  # .sort_values(pc).iloc[:, range(20)]
    pvals_p = pvals  # .reindex(corrs_p.index).iloc[:, range(20)]
    qvals_p = qvals  # .reindex(corrs_p.index).iloc[:, range(20)]
    annot = qvals_p < 1e-5  # .replace({False: "", True: "*"})

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.5 * 4, 15),
    )
    sns.heatmap(
        corrs_p,
        ax=axes[0],
        square=True,
        annot=annot,
        cmap="RdBu_r",
        center=0,
        yticklabels=True,
        cbar_kws=dict(label="Pearson correlation"),
    )
    lpv = log_pvalues(pvals_p)
    v = (lpv).values.max()
    sns.heatmap(
        lpv,
        ax=axes[1],
        vmax=v,
        square=True,
        yticklabels=True,
        cbar_kws=dict(label="-log10(p)"),
    )
    lqv = log_pvalues(qvals_p)
    sns.heatmap(
        lqv,
        ax=axes[2],
        vmax=v,
        square=True,
        yticklabels=True,
        cbar_kws=dict(label="-log10(FDR(p))"),
    )
    sns.heatmap(
        (corrs_p > 0).astype(int).replace(0, -1) * lqv,
        ax=axes[3],
        square=True,
        annot=annot,
        cmap="RdBu_r",
        center=0,
        yticklabels=True,
        cbar_kws=dict(label="Signed(-log10(p))"),
    )
    for ax in [axes[0], axes[3]]:
        for i, c in enumerate(ax.get_children()):
            if isinstance(c, matplotlib.text.Text):
                if c.get_text() == "0":
                    c.set_visible(False)
                    # ax.get_children().pop(i)
                elif c.get_text() == "1":
                    c.set_text("*")
    for ax in axes:
        ax.set(xlabel="Cell types", ylabel="Factor")
    fig.savefig(
        output_dir / f"cell_type_associations.heatmap.svg",
        **figkws,
    )

    #

    # Volcano plots and rank vs value plots
    for ct in corrs_roi.columns:
        corrs_roi = corrs_roi.sort_values(ct)
        pvals = pvals.reindex(corrs_roi.index)
        qvals = qvals.reindex(corrs_roi.index)
        rank = corrs_roi[ct].rank(ascending=True)

        qsum = (1 - qvals).sum(1).sort_values()
        qrank = qsum.rank(ascending=False)
        v = corrs_roi[ct].abs().max()
        v += v / 10.0

        fig, axes = plt.subplots(1, 3, figsize=(3 * 4, 2))
        axes[0].scatter(
            qrank, qsum, s=5 + (7.5 ** (qsum / 10)), c=qsum, cmap="inferno"
        )
        for i in qsum.tail(20).index:
            axes[0].text(qrank.loc[i], qsum.loc[i], s=i, rotation=90, va="top")
        axes[0].set(xlabel="Rank", ylabel="Sum of associations")

        axes[1].axhline(0, linestyle="--", color="grey")
        axes[1].scatter(
            rank,
            corrs_roi[ct],
            s=5 + (50 ** (1 - pvals[ct])),
            c=corrs_roi[ct],
            vmin=-v,
            vmax=v,
            cmap="RdBu_r",
        )
        for i in corrs_roi[ct].abs().sort_values().tail(20).index:
            axes[1].text(
                rank.loc[i],
                corrs_roi.loc[i, ct],
                s=i,
                rotation=90,
                va="bottom" if corrs_roi.loc[i, ct] < 0 else "top",
            )
        axes[1].set(xlabel="Rank", ylabel="Pearson correlation")

        axes[2].axvline(0, linestyle="--", color="grey")
        lpv = log_pvalues(pvals[ct])
        axes[2].scatter(
            corrs_roi[ct],
            lpv,
            c=corrs_roi[ct],
            cmap="RdBu_r",
            vmin=-v,
            vmax=v,
        )
        for i in corrs_roi[ct].abs().sort_values().tail(20).index:
            axes[2].text(
                corrs_roi.loc[i, ct],
                lpv.loc[i],
                s=i,
                ha="left" if corrs_roi.loc[i, ct] < 0 else "right",
            )
        axes[2].set(
            xlabel="Pearson correlation", ylabel="-log10(p-value)", xlim=(-v, v)
        )
        fig.savefig(
            output_dir / f"cell_type_associations.{ct}.scatter_volcano.svg",
            **figkws,
        )

        p = (
            agg_counts["Mast cells "]
            .to_frame()
            .join(cont_roi["Ddimer_mgperL"])
            .dropna()
        )
        p = p.groupby(list(map(lambda x: x[0], p.index.str.split("-")))).mean()
        sns.regplot(data=p, x="Mast cells ", y="Ddimer_mgperL")

        p = agg_counts["Mast cells "].to_frame().join(cont_roi["IL6"]).dropna()
        p = p.groupby(list(map(lambda x: x[0], p.index.str.split("-")))).mean()
        sns.regplot(data=p, x="Mast cells ", y="IL6")

        p = (
            agg_counts["Endothelial cells "]
            .to_frame()
            .join(cont_roi["IL6"])
            .dropna()
        )
        p = p.groupby(list(map(lambda x: x[0], p.index.str.split("-")))).mean()
        sns.regplot(data=p, x="Endothelial cells ", y="IL6")


def log_pvalues(x, f=0.1):
    """
    Calculate -log10(p-value) of array.

    Replaces infinite values with:

    .. highlight:: python
    .. code-block:: python

        max(x) + max(x) * f

    that is, fraction ``f`` more than the maximum non-infinite -log10(p-value).

    Parameters
    ----------
    x : :class:`pandas.Series`
        Series with numeric values
    f : :obj:`float`
        Fraction to augment the maximum value by if ``x`` contains infinite values.

        Defaults to 0.1.

    Returns
    -------
    :class:`pandas.Series`
        Transformed values.
    """
    if isinstance(x, (float, int)):
        r = -np.log10(x)
        if r == np.inf:
            r = 300
        return r
    ll = -np.log10(x)
    rmax = ll[ll != np.inf].max()
    return ll.replace(np.inf, rmax + rmax * f)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
