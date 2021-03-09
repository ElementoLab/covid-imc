#! /usr/bin/env python

"""
Analysis of revision data with more depth in the immune compartment.
"""

import sys
import datetime
from argparse import ArgumentParser, Namespace
import json
from dataclasses import dataclass

from tqdm import tqdm
from joblib import parallel_backend  # type: ignore[import]
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import scanpy as sc

from imc import Project
from imc.types import Path, DataFrame, Array
from imc.graphics import close_plots, rasterize_scanpy, add_centroids
from imc.utils import z_score

from seaborn_extensions import clustermap, swarmboxenplot


args: Namespace


def main() -> int:
    global args

    # Parse arguments
    args = get_parser().parse_args()
    args.resolutions = [float(x) for x in args.resolutions]

    prj = get_project()

    illustrations(prj)

    phenotyping(prj)

    replot_with_classic_cell_types(prj)

    metacluster_expression(prj)

    intra_metacluster(prj)

    example_visualizations(prj)

    return 0


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resolutions", default=[0.5, 1.0, 1.5, 2.0, 3.0], nargs="+"
    )
    parser.add_argument("--algos", default=["umap"], nargs="+")
    return parser


def get_project() -> Project:
    prj = Project(name="imc_revision")
    prj.samples = [s for s in prj if "2021" in s.name]
    for r in prj.rois:
        r.set_channel_exclude(consts.exclude_channels)
    return prj
    # for s in prj:
    #     s.rois = [r for r in s if r._get_input_filename("cell_mask").exists()]


@close_plots
def illustrations(prj: Project) -> None:
    from csbdeep.utils import normalize  # type: ignore[import]

    (consts.output_dir / "full_stacks").mkdir()
    (consts.output_dir / "illustration").mkdir()

    for r in tqdm(prj.rois):
        output_f = consts.output_dir / "full_stacks" / r.name + ".pdf"
        if output_f.exists():
            continue
        fig = r.plot_channels()
        fig.savefig(output_f, **consts.figkws)

    for r in tqdm(prj.rois):
        output_f = consts.output_dir / "illustration" / r.name + ".svg"
        if output_f.exists():
            continue
        fig = r.plot_probabilities_and_segmentation()
        fig.savefig(output_f, **consts.figkws)

    # Specific example
    roi_name = "A20_58_20210122_ActivationPanel-01"
    x, y = (600, 200), (950, 450)
    r = prj.get_rois(roi_name)
    q = r._get_channels(
        ["CD31", "CD39", "DNA1"], minmax=True, log=True, smooth=1
    )[1]
    q2 = np.moveaxis(q, 0, -1)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(normalize(q2))
    ax.set(title=roi_name, xlim=x, ylim=y)
    fig.savefig(
        "Red:CD31-Green:CD39-Blue:DNA.A20_58_example.png", **consts.figkws
    )


def qc(prj: Project) -> None:
    (consts.output_dir / "qc").mkdir()
    output_prefix = consts.output_dir / "qc" / "channel_summary."

    c = prj.rois[0].channel_labels
    exc = [x for x in c if x in consts.exclude_channels]

    prj.channel_summary(output_prefix=output_prefix, channel_exclude=exc)


@close_plots
def phenotyping(prj: Project) -> None:
    (consts.output_dir / "phenotyping").mkdir()
    # output_prefix = consts.output_dir / "phenotyping" / prj.name + f".{cur_date}."
    output_prefix = consts.output_dir / "phenotyping" / prj.name + "."

    quant_f = output_prefix + "quantification.pq"
    if not quant_f.exists() or args.overwrite:
        prj.quantify_cells()
        quant = prj.quantification
        prj.quantification.to_parquet(quant_f)
    quant = pd.read_parquet(quant_f)

    quant_mean = quant.groupby("sample").mean()
    quant = quant.drop(consts.exclude_channels, axis=1)

    # Collapse redudant channels
    quant["DNA"] = quant.loc[:, quant.columns.str.contains("DNA")].mean(1)
    quant["Ki67"] = quant["Ki67(Pt196)"]
    quant = quant.drop(
        quant.columns[
            quant.columns.str.contains(r"DNA\d\(")
            | quant.columns.str.contains(r"Ki67\(")
        ],
        axis=1,
    )

    # filter out cells
    quant_ff = quant_f.replace_(".pq", ".filtered.pq")
    if not quant_ff.exists() or args.overwrite:
        exclude = filter_out_cells(
            quant, plot=True, output_prefix=output_prefix
        )
        tqdm.write(
            f"Filtering out {exclude.sum()} cells ({(exclude.sum() / exclude.shape[0]) * 100:.2f} %)"
        )

        quant = quant.loc[~exclude, :]
        quant.to_parquet(quant_ff)
    quant = pd.read_parquet(quant_ff)

    # Process, project, cluster
    h5ad_f = output_prefix + "sample_zscore.h5ad"
    if not h5ad_f.exists() or args.overwrite:
        # Drop unwanted channels and redundant morphological features
        q = quant.drop(["perimeter", "major_axis_length"], axis=1).reset_index()
        id_cols = ["sample", "roi", "obj_id"]

        # get measure of overal intensity
        tech = q.merge(
            quant_mean[consts.exclude_channels[:-2]]
            .apply(z_score)
            .mean(1)
            .rename("tech"),
            left_on="sample",
            right_index=True,
        )["tech"]

        # # Z-score by sample:
        from src.utils import z_score_by_column

        zquant = z_score_by_column(
            quant, "sample", clip=(-2.5, 10)
        ).reset_index()
        a = AnnData(
            zquant.drop(id_cols + ["area"], axis=1),
            obs=zquant[id_cols + ["area"]].join(tech),
        )

        # # keep track of raw untransformed values
        q = quant[a.var.index].reset_index()
        q = q.reindex(a.obs.index.astype(int)).set_index("obj_id")
        r = AnnData(q.reset_index(drop=True))
        a.raw = r
        sc.pp.scale(a)

        # Proceed with dimres + clustering
        sc.pp.pca(a)
        with parallel_backend("threading", n_jobs=12):
            sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca")
        with parallel_backend("threading", n_jobs=12):
            sc.tl.umap(a, gamma=25)

        for res in tqdm(args.resolutions, desc="resolution"):
            sc.tl.leiden(a, resolution=res, key_added=f"cluster_{res}")
            a.obs[f"cluster_{res}"] = pd.Categorical(
                a.obs[f"cluster_{res}"].astype(int) + 1
            )
        sc.write(h5ad_f, a)

    a = sc.read(h5ad_f)
    a = a[a.obs.sample(frac=1).index, :]

    # output_prefix = output_prefix.replace_(cur_date, f"{cur_date}.raw.")
    output_prefix += "sample_zscore."

    # Plot cluster phenotypes
    for res in args.resolutions:
        # # get mean per cluster
        m = a.to_df().groupby(a.obs[f"cluster_{res}"]).mean()
        umap_pos = (
            pd.DataFrame(
                a.obsm["X_umap"], index=a.obs.index, columns=["UMAP1", "UMAP2"]
            )
            .groupby(a.obs[f"cluster_{res}"])
            .mean()
        )
        m = m.join(umap_pos)
        # mr = (
        #     AnnData(a.raw.X, var=a.var, obs=a.obs)
        #     .to_df()
        #     .groupby(a.obs[f"cluster_{res}"])
        #     .mean()
        # )
        # mr = mr.join(umap_pos)

        # # get normalized proportions per disease group
        annot = a.obs.merge(consts.phenotypes.to_frame().reset_index())
        ct = annot["phenotypes"].value_counts()
        ct /= ct.sum()
        c = annot.groupby([f"cluster_{res}", "phenotypes"]).size()
        p = c.groupby(level=0).apply(lambda x: x / x.sum())
        p = p.to_frame().pivot_table(
            index=f"cluster_{res}", columns="phenotypes", values=0
        )
        p = np.log2(p / ct)

        for conf in ["abs", "z"]:
            # grid = clustermap(
            #     mr, row_colors=p, config=conf, figsize=(8, 4 * max(1, res))
            # )
            # grid.fig.savefig(
            #     output_prefix
            #     + f"phenotypes.cluster_{res}.clustermap.raw.{conf}.svg",
            #     **consts.figkws,
            # )

            grid = clustermap(
                m, row_colors=p, config=conf, figsize=(8, 4 * max(1, res))
            )
            grid.fig.savefig(
                output_prefix
                + f"phenotypes.cluster_{res}.clustermap.norm.{conf}.svg",
                **consts.figkws,
            )

    # Plot projections
    # vmin = [min(x, 0.0) for x in np.percentile(a.raw.X, 1, axis=0).tolist()]
    # vmax = [max(x, 1.0) for x in np.percentile(a.raw.X, 99, axis=0).tolist()]
    # vmin = np.percentile(a.raw.X, 1, axis=0).tolist()
    vmin = None
    vmax = np.percentile(a.raw.X, 99, axis=0).tolist()
    # notes:
    ## if scaling values clip both ends to percentiles
    ## if using log or raw original values clip top to percentiles

    color = (
        a.var.index.tolist()
        + ["area", "sample"]
        + [f"cluster_{res}" for res in args.resolutions]
    )
    for algo in args.algos:
        # norm values
        f = output_prefix + f"{algo}.z.svgz"
        projf = getattr(sc.pl, algo)
        axes = projf(
            a,
            color=color,
            show=False,
            use_raw=False,
        )
        fig = axes[0].figure
        for ax, res in zip(axes[-len(args.resolutions) :], args.resolutions):
            add_centroids(a, res=res, ax=ax)
        rasterize_scanpy(fig)
        fig.savefig(f, **consts.figkws)

        # original values
        f = output_prefix + f"{algo}.raw.svgz"
        projf = getattr(sc.pl, algo)
        axes = projf(
            a,
            color=color,
            show=False,
            vmin=vmin,
            vmax=vmax
            + [np.percentile(a.obs["area"], 99)]
            + [None]
            + [None] * (len(args.resolutions)),
            use_raw=True,
        )
        fig = axes[0].figure
        for ax, res in zip(axes[-len(args.resolutions) :], args.resolutions):
            add_centroids(a, res=res, ax=ax)
        rasterize_scanpy(fig)
        fig.savefig(f, **consts.figkws)


def replot_with_classic_cell_types(prj) -> None:
    (consts.output_dir / "phenotyping").mkdir()
    output_prefix = consts.output_dir / "phenotyping" / prj.name + "."

    res = 2.0
    h5ad_f = output_prefix + "sample_zscore.labeled.h5ad"
    a = sc.read(h5ad_f)
    a = a[a.obs.sample(frac=1).index, :]

    # Redo UMAP with only filtered cell types
    a2 = a[~a.obs[f"metacluster_labels_{res}"].str.startswith("?"), :]
    sc.pp.scale(a2)
    sc.pp.pca(a2)
    with parallel_backend("threading", n_jobs=12):
        sc.pp.neighbors(a2, n_neighbors=15, use_rep="X_pca")
    with parallel_backend("threading", n_jobs=12):
        sc.tl.umap(a2, gamma=25)

    # Replot projections
    vmin = None
    vmax = np.percentile(a2.raw.X, 99, axis=0).tolist()
    color = (
        a2.var.index.tolist()
        + ["area", "sample", "disease", "phenotypes"]
        + [f"metacluster_labels_{res}"]
    )
    for algo in args.algos:
        # norm values
        f = output_prefix + f"{algo}.filtered.z.svgz"
        projf = getattr(sc.pl, algo)
        axes = projf(
            a2,
            color=color,
            show=False,
            use_raw=False,
        )
        fig = axes[0].figure
        for ax, res in zip(axes[-len(args.resolutions) :], args.resolutions):
            add_centroids(a2, res=res, ax=ax)
        rasterize_scanpy(fig)
        fig.savefig(f, **consts.figkws)

        # original values
        f = output_prefix + f"{algo}.filtered.raw.svgz"
        projf = getattr(sc.pl, algo)
        axes = projf(
            a2,
            color=color,
            show=False,
            vmin=vmin,
            vmax=vmax
            + [np.percentile(a2.obs["area"], 99)]
            + [None, None, None, None],
            use_raw=True,
        )
        fig = axes[0].figure
        for ax, res in zip(axes[-len(args.resolutions) :], args.resolutions):
            add_centroids(a2, res=res, ax=ax)
        rasterize_scanpy(fig)
        fig.savefig(f, **consts.figkws)

    # output_prefix = output_prefix.replace_(cur_date, f"{cur_date}.raw.")
    output_prefix += "sample_zscore."

    # Plot cluster phenotypes
    res = 2.0
    # # get mean per cluster
    m = a2.to_df().groupby(a2.obs[f"metacluster_labels_{res}"]).mean()
    # # get normalized proportions per disease group
    annot = a2.obs.merge(consts.phenotypes.to_frame().reset_index())
    ct = annot["phenotypes"].value_counts()
    ct /= ct.sum()
    c = annot.groupby([f"metacluster_labels_{res}", "phenotypes"]).size()
    p = c.groupby(level=0).apply(lambda x: x / x.sum())
    p = p.to_frame().pivot_table(
        index=f"metacluster_labels_{res}", columns="phenotypes", values=0
    )
    p = np.log2(p / ct)

    conf = "z"
    grid = clustermap(m, row_colors=p, config=conf, figsize=(8, 3.4))
    grid.fig.savefig(
        output_prefix
        + f"phenotypes.filtered.metacluster_labels_{res}.clustermap.norm.{conf}.svg",
        **consts.figkws,
    )


def metacluster_expression(prj: Project) -> None:
    (consts.output_dir / "phenotyping").mkdir()
    output_prefix = consts.output_dir / "phenotyping" / prj.name + "."

    quant_ff = output_prefix + "quantification.filtered.pq"
    quant = pd.read_parquet(quant_ff)

    # Drop unwanted channels and redundant morphological features
    h5ad_f = output_prefix + "sample_zscore.h5ad"
    a = sc.read(h5ad_f)
    a.obs = a.obs.merge(consts.phenotypes.reset_index())
    a.obs["disease"] = pd.Categorical(
        a.obs["phenotypes"].str.split("_").apply(lambda x: x[0]).values,
        categories=["Healthy", "COVID19"],
        ordered=True,
    )

    res = 2.0
    a.obs[f"cluster_labels_{res}"] = a.obs[f"cluster_{res}"].replace(
        consts.cluster_idents[res]
    )
    a.obs[f"metacluster_labels_{res}"] = (
        a.obs[f"cluster_labels_{res}"].str.extract(r"(.*) \(")[0].values
    )
    a.obs[f"cluster_labels_{res}"].value_counts().filter(regex=r"^[^\?]")
    a.obs[f"metacluster_labels_{res}"].value_counts().filter(regex=r"^[^\?]")

    h5ad_f = output_prefix + "sample_zscore.labeled.h5ad"
    sc.write(h5ad_f, a)

    # Cell type abundances
    count = a.obs.groupby(["roi", f"cluster_labels_{res}"]).size()
    total = a.obs.groupby(["roi"]).size()
    area = pd.Series(
        {roi.name: roi.area for roi in prj.rois}, name="area"
    ).rename_axis(index="roi")

    # normalize by total or area
    perc = ((count / total) * 100).rename("percentage")
    exte = ((count / area) * 1e6).rename("absolute")

    count_red = a.obs.groupby(["roi", f"metacluster_labels_{res}"]).size()
    perc_red = ((count_red / total) * 100).rename("percentage")
    exte_red = ((count_red / area) * 1e6).rename("absolute")

    # get roi_attributes
    roi_attributes = (
        a.obs[["roi", "sample", "phenotypes", "disease"]]
        .drop_duplicates()
        .set_index("roi")
    )

    # Plot cluster abundance per disease group
    _stats = list()
    for group, ext in [("cluster", ""), ("metacluster", "_red")]:
        for factor in ["disease", "phenotypes"]:
            for name, dt in [("percentage", "perc"), ("absolute", "exte")]:
                df = locals()[dt + ext]
                p = df.to_frame().pivot_table(
                    index="roi", columns=f"{group}_labels_{res}", values=name
                )
                kws = dict(
                    data=p.join(roi_attributes),
                    x=factor,
                    y=p.columns,
                    plot_kws=dict(palette=consts.colors[factor]),
                )
                fig, stats = swarmboxenplot(**kws)
                _stats.append(
                    stats.assign(group=group, factor=factor, name=name)
                )
                fig.savefig(
                    output_prefix
                    + f"phenotypes.{group}s.abundance.{name}.by_{factor}.swarmboxenplot.svg",
                    **consts.figkws,
                )

    stats = pd.concat(_stats)
    stats.to_csv(
        output_prefix + "abundance.differential_testing.csv", index=False
    )

    # Some single-cell heatmaps
    a.obs.index = a.obs.index.astype(str)
    a2 = a[~a.obs[f"metacluster_labels_{res}"].str.startswith("?").values, :]
    a2.X = z_score(a2.X.T).T
    a2 = a2[:, ~a2.var.index.isin(consts.tech_channels)]

    # Get clustered order for markers
    marker_order = clustermap(
        a2.to_df()
        .groupby(a2.obs[f"metacluster_labels_{res}"].values)
        .mean()
        .corr()
    ).dendrogram_row.reordered_ind

    fig = sc.pl.heatmap(
        a2,
        a2.var.index[marker_order],
        groupby=f"metacluster_labels_{res}",
        use_raw=False,
        cmap="RdBu_r",
        vmin=-6,
        vmax=6,
        show=False,
    )["heatmap_ax"].figure
    fig.savefig(
        output_prefix + "phenotypes.metaclusters.expression.heatmap.svg",
        **consts.figkws,
    )

    # ## same but by metacluster
    # metaclusters = a2.obs[f"metacluster_labels_{res}"].unique()
    # for metacluster in metaclusters:
    #     a3 = a2[a2.obs[f'metacluster_labels_{res}'] == metacluster, :]
    #     fig = sc.pl.heatmap(
    #         a3,
    #         functional_markers,
    #         groupby="phenotypes",
    #         use_raw=False,
    #         cmap="RdBu_r",
    #         vmin=-6,
    #         vmax=6,
    #         show=False,
    #     )["heatmap_ax"].figure
    #     fig.savefig(
    #         output_prefix + "phenotypes.metaclusters.expression.heatmap.svg",
    #         **consts.figkws,
    #     )

    # Now aggregated by cluster
    for factor in ["", "disease", "phenotypes"]:
        groups = [f"metacluster_labels_{res}"] + (
            [factor] if factor != "" else []
        )
        kws = dict(
            adata=a[~a.obs[f"metacluster_labels_{res}"].str.startswith("?"), :],
            var_names=a.var.index[marker_order],
            groupby=groups,
            use_raw=False,
            cmap="RdBu_r",
            show=False,
            vmin=-6,
            vmax=6,
        )
        fig = sc.pl.matrixplot(
            **kws,
        )["mainplot_ax"].figure
        fig.savefig(
            output_prefix
            + f"phenotypes.metaclusters.expression.by_{factor}.heatmap.svg",
            **consts.figkws,
        )
        fig = sc.pl.dotplot(**kws)["mainplot_ax"].figure
        fig.savefig(
            output_prefix
            + f"phenotypes.metaclusters.expression.by_{factor}.dotplot.svg",
            **consts.figkws,
        )

    for factor in ["", "disease", "phenotypes"]:
        groups = [f"metacluster_labels_{res}"] + (
            [factor] if factor != "" else []
        )
        fig = sc.pl.stacked_violin(
            a[~a.obs[f"metacluster_labels_{res}"].str.startswith("?"), :],
            a.var.index[marker_order],
            groupby=groups,
            use_raw=False,
            show=False,
        )["mainplot_ax"].figure
        fig.savefig(
            output_prefix
            + f"phenotypes.metaclusters.expression.by_{factor}.stacked_violinplot.svg",
            **consts.figkws,
        )

    # Test for differential expression within each metacluster between disease groups
    metaclusters = a2.obs[f"metacluster_labels_{res}"].unique()
    a.obs["disease"] = a.obs["phenotypes"].str.split("_").apply(lambda x: x[0])
    _diff_res = list()
    for factor in ["phenotypes", "disease"]:
        for metacluster in metaclusters:
            a3 = a[a.obs[f"metacluster_labels_{res}"] == metacluster, :]
            a3.X += abs(a3.X.min())
            groups = a3.obs[factor].unique()[1:]
            # sc.tl.rank_genes_groups(
            #     a3,
            #     factor,
            #     use_raw=False,
            #     reference="Healthy",
            #     # method="t-test_overestim_var",
            #     method="wilcoxon",
            # )
            sc.tl.rank_genes_groups(
                a3,
                factor,
                use_raw=True,
                reference="Healthy",
                method="t-test_overestim_var",
                # method="wilcoxon",
            )
            _diff_res.append(
                pd.concat(
                    [
                        pd.DataFrame(
                            {
                                "marker": a3.uns["rank_genes_groups"]["names"][
                                    group
                                ],
                                "logfoldchanges": a3.uns["rank_genes_groups"][
                                    "logfoldchanges"
                                ][group],
                                "pvals": a3.uns["rank_genes_groups"]["pvals"][
                                    group
                                ],
                                "pvals_adj": a3.uns["rank_genes_groups"][
                                    "pvals_adj"
                                ][group],
                            }
                        ).assign(
                            metacluster=metacluster, group=group, factor=factor
                        )
                        for group in groups
                    ]
                )
            )
    diff_res = pd.concat(_diff_res)
    diff_res.to_csv(
        output_prefix
        + "phenotypes.metaclusters.expression.differential_testing.csv"
    )
    # diff_res = pd.read_csv(
    #     output_prefix
    #     + "phenotypes.metaclusters.expression.differential_testing.csv",
    #     index_col=0
    # )

    # # Test for differential expression within each metacluster between disease groups
    # n_random = 25  # TODO: run with higher N

    # a.obs["disease"] = a.obs["phenotypes"].str.split("_").apply(lambda x: x[0])
    # metaclusters = a2.obs[f"metacluster_labels_{res}"].unique()
    # _diff_res = list()
    # for factor in ["phenotypes", "disease"]:
    #     for metacluster in metaclusters:
    #         a3 = a[a.obs[f"metacluster_labels_{res}"] == metacluster, :]
    #         a3.X += abs(a3.X.min())
    #         groups = a3.obs[factor].unique()[1:]
    #         n = a3.obs["sample"].value_counts().min()

    #         for i in range(n_random):
    #             cells = list()
    #             for sample in a3.obs["sample"].unique():
    #                 cells += (
    #                     a3.obs.query(f"sample == '{sample}'")
    #                     .sample(n=n)
    #                     .index.tolist()
    #                 )
    #             a4 = a3[cells, :]

    #             sc.tl.rank_genes_groups(
    #                 a4,
    #                 factor,
    #                 use_raw=False,
    #                 reference="Healthy",
    #                 method="t-test_overestim_var",
    #                 # method="wilcoxon",
    #             )
    #             _diff_res.append(
    #                 pd.concat(
    #                     [
    #                         pd.DataFrame(
    #                             {
    #                                 "marker": a4.uns["rank_genes_groups"][
    #                                     "names"
    #                                 ][group],
    #                                 "logfoldchanges": a4.uns[
    #                                     "rank_genes_groups"
    #                                 ]["logfoldchanges"][group],
    #                                 "pvals": a4.uns["rank_genes_groups"][
    #                                     "pvals"
    #                                 ][group],
    #                                 "pvals_adj": a4.uns["rank_genes_groups"][
    #                                     "pvals_adj"
    #                                 ][group],
    #                             }
    #                         ).assign(
    #                             metacluster=metacluster,
    #                             group=group,
    #                             factor=factor,
    #                             iter=i,
    #                         )
    #                         for group in groups
    #                     ]
    #                 )
    #             )
    # diff_res = pd.concat(_diff_res)

    # diff_res.to_csv(
    #     output_prefix
    #     + "phenotypes.metaclusters.expression.differential_testing.csv"
    # )

    # import scipy
    # diff_res = (
    #     diff_res.groupby(["marker", "group", "factor", "metacluster"]).agg(
    #         {
    #             "logfoldchanges": np.mean,
    #             "pvals": lambda x: scipy.stats.combine_pvalues(x)[1],
    #             "pvals_adj": lambda x: scipy.stats.combine_pvalues(x)[1],
    #         }
    #     )
    #     # .drop("iter", 1)
    #     .reset_index()
    # )

    # # Simple mann-whitney
    # metaclusters = a2.obs[f"metacluster_labels_{res}"].unique()
    # a.obs["disease"] = a.obs["phenotypes"].str.split("_").apply(lambda x: x[0])
    # _diff_res = list()
    # for factor in ["phenotypes", "disease"]:
    #     for metacluster in metaclusters:
    #         a3 = a[a.obs[f"metacluster_labels_{res}"] == metacluster, :]
    #         # x = a3.to_df().join(a3.obs[['roi']]).groupby(['roi']).mean()[consts.functional_markers]
    #         x = a3.raw.to_adata().to_df().join(a3.obs[['roi']]).groupby(['roi']).mean()[consts.functional_markers]
    #         fig, stats = swarmboxenplot(data=x.join(roi_attributes), x=factor, y=x.columns)
    #         _diff_res.append(stats.assign(metacluster=metacluster, factor=factor))
    #         plt.close('all')
    # diff_res = pd.concat(_diff_res)
    # diff_res.to_csv(
    #     output_prefix
    #     + "phenotypes.metaclusters.expression.differential_testing.raw.mannwhitney.csv"
    # )
    # diff_res.to_csv(
    #     output_prefix
    #     + "phenotypes.metaclusters.expression.differential_testing.zscore.mannwhitney.csv"
    # )
    # # adapt to fit scanpy diff results
    # diff_res = diff_res.rename(columns={'p-unc': 'pvals', 'p-cor': 'pvals_adj', 'hedges': 'logfoldchanges', "Variable": 'marker', "factor": "group"})
    # diff_res['logfoldchanges'] *= -1
    # diff_res = diff_res.loc[diff_res['A'] != 'Healthy']
    # diff_res.loc[diff_res['group'] == 'disease', 'group'] = "COVID19"
    # diff_res.loc[diff_res['group'] == 'phenotypes', 'group'] = diff_res.loc[diff_res['group'] == 'phenotypes', 'B']

    diff_res = diff_res.dropna()
    diff_res["-logp"] = -np.log10(diff_res["pvals"])
    v = diff_res["-logp"].replace(np.inf, np.nan).dropna().max()
    diff_res["-logp"] = diff_res["-logp"].replace(np.inf, v)

    diff_res["marker_name"] = (
        diff_res["marker"].str.split("(").apply(lambda x: x[0])
    )

    diff_res = diff_res.loc[diff_res["marker"].isin(consts.functional_markers)]

    # Heatmap of log fold changes + pvalues
    lfc = diff_res.pivot_table(
        index=["metacluster", "group"],
        columns="marker",
        values="logfoldchanges",
    ).loc[:, consts.functional_markers]
    p = diff_res.pivot_table(
        index=["metacluster", "group"],
        columns="marker",
        values="pvals_adj",
    ).loc[:, consts.functional_markers]
    sigs = p < 1e-15
    sigs = p < 0.05

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        lfc,
        annot=sigs,
        center=0,
        cmap="RdBu_r",
        # vmin=-10,
        # vmax=10,
        xticklabels=True,
        yticklabels=True,
        cbar_kws=dict(label="log fold change\n(over healthy)"),
        ax=ax,
    )
    for i, c in enumerate(ax.get_children()):
        if isinstance(c, matplotlib.text.Text):
            if c.get_text() == "0":
                c.set_visible(False)
                # ax.get_children().pop(i)
            elif c.get_text() == "1":
                c.set_text("*")

    fig.savefig(
        output_prefix
        + "phenotypes.metaclusters.expression.differential_testing.joint_stats.heatmap.svg",
        **consts.figkws,
    )

    # Volcano plots
    diff_res["-logp"] = (-np.log10(diff_res["pval"])).replace(np.inf, 16)
    diff_res = diff_res.rename(
        columns={
            "contrast": "group",
            "cell_type": "metacluster",
            "gene": "marker_name",
            "log2fc": "logfoldchanges",
        }
    )
    diff_res["group"] = (
        diff_res["group"]
        .replace("COVID19_all", "COVID19")
        .replace("all", "COVID19")
    )
    diff_res["factor"] = "disease"
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(4 * 2.7, 4 * 2.7),
        sharex=False,
        sharey=False,
        gridspec_kw=dict(hspace=0.5, wspace=0.5),
    )
    for ax in axes.flat:
        ax.axvline(0, linestyle="--", color="grey")
        ax.axhline(5, linestyle="--", color="grey", linewidth=0.3)
    for metacluster, ax in zip(metaclusters, axes.flat):
        p = diff_res.query(
            f"metacluster == '{metacluster}' & group == 'COVID19' & factor == 'disease'"
        )
        v = p["logfoldchanges"].abs().max()
        v += v * 0.1
        ax.scatter(
            p["logfoldchanges"],
            p["-logp"],
            s=10,
            alpha=0.5,
            c=p["logfoldchanges"],
            cmap="coolwarm",
            vmin=-v,
            vmax=v,
        )
        ax.set(title=metacluster)
        top = (
            p[["logfoldchanges", "-logp"]]
            .abs()
            .apply(z_score)
            .mean(1)
            .sort_values()
        )
        for x in top.tail(5).index:
            ha = "right" if p.loc[x, "logfoldchanges"] < 0 else "left"
            ax.text(
                p.loc[x, "logfoldchanges"],
                p.loc[x, "-logp"],
                s=p.loc[x, "marker_name"],
                ha=ha,
            )
        ax.set(xlim=(-v, v))
    axes[2][-1].set_visible(False)
    axes[1][0].set(ylabel="-log(p-value)")
    axes[-1][1].set(xlabel="log(fold-change)")
    axes[0][1].set_title("COVID19 vs Healthy\n" + axes[0][1].get_title())
    fig.savefig(
        output_prefix
        + "phenotypes.metaclusters.expression.differential_testing.volcano_plots.disease.svg",
        **consts.figkws,
    )

    fig, axes = plt.subplots(
        3 * 2,
        3,
        figsize=(4 * 2.7, 4 * 2.7 * 2),
        sharex=False,
        sharey=False,
        gridspec_kw=dict(hspace=0.5, wspace=0.5),
    )
    for ax in axes.flat:
        ax.axvline(0, linestyle="--", color="grey")
        ax.axhline(5, linestyle="--", color="grey", linewidth=0.3)
    for metacluster, ax in zip(metaclusters, axes[:3].flat):
        p = diff_res.query(
            f"metacluster == '{metacluster}' & group == 'COVID19_early' & factor == 'phenotypes'"
        )
        v = p["logfoldchanges"].abs().max()
        v += v * 0.1
        ax.scatter(
            p["logfoldchanges"],
            p["-logp"],
            s=10,
            alpha=0.5,
            c=p["logfoldchanges"],
            cmap="coolwarm",
            vmin=-v,
            vmax=v,
        )
        ax.set(title=metacluster)
        top = (
            p[["logfoldchanges", "-logp"]]
            .abs()
            .apply(z_score)
            .mean(1)
            .sort_values()
        )
        for x in top.tail(5).index:
            ha = "right" if p.loc[x, "logfoldchanges"] < 0 else "left"
            ax.text(
                p.loc[x, "logfoldchanges"],
                p.loc[x, "-logp"],
                s=p.loc[x, "marker_name"],
                ha=ha,
            )
        ax.set(xlim=(-v, v))
    axes[2][-1].set_visible(False)
    for metacluster, ax in zip(metaclusters, axes[3:].flat):
        p = diff_res.query(
            f"metacluster == '{metacluster}' & group == 'COVID19_late'"
        )
        v = p["logfoldchanges"].abs().max()
        v += v * 0.1
        ax.scatter(
            p["logfoldchanges"],
            p["-logp"],
            s=10,
            alpha=0.5,
            c=p["logfoldchanges"],
            cmap="coolwarm",
            vmin=-v,
            vmax=v,
        )
        ax.set(title=metacluster)
        top = (
            p[["logfoldchanges", "-logp"]]
            .abs()
            .apply(z_score)
            .mean(1)
            .sort_values()
        )
        for x in top.tail(5).index:
            ha = "right" if p.loc[x, "logfoldchanges"] < 0 else "left"
            ax.text(
                p.loc[x, "logfoldchanges"],
                p.loc[x, "-logp"],
                s=p.loc[x, "marker"],
                ha=ha,
            )
        ax.set(xlim=(-v, v))
    axes[-1][-1].set_visible(False)
    axes[1][0].set(ylabel="-log(p-value)")
    axes[-2][0].set(ylabel="-log(p-value)")
    axes[-1][1].set(xlabel="log(fold-change)")
    axes[0][1].set_title("COVID19_early vs Healthy\n" + axes[0][1].get_title())
    axes[-3][1].set_title("COVID19_late vs Healthy\n" + axes[-3][1].get_title())
    fig.savefig(
        output_prefix
        + "phenotypes.metaclusters.expression.differential_testing.volcano_plots.phenotypes.svg",
        **consts.figkws,
    )

    # Violinplots
    diff_res = pd.read_csv(
        output_prefix
        + "phenotypes.metaclusters.expression.differential_testing.csv",
        index_col=0,
    )

    # diff_res = (
    #     diff_res.groupby(["marker", "group", "factor", "metacluster"])
    #     .mean()
    #     .drop("iter", 1)
    #     .reset_index()
    # )
    diff_res = diff_res.dropna()
    diff_res["-logp"] = -np.log10(diff_res["pvals"])
    v = diff_res["-logp"].replace(np.inf, np.nan).dropna().max()
    diff_res["-logp"] = diff_res["-logp"].replace(np.inf, v)

    metaclusters = a2.obs[f"metacluster_labels_{res}"].unique()
    for metacluster in metaclusters:
        for factor in ["disease", "phenotypes"]:
            a3 = a2[a2.obs[f"metacluster_labels_{res}"] == metacluster, :]
            a3.obs[factor] = pd.Categorical(
                a3.obs[factor],
                categories=a2.obs[factor].cat.categories,
                ordered=True,
            )
            kws = dict(
                groupby=factor,
                use_raw=True,
                stripplot=False,
                multi_panel=True,
                order=a3.obs[factor].cat.categories,
                show=False,
            )
            fig = sc.pl.violin(a3, a3.var.index, **kws)[0].figure
            fig.savefig(
                output_prefix
                + f"phenotypes.metaclusters.expression.{metacluster}.{factor}.violinplots.svg",
                **consts.figkws,
            )
            for group in a3.obs[factor].cat.categories[1:]:
                p = diff_res.query(
                    f"metacluster == '{metacluster}' & group == '{group}'"
                )
                p = p.loc[p["marker"].isin(consts.functional_markers), :]
                top = p.loc[
                    (
                        p[["logfoldchanges", "-logp"]]
                        .abs()
                        .apply(z_score)
                        .mean(1)
                        .sort_values()
                        .tail(3)
                        .index
                    ),
                    "marker",
                ][::-1]
                fig = sc.pl.violin(a3, top, **kws)[0].figure
                fig.savefig(
                    output_prefix
                    + f"phenotypes.metaclusters.expression.{metacluster.replace(' ', '')}.{factor}.violinplots.top_diff_{group}.svg",
                    **consts.figkws,
                )


def differential_diffxpy(prj) -> None:
    import diffxpy.api as de

    output_prefix = consts.output_dir / "phenotyping" / prj.name + "."

    # load cell types from h5ad
    h5ad_f = (
        consts.output_dir / "phenotyping" / prj.name
        + "."
        + "sample_zscore.labeled.h5ad"
    )
    a = sc.read(h5ad_f).raw.to_adata()

    _diff_res = list()
    for group in ["all", "COVID19_early", "COVID19_late", "between"]:
        if group == "all":
            a2 = a.copy()
        elif group == "between":
            a2 = a[~a.obs["phenotypes"].isin(["Healthy"])]
            a2.obs["phenotypes"] = a2.obs[
                "phenotypes"
            ].cat.remove_unused_categories()
        else:
            a2 = a[a.obs["phenotypes"].isin(["Healthy", group])]
            a2.obs["phenotypes"] = a2.obs[
                "phenotypes"
            ].cat.remove_unused_categories()

        a2.obs["p"] = a2.obs["phenotypes"].cat.codes

        part = de.test.partition(data=a2, parts="metacluster_labels_2.0")
        test_part = part.wald(formula_loc="~ 1 + p", factor_loc_totest="p")
        diff_res = pd.concat(
            [
                r.summary().assign(cell_type=n)
                for n, r in zip(test_part.partitions, test_part.tests)
            ]
        ).assign(contrast=group)
        _diff_res.append(diff_res)

    diff_res = pd.concat(_diff_res)
    diff_res.to_csv(
        output_prefix
        + "phenotypes.metaclusters.expression.differential_testing.raw_values.wald.csv"
    )

    diff_res = pd.read_csv(
        output_prefix
        + "phenotypes.metaclusters.expression.differential_testing.raw_values.wald.csv",
        index_col=0,
    )
    diff_res = diff_res.loc[
        diff_res["contrast"].isin(["all", "COVID19_early", "COVID19_late"])
    ]
    diff_res["contrast"] = diff_res["contrast"].replace("all", "COVID19_all")
    diff_res["gene"] = [x.split("(")[0] for x in diff_res["gene"]]

    funct = [x.split("(")[0] for x in consts.functional_markers]
    lfc = diff_res.pivot_table(
        index=["cell_type", "contrast"],
        columns="gene",
        values="log2fc",
    ).loc[:, funct]
    p = diff_res.pivot_table(
        index=["cell_type", "contrast"],
        columns="gene",
        values="pval",
    ).loc[:, funct]
    padj = diff_res.pivot_table(
        index=["cell_type", "contrast"],
        columns="gene",
        values="qval",
    ).loc[:, funct]
    sigs = padj < 1e-25

    grid = clustermap(
        lfc,
        annot=sigs,
        center=0,
        cmap="RdBu_r",
        xticklabels=True,
        yticklabels=True,
        cbar_kws=dict(label="log fold change\n(over healthy)"),
        row_cluster=False,
        col_cluster=False,
        vmin=-3.5,
        vmax=3.5,
        # col_colors=np.log1p(a.to_df()[funct].mean())
        # .rename("Channel mean")
        # .clip(0, 1),
    )
    for i, c in enumerate(grid.ax_heatmap.get_children()):
        if isinstance(c, matplotlib.text.Text):
            if c.get_text() == "0":
                c.set_visible(False)
                # ax.get_children().pop(i)
            elif c.get_text() == "1":
                c.set_text("*")
    grid.fig.savefig(
        output_prefix
        + f"phenotypes.metaclusters.expression.differential_testing.raw_values.wald.joint_stats.heatmap.joint.svg",
        **consts.figkws,
    )

    grid = clustermap(
        lfc.loc[:, "COVID19_all", :],
        annot=sigs.loc[:, "COVID19_all", :],
        center=0,
        cmap="RdBu_r",
        xticklabels=True,
        yticklabels=True,
        cbar_kws=dict(label="log fold change\n(over healthy)"),
        row_cluster=False,
        col_cluster=False,
        vmin=-3.5,
        vmax=3.5,
        # col_colors=np.log1p(a.to_df()[funct].mean())
        # .rename("Channel mean")
        # .clip(0, 1),
        figsize=(6, 4),
    )
    for i, c in enumerate(grid.ax_heatmap.get_children()):
        if isinstance(c, matplotlib.text.Text):
            if c.get_text() == "0":
                c.set_visible(False)
                # ax.get_children().pop(i)
            elif c.get_text() == "1":
                c.set_text("*")
    grid.fig.savefig(
        output_prefix
        + f"phenotypes.metaclusters.expression.differential_testing.raw_values.wald.joint_stats.heatmap.joint.both.svg",
        **consts.figkws,
    )

    for group in ["all", "COVID19_early", "COVID19_late", "between"]:
        diff = diff_res.query(f"contrast == '{group}'")

        # Heatmap of log fold changes + pvalues
        lfc = diff.pivot_table(
            index=["cell_type"],
            columns="gene",
            values="log2fc",
        ).loc[:, consts.functional_markers]
        p = diff.pivot_table(
            index=["cell_type"],
            columns="gene",
            values="pval",
        ).loc[:, consts.functional_markers]
        padj = diff.pivot_table(
            index=["cell_type"],
            columns="gene",
            values="qval",
        ).loc[:, consts.functional_markers]
        sigs = padj < 1e-10

        grid = clustermap(
            lfc,
            annot=sigs,
            center=0,
            cmap="RdBu_r",
            xticklabels=True,
            yticklabels=True,
            cbar_kws=dict(label="log fold change\n(over healthy)"),
            row_cluster=False,
            col_cluster=False,
            vmin=-3.5,
            vmax=3.5,
            col_colors=np.log1p(a.to_df()[consts.functional_markers].mean())
            .rename("Channel mean")
            .clip(0, 1),
        )
        for i, c in enumerate(grid.ax_heatmap.get_children()):
            if isinstance(c, matplotlib.text.Text):
                if c.get_text() == "0":
                    c.set_visible(False)
                    # ax.get_children().pop(i)
                elif c.get_text() == "1":
                    c.set_text("*")

        grid.fig.savefig(
            output_prefix
            + f"phenotypes.metaclusters.expression.differential_testing.raw_values.wald.joint_stats.heatmap.{group}.svg",
            **consts.figkws,
        )

    # with statsmodels
    import statsmodels.formula.api as smf
    import pingouin as pg

    df = a.to_df().join(a.obs[["phenotypes", "metacluster_labels_2.0"]])
    _res = list()
    for ct in a.obs["metacluster_labels_2.0"].unique():
        df2 = df.loc[
            df["metacluster_labels_2.0"] == ct,
            consts.functional_markers + ["phenotypes"],
        ]
        df2.columns = [x.split("(")[0] for x in df2.columns]

        for m in df2.columns.drop(["phenotypes"]):
            res = smf.ols(f"{m} ~ phenotypes", df2).fit()
            _res.append(
                pd.DataFrame(
                    {
                        "coef": res.params,
                        "p-value": res.pvalues,
                        "marker": m,
                        "cell_type": ct,
                    }
                )
            )

    diff_res = pd.concat(_res).drop("Intercept").rename_axis("contrast")
    diff_res["qval"] = pg.multicomp(diff_res["p-value"].values, method="bonf")[
        1
    ]
    funct = [x.split("(")[0] for x in consts.functional_markers]
    lfc = diff_res.pivot_table(
        index=["cell_type", "contrast"],
        columns="marker",
        values="coef",
    ).loc[:, funct]
    p = diff_res.pivot_table(
        index=["cell_type", "contrast"],
        columns="marker",
        values="p-value",
    ).loc[:, funct]
    padj = diff_res.pivot_table(
        index=["cell_type", "contrast"],
        columns="marker",
        values="qval",
    ).loc[:, funct]
    sigs = padj < 1e-25

    grid = clustermap(
        lfc,
        annot=sigs,
        center=0,
        cmap="RdBu_r",
        xticklabels=True,
        yticklabels=True,
        cbar_kws=dict(label="log fold change\n(over healthy)"),
        row_cluster=False,
        col_cluster=False,
        vmin=-3.5,
        vmax=3.5,
        col_colors=np.log1p(a.to_df()[consts.functional_markers].mean())
        .rename("Channel mean")
        .clip(0, 1),
    )
    for i, c in enumerate(grid.ax_heatmap.get_children()):
        if isinstance(c, matplotlib.text.Text):
            if c.get_text() == "0":
                c.set_visible(False)
                # ax.get_children().pop(i)
            elif c.get_text() == "1":
                c.set_text("*")
    grid.fig.savefig(
        output_prefix
        + f"phenotypes.metaclusters.expression.differential_testing.raw_values.wald.statsmodels.heatmap.joint.svg",
        **consts.figkws,
    )

    # Pseudobulk approach
    df = (
        a.to_df()
        .join(a.obs[["phenotypes", "roi", "metacluster_labels_2.0"]])
        .groupby(["phenotypes", "roi", "metacluster_labels_2.0"])
        .mean()
        .reset_index()
    )
    _res = list()
    for ct in a.obs["metacluster_labels_2.0"].unique():
        df2 = df.loc[
            df["metacluster_labels_2.0"] == ct,
            consts.functional_markers + ["phenotypes"],
        ]
        df2.columns = [x.split("(")[0] for x in df2.columns]

        for m in df2.columns.drop(["phenotypes"]):
            res = smf.ols(f"{m} ~ phenotypes", df2).fit()
            _res.append(
                pd.DataFrame(
                    {
                        "coef": res.params,
                        "p-value": res.pvalues,
                        "marker": m,
                        "cell_type": ct,
                    }
                )
            )

    diff_res = pd.concat(_res).drop("Intercept").rename_axis("contrast")
    diff_res["qval"] = pg.multicomp(diff_res["p-value"].values, method="bonf")[
        1
    ]
    funct = [x.split("(")[0] for x in consts.functional_markers]
    lfc = diff_res.pivot_table(
        index=["cell_type", "contrast"],
        columns="marker",
        values="coef",
    ).loc[:, funct]
    p = diff_res.pivot_table(
        index=["cell_type", "contrast"],
        columns="marker",
        values="p-value",
    ).loc[:, funct]
    padj = diff_res.pivot_table(
        index=["cell_type", "contrast"],
        columns="marker",
        values="qval",
    ).loc[:, funct]
    sigs = padj < 0.05

    grid = clustermap(
        lfc,
        annot=sigs,
        center=0,
        cmap="RdBu_r",
        xticklabels=True,
        yticklabels=True,
        cbar_kws=dict(label="log fold change\n(over healthy)"),
        row_cluster=False,
        col_cluster=False,
        vmin=-3.5,
        vmax=3.5,
        col_colors=np.log1p(a.to_df()[consts.functional_markers].mean())
        .rename("Channel mean")
        .clip(0, 1),
    )
    for i, c in enumerate(grid.ax_heatmap.get_children()):
        if isinstance(c, matplotlib.text.Text):
            if c.get_text() == "0":
                c.set_visible(False)
                # ax.get_children().pop(i)
            elif c.get_text() == "1":
                c.set_text("*")
    grid.fig.savefig(
        output_prefix
        + f"phenotypes.metaclusters.expression.differential_testing.raw_values.wald.statsmodels_pseudobulk.heatmap.joint.svg",
        **consts.figkws,
    )


def threshold_positiveness():
    import yaml
    from imc.operations import (
        get_best_mixture_number,
        get_threshold_from_gaussian_mixture,
    )

    (consts.output_dir / "gating").mkdir()

    # load quantification
    prefix = consts.output_dir / "phenotyping" / prj.name + "."
    quant_ff = prefix + "quantification.filtered.pq"
    quant = pd.read_parquet(quant_ff)

    ids = ["sample", "roi"]
    quant = pd.concat([np.log1p(quant.drop(ids, axis=1)), quant[ids]], axis=1)

    # load cell types from h5ad
    h5ad_f = (
        consts.output_dir / "phenotyping" / prj.name
        + "."
        + "sample_zscore.labeled.h5ad"
    )
    a = sc.read(h5ad_f)

    # remove excluded channels
    exc = prj.rois[0].channel_exclude[prj.rois[0].channel_exclude].index
    quant = quant.drop(exc, axis=1, errors="ignore")

    # # Univariate gating of each channel per sample
    # thresholds_file = consts.output_dir / "thresholds.activation.json"
    # mixes_file = consts.output_dir / "mixes.activation.json"
    # if not (thresholds_file.exists() and thresholds_file.exists()):
    #     mixes = dict()
    #     thresholds = dict()
    #     for m in quant.columns.drop(ids):
    #         if m not in thresholds:
    #             mixes[m] = get_best_mixture_number(quant[m], 2, 8)
    #             thresholds[m] = get_threshold_from_gaussian_mixture(
    #                 quant[m], None, mixes[m]
    #             ).to_dict()
    #     json.dump(thresholds, open(thresholds_file, "w"), indent=4)
    #     json.dump(mixes, open(mixes_file, "w"), indent=4)
    # thresholds = json.load(open(thresholds_file))
    # mixes = json.load(open(mixes_file))

    # # Make dataframe with population for each marker
    # gating_file = consts.output_dir / "gating" / "positive.pq"
    # if not gating_file.exists():
    #     pos = pd.DataFrame(index=quant.index, columns=consts.functional_markers)
    #     for m in consts.functional_markers:
    #         name = m.split("(")[0]
    #         o = sorted(thresholds[m])
    #         if mixes[m] == 2:
    #             pos[m] = quant[m] > thresholds[m][o[0]]
    #         else:
    #             pos[m] = quant[m] > thresholds[m][o[-1]]
    #             sel = pos[m] == False
    #             pos.loc[sel, m] = quant.loc[sel, m] > thresholds[m][o[-2]]
    #     pos = pd.concat([pos, quant[ids]], axis=1)
    #     pos.to_parquet(gating_file)
    # pos = pd.read_parquet(gating_file)
    # pos.index.name = "obj_id"

    # # Univariate gating of each channel (per sample)
    # thresholds_file = (
    #     consts.output_dir / "thresholds.activation.per_sample.yaml"
    # )
    # mixes_file = consts.output_dir / "mixes.activation.per_sample.yaml"
    # if not (thresholds_file.exists() and thresholds_file.exists()):
    #     mixes = dict()
    #     thresholds = dict()
    #     for m in consts.functional_markers:
    #         for s in quant["sample"].unique():
    #             if (m, s) not in thresholds:
    #                 y = quant.query(f"sample == '{s}'")
    #                 mixes[(m, s)] = get_best_mixture_number(y[m], 2, 8)
    #                 thresholds[(m, s)] = get_threshold_from_gaussian_mixture(
    #                     y[m], None, mixes[(m, s)]
    #                 ).to_dict()
    #     yaml.dump(thresholds, open(thresholds_file, "w"))
    #     yaml.dump(mixes, open(mixes_file, "w"))
    # thresholds = yaml.load(open(thresholds_file))
    # mixes = yaml.load(open(mixes_file))

    # # Make dataframe with population for each marker
    # gating_file = consts.output_dir / "gating" / "positive.per_sample.pq"
    # if not gating_file.exists():
    #     _pos = list()
    #     for s in quant["sample"].unique():
    #         y = quant.query(f"sample == '{s}'")
    #         pos = pd.DataFrame(index=y.index, columns=consts.functional_markers)
    #         for m in consts.functional_markers:
    #             o = sorted(thresholds[(m, s)])
    #             if mixes[(m, s)] == 2:
    #                 pos[m] = y[m] > thresholds[(m, s)][o[0]]
    #             else:
    #                 pos[m] = y[m] > thresholds[(m, s)][o[-1]]
    #                 sel = pos[m] == False
    #                 pos.loc[sel, m] = y.loc[sel, m] > thresholds[(m, s)][o[-2]]
    #         _pos.append(pd.concat([pos, y[ids]], axis=1))
    #     pos = pd.concat(_pos, axis=0)
    #     pos.to_parquet(gating_file)
    # pos = pd.read_parquet(gating_file)

    # # Univariate gating of each channel per sample (with Z-scored data)
    # zquant = a.to_df()
    # thresholds_file = consts.output_dir / "thresholds.activation.zscore.json"
    # mixes_file = consts.output_dir / "mixes.activation.zscore.json"
    # if not (thresholds_file.exists() and thresholds_file.exists()):
    #     mixes = dict()
    #     thresholds = dict()
    #     for m in consts.functional_markers:
    #         if m not in thresholds:
    #             mixes[m] = get_best_mixture_number(zquant[m], 2, 8)
    #             thresholds[m] = get_threshold_from_gaussian_mixture(
    #                 zquant[m], None, mixes[m]
    #             ).to_dict()
    #     json.dump(thresholds, open(thresholds_file, "w"), indent=4)
    #     json.dump(mixes, open(mixes_file, "w"), indent=4)
    # thresholds = json.load(open(thresholds_file))
    # mixes = json.load(open(mixes_file))

    # # Make dataframe with population for each marker
    # gating_file = consts.output_dir / "gating" / "positive.z_score.pq"
    # if not gating_file.exists():
    #     pos = pd.DataFrame(index=quant.index, columns=consts.functional_markers)
    #     for m in consts.functional_markers:
    #         name = m.split("(")[0]
    #         o = sorted(thresholds[m])
    #         if mixes[m] == 2:
    #             pos[m] = quant[m] > thresholds[m][o[0]]
    #         else:
    #             pos[m] = quant[m] > thresholds[m][o[-1]]
    #             sel = pos[m] == False
    #             pos.loc[sel, m] = quant.loc[sel, m] > thresholds[m][o[-2]]
    #     pos = pd.concat([pos, quant[ids]], axis=1)
    #     pos.to_parquet(gating_file)
    # pos = pd.read_parquet(gating_file)
    # pos.index.name = "obj_id"

    # p = pos.merge(roi_attributes["phenotypes"].reset_index())
    po = pos.groupby("roi").sum()
    total = pos.groupby("roi").size()
    perc = (po.T / total).T * 100

    fig, stats = swarmboxenplot(
        data=perc.join(roi_attributes),
        x="phenotypes",
        y=consts.functional_markers,
    )

    # by cell type
    m = pos.merge(a.obs, on=["sample", "roi", "obj_id"])
    po = m.groupby(["metacluster_labels_2.0", "roi"])[
        consts.functional_markers
    ].sum()
    total = m.groupby(["metacluster_labels_2.0", "roi"]).size()
    perc = (po.T / total).T.fillna(0) * 100

    grid = clustermap(perc.groupby(level=0).mean())

    p = (
        perc.join(roi_attributes["phenotypes"])
        .groupby(["metacluster_labels_2.0", "phenotypes"])
        .mean()
    )
    grid = clustermap(p, row_cluster=False)
    grid = clustermap(p, row_cluster=False, col_cluster=False)
    grid = clustermap(p.T, config="z", row_cluster=False, col_cluster=False)

    grid = clustermap(p, config="z", row_cluster=False, col_cluster=False)

    # fig, stats = swarmboxenplot(
    #     data=perc.join(roi_attributes),
    #     x="phenotypes",
    #     y=consts.functional_markers,
    # )

    # Try using hard thresholds (doesn't work)
    # zquant = a.to_df()
    # zquant.index = zquant.index.astype(str)
    # pos = (zquant[consts.functional_markers] > 3).join(a.obs[ids])
    # pos.index = pos.index.astype(int)
    # pos.index.name = 'obj_id'


def intra_metacluster(prj: Project) -> None:
    (consts.output_dir / "refined_cell_types").mkdir()
    prefix = consts.output_dir / "phenotyping" / prj.name + "."
    h5ad_f = prefix + "sample_zscore.labeled.h5ad"
    a = sc.read(h5ad_f)

    res = 2.0
    metaclusters = a.obs[f"metacluster_labels_{res}"].unique()
    metaclusters = [
        "T cells",
        "Neutrophils",
        "Macrophages|Monocytes",
        "Endothelial",
        "Epithelial",
        "B cells",
        "NK cells",
    ]

    for metacluster in metaclusters:
        output_prefix = (
            consts.output_dir
            / "refined_cell_types"
            / metacluster.replace(" ", "_").replace("|", "-")
            + "."
        )

        ta = a[a.obs[f"metacluster_labels_{res}"].str.contains(metacluster), :]
        ta = ta[:, ~ta.var.index.isin(consts.tech_channels)]

        # sc.pp.scale(ta)
        sc.pp.pca(ta)
        with parallel_backend("threading", n_jobs=12):
            sc.pp.neighbors(ta, n_neighbors=15, use_rep="X_pca")
        with parallel_backend("threading", n_jobs=12):
            sc.tl.umap(ta, gamma=25)

        nres = 1.0
        sc.tl.leiden(ta, resolution=nres, key_added=f"refined_cluster_{nres}")
        ta.obs[f"refined_cluster_{nres}"] = pd.Categorical(
            ta.obs[f"refined_cluster_{nres}"].astype(int) + 1
        )
        ta.uns["phenotypes_colors"] = consts.colors["phenotypes"]

        # UMAPs
        # vmin = None
        # vmax = np.percentile(ta.raw.X, 99, axis=0).tolist() + [None] * 4
        # fig = sc.pl.umap(
        #     ta,
        #     # color=functional_markers
        #     color=ta.var.index.tolist()
        #     + [
        #         f"cluster_{res}",
        #         f"refined_cluster_{nres}",
        #         "phenotypes",
        #         "sample",
        #     ],
        #     use_raw=True,
        #     vmin=vmin,
        #     vmax=vmax,
        #     show=False,
        # )[0].figure
        # rasterize_scanpy(fig)
        # fig.savefig(output_prefix + "umap.markers.raw.svgz", **consts.figkws)

        fig = sc.pl.umap(
            ta,
            # color=functional_markers
            color=ta.var.index.tolist()
            + [
                f"cluster_{res}",
                f"refined_cluster_{nres}",
                "phenotypes",
                "sample",
            ],
            use_raw=False,
            show=False,
        )[0].figure
        rasterize_scanpy(fig)
        fig.savefig(output_prefix + "umap.markers.svgz", **consts.figkws)

        # Plot cluster phenotypes
        # # get mean per cluster
        m = ta.to_df().groupby(ta.obs[f"cluster_{nres}"]).mean()
        umap_pos = (
            pd.DataFrame(
                ta.obsm["X_umap"],
                index=ta.obs.index,
                columns=["UMAP1", "UMAP2"],
            )
            .groupby(ta.obs[f"cluster_{nres}"])
            .mean()
        )
        m = m.join(umap_pos)

        # # get normalized proportions per disease group
        ct = ta.obs["phenotypes"].value_counts()
        ct /= ct.sum()
        c = ta.obs.groupby([f"refined_cluster_{nres}", "phenotypes"]).size()
        p = c.groupby(level=0).apply(lambda x: x / x.sum())
        p = p.to_frame().pivot_table(
            index=f"refined_cluster_{nres}", columns="phenotypes", values=0
        )
        p = np.log(p / ct)

        grid = clustermap(
            m, row_colors=p, config="z", figsize=(8, 4 * max(1, res))
        )
        grid.fig.savefig(
            output_prefix + "clustermap.svg",
            **consts.figkws,
        )


def filter_out_cells(
    quant: DataFrame, plot=True, output_prefix: Path = None
) -> Array:
    from imc.operations import get_population
    from imc.utils import minmax_scale
    from mpl_toolkits.mplot3d import Axes3D

    # create combined score for artifact likelihood
    score = minmax_scale(
        (minmax_scale(quant["solidity"]) * 2)
        * (1 - minmax_scale(quant["area"]))
        * (1 - minmax_scale(quant["DNA"]))
    )

    # get population with highest score
    ## KMeans with k == 3 also works well but we have to assume smallest cluster is to remove
    # from sklearn.cluster import KMeans
    # al = KMeans(3)
    # al.fit(score.values.reshape((-1, 1)))
    # c = al.predict(score.values.reshape((-1, 1)))
    # to_filter = c == pd.Series(c).value_counts().idxmin()

    ## Mixture of gaussians
    to_filter = get_population(score, min_mix=3)

    if plot:
        assert output_prefix is not None
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for sel, edgecolor in [(to_filter, "red"), (~to_filter, "black")]:
            s = ax.scatter(
                quant.loc[sel]["solidity"],
                np.sqrt(quant.loc[sel]["area"]),
                np.log1p(quant.loc[sel]["DNA"]),
                s=2,
                alpha=0.25,
                c=score[sel],
                edgecolors=edgecolor,
                linewidths=0.25,
                rasterized=True,
            )
        fig.colorbar(s, ax=ax, label="score")
        ax.set(xlabel="solidity", ylabel="area", zlabel="DNA")
        fig.savefig(output_prefix + "3d_scatter.svg", **consts.figkws)

        fig, axes = plt.subplots(1, 3, figsize=(3 * 4, 1 * 4), sharey=True)
        for ax, var in zip(axes, ["area", "DNA", "solidity"]):
            sns.distplot(
                quant[var][~to_filter], label="To keep", kde=False, ax=ax
            )
            sns.distplot(
                quant[var][to_filter], label="To remove", kde=False, ax=ax
            )
            ax.set_xlabel(var)
            ax.legend()
        axes[0].set_ylabel("Cell number")
        fig.savefig(
            output_prefix + "per_variable_histogram.svg", **consts.figkws
        )

    return to_filter


def example_visualizations(prj) -> None:
    from imc.graphics import get_grid_dims
    from csbdeep.utils import normalize

    output_dir = consts.output_dir / "example_visualizations"
    output_dir.mkdir()

    examples = [
        # ((roi_name, example_name), (pos=((y2, y2), (x2, x1)), markers))
        (
            (
                "A20_77_20210121_ActivationPanel-06",
                "S100A9_reduct_in_monos_covid",
            ),
            (
                None,
                [
                    "CD14",
                    "S100A9",
                    "DNA",
                ],
            ),
        ),
        (
            (
                "A20_77_20210121_ActivationPanel-06",
                "S100A9_reduct_in_monos_covid_zoom",
            ),
            (
                ((1200, 940), (1000, 740)),
                [
                    "CD14",
                    "S100A9",
                    "DNA",
                ],
            ),
        ),
        (
            (
                "A19_33_20210121_ActivationPanel-04",
                "S100A9_high_in_monos_healthy3",
            ),
            (
                None,
                [
                    "CD14",
                    "S100A9",
                    "DNA",
                ],
            ),
        ),
        (
            (
                "A19_33_20210121_ActivationPanel-04",
                "S100A9_high_in_monos_healthy3_zoom",
            ),
            (
                ((400, 140), (700, 440)),
                [
                    "CD14",
                    "S100A9",
                    "DNA",
                ],
            ),
        ),
        (
            ("A20_77_20210121_ActivationPanel-05", "HLADR_in_keratin_covid"),
            (
                None,
                [
                    "HLADR",
                    "Keratin818",
                    "DNA",
                ],
            ),
        ),
        (
            (
                "A20_77_20210121_ActivationPanel-05",
                "HLADR_in_keratin_covid_zoom",
            ),
            (
                ((260, 20), (790, 550)),
                [
                    "HLADR",
                    "Keratin818",
                    "DNA",
                ],
            ),
        ),
        (
            (
                "A19_33_20210121_ActivationPanel-04",
                "HLADR_not_in_keratin_healthy",
            ),
            (
                None,
                [
                    "HLADR",
                    "Keratin818",
                    "DNA",
                ],
            ),
        ),
        (
            (
                "A19_33_20210121_ActivationPanel-04",
                "HLADR_not_in_keratin_healthy_zoom",
            ),
            (
                ((460, 220), (520, 280)),
                [
                    "HLADR",
                    "Keratin818",
                    "DNA",
                ],
            ),
        ),
        (
            ("A20_58_20210122_ActivationPanel-06", "pNFkbp65_in_monos_covid"),
            (None, ["CD16", "pNFkbp65", "DNA"]),
        ),
        (
            ("A20_58_20210122_ActivationPanel-06", "VISTA_not_in_Tcells"),
            (None, ["VISTA", "CD3(", "CD15"]),
        ),
        (
            ("A20_58_20210122_ActivationPanel-06", "TIM3_not_in_Tcells"),
            (None, ["TIM3", "CD3(", "CD15"]),
        ),
        (
            ("A20_58_20210122_ActivationPanel-06", "PDL1_not_in_Tcells"),
            (None, ["PDL1", "CD3(", "CD15"]),
        ),
        (
            ("A20_58_20210122_ActivationPanel-06", "PD1_not_in_Tcells"),
            (None, ["PD1", "CD3(", "CD15"]),
        ),
        (
            ("A20_58_20210122_ActivationPanel-08", "VISTA_not_in_Tcells2"),
            (None, ["VISTA", "CD3(", "CD15"]),
        ),
        (
            ("A20_58_20210122_ActivationPanel-08", "VISTA_not_in_Tcells2_zoom"),
            (((600, 340), (1200, 940)), ["VISTA", "CD3(", "CD15"]),
        ),
    ]

    examples = [
        (
            ("A19_33_20210121_ActivationPanel-02", "S100A9_low_in_Healthy"),
            (None, ["S100A9", "CD15", "Keratin818"]),
        ),
        (
            ("A20_47_20210120_ActivationPanel-07", "S100A9_high_in_COVIDearly"),
            (None, ["S100A9", "CD15", "Keratin818"]),
        ),
        (
            ("A20_77_20210121_ActivationPanel-04", "S100A9_high_in_COVIDlate"),
            (None, ["S100A9", "CD15", "Keratin818"]),
        ),
        (
            ("A19_33_20210121_ActivationPanel-02", "S100A9_low_in_Healthy2"),
            (None, ["S100A9", "CD15", "DNA"]),
        ),
        (
            ("A20_47_20210120_ActivationPanel-07", "S100A9_high_in_COVIDearly"),
            (None, ["S100A9", "CD15", "DNA"]),
        ),
        (
            ("A20_77_20210121_ActivationPanel-04", "S100A9_high_in_COVIDlate2"),
            (None, ["S100A9", "CD15", "DNA"]),
        ),
    ]

    for example in examples:
        (roi_name, example_name), (pos, markers) = example
        roi = prj.get_rois(roi_name)
        fig1 = roi.plot_channels(
            markers, equalize=False, position=pos, smooth=3
        )
        fig1.savefig(
            output_dir / f"examples.{example_name}.separate.svg",
            **consts.figkws,
        )

        fig2 = roi.plot_channels(
            markers[:3],
            equalize=False,
            position=pos,
            merged=True,
            # smooth=1
        )
        fig2.savefig(
            output_dir / f"examples.{example_name}.merged.svg", **consts.figkws
        )

        # plot manually
        from imc.graphics import add_scale
        from skimage.filters import gaussian

        p = np.asarray(
            [
                gaussian(normalize(x), sigma=1)
                for x in roi._get_channels(markers[:3])[1]
            ]
        )
        if pos is not None:
            p = p[:, slice(pos[0][1], pos[0][0]), slice(pos[1][1], pos[1][0])]
        fig3, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(np.moveaxis(normalize(p), 0, -1))
        add_scale(ax)
        ax.axis("off")
        fig3.savefig(
            output_dir / f"examples.{example_name}.merged.smooth.svg",
            **consts.figkws,
        )


@dataclass
class consts:
    cur_date = f"{datetime.datetime.now().date()}"
    figkws = dict(bbox_inches="tight", dpi=300)

    metadata_dir = Path("metadata")
    data_dir = Path("data")
    results_dir = Path("results")
    output_dir = results_dir / "imc_revision"
    output_dir.mkdir()

    # Sample-specific
    phenotypes = (
        pd.Series(
            {
                "A19_33_20210121_ActivationPanel": "Healthy",
                "A19_33_20210122_ActivationPanel": "Healthy",
                "S19_6699_20210120_ActivationPanel": "Healthy",
                "A20_47_20210120_ActivationPanel": "COVID19_early",
                "A20_58_20210122_ActivationPanel": "COVID19_early",
                "A20_56_20210120_ActivationPanel": "COVID19_late",
                "A20_77_20210121_ActivationPanel": "COVID19_late",
            },
            name="phenotypes",
        )
        .rename_axis(index="sample")
        .astype(
            pd.CategoricalDtype(
                ordered=True,
                categories=["Healthy", "COVID19_early", "COVID19_late"],
            )
        )
    )
    colors = {
        "phenotypes": [
            matplotlib.colors.to_hex(x)
            for x in np.asarray(sns.color_palette("tab10"))[[2, 4, 3]]
        ],
        "disease": [
            matplotlib.colors.to_hex(x)
            for x in np.asarray(sns.color_palette("tab10"))[[2, 3]]
        ],
    }

    # Load cluster assignments
    cluster_idents = json.load(
        open(metadata_dir / "imc_revision.cluster_identities.json")
    )
    cluster_idents = {
        float(res): {int(c): n for c, n in clusts.items()}
        for res, clusts in cluster_idents.items()
    }

    # Subset markers
    exclude_channels = [
        "80ArAr(ArAr80)",
        "129Xe(Xe129)",
        "190BCKG(BCKG190)",
        "<EMPTY>(Pb204)",
    ]
    tech_channels = [
        "perimeter",
        "DNA",
        "major_axis_length",
        "eccentricity",
        "solidity",
        "DNA",
        "HistoneH3(In113)",
    ]
    functional_markers = [
        "pH3s28(In115)",
        "Ki67",
        "CD45RO(Nd142)",
        "GATA3(Dy164)",
        "TBet(Sm149)",
        "GranzymeB(Er167)",
        "pNFkbp65(Er166)",
        "CD27(Yb171)",
        "CD86(Sm152)",
        "CD44(Eu153)",
        "CD127(Er168)",
        "CD123(Tm169)",
        "CD38(Pr141)",
        "CD161(Gd158)",
        "S100A9(Yb173)",
        "HLADR(Ho165)",
        "CleavedCaspase3(Yb172)",
        "PD1(Nd150)",
        "PDL1(Lu175)",
        "VISTA(Gd160)",
        "TIM3(Nd145)",
    ]


"""
# To quickly convert svgz to pdf (faster than saving as pdf directly)
for F in `find results/imc_revision -name "*.svgz"`
do
    if [ ! -f ${F/svgz/pdf} ]; then 
        echo $F
        sleep 1
        inkscape -o ${F/svgz/pdf} $F 2> /dev/null
    fi
done
"""


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
