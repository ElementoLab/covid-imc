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

    metacluster_expression(prj)

    intra_metacluster(prj)

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

    # Replot UMAP with labels
    fig = sc.pl.umap(a, color=[f"metacluster_labels_{res}"], show=False).figure
    rasterize_scanpy(fig)
    fig.savefig(
        output_prefix + "phenotypes.umap.colored_by_metacluster.svg",
        **consts.figkws,
    )

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
                fig.savefig(
                    output_prefix
                    + f"phenotypes.{group}s.abundance.{name}.by_{factor}.swarmboxenplot.svg",
                    **consts.figkws,
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

    # Now aggregated by clsuter
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
            sc.tl.rank_genes_groups(
                a3,
                factor,
                use_raw=False,
                reference="Healthy",
                # method="t-test_overestim_var",
                method="wilcoxon",
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

    # Test for differential expression within each metacluster between disease groups
    n_random = 25

    a.obs["disease"] = a.obs["phenotypes"].str.split("_").apply(lambda x: x[0])
    metaclusters = a2.obs[f"metacluster_labels_{res}"].unique()
    _diff_res = list()
    for factor in ["phenotypes", "disease"]:
        for metacluster in metaclusters:
            a3 = a[a.obs[f"metacluster_labels_{res}"] == metacluster, :]
            a3.X += abs(a3.X.min())
            groups = a3.obs[factor].unique()[1:]
            n = a3.obs["sample"].value_counts().min()

            for i in range(n_random):
                cells = list()
                for sample in a3.obs["sample"].unique():
                    cells += (
                        a3.obs.query(f"sample == '{sample}'")
                        .sample(n=n)
                        .index.tolist()
                    )
                a4 = a3[cells, :]

                sc.tl.rank_genes_groups(
                    a4,
                    factor,
                    use_raw=False,
                    reference="Healthy",
                    # method="t-test_overestim_var",
                    method="wilcoxon",
                )
                _diff_res.append(
                    pd.concat(
                        [
                            pd.DataFrame(
                                {
                                    "marker": a4.uns["rank_genes_groups"][
                                        "names"
                                    ][group],
                                    "logfoldchanges": a4.uns[
                                        "rank_genes_groups"
                                    ]["logfoldchanges"][group],
                                    "pvals": a4.uns["rank_genes_groups"][
                                        "pvals"
                                    ][group],
                                    "pvals_adj": a4.uns["rank_genes_groups"][
                                        "pvals_adj"
                                    ][group],
                                }
                            ).assign(
                                metacluster=metacluster,
                                group=group,
                                factor=factor,
                                iter=i,
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

    diff_res = (
        diff_res.groupby(["marker", "group", "factor", "metacluster"])
        .mean()
        .drop("iter", 1)
        .reset_index()
    )

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
    )
    p = diff_res.pivot_table(
        index=["metacluster", "group"],
        columns="marker",
        values="pvals_adj",
    )
    sigs = p < 1e-5

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        lfc,
        annot=sigs,
        center=0,
        cmap="RdBu_r",
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

    diff_res = (
        diff_res.groupby(["marker", "group", "factor", "metacluster"])
        .mean()
        .drop("iter", 1)
        .reset_index()
    )
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
                use_raw=False,
                stripplot=False,
                multi_panel=True,
                order=a3.obs[factor].cat.categories,
                show=False,
            )
            fig = sc.pl.violin(a3, a3.var.index)[0].figure
            fig.savefig(
                output_prefix
                + f"phenotypes.metaclusters.expression.{metacluster}.violinplots.svg",
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
                    + f"phenotypes.metaclusters.expression.{metacluster.replace(' ', '')}.violinplots.top_diff_{group}.svg",
                    **consts.figkws,
                )


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
        "CD38(Pr141)",
        "CD45RO(Nd142)",
        "TIM3(Nd145)",
        "TBet(Sm149)",
        "PD1(Nd150)",
        "CD86(Sm152)",
        "CD44(Eu153)",
        "FoxP3(Gd155)",
        "CD161(Gd158)",
        "VISTA(Gd160)",
        "GATA3(Dy164)",
        "HLADR(Ho165)",
        "pNFkbp65(Er166)",
        "GranzymeB(Er167)",
        "CD127(Er168)",
        "CD123(Tm169)",
        "CD27(Yb171)",
        "CleavedCaspase3(Yb172)",
        "S100A9(Yb173)",
        "PDL1(Lu175)",
        "Ki67",
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
