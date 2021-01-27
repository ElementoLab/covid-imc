#! /usr/bin/env python

"""
Analysis of revision data looking in more depth at the immune compartment.
"""

import sys
import datetime
from argparse import ArgumentParser, Namespace
import json

from tqdm import tqdm
from joblib import parallel_backend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import scanpy as sc

from imc import Project
from imc.types import Path, DataFrame, Array
from imc.graphics import close_plots, rasterize_scanpy, add_centroids
from imc.utils import z_score

from seaborn_extensions import clustermap, swarmboxenplot


cur_date = f"{datetime.datetime.now().date()}"
figkws = dict(bbox_inches="tight", dpi=300)

metadata_dir = Path("metadata")
data_dir = Path("data")
results_dir = Path("results")
output_dir = results_dir / "imc_revision"
output_dir.mkdir()


exclude_channels = [
    "80ArAr(ArAr80)",
    "129Xe(Xe129)",
    "190BCKG(BCKG190)",
    "<EMPTY>(Pb204)",
]

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

cluster_idents = json.load(
    open(metadata_dir / "imc_revision.cluster_identities.json")
)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resolutions", default=[0.5, 1.0, 1.5, 2.0, 3.0], nargs="+"
    )
    parser.add_argument("--algos", default=["umap"], nargs="+")
    return parser


args = get_parser().parse_args()
args.resolutions = [float(x) for x in args.resolutions]


def main() -> int:
    prj = Project(name="imc_revision")
    prj.samples = [s for s in prj if "2021" in s.name]
    for r in prj.rois:
        r.set_channel_exclude(exclude_channels)
    # for s in prj:
    #     s.rois = [r for r in s if r._get_input_filename("cell_mask").exists()]

    illustrations(prj)

    # qc(prj)

    phenotyping(prj)

    cell_types(prj)

    return 0


@close_plots
def illustrations(prj: Project) -> None:
    from csbdeep.utils import normalize

    (output_dir / "full_stacks").mkdir()
    (output_dir / "illustration").mkdir()

    for r in tqdm(prj.rois):
        output_f = output_dir / "full_stacks" / r.name + ".pdf"
        if output_f.exists():
            continue
        fig = r.plot_channels()
        fig.savefig(output_f, **figkws)

    for r in tqdm(prj.rois):
        output_f = output_dir / "illustration" / r.name + ".svg"
        if output_f.exists():
            continue
        fig = r.plot_probabilities_and_segmentation()
        fig.savefig(output_f, **figkws)

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
    fig.savefig("Red:CD31-Green:CD39-Blue:DNA.A20_58_example.png", **figkws)


def qc(prj: Project) -> None:
    (output_dir / "qc").mkdir()
    output_prefix = output_dir / "qc" / "channel_summary."

    c = prj.rois[0].channel_labels
    exc = [x for x in c if x in exclude_channels]

    prj.channel_summary(output_prefix=output_prefix, channel_exclude=exc)


def phenotyping(prj: Project) -> None:
    (output_dir / "phenotyping").mkdir()
    output_prefix = output_dir / "phenotyping" / prj.name + f".{cur_date}."

    quant_f = output_prefix + "quantification.pq"
    if not quant_f.exists() or args.overwrite:
        prj.quantify_cells()
        quant = prj.quantification
        prj.quantification.to_parquet(quant_f)
    quant = pd.read_parquet(quant_f)

    quant_mean = quant.groupby("sample").mean()
    quant = quant.drop(exclude_channels, axis=1)

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
    h5ad_f = output_prefix + prj.name + ".sample_zscore.h5ad"
    if not h5ad_f.exists() or args.overwrite:
        # Drop unwanted channels and redundant morphological features
        q = quant.drop(["perimeter", "major_axis_length"], axis=1).reset_index()
        id_cols = ["sample", "roi", "obj_id"]

        # get measure of overal intensity
        tech = q.merge(
            quant_mean[exclude_channels[:-2]]
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
    output_prefix = output_prefix.replace_(
        cur_date, f"{cur_date}.sample_zscore"
    )

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
        mr = (
            AnnData(a.raw.X, var=a.var, obs=a.obs)
            .to_df()
            .groupby(a.obs[f"cluster_{res}"])
            .mean()
        )
        mr = mr.join(umap_pos)

        # # get normalized proportions per disease group
        annot = a.obs.merge(phenotypes.to_frame().reset_index())
        ct = annot["phenotypes"].value_counts()
        ct /= ct.sum()
        c = annot.groupby([f"cluster_{res}", "phenotypes"]).size()
        p = c.groupby(level=0).apply(lambda x: x / x.sum())
        p = p.to_frame().pivot_table(
            index=f"cluster_{res}", columns="phenotypes", values=0
        )
        p = p / ct

        for conf in ["abs", "z"]:
            grid = clustermap(
                mr, row_colors=p, config=conf, figsize=(8, 5 * max(1, res))
            )
            grid.fig.savefig(
                output_prefix
                + f"phenotypes.cluster_{res}.clustermap.raw.{conf}.svg",
                **figkws,
            )

            grid = clustermap(
                m, row_colors=p, config=conf, figsize=(8, 5 * max(1, res))
            )
            grid.fig.savefig(
                output_prefix
                + f"phenotypes.cluster_{res}.clustermap.norm.{conf}.svg",
                **figkws,
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
        f = output_prefix + f"{algo}.pdf"
        projf = getattr(sc.pl, algo)
        axes = projf(
            a,
            color=color,
            show=False,
            vmin=vmin,  # + [None] * len(args.resolutions),
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

        fig.savefig(f, **figkws)


def cell_types(prj) -> None:
    (output_dir / "phenotyping").mkdir()
    cur_date = "2021-01-25"
    output_prefix = output_dir / "phenotyping" / prj.name + f".{cur_date}."

    quant_ff = output_prefix + "quantification.filtered.pq"
    quant = pd.read_parquet(quant_ff)

    # Drop unwanted channels and redundant morphological features
    h5ad_f = output_prefix + prj.name + ".sample_zscore.h5ad"

    a = sc.read(h5ad_f)

    res = 2.0
    a.obs[f"cluster_labels_{res}"] = a.obs[f"cluster_{res}"].replace(
        cluster_idents[res]
    )
    a.obs[f"metacluster_labels_{res}"] = (
        a.obs[f"cluster_labels_{res}"].str.extract(r"(.*) \(")[0].values
    )
    a.obs[f"cluster_labels_{res}"].value_counts().filter(regex=r"^[^\?]")
    a.obs[f"metacluster_labels_{res}"].value_counts().filter(regex=r"^[^\?]")

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
    roi_attributes = pd.DataFrame(
        {
            roi.name: [roi.sample.name, phenotypes[roi.sample.name]]
            for roi in prj.rois
        },
        index=["sample", "phenotypes"],
    ).T.rename_axis(index="roi")
    roi_attributes["phenotypes"] = pd.Categorical(
        roi_attributes["phenotypes"],
        categories=["Healthy", "COVID19_early", "COVID19_late"],
        ordered=True,
    )

    # Plot cluster abundance per disease group
    tech_vars = [
        "perimeter",
        "DNA",
        "major_axis_length",
        "eccentricity",
        "solidity",
        "DNA",
        "HistoneH3(In113)",
    ]
    p = perc.to_frame().pivot_table(
        index="roi", columns=f"cluster_labels_{res}", values="percentage"
    )
    fig, stats = swarmboxenplot(
        data=p.join(roi_attributes), x="phenotypes", y=p.columns
    )
    p = exte.to_frame().pivot_table(
        index="roi", columns=f"cluster_labels_{res}", values="absolute"
    )
    fig, stats = swarmboxenplot(
        data=p.join(roi_attributes), x="phenotypes", y=p.columns
    )

    p = perc_red.to_frame().pivot_table(
        index="roi", columns=f"metacluster_labels_{res}", values="percentage"
    )
    fig, stats = swarmboxenplot(
        data=p.join(roi_attributes), x="phenotypes", y=p.columns
    )
    p = exte_red.to_frame().pivot_table(
        index="roi", columns=f"metacluster_labels_{res}", values="absolute"
    )
    fig, stats = swarmboxenplot(
        data=p.join(roi_attributes), x="phenotypes", y=p.columns
    )

    # Some single-cell heatmaps
    a2 = a[~a.obs[f"metacluster_labels_{res}"].str.startswith("?"), :]
    a2.X = z_score(a2.X.T).T
    a2 = a2[:, ~a2.var.index.isin(tech_vars)]

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
        f"metacluster_labels_{res}",
        use_raw=False,
        cmap="RdBu_r",
        vmin=-6,
        vmax=6,
        show=False,
    )["heatmap_ax"].figure
    fig.savefig(
        output_prefix + "phenotypes.metaclusters.expression.heatmap.svg",
        **figkws,
    )

    # dotplot
    fig = sc.pl.dotplot(
        a2,
        a2.var.index[marker_order],
        groupby=f"metacluster_labels_{res}",
        use_raw=False,
        show=False,
    )["mainplot_ax"].figure
    fig.savefig(
        output_prefix + "phenotypes.metaclusters.expression.dotplot.svg",
        **figkws,
    )

    # Test for diffeerntial expression within each metacluster between disease groups
    metaclusters = a2.obs[f"metacluster_labels_{res}"].unique()
    _diff_res = list()
    for metacluster in metaclusters:
        a3 = a2[a2.obs[f"metacluster_labels_{res}"] == metacluster, :]
        a3.obs["phenotypes"] = a3.obs["sample"].replace(phenotypes)
        # sc.pl.heatmap(a3, a3.var.index, "phenotypes", use_raw=False, dendrogram=True)
        sc.tl.rank_genes_groups(
            a3,
            "phenotypes",
            use_raw=False,
            reference="Healthy",
            method="t-test_overestim_var",
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
                    ).assign(metacluster=metacluster, group=group)
                    for group in ["COVID19_late", "COVID19_early"]
                ]
            )
        )

    diff_res = pd.concat(_diff_res)
    diff_res.to_csv(
        output_prefix
        + "phenotypes.metaclusters.expression.differential_testing.csv"
    )
    diff_res = diff_res.dropna()
    diff_res["-logp"] = -np.log10(diff_res["pvals"])
    v = diff_res["-logp"].replace(np.inf, np.nan).dropna().max()
    diff_res["-logp"] = diff_res["-logp"].replace(np.inf, v)

    diff_res["marker"] = diff_res["marker"].str.split("(").apply(lambda x: x[0])

    # Volcano plots
    fig, axes = plt.subplots(
        3 * 2, 3, figsize=(3 * 2.7, 3 * 2.7 * 2), sharex=False, sharey=False
    )
    for ax in axes.flat:
        ax.axvline(0, linestyle="--", color="grey")
    for metacluster, ax in zip(metaclusters, axes[:3].flat):
        p = diff_res.query(
            f"metacluster == '{metacluster}' & group == 'COVID19_early'"
        )
        ax.scatter(
            p["logfoldchanges"],
            p["-logp"],
            s=10,
            alpha=0.5,
            c=p["logfoldchanges"],
            cmap="coolwarm",
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
            ax.text(
                p.loc[x, "logfoldchanges"],
                p.loc[x, "-logp"],
                s=p.loc[x, "marker"],
            )
    axes[2][-1].axis("off")
    for metacluster, ax in zip(metaclusters, axes[3:].flat):
        p = diff_res.query(
            f"metacluster == '{metacluster}' & group == 'COVID19_late'"
        )
        ax.scatter(
            p["logfoldchanges"],
            p["-logp"],
            s=10,
            alpha=0.5,
            c=p["logfoldchanges"],
            cmap="coolwarm",
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
            ax.text(
                p.loc[x, "logfoldchanges"],
                p.loc[x, "-logp"],
                s=p.loc[x, "marker"],
            )
    axes[-1][-1].axis("off")
    axes[2][0].set(ylabel="-log(p-value)")
    axes[-1][1].set(xlabel="log(fold-change)")
    fig.savefig(
        output_prefix
        + "phenotypes.metaclusters.expression.differential_testing.volcano_plots.svg",
        **figkws,
    )

    #

    # Violinplots
    diff_res = pd.read_csv(
        output_prefix
        + "phenotypes.metaclusters.expression.differential_testing.csv",
        index_col=0,
    )
    diff_res = diff_res.dropna()
    diff_res["-logp"] = -np.log10(diff_res["pvals"])
    v = diff_res["-logp"].replace(np.inf, np.nan).dropna().max()
    diff_res["-logp"] = diff_res["-logp"].replace(np.inf, v)

    metaclusters = a2.obs[f"metacluster_labels_{res}"].unique()
    for metacluster in metaclusters:
        a3 = a2[a2.obs[f"metacluster_labels_{res}"] == metacluster, :]
        a3.obs["phenotypes"] = a3.obs["sample"].replace(phenotypes)

        fig = sc.pl.violin(
            a3,
            a3.var.index,
            groupby="phenotypes",
            use_raw=False,
            stripplot=False,
            multi_panel=True,
            order=phenotypes.cat.categories,
            show=False,
        )[0].figure
        fig.savefig(
            output_prefix
            + f"phenotypes.metaclusters.expression.{metacluster}.violinplots.svg",
            **figkws,
        )
        for group in ["COVID19_early", "COVID19_late"]:
            p = diff_res.query(
                f"metacluster == '{metacluster}' & group == '{group}'"
            )
            top = p.loc[
                (
                    p[["logfoldchanges", "-logp"]]
                    .abs()
                    .apply(z_score)
                    .mean(1)
                    .sort_values()
                    .tail(5)
                    .index
                ),
                "marker",
            ][::-1]
            fig = sc.pl.violin(
                a3,
                top,
                groupby="phenotypes",
                use_raw=False,
                stripplot=False,
                multi_panel=True,
                order=phenotypes.cat.categories,
                show=False,
            )[0].figure
            fig.savefig(
                output_prefix
                + f"phenotypes.metaclusters.expression.{metacluster}.violinplots.top_diff_{group}.svg",
                **figkws,
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
        fig.savefig(output_prefix + "3d_scatter.svg", **figkws)

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
        fig.savefig(output_prefix + "per_variable_histogram.svg", **figkws)

    return to_filter


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
