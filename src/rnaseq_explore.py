# coding: utf-8

import sys
from typing import Tuple
from functools import wraps

import parmap
import numpy as np
import pandas as pd
import imageio
import PIL
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import scanpy as sc
import gseapy as gp

from ngs_toolkit.general import enrichr, query_biomart

from seaborn_extensions import activate_annotated_clustermap, swarmboxenplot

from imc.types import Path, DataFrame, Series
from imc.graphics import get_grid_dims


activate_annotated_clustermap()

PIL.Image.MAX_IMAGE_PIXELS = 933120000

figkws = dict(dpi=300, bbox_inches="tight")
data_dir = Path("data") / "rna-seq"
output_dir = Path("results") / "rna-seq"
output_dir.mkdir(exist_ok=True)

database_dir = Path("data") / "gene_set_libraries"
scrnaseq_dir = Path("data") / "krasnow_scrna-seq"

CLINVARS = [
    "disease",
    "phenotypes",
    "pcr_spike_positive",
    # "days_of_disease",
    # "gender",
]

CELL_GROUPS = {
    "Basal": "Basal|Goblet",
    "Endothelial": "Artery|Vein|Vessel|Capilary",
    "Vascular": "Fibroblast|Pericyte|Smooth|Fibromyocyte|Myofibroblast",
    "Epithelial": "Epithelial",
    "Myeloid": "Monocyte|Macrophage|Dendritic",
    "Ciliated": "Ciliated|Ionocyte",
    "Lymphoid": "CD4|CD8|Plasma|B",
}


def main() -> int:
    # load data and plot sample correlations
    x, y = load_X_Y_data()
    plot_correlation(x, y)
    # # map to gene names
    # g = map_x_to_gene_name_space(x)
    # In the current dataset, the genes are already in gene name space
    g = x

    # map to signature space
    enr = rnaseq_to_ssgsea_hallmark_space(g)
    c = rnaseq_to_ssgsea_cell_type_space(g)

    # plot signature enrichments
    plot_ssGSEA(enr, c, y)
    agg = aggregate_ssGSEA_by_imc_cell_types(c)
    plot_aggregated_ssGSEA_by_imc_cell_types(agg, y)
    unsupervised_ssGSEA_space(agg, y)

    # use scRNA-seq as reference rather than gene sets
    dg = get_hca_lung_reference(g.index.to_series())
    inspect_hca_lung_reference(dg)

    use_hca_lung_reference(g, dg, y)

    return 0


def close_plots(func):
    """
    Decorator to close all plots on function exit.
    """

    @wraps(func)
    def close(*args, **kwargs):
        func(*args, **kwargs)
        plt.close("all")

    return close


def load_X_Y_data() -> Tuple[DataFrame, DataFrame]:
    """
    Read up expression data (X), metadata (Y) and match them by index.
    """

    # metadata and data
    x = pd.read_table(
        data_dir / "AutopsyRNAseq_Lung_counts.txt.gz", index_col=0
    )
    y = pd.read_table(
        data_dir / "AutopsyRNAseq_Lung_design.txt.gz", index_col=0
    )

    y["sample_id"] = y.index
    y["disease"] = y["STATUS"]

    # in the absence of more clinical data for the new sample set
    # we'll use a proxy for time since disease start as whether
    # the patient has cleared the virus or not
    y["pcr_spike_positive"] = y["Spike"]
    y["phenotypes"] = y["Spike"].isna().replace(True, np.nan) + (
        y["Spike"] == "Positive"
    ).astype(float)
    y["phenotypes"] = y["phenotypes"].replace({1: "Early", 0: "Late"})
    y.loc[y["disease"] == "Control", "phenotypes"] = "Control"

    y["phenotypes"] = pd.Categorical(
        y["phenotypes"], ordered=True, categories=["Control", "Early", "Late"],
    )

    # align indeces of data and metadata
    x = x.reindex(columns=y.index)
    return x, y


def map_x_to_gene_name_space(x: DataFrame) -> DataFrame:
    # map to gene symbols by max isoform expression
    xx = x.groupby(x.index.str.extract(r"(.*)\.\d+")[0].values).max()
    gene_map = query_biomart()
    g = (
        xx.join(gene_map.set_index("ensembl_gene_id")["external_gene_name"])
        .groupby("external_gene_name")
        .max()
    )
    return g


@close_plots
def rnaseq_unsupervised(x, y) -> None:
    a = AnnData(x.T, obs=y.assign(nreads=x.sum()))
    sc.pp.log1p(a)
    sc.pp.normalize_total(a)
    sc.pp.scale(a)
    sc.tl.pca(a)
    sc.pp.neighbors(a)
    sc.tl.umap(a)

    fig = sc.pl.pca(a, color=["nreads"] + CLINVARS, s=150, show=False)[0].figure
    fig.savefig(output_dir / "gene_expression.pca.svg")

    fig = sc.pl.umap(a, color=["nreads"] + CLINVARS, s=150, show=False)[
        0
    ].figure
    fig.savefig(output_dir / "gene_expression.umap.svg")


def ssgsea(sample: str, database: str, x: DataFrame) -> DataFrame:
    res = gp.ssgsea(x.loc[:, sample], database)
    return res.resultsOnSamples["sample1"]


def rnaseq_to_ssgsea_hallmark_space(g) -> DataFrame:
    db = (database_dir / "h.all.v7.2.symbols.gmt").as_posix()
    output_file = output_dir / "rna-seq.ssGSEA_enrichment.h.all.v7.2.csv"
    if not output_file.exists():
        r = parmap.map(ssgsea, g.columns, database=db, x=g)
        enr = pd.concat(r, axis=1)
        enr.columns = g.columns
        enr.to_csv(output_file)
    enr = pd.read_csv(output_file, index_col=0)
    return enr


def rnaseq_to_ssgsea_cell_type_space(g) -> DataFrame:
    db = (database_dir / "c8.all.v7.2.symbols.gmt").as_posix()
    output_file = output_dir / "rna-seq.ssGSEA_enrichment.c8.all.v7.2.csv"
    if not output_file.exists():
        r = parmap.map(ssgsea, g.columns, database=db, x=g)
        ct = pd.concat(r, axis=1)
        ct.columns = g.columns
        ct.to_csv(output_file)
    ct = pd.read_csv(output_file, index_col=0)  # shape: (272, 14)

    db = (database_dir / "scsig.all.v1.0.1.symbols.gmt").as_posix()
    output_file = output_dir / "rna-seq.ssGSEA_enrichment.scsig.all.v1.0.1.csv"
    if not output_file.exists():
        r = parmap.map(ssgsea, g.columns, database=db, x=g)
        ct2 = pd.concat(r, axis=1)
        ct2.columns = g.columns
        ct2.to_csv(output_file)
    ct2 = pd.read_csv(output_file, index_col=0)  # shape: (233, 14)

    return pd.concat([ct, ct2])


def filter_samples_by_enrichment_threshold(df, enr, threshold=5):
    """
    Remove samples(s) which sum of enrichments is `threshold` stds from mean.

    Assumes (features, samples) shape.
    # this was only one sample: "CA_Lu_8"
    """
    s = enr.sum(0)
    sz = (s - s.mean()) / s.std()
    sel = sz[sz.abs() < 5].index
    return df.reindex(columns=sel)


@close_plots
def plot_correlation(x, y, enr=None) -> None:
    if enr is not None:
        x = filter_samples_by_enrichment_threshold(x, enr)
    df2 = np.log1p(x)
    df3 = (df2 / df2.sum()) * 1e4

    corrs = df3.corr()

    v = corrs.values.min()
    v -= v * 0.1
    grid = sns.clustermap(
        corrs,
        vmin=v,
        cmap="RdBu_r",
        xticklabels=False,
        yticklabels=True,
        metric="correlation",
        col_colors=y[CLINVARS],
        cbar_kws=dict(label="Pearson correlation"),
        dendrogram_ratio=0.1,
    )
    grid.savefig(
        output_dir / "rna-seq.pairwise_correlation.clustermap.svg", **figkws,
    )


@close_plots
def plot_ssGSEA(enr, c, y) -> None:
    grid = sns.clustermap(
        enr,
        col_colors=y[CLINVARS],
        robust=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws=dict(label="ssGSEA score\n(Z-score)"),
    )
    grid.savefig(
        output_dir / "ssGSEA_enrichment.hallmark.all_ROIs.clustermap.svg",
        **figkws,
    )
    grid = sns.clustermap(
        enr,
        center=0,
        cmap="RdBu_r",
        row_colors=enr.mean(1).rename("Mean ssGSEA score"),
        col_colors=y[CLINVARS],
        z_score=0,
        metric="correlation",
        robust=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws=dict(label="ssGSEA score\n(Z-score)"),
    )
    grid.savefig(
        output_dir
        / "ssGSEA_enrichment.hallmark.all_ROIs.clustermap.z_score.svg",
        **figkws,
    )

    # One example
    fig, stats = swarmboxenplot(
        data=enr.T.join(y),
        x="phenotypes",
        y="HALLMARK_IL6_JAK_STAT3_SIGNALING",
        # hue='phenotypes',
        test=True,
        test_kws=dict(parametric=False),
    )
    fig.savefig(
        output_dir / "ssGSEA_enrichment.IL6_signature.swarmboxenplot.svg",
        **figkws,
    )

    enr = filter_samples_by_enrichment_threshold(enr, enr)

    n, m = get_grid_dims(len(enr.index))
    fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 4), sharex=True)
    _stats = list()
    for i, _path in enumerate(enr.index):
        ax = axes.flatten()[i]
        s = swarmboxenplot(
            data=enr.T.join(y),
            x="phenotypes",
            y=_path,
            test_kws=dict(parametric=False),
            ax=ax,
        )
        ax.set(title=_path, xlabel=None, ylabel=None)
        _stats.append(s.assign(signature=_path))
    for ax in axes.flatten()[i + 1 :]:
        ax.axis("off")
    fig.savefig(
        output_dir / "ssGSEA_enrichment.hallmark_signatures.swarmboxenplot.svg",
        **figkws,
    )
    stats = pd.concat(_stats)
    stats.to_csv(
        output_dir
        / "ssGSEA_enrichment.hallmark_signatures.mann-whitney_test.csv",
        index=False,
    )

    ct = c.iloc[:272, :]
    ct2 = c.iloc[272:, :]

    kws = dict(
        center=0,
        cmap="RdBu_r",
        cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
        rasterized=True,
        z_score=0,
        metric="correlation",
        col_colors=y[CLINVARS],
    )
    for d, label in [
        (ct, "c8.all.v7.2"),
        (ct2, "all.v1.0.1"),
        (c, "cell_type"),
    ]:
        grid = sns.clustermap(
            d, cbar_kws=dict(label="ssGSEA enrichment"), rasterized=True,
        )
        grid.savefig(
            output_dir / f"ssGSEA_enrichment.{label}.all_ROIs.clustermap.svg",
            **figkws,
        )

        grid = sns.clustermap(d, **kws)
        grid.savefig(
            output_dir
            / f"ssGSEA_enrichment.{label}.all_ROIs.clustermap.zscore.svg",
            **figkws,
        )


def aggregate_ssGSEA_by_imc_cell_types(c: DataFrame) -> DataFrame:
    cells = [
        "Epithelial",
        "Mesenchymal",
        "Fibroblast",
        "Smooth_muscle",
        "Club",
        "CD4_T",
        "CD8_T",
        "NK_cell",
        "Macrophage",
        "Monocyte",
        "Neutrophil",
        "B_cell",
        "Mast",
        "Dendritic",
    ]
    _agg = dict()
    for cell in cells:
        _agg[cell] = c.loc[c.index.str.contains(cell, case=False)].mean()
    agg = pd.DataFrame(_agg)
    return agg


@close_plots
def plot_aggregated_ssGSEA_by_imc_cell_types(agg, y) -> None:

    agg = filter_samples_by_enrichment_threshold(agg.T, agg.T).dropna().T

    grid = sns.clustermap(
        agg.T.dropna(),
        col_colors=y[CLINVARS],
        cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
        figsize=(5, 4),
        xticklabels=False,
    )
    grid.savefig(
        output_dir
        / "ssGSEA_enrichment.cell_type_aggregated.all_ROIs.clustermap.svg",
        **figkws,
    )
    grid = sns.clustermap(
        agg.T.dropna(),
        z_score=0,
        metric="correlation",
        center=0,
        cmap="RdBu_r",
        col_colors=y[CLINVARS],
        cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
        figsize=(5, 4),
        xticklabels=False,
    )
    grid.savefig(
        output_dir
        / "ssGSEA_enrichment.cell_type_aggregated.all_ROIs.clustermap.z_score.svg",
        **figkws,
    )

    n, m = get_grid_dims(agg.shape[1])
    fig, axes = plt.subplots(n, m, figsize=(m * 3, n * 3))
    for i, c in enumerate(agg.columns):
        swarmboxenplot(
            data=agg[[c]].join(y), x="phenotypes", y=c, ax=axes.flatten()[i]
        )
        axes.flatten()[i].set(title=c)
    for ax in axes.flat[i + 1 :]:
        ax.axis("off")

    fig.savefig(
        output_dir
        / "ssGSEA_enrichment.cell_type_aggregated.all_ROIs.swarmboxenplot.svg",
        **figkws,
    )

    # Scatter days of disease vs signatures
    # n, m = get_grid_dims(agg.shape[1])
    # fig, axes = plt.subplots(n, m, figsize=(m * 3, n * 3))
    # for i, c in enumerate(agg.columns):
    #     ax = axes.flatten()[i]
    #     p = agg.join(y[["disease", "days_of_disease"]])
    #     p.loc[p["disease"] == "Control", "days_of_disease"] = 0
    #     ax.scatter(p["days_of_disease"], p[c])
    #     ax.set(title=c)


@close_plots
def unsupervised_ssGSEA_space(agg, y) -> None:
    # Unsupervised dimres

    agg = filter_samples_by_enrichment_threshold(agg.T, agg.T).dropna().T

    # # signature space
    a = AnnData(agg, obs=agg.join(y))
    sc.tl.pca(a)

    fig = sc.pl.pca(
        a,
        color=CLINVARS,
        components=["1,2", "2,3", "3,4", "4,5"],
        show=False,
        s=150,
    )[0].figure
    fig.savefig(output_dir / "signature_space.pca.svg")
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    fig = sc.pl.umap(a, color=CLINVARS, s=150, show=False)[0].figure
    fig.savefig(output_dir / "signature_space.umap.svg")

    # # cell type space
    aggz = ((agg.T - agg.mean(1)) / agg.std(1)).T

    a = AnnData(aggz, obs=aggz.join(y))
    sc.tl.pca(a)
    fig = sc.pl.pca(
        a,
        color=CLINVARS,
        components=["1,2", "2,3", "3,4", "4,5"],
        show=False,
        s=150,
    )[0].figure
    fig.savefig(output_dir / "cell_type_space.zscore.pca.svg")
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    fig = sc.pl.umap(a, color=CLINVARS, s=150, show=False)[0].figure
    fig.savefig(output_dir / "cell_type_space.zscore.umap.svg")


def get_hca_lung_reference(rnaseq_genes: Series) -> DataFrame:
    # Try to use scRNA-seq as reference
    mean_file = scrnaseq_dir / "krasnow_hlca_10x.average.expression.csv"
    mean = pd.read_csv(mean_file, index_col=0)

    zm = ((mean.T - mean.mean(1)) / mean.std(1)).T.dropna()

    rnaseq_genes[~rnaseq_genes.isin(zm.index)]

    dg = zm.reindex(rnaseq_genes).dropna()
    return dg


@close_plots
def inspect_hca_lung_reference(dg: DataFrame) -> None:
    _super_means = dict()
    for group, groupstr in CELL_GROUPS.items():
        i = dg.columns.str.contains(groupstr)
        _super_means["SG_" + group] = dg.loc[:, i].mean(1)
    super_means = pd.DataFrame(_super_means)

    super_corrs = (
        dg.join(super_means).corr().loc[dg.columns, super_means.columns]
    )

    for ext, colcolors in [("", None), (".with_supergroups", super_corrs)]:
        grid = sns.clustermap(
            dg.T,
            center=0,
            cmap="RdBu_r",
            yticklabels=True,
            robust=True,
            rasterized=True,
            dendrogram_ratio=0.1,
            cbar_kws=dict(label="Expression Z-score"),
            row_colors=colcolors,
        )
        grid.ax_heatmap.set(xlabel=f"RNA-seq genes only (n = {dg.shape[0]})")
        grid.savefig(
            output_dir / f"krasnow_scRNA_mean.clustermap{ext}.svg", **figkws
        )

    a = AnnData(dg.T, obs=super_corrs)
    sc.pp.pca(a)
    sc.pp.neighbors(a)
    sc.tl.umap(a, gamma=0.0001)
    sc.tl.umap(a)
    axes = sc.pl.umap(a, color=super_corrs.columns, show=False)
    for i, ax in enumerate(axes):
        for t, xy in zip(a.obs.index, a.obsm["X_umap"]):
            ax.text(*xy, s=t, fontsize=4)
    fig = axes[0].figure
    fig.savefig(output_dir / f"krasnow_scRNA_mean.umap.svg", **figkws)

    # Same clustermaps as above, less genes
    sc.pp.highly_variable_genes(a)
    dgv = dg.loc[a.var["highly_variable"], :]
    for ext, colcolors in [("", None), (".with_supergroups", super_corrs)]:
        grid = sns.clustermap(
            dgv.T,
            center=0,
            cmap="RdBu_r",
            yticklabels=True,
            robust=True,
            rasterized=True,
            dendrogram_ratio=0.1,
            cbar_kws=dict(label="Expression Z-score"),
            row_colors=colcolors,
        )
        grid.ax_heatmap.set(
            xlabel=f"RNA-seq genes, highly variable (n = {dgv.shape[0]})"
        )
        grid.savefig(
            output_dir
            / f"krasnow_scRNA_mean.highly_variable.clustermap{ext}.svg",
            **figkws,
        )


@close_plots
def use_hca_lung_reference(g, dg, y) -> None:
    # Try to deconvolve
    output_prefix = output_dir / "krasnow_scRNA_deconvolve.correlation"
    dc = g.join(dg).corr().loc[g.columns, dg.columns]

    sign = (dc > 0).astype(int).replace(0, -1)
    dcs = dc.abs() ** (1 / 3) * sign

    kws = dict(
        yticklabels=True,
        robust=True,
        figsize=(5, 10),
        col_colors=y[CLINVARS],
        metric="correlation",
        cbar_kws=dict(label="Pearson correlation"),
        dendrogram_ratio=0.1,
        rasterized=True,
    )

    grid = sns.clustermap(dc.T, center=0, cmap="RdBu_r", **kws)
    grid.savefig(
        output_prefix + ".clustermap.svg", **figkws,
    )

    grid = sns.clustermap(
        dc.T, standard_scale=1, center=0, cmap="RdBu_r", **kws
    )
    grid.savefig(
        output_prefix + ".clustermap.std_scale.svg", **figkws,
    )

    p = (dcs.T + 1) / 2
    p -= p.min()
    normdcs = p / p.sum()

    grid = sns.clustermap(normdcs * 100, **kws)
    grid.savefig(
        output_prefix + ".clustermap.norm.svg", **figkws,
    )

    a = AnnData(dcs, obs=y)
    sc.pp.pca(a)
    sc.pp.neighbors(a)
    sc.tl.umap(a)  # , gamma=0.0001)
    axes = sc.pl.pca(a, color=CLINVARS, show=False)
    fig = axes[0].figure
    fig.savefig(output_prefix + ".pca.svg", **figkws)

    axes = sc.pl.umap(a, color=CLINVARS, show=False)
    fig = axes[0].figure
    fig.savefig(output_prefix + ".umap.svg", **figkws)

    colors = {
        "phenotypes": np.asarray(sns.color_palette("tab10"))[[2, 0, 1, 5, 4, 3]]
    }

    n, m = get_grid_dims(dcs.shape[1])
    fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 4), sharex=True)
    _stats = list()
    for i, c in enumerate(dcs.columns):
        ax = axes.flatten()[i]
        stats = swarmboxenplot(
            data=dcs[[c]].join(y),
            x="phenotypes",
            y=c,
            ax=ax,
            plot_kws=dict(palette=colors["phenotypes"]),
            test_kws=dict(parametric=False),
        )
        ax.set(title=c, xlabel=None, ylabel=True)
        means = (
            dcs[[c]]
            .join(y["phenotypes"])
            .groupby("phenotypes")
            .mean()
            .squeeze()
        )
        _stats.append(stats.assign(cell_type=c, **means.to_dict()))
    for ax in axes.flatten()[i + 1 :]:
        ax.axis("off")
    fig.savefig(
        output_prefix + f".correlation.swarmboxenplot.svg", **figkws,
    )
    stats = pd.concat(_stats).reset_index(drop=True)
    stats.to_csv(output_prefix + ".correlation.csv", index=False)

    ### volcano plot
    combs = stats[["A", "B"]].drop_duplicates().reset_index(drop=True)
    stats["hedges"] *= -1
    stats["logp-unc"] = -np.log10(stats["p-unc"].fillna(1))
    stats["logp-cor"] = -np.log10(stats["p-cor"].fillna(1))
    stats["p-cor-plot"] = (stats["logp-cor"] / stats["logp-cor"].max()) * 5
    n, m = get_grid_dims(combs.shape[0])
    fig, axes = plt.subplots(n, m, figsize=(4 * m, 4 * n), squeeze=False)
    for idx, (a, b) in combs.iterrows():
        ax = axes.flatten()[idx]
        p = stats.query(f"A == '{a}' & B == '{b}'")
        ax.axvline(0, linestyle="--", color="grey")
        ax.scatter(
            p["hedges"],
            p["logp-unc"],
            c=p["hedges"],
            s=5 + (2 ** p["p-cor-plot"]),
            cmap="coolwarm",
        )
        ax.set(title=f"{b} / {a}", ylabel=None, xlabel=None)
        for t in p.query(f"`p-cor` < 0.05").index:
            ax.text(
                p.loc[t, "hedges"],
                p.loc[t, "logp-unc"],
                s=p.loc[t, "cell_type"],
                ha="right" if p.loc[t, "hedges"] > 0 else "left",
            )
    for ax in axes.flatten()[idx + 1 :]:
        ax.axis("off")

    for ax in axes[:, 0]:
        ax.set(ylabel="-log10(p-val)")
    for ax in axes[-1, :]:
        ax.set(xlabel="Hedges' g")
    fig.savefig(output_prefix + ".volcano.svg", **figkws)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\t - Exiting due to user interruption.")
        sys.exit(1)
