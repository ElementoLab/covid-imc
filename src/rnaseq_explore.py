# coding: utf-8

import sys
from typing import Tuple

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

CLINVARS = [
    "disease",
    "phenotypes",
    "pcr_spike_positive",
    "days_of_disease",
    "gender",
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


def ssgsea(sample: str, database: str, x: DataFrame) -> DataFrame:
    res = gp.ssgsea(x.loc[:, sample], database)
    return res.resultsOnSamples["sample1"]


def main():
    # load data and plot sample correlations
    x, y = load_X_Y_data()
    plot_correlation(x, y)
    # map to gene names
    g = map_x_to_gene_name_space(x)

    # map to signature space
    enr = geomx_to_ssgsea_hallmark_space(g)
    c = geomx_to_ssgsea_cell_type_space(g)

    # plot signature enrichments
    plot_ssGSEA(enr, c, y)
    agg = aggregate_ssGSEA_by_imc_cell_types(c)
    plot_aggregated_ssGSEA(agg, y)

    # use scRNA-seq as reference rather than gene sets
    dg = get_hca_lung_reference(g.index.to_series())
    inspect_hca_lung_reference(dg)

    use_hca_lung_reference(g, dg, y)


def load_X_Y_data() -> Tuple[DataFrame, DataFrame]:
    """
    Read up expression data (X), metadata (Y) and match them by index.
    """

    # metadata and data
    x = pd.read_table(data_dir / "RNAseq_counts_forIMC.txt", index_col=0)
    y = (
        pd.read_table(data_dir / "RNAseq_metadata_forIMC.txt")
        .set_index("SeqID")
        .sort_index()
    )
    y["sample_id"] = y["SampleID"]
    y["disease"] = y["STATUS"]

    # # load extra metadata from patient files
    y_extra = pd.read_excel(
        "metadata/original/Hyperion samples.original.20200820.xlsx",
        na_values=["na"],
    )
    y_extra["Autopsy Code"] = y_extra["Autopsy Code"].str.capitalize()
    y_extra["pcr_spike_positive"] = y_extra[
        "Sample Classification"
    ].str.endswith("Positive")
    y_extra["phenotypes"] = (
        y_extra["Days of disease"].replace("na", np.nan).convert_dtypes() < 30
    ).replace({True: "Early", False: "Late"})

    y_extra = y_extra.rename(
        columns={
            "Days of disease": "days_of_disease",
            "Days Intubated": "days_intubated",
            "lung WEIGHT g": "lung_weight_grams",
            "AGE (years)": "age_years",
            "GENDER (M/F)": "gender",
            "RACE": "race",
            "SMOKE (Y/N)": "smoker",
            "Fever (Tmax)": "fever",
            "Cough": "cough",
            "Shortness of breath": "shortness_of_breath",
            "COMORBIDITY (Y/N; spec)": "comorbidities",
            "TREATMENT": "treatment",
        }
    )

    cols = [
        "Autopsy Code",
        "phenotypes",
        "pcr_spike_positive",
        "days_of_disease",
        "days_intubated",
        "days_intubated",
        "lung_weight_grams",
        "age_years",
        "gender",
        "race",
        "smoker",
        "fever",
        "cough",
        "shortness_of_breath",
        "comorbidities",
        "treatment",
        "PLT/mL",
        "D-dimer (mg/L)",
        "WBC",
        "LY%",
        "PMN %",
    ]

    # merge metadata
    y = (
        y.reset_index()
        .merge(
            y_extra[cols],
            left_on="AutopsyCode",
            right_on="Autopsy Code",
            how="left",
        )
        .set_index("SeqID")
    )
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


def geomx_unsupervised(x, y) -> None:
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


def geomx_to_ssgsea_hallmark_space(g) -> DataFrame:
    output_file = output_dir / "rna-seq.ssGSEA_enrichment.h.all.v7.2.csv"
    if not output_file.exists():
        r = parmap.map(
            ssgsea, g.columns, database="h.all.v7.2.symbols.gmt", x=g
        )
        enr = pd.concat(r, axis=1)
        enr.columns = g.columns
        enr.to_csv(output_file)
    enr = pd.read_csv(output_file, index_col=0)
    return enr


def geomx_to_ssgsea_cell_type_space(g) -> DataFrame:
    output_file = output_dir / "rna-seq.ssGSEA_enrichment.c8.all.v7.2.csv"
    if not output_file.exists():
        r = parmap.map(
            ssgsea, g.columns, database="c8.all.v7.2.symbols.gmt", x=g
        )
        ct = pd.concat(r, axis=1)
        ct.columns = g.columns
        ct.to_csv(output_file)
    ct = pd.read_csv(output_file, index_col=0)  # shape: (272, 14)

    output_file = output_dir / "rna-seq.ssGSEA_enrichment.scsig.all.v1.0.1.csv"
    if not output_file.exists():
        r = parmap.map(
            ssgsea, g.columns, database="scsig.all.v1.0.1.symbols.gmt", x=g
        )
        ct2 = pd.concat(r, axis=1)
        ct2.columns = g.columns
        ct2.to_csv(output_file)
    ct2 = pd.read_csv(output_file, index_col=0)  # shape: (233, 14)

    return pd.concat([ct, ct2])


def plot_correlation(x, y) -> None:
    df2 = np.log1p(x)
    df3 = (df2 / df2.sum()) * 1e4

    # Intra patient, inter location correlation

    means = (
        df3.T.join(y[["disease", "SampleID"]])
        .groupby(["disease", "SampleID"])
        .mean()
        .T
    )

    corrs = means.corr()

    v = corrs.values.min()
    v -= v * 0.1
    grid = sns.clustermap(corrs, vmin=v, cmap="RdBu_r",)
    # sns.histplot(x.mean())


def plot_ssGSEA(enr, c, y) -> None:
    grid = sns.clustermap(
        enr,
        center=0,
        cmap="RdBu_r",
        # col_colors=y[CLINVARS],
        z_score=0,
        metric="correlation",
    )
    grid.savefig(
        output_dir / "ssGSEA_enrichment.h.all.v7.2.all_ROIs.clustermap.svg",
        **figkws,
    )

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

    ct = c.iloc[:272, :]
    ct2 = c.iloc[272:, :]

    kws = dict(
        center=0,
        cmap="RdBu_r",
        cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
        rasterized=True,
        z_score=0,
    )
    for d, label in [
        (ct, "c8.all.v7.2"),
        (ct2, "all.v1.0.1"),
        (c, "cell_type"),
    ]:
        grid = sns.clustermap(d, **kws)
        grid.savefig(
            output_dir / f"ssGSEA_enrichment.{label}.all_ROIs.clustermap.svg",
            **figkws,
        )

    grid = sns.clustermap(
        ((c.T - c.mean(1)) / c.std(1)).T,
        center=0,
        cmap="RdBu_r",
        # col_colors=y[CLINVARS],
        cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
        rasterized=True,
    )
    grid.savefig(
        output_dir
        / "ssGSEA_enrichment.cell_type.all_ROIs.clustermap.z_score.svg",
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


def plot_aggregated_ssGSEA(agg, y) -> None:
    grid = sns.clustermap(
        agg.T.dropna(),
        metric="correlation",
        center=0,
        cmap="RdBu_r",
        # col_colors=y[CLINVARS],
        cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
        figsize=(5, 4),
        xticklabels=False,
    )
    grid.savefig(
        output_dir
        / "ssGSEA_enrichment.cell_type_aggregated.all_ROIs.clustermap.svg",
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

    n, m = get_grid_dims(agg.shape[1])
    fig, axes = plt.subplots(n, m, figsize=(m * 3, n * 3))
    for i, c in enumerate(agg.columns):
        ax = axes.flatten()[i]
        p = agg.join(y[["disease", "days_of_disease"]])
        p.loc[p["disease"] == "Control", "days_of_disease"] = 0
        ax.scatter(p["days_of_disease"], p[c])
        ax.set(title=c)

    grid = sns.clustermap(
        agg.T.dropna(),
        metric="correlation",
        center=0,
        cmap="RdBu_r",
        # col_colors=y[CLINVARS],
        cbar_kws=dict(label="ssGSEA enrichment\nZ-score)"),
        # figsize=(5, 4),
    )

    grid.savefig(
        output_dir
        / "ssGSEA_enrichment.cell_type_aggregated.sample_reduced.all_ROIs.clustermap.svg",
        **figkws,
    )


def unsupervised_ssGSEA_space(agg, y) -> None:
    # Unsupervised dimres

    # # signature space
    a = AnnData(agg.T.dropna().T, obs=agg.join(y))
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
    aggz = (agg.T - agg.mean(1)) / agg.std(1)

    a = AnnData(aggz.T.dropna().T, obs=aggz.join(y))
    sc.tl.pca(a)
    fig = sc.pl.pca(
        a,
        color=CLINVARS,
        components=["1,2", "2,3", "3,4", "4,5"],
        show=False,
        s=150,
    )[0].figure
    fig.savefig(output_dir / "cell_type_space.pca.svg")
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    fig = sc.pl.umap(a, color=CLINVARS, s=150, show=False)[0].figure
    fig.savefig(output_dir / "cell_type_space.umap.svg")


def get_hca_lung_reference(geomx_genes: Series) -> DataFrame:
    # Try to use scRNA-seq as reference
    input_dir = Path("~/Downloads").expanduser()
    mean_file = input_dir / "krasnow_hlca_10x.average.expression.csv"
    mean = pd.read_csv(mean_file, index_col=0)

    zm = ((mean.T - mean.mean(1)) / mean.std(1)).T.dropna()

    geomx_genes[~geomx_genes.isin(zm.index)]

    dg = zm.reindex(geomx_genes).dropna()
    return dg


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
        col_colors=y[["phenotypes", "days_of_disease", "Spike"]],
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
    axes = sc.pl.pca(
        a, color=["phenotypes", "days_of_disease", "Spike"], show=False
    )
    fig = axes[0].figure
    fig.savefig(output_prefix + ".pca.svg", **figkws)

    axes = sc.pl.umap(
        a, color=["phenotypes", "days_of_disease", "Spike"], show=False
    )
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
    fig, axes = plt.subplots(n, m, figsize=(4 * m, 4 * n))
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
