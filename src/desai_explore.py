# coding: utf-8

import io
import sys
import json
from functools import wraps
from typing import Tuple

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parmap
import requests
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from imc.graphics import get_grid_dims
from imc.types import DataFrame, Path, Series
from ngs_toolkit.general import enrichr, query_biomart
from seaborn_extensions import (
    activate_annotated_clustermap,
    clustermap,
    swarmboxenplot,
    volcano_plot,
)

# activate_annotated_clustermap()

figkws = dict(dpi=300, bbox_inches="tight")
# plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['savefig.bbox_inches']

metadata_dir = Path("metadata")
metadata_dir.mkdir(exist_ok=True)
data_dir = Path("data") / "desai"
data_dir.mkdir(exist_ok=True)
output_dir = Path("results") / "desai"
output_dir.mkdir(exist_ok=True)

database_dir = Path("data") / "gene_set_libraries"
scrnaseq_dir = Path("data") / "krasnow_scrna-seq"


config = json.load(open(metadata_dir / "desai.config.json", "r"))
colors = {"phenotypes": np.asarray(sns.color_palette("tab10"))[[2, 4, 3]]}


def main() -> int:
    # get, parse and assemble data from "source data" files
    full_meta = get_source_data()

    # get deconvolution predictions
    dx, dy = get_geomx_deconvolution()
    # # further augment metadata with source data
    dy = dy.merge(full_meta, left_on="case_id", right_index=True)
    # # plot
    plot_deconvolution(dx, dy)

    # Bulk RNA-seq data
    # # get bulk data
    bx, by = get_bulk_rna_data()
    # # further augment metadata with source data
    by = by.merge(full_meta, how="left", left_on="case_id", right_index=True)

    # # plot
    plot_bulk_unsupervised(bx, by)

    # load data and plot sample correlations
    gx, gy = get_geomx_X_Y_data()
    # # further augment metadata with source data
    gy = gy.merge(full_meta, left_on="case_id", right_index=True)

    # # plot
    plot_geomx_unsupervised(gx, gy)

    # Compare GeoMx data with IMC

    # ssGSEA space
    bz = to_ssGSEA_pathway_space(bx, "bulk_rna-seq")
    gz = to_ssGSEA_pathway_space(gx, "geomx")

    bct = to_ssGSEA_cell_type_space(bx, "bulk_rna-seq")

    # # plot
    plot_pathways(bz, by, "Bulk")
    plot_pathways(gz, gy, "geomx")

    plot_pathways(bct, by, "Bulk", space_type="cell_type")
    # TODO: aggregate cell type signatures
    # agg_bct = aggregate_cell_type_signatures(bct)
    # plot_pathways(agg_bct, by, "Bulk", space_type="aggregated_cell_type")

    # Try to use scRNA-seq cell type reference
    scrna_ref = get_hca_lung_reference(bx.index)

    return 0


def _download_source_data(source_data_file=data_dir / "source_data.xlsx"):
    if not source_data_file.exists():
        with requests.get(config["SOURCE_DATA_URL"]) as req:
            with open(source_data_file, "wb") as handle:
                handle.write(req.content)


def get_source_data() -> DataFrame:
    """
    Read up expression data (X), metadata (Y) and match them by index.
    """

    def fix_axis(df):
        df.columns = df.columns.str.strip()
        df.index = (
            df.index.str.strip().str.replace("cse", "case").str.capitalize()
        )
        return df.sort_index()

    def stack(df):
        cols = range(len(df.columns.levels[0]))
        return pd.concat(
            [
                df.loc[:, df.columns.levels[0][i]]
                .stack()
                .reset_index(level=1, drop=True)
                for i in cols
            ],
            axis=1,
        ).rename(columns=dict(zip(cols, df.columns.levels[0])))

    # Y: metadata
    source_data_file = data_dir / "source_data.xlsx"
    _download_source_data(source_data_file)

    # metadata and data
    viral_load = fix_axis(
        pd.read_excel(source_data_file, index_col=0, sheet_name="figure 1b")
    )
    time_since_symptoms = fix_axis(
        pd.read_excel(source_data_file, index_col=0, sheet_name="figure 1 d")
    )
    struc_cell_content = pd.read_excel(
        source_data_file, index_col=0, sheet_name="figure 1e", header=[0, 1]
    )
    struc_cell_content = fix_axis(stack(struc_cell_content))

    lymph_content = pd.read_excel(
        source_data_file, index_col=0, sheet_name="figure 4 b", header=[0, 1]
    )
    lymph_content = fix_axis(stack(lymph_content))

    myelo_content = pd.read_excel(
        source_data_file, index_col=0, sheet_name="figure 4 d"
    )
    myelo_content.columns = myelo_content.columns.str.replace(
        "Macrophages", "Macrophage"
    )
    myelo_content = myelo_content.loc[~myelo_content.isnull().all(1)]
    myelo_content.columns = pd.MultiIndex.from_frame(
        myelo_content.columns.str.extract(r"^(.*) ?.* (.*)$")
    )
    myelo_content = fix_axis(stack(myelo_content))
    myelo_content.columns = myelo_content.columns + " / %"
    myelo_content.index = (
        myelo_content.index.str.replace("Case ", "Case")
        .str.replace("Case", "Case ")
        .str.replace(" ", ".")
        .str.replace("-", ".")
        .str.replace(r"\.\.", ".")
        .str.extract(r"(Case\..*?\.).*")[0]
        .str.replace(r"\.", " ")
        .str.strip()
    )
    myelo_content_red = myelo_content.groupby(level=0).mean()

    deconv = pd.read_excel(
        source_data_file, index_col=0, sheet_name="Supplementary figure 5"
    ).T
    deconv = deconv.loc[~deconv.isnull().all(1)]
    deconv_red = deconv.copy()
    deconv_red.index = (
        (
            (deconv_red.index.str.replace("Case ", "Case")).str.extract(
                r"(.*)?-"
            )[0]
        )
        .fillna(pd.Series(deconv_red.index))
        .str.replace("Case", "Case ")
        .str.replace("NegControl", "NegControl ")
        .str.strip()
    )
    deconv_red = fix_axis(deconv_red)
    deconv_red = deconv_red.groupby(level=0).mean()

    diffs = pd.read_excel(source_data_file, sheet_name="figure 7 a")
    effectsize = diffs.pivot_table(index="gene", columns="tiss", values="est")
    signif = diffs.pivot_table(index="gene", columns="tiss", values="fdr")
    signif = -np.log10(signif)

    full_meta = pd.concat(
        [
            time_since_symptoms,
            viral_load,
            struc_cell_content,
            lymph_content,
            myelo_content_red,
            deconv_red,
        ],
        axis=1,
    )
    full_meta.index.name = "case_id"
    full_meta.index = full_meta.index.str.replace("Negcontrol", "Control")
    # match metadata var names to our data
    full_meta["disease"] = (
        full_meta.index.to_series()
        .str.contains("Case")
        .replace({True: "COVID19", False: "Healthy"})
    )
    full_meta["phenotypes"] = (
        full_meta["Duration (days)"].convert_dtypes() < 15
    ).replace({True: "Early", False: "Late"})
    full_meta.loc[full_meta["disease"] == "Healthy", "phenotypes"] = "Healthy"
    full_meta["pcr_spike_positive"] = full_meta["Virus High/Low"]
    full_meta["days_of_disease"] = full_meta["Duration (days)"]
    full_meta["phenotypes"] = pd.Categorical(
        full_meta["phenotypes"],
        ordered=True,
        categories=["Healthy", "Early", "Late"],
    )
    full_meta["pcr_spike_positive"] = pd.Categorical(
        full_meta["pcr_spike_positive"],
        ordered=True,
        categories=["Low", "High"],
    )
    return full_meta


def get_geomx_deconvolution() -> Tuple[DataFrame, DataFrame]:
    source_data_file = data_dir / "source_data.xlsx"
    _download_source_data(source_data_file)

    x = pd.read_excel(
        source_data_file, index_col=0, sheet_name="Supplementary figure 5"
    ).T
    x = x.loc[~x.isnull().all(1)]

    x = pd.read_table(config["DECONVOLUTION_URL"], index_col=0)
    x = x.loc[~x.isnull().all(1), :]

    # Y: metadata
    prjmeta, samplemeta = series_matrix2csv(config["RNA_SERIES_URL"])

    samplemeta["roi_id"] = samplemeta["description_2"].str.replace(".dcc", "")
    samplemeta["case_id"] = (
        samplemeta["characteristics_ch1"]
        .str.replace("case number: ", "")
        .str.extract(r"(.* \d+) ?")[0]
    )

    y = samplemeta.set_index("roi_id")[["case_id"]]

    # Match x and y
    y = y.reindex(x.columns)
    x = x.reindex(y.index, axis=1)
    return x, y


def get_bulk_rna_data() -> Tuple[DataFrame, DataFrame]:
    """
    Read up expression data (X), metadata (Y) and match them by index.
    """

    # X: expression data
    x = pd.read_table(config["BULK_RNA_DESEQ"]).sort_index()
    x.columns = (
        x.columns.str.replace("-NYC", "")
        .rename("sample_id")
        .str.replace("NegControl", "Control ")
    )

    # Y: metadata
    prjmeta, samplemeta = series_matrix2csv(config["BULK_RNA_SERIES_URL"])
    samplemeta["title"] = samplemeta["title"].str.replace(
        "NegControl", "Control "
    )

    assert samplemeta["title"].isin(x.columns).all()
    # assert x.columns.isin(samplemeta["title"]).all()
    # ['case3-liver1', 'case10-liver1', 'case12-liver1'] missing!
    x.columns[~x.columns.isin(samplemeta["title"])]

    # Expand name into components
    y = x.columns.str.split("-").to_series().apply(pd.Series)
    y.columns = ["case_id", "tissue"]
    y.index = x.columns
    y["case_id"] = y["case_id"].str.replace("case", "Case ").str.capitalize()
    y["replicate"] = y["tissue"].str.extract(r"(\d)")[0].fillna("1").astype(int)
    y["tissue"] = y["tissue"].str.extract(r"(\D+)")[0]

    # Join with metadata
    y = (
        y.reset_index()
        .merge(samplemeta, left_on="sample_id", right_on="title", how="left")
        .set_index("sample_id")
    )

    # Match x and y
    y = y.reindex(x.columns)
    x = x.reindex(y.index, axis=1)

    return x, y


def get_geomx_X_Y_data(
    data_type: str = "RNA",
) -> Tuple[DataFrame, DataFrame]:
    """
    Read up expression data (X), metadata (Y) and match them by index.
    """

    def fix_axis(df):
        df.columns = df.columns.str.strip()
        df.index = (
            df.index.str.strip().str.replace("cse", "case").str.capitalize()
        )
        return df.sort_index()

    def stack(df):
        cols = range(len(df.columns.levels[0]))
        return pd.concat(
            [
                df.loc[:, df.columns.levels[0][i]]
                .stack()
                .reset_index(level=1, drop=True)
                for i in cols
            ],
            axis=1,
        ).rename(columns=dict(zip(cols, df.columns.levels[0])))

    # X: expression data
    input_file = (
        config["RNA_MATRIX_URL"]
        if data_type == "RNA"
        else config["PROTEIN_MATRIX_URL"]
    )
    x = pd.read_csv(input_file, sep="\t", index_col=0)
    # # # the sample_ids have a strange naming where the same well appears
    # # # multiple times with a suffix
    # x.columns = x.columns.str.extract(r"(DSP-\d+-\w\d+).*")[0].rename("roi_id")

    # Y: metadata
    input_file = (
        config["RNA_SERIES_URL"]
        if data_type == "RNA"
        else config["PRO_SERIES_URL"]
    )
    prjmeta, samplemeta = series_matrix2csv(input_file)

    samplemeta["roi_id"] = samplemeta["description_2"].str.replace(".dcc", "")
    samplemeta["case_id"] = (
        samplemeta["characteristics_ch1"]
        .str.replace("case number: ", "")
        .str.extract(r"(.* \d+) ?")[0]
    )

    samplemeta.loc[
        samplemeta["title"].str.contains("PanCK"), "panKeratin"
    ] = samplemeta["title"].str.endswith("+")
    samplemeta["note"] = samplemeta["characteristics_ch1_2"].str.replace(
        "note: ", ""
    )

    assert x.columns.isin(samplemeta["roi_id"]).all()
    assert samplemeta["roi_id"].isin(x.columns).all()

    # Join with metadata
    y = samplemeta.set_index("roi_id")[
        ["case_id", "panKeratin", "note", "geo_accession"]
    ]

    # Match x and y
    y = y.reindex(x.columns)
    x = x.reindex(y.index, axis=1)

    return x, y


def close_plots(func):
    """
    Decorator to close all plots on function exit.
    """

    @wraps(func)
    def close(*args, **kwargs):
        func(*args, **kwargs)
        plt.close("all")

    return close


@close_plots
def plot_time_and_viral_load(full_meta):
    color = [
        "red" if o == "High" else "blue" for o in full_meta["Virus High/Low"]
    ]

    for label, regparams in [
        ("order_1", dict(order=1)),
        ("order_2", dict(order=2)),
        ("order_3", dict(order=3)),
        ("lowess", dict(lowess=True)),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.scatter(
            full_meta["Duration (days)"],
            full_meta["Viral load% by RNA ISH"],
            c=color,
        )
        sns.regplot(
            full_meta["Duration (days)"],
            full_meta["Viral load% by RNA ISH"],
            scatter=False,
            truncate=False,
            ax=ax,
            **regparams,
        )
        ax.set(
            xlabel="Time since symptoms (days)",
            ylabel="Viral load% by RNA ISH",
            ylim=(-5, 125),
        )
        fig.savefig(output_dir / f"time_vs_viral_load.{label}.svg", **figkws)


@close_plots
def plot_time_and_cell_type_abundance(y, order: int = 1):
    ct_cols = y.columns.where(y.columns.str.contains(" / mm2")).dropna()
    n, m = get_grid_dims(len(ct_cols))
    fig, axes = plt.subplots(n, m, figsize=(3 * m, 3 * n), sharex=True)
    for i, ct in enumerate(ct_cols):
        ax = axes.flatten()[i]
        ax.scatter(y["Duration (days)"], y[ct])
        sns.regplot(y["Duration (days)"], y[ct], ax=ax, order=order)
        ax.set(title=ct, ylabel=None, xlim=(-1, None))
    for ax in axes[:, 0]:
        ax.set_ylabel("Cells per mm2 (IHC)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time since symptoms (days)")
    fig.savefig(output_dir / f"time_vs_lymphoid_abundance.svg", **figkws)

    ct_cols = y.columns.where(y.columns.str.contains(" / %")).dropna()
    n, m = get_grid_dims(len(ct_cols))
    fig, axes = plt.subplots(n, m, figsize=(3 * m, 3 * n), sharex=True)
    for i, ct in enumerate(ct_cols):
        ax = axes.flatten()[i]
        ax.scatter(y["Duration (days)"], y[ct])
        sns.regplot(y["Duration (days)"], y[ct], ax=ax, order=order)
        ax.set(title=ct, ylabel=None, xlim=(-1, None))
    for ax in axes[:, 0]:
        ax.set_ylabel("% abundance (CYBERSORT)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time since symptoms (days)")
    fig.savefig(output_dir / f"time_vs_myeloid_abundance.svg", **figkws)


@close_plots
def plot_deconvolution(x, y):
    grid = clustermap(
        x,
        col_colors=y[config["CLINVARS"]],
        cbar_kws=dict(label="Fraction (CYBERSORT)"),
        dendrogram_ratio=0.1,
        xticklabels=False,
        rasterized=True,
        figsize=(10, 6),
    )
    grid.ax_heatmap.set_xlabel(f"GeoMx ROIs (n = {x.shape[1]})")
    grid.savefig(
        output_dir / "deconvolution.clustermap.svg",
        **figkws,
    )
    grid = clustermap(
        x,
        col_colors=y[config["CLINVARS"]],
        cbar_kws=dict(label="Fraction (CYBERSORT)\n(Z-score)"),
        dendrogram_ratio=0.1,
        xticklabels=False,
        rasterized=True,
        z_score=0,
        metric="correlation",
        cmap="RdBu_r",
        center=0,
        robust=True,
        figsize=(10, 6),
    )
    grid.ax_heatmap.set_xlabel(f"GeoMx ROIs (n = {x.shape[1]})")
    grid.savefig(
        output_dir / "deconvolution.z_score.clustermap.svg",
        **figkws,
    )

    fig, stats = swarmboxenplot(
        data=x.T.join(y),
        x="phenotypes",
        y=x.index,
        test_kws=dict(parametric=False),
        plot_kws=dict(palette=colors["phenotypes"]),
    )
    fig.savefig(
        output_dir / "deconvolution.by_disease_group.swarmboxenplot.svg",
        **figkws,
    )
    stats.to_csv(
        output_dir / "deconvolution.by_disease_group.mann-whitney_test.csv",
        index=False,
    )


@close_plots
def plot_bulk_unsupervised(x, y):

    a = AnnData(x.T, obs=y)
    sc.pp.highly_variable_genes(a)
    variable = a.var["highly_variable"].loc[lambda x: x == True].index

    grid = clustermap(
        x.loc[variable, :],
        col_colors=y[config["CLINVARS"]],
        cbar_kws=dict(label="Fraction (CYBERSORT)"),
        dendrogram_ratio=0.1,
        xticklabels=False,
        rasterized=True,
        figsize=(10, 6),
    )
    grid.ax_heatmap.set_xlabel(f"Bulk RNA-seq (n = {x.shape[1]})")
    grid.savefig(
        output_dir / "bulk_rna-seq.highly_variable_genes.clustermap.svg",
        **figkws,
    )
    grid = clustermap(
        x.loc[variable, :],
        col_colors=y[config["CLINVARS"]],
        cbar_kws=dict(label="Fraction (CYBERSORT)\n(Z-score)"),
        dendrogram_ratio=0.1,
        xticklabels=False,
        rasterized=True,
        z_score=1,
        metric="correlation",
        cmap="RdBu_r",
        center=0,
        robust=True,
        figsize=(10, 6),
    )
    grid.ax_heatmap.set_xlabel(f"Bulk RNA-seq (n = {x.shape[1]})")
    grid.savefig(
        output_dir / "bulk_rna-seq.z_score.clustermap.svg",
        **figkws,
    )

    a = AnnData(x.T, obs=y)
    sc.pp.scale(a)
    sc.pp.pca(a)
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    sc.pl.pca(a, color=config["CLINVARS"])
    sc.pl.umap(a, color=config["CLINVARS"])


def ssgsea(sample: str, database: str, x: DataFrame) -> DataFrame:
    res = gp.ssgsea(x.loc[:, sample], database)
    return res.resultsOnSamples["sample1"]


def rnaseq_to_ssgsea_hallmark_space(g, output_file=None) -> DataFrame:
    db = (database_dir / "h.all.v7.2.symbols.gmt").as_posix()
    if output_file is None:
        output_file = output_dir / "ssGSEA_enrichment.h.all.v7.2.csv"
    if not output_file.exists():
        r = parmap.map(ssgsea, g.columns, database=db, x=g)
        enr = pd.concat(r, axis=1)
        enr.columns = g.columns
        enr.to_csv(output_file)
    enr = pd.read_csv(output_file, index_col=0)
    return enr


def to_ssGSEA_pathway_space(x, data_type) -> DataFrame:
    output_file = output_dir / f"{data_type}.ssGSEA_enrichment.h.all.v7.2.csv"
    return rnaseq_to_ssgsea_hallmark_space(x, output_file)


def rnaseq_to_ssgsea_cell_type_space(g, output_prefix=None) -> DataFrame:
    if output_prefix is None:
        output_prefix = output_dir / "ssGSEA_enrichment"

    db = (database_dir / "c8.all.v7.2.symbols.gmt").as_posix()
    output_file = output_prefix + ".c8.all.v7.2.csv"
    if not output_file.exists():
        r = parmap.map(ssgsea, g.columns, database=db, x=g)
        ct = pd.concat(r, axis=1)
        ct.columns = g.columns
        ct.to_csv(output_file)
    ct = pd.read_csv(output_file, index_col=0)

    db = (database_dir / "scsig.all.v1.0.1.symbols.gmt").as_posix()
    output_file = output_prefix + ".scsig.all.v1.0.1.csv"
    if not output_file.exists():
        r = parmap.map(ssgsea, g.columns, database=db, x=g)
        ct2 = pd.concat(r, axis=1)
        ct2.columns = g.columns
        ct2.to_csv(output_file)
    ct2 = pd.read_csv(output_file, index_col=0)  # shape: (233, 14)

    return pd.concat([ct, ct2])


def to_ssGSEA_cell_type_space(x, data_type) -> DataFrame:
    output_prefix = output_dir / f"{data_type}.ssGSEA_enrichment.cell_type"
    return rnaseq_to_ssgsea_cell_type_space(x, output_prefix)


@close_plots
def plot_geomx_unsupervised(x, y):
    """
    TODO: get tissue origin of GeoMx ROIs, see data in that context.
    """

    x2 = np.log1p(x)
    x3 = (x2 / x2.sum()) * 1e4

    xz = ((x3.T - x3.mean(1)) / x3.std(1)).T

    a = AnnData(x2.T.sort_index(), obs=y.sort_index())
    sc.pp.highly_variable_genes(a)
    variable = a.var["highly_variable"].loc[lambda x: x == True].index

    zs = xz.abs().sum(1).sort_values()
    variable = zs.index[zs >= zs.quantile(0.75)]

    grid = clustermap(
        x3,
        col_colors=y[config["CLINVARS"]],
        cbar_kws=dict(label="Expression"),
        dendrogram_ratio=0.1,
        robust=True,
        xticklabels=False,
        rasterized=True,
        figsize=(10, 6),
    )
    grid.ax_heatmap.set_xlabel(f"GeoMx ROIs (n = {x.shape[1]})")
    grid.savefig(
        output_dir / "GeoMx.all_genes.clustermap.svg",
        **figkws,
    )
    grid = clustermap(
        x3,
        col_colors=y[config["CLINVARS"]],
        cbar_kws=dict(label="Expression\n(Z-score)"),
        dendrogram_ratio=0.1,
        xticklabels=False,
        rasterized=True,
        z_score=0,
        metric="correlation",
        cmap="RdBu_r",
        center=0,
        robust=True,
        figsize=(10, 6),
    )
    grid.ax_heatmap.set_xlabel(f"GeoMx ROIs (n = {x.shape[1]})")
    grid.savefig(
        output_dir / "GeoMx.all_genes.z_score.clustermap.svg",
        **figkws,
    )

    grid = clustermap(
        x3.loc[variable, :],
        col_colors=y[config["CLINVARS"]],
        cbar_kws=dict(label="Expression"),
        dendrogram_ratio=0.1,
        robust=True,
        xticklabels=False,
        rasterized=True,
        figsize=(10, 6),
    )
    grid.ax_heatmap.set_xlabel(f"GeoMx ROIs (n = {x.shape[1]})")
    grid.savefig(
        output_dir / "GeoMx.highly_variable_genes.clustermap.svg",
        **figkws,
    )
    grid = clustermap(
        x3.loc[variable, :],
        col_colors=y[config["CLINVARS"]],
        cbar_kws=dict(label="Expression\n(Z-score)"),
        dendrogram_ratio=0.1,
        xticklabels=False,
        rasterized=True,
        z_score=0,
        metric="correlation",
        cmap="RdBu_r",
        center=0,
        robust=True,
        figsize=(10, 6),
    )
    grid.ax_heatmap.set_xlabel(f"GeoMx ROIs (n = {x.shape[1]})")
    grid.savefig(
        output_dir / "GeoMx.highly_variable_genes.z_score.clustermap.svg",
        **figkws,
    )

    a = AnnData(x3.T, obs=y)
    sc.pp.scale(a)
    sc.pp.pca(a)
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    sc.pl.pca(a, color=config["CLINVARS"])
    sc.pl.umap(a, color=config["CLINVARS"] + ["case_id"])


@close_plots
def plot_pathways(x, y, data_type, space_type="hallmark"):

    if data_type == "Bulk":
        obs_lab = "Bulk RNA-seq samples"
        dt = "bulk_rna-seq"
    else:
        obs_lab = "GeoMx ROIs"
        dt = "geomx"

    grid = clustermap(
        x,
        col_colors=y[config["CLINVARS"]],
        cbar_kws=dict(label="ssGSEA score"),
        config="abs",
        figsize=(10, 6),
    )
    grid.ax_heatmap.set_xlabel(f"{obs_lab} (n = {x.shape[1]})")
    grid.savefig(
        output_dir / f"{dt}.ssGSEA_space.{space_type}clustermap.svg",
        **figkws,
    )
    grid = clustermap(
        x,
        col_colors=y[config["CLINVARS"]],
        cbar_kws=dict(label="ssGSEA score\n(Z-score)"),
        config="z",
        z_score=0,
        figsize=(10, 6),
    )
    grid.ax_heatmap.set_xlabel(f"{obs_lab} (n = {x.shape[1]})")
    grid.savefig(
        output_dir / f"{dt}.ssGSEA_space.{space_type}z_score.clustermap.svg",
        **figkws,
    )

    fig, stats = swarmboxenplot(
        data=x.T.join(y),
        x="phenotypes",
        y=x.index,
        test_kws=dict(parametric=False),
        plot_kws=dict(palette=colors["phenotypes"]),
    )
    fig.savefig(
        output_dir
        / f"{dt}.ssGSEA_space.{space_type}by_disease_group.swarmboxenplot.svg",
        **figkws,
    )
    stats.to_csv(
        output_dir
        / f"{dt}.ssGSEA_space.{space_type}.by_disease_group.mann-whitney_test.csv",
        index=False,
    )


def aggregate_cell_type_signatures(x) -> DataFrame:
    ...


def _get_geomx_dataset() -> DataFrame:
    ...


@close_plots
def compare_geomx_datasets():
    ...
    _get_geomx_dataset()


@close_plots
def compare_effect_sizes_between_imc_and_geomx():
    imcoef = pd.read_csv(
        Path("results")
        / "cell_type"
        / "clustering.roi_zscored.filtered.fraction.cluster_1.0.differences.csv"
    )
    imcoef = imcoef.loc[
        lambda x: (~x["cell_type"].str.contains(" - "))
        & (x["Contrast"] == "phenotypes")
        & (x["measure"] == "percentage")
        & (x["grouping"] == "roi")
    ]
    imcoef["cell_type"] = (
        imcoef["cell_type"]
        .str.replace(" cells", "")
        .str.replace("-cells", "")
        .str.replace(r"s$", "")
    )
    # expand T cells into

    desai_geomx = pd.read_csv(
        output_dir / "deconvolution.by_disease_group.mann-whitney_test.csv"
    ).rename(columns={"signature": "cell_type"})
    # Missing in Desai dataset: 'Mesenchymal', 'CD4/CD8'
    desai_to_imc_cell_type_mapping = {
        "Alveolar epithelial": "Epithelial",
        "B cell": "B",
        "blood vessel cell": "Endothelial",
        "Ciliated cell": None,
        "DC": "Dendritic",
        "Fibroblast": "Fibroblast",
        "Lymph vessel cell": "Endothelial",
        "Macrophage": "Macrophage",
        "Mast cell": "Mast",
        "Monocyte": "Monocyte",
        "Muscle cell": "Smooth muscle",
        "Neutrophil": "Neutrophil",
        "NK": "NK",
        "pDC": None,
        "Plasma cell": None,
        "T": None,  # this would be ["CD4 T", "CD8 T"]
    }
    desai_geomx["cell_type"] = desai_geomx["cell_type"].replace(
        desai_to_imc_cell_type_mapping
    )

    # plot
    for (a1, a2), (b1, b2), label in [
        (("Early", "Late"), ("COVID19_early", "COVID19_late"), "Late-vs-Early"),
        # (("Normal", "Late"), ("Healthy", "COVID19_late"), "Late-vs-Healthy"),
        # (("Normal", "Early"), ("Healthy", "COVID19_early"), "Early-vs-Healthy"),
        # (("Normal", "Flu"), ("Healthy", "Flu"), "Flu-vs-Healthy"),
        # (("Normal", "ARDS"), ("Healthy", "ARDS"), "ARDS-vs-Healthy"),
        # (
        #     ("Normal", "Pneumonia"),
        #     ("Healthy", "Pneumonia"),
        #     "Pneumonia-vs-Healthy",
        # ),
    ]:
        a = desai_geomx.loc[
            (desai_geomx["A"] == a1) & (desai_geomx["B"] == a2)
        ][["cell_type", "hedges", "p-cor"]].set_index("cell_type")
        a["mlogq"] = -np.log10(a["p-cor"])
        b = imcoef.loc[(imcoef["A"] == b1) & (imcoef["B"] == b2)][
            ["cell_type", "hedges", "p-cor"]
        ].set_index("cell_type")
        b["mlogq"] = -np.log10(b["p-cor"])

        b = b.reindex(a.index).dropna().groupby(level=0).mean()
        a = a.groupby(level=0).mean().reindex(b.index).dropna()

        fig, axes = plt.subplots(1, 3, figsize=(3 * 4.25, 1 * 3.75))
        for i, meas in enumerate(["hedges", "p-cor", "mlogq"]):
            d = a[meas].rename("GeoMx").to_frame().join(
                b[meas].rename("IMC")
            ) * (-1 if meas == "hedges" else 1)
            mm = d.mean(1)
            s = pg.corr(d["IMC"], d["GeoMx"]).squeeze()
            vmax = d.abs().values.max()
            vmax += vmax * 0.1
            if meas == "hedges":
                vmin = -vmax
            else:
                vmin = -(vmax * 0.05)

            # coef, inter = np.polyfit(d["IMC"], d["GeoMx"], 1)
            # axes[i].plot(d["IMC"], coef * d["IMC"] + inter, color='black', linestyle="--")
            # axes[i].fill_between(d["IMC"], coef * d["IMC"] + inter, coef * d["IMC"] + inter, color='black', alpha=0.25)

            sns.regplot(
                d["IMC"],
                d["GeoMx"],
                line_kws=dict(alpha=0.1, color="black"),
                color="black",
                ax=axes[i],
            )
            axes[i].scatter(
                d["IMC"],
                d["GeoMx"],
                c=mm,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
            )
            axes[i].axhline(0, linestyle="--", color="grey")
            axes[i].axvline(0, linestyle="--", color="grey")
            axes[i].set(
                title=f"{meas}\nr = {s['r']:.3f}; 95% CI: {s['CI95%']}\np = {s['p-val']:.3f}",
                xlabel="IMC",
                ylabel="GeoMx",
                xlim=(vmin, vmax),
                ylim=(vmin, vmax),
            )
            for t in d.index:
                axes[i].text(
                    d.loc[t, "IMC"],
                    d.loc[t, "GeoMx"],
                    s=t,
                    ha="left" if mm.loc[t] < 0 else "right",
                )
        fig.savefig(
            output_dir
            / f"coefficient_comparison.imc_with_desai_geomx.{label}.scatter.svg",
            **figkws,
        )


def get_hca_lung_reference(rnaseq_genes: Series) -> DataFrame:
    # Try to use scRNA-seq as reference
    mean_file = scrnaseq_dir / "krasnow_hlca_10x.average.expression.csv"
    mean = pd.read_csv(mean_file, index_col=0)

    zm = ((mean.T - mean.mean(1)) / mean.std(1)).T.dropna()

    rnaseq_genes[~rnaseq_genes.isin(zm.index)]

    dg = zm.reindex(rnaseq_genes).dropna()
    return dg


@close_plots
def use_hca_lung_reference(x, y, ref) -> None:
    # Try to deconvolve
    output_prefix = output_dir / "krasnow_scRNA_deconvolve.correlation"
    dc = x.join(ref).corr().loc[x.columns, ref.columns]

    sign = (dc > 0).astype(int).replace(0, -1)
    dcs = dc.abs() ** (1 / 3) * sign

    kws = dict(
        yticklabels=True,
        robust=True,
        figsize=(5, 10),
        col_colors=y[config["CLINVARS"]],
        metric="correlation",
        cbar_kws=dict(label="Pearson correlation"),
        dendrogram_ratio=0.1,
        rasterized=True,
    )

    grid = clustermap(dc.T, center=0, cmap="RdBu_r", **kws)
    grid.savefig(
        output_prefix + ".clustermap.svg",
        **figkws,
    )

    grid = clustermap(dc.T, z_score=0, center=0, cmap="RdBu_r", **kws)
    grid.savefig(
        output_prefix + ".clustermap.z_score.svg",
        **figkws,
    )

    grid = clustermap(dc.T, standard_scale=1, cmap="Reds", **kws)
    grid.savefig(
        output_prefix + ".clustermap.std_scale.svg",
        **figkws,
    )

    p = (dcs.T + 1) / 2
    p -= p.min()
    normdcs = p / p.sum()

    grid = clustermap(normdcs * 100, **kws)
    grid.savefig(
        output_prefix + ".clustermap.norm.svg",
        **figkws,
    )

    a = AnnData(dcs, obs=y)
    sc.pp.pca(a)
    sc.pp.neighbors(a)
    sc.tl.umap(a)  # , gamma=0.0001)
    axes = sc.pl.pca(a, color=config["CLINVARS"], show=False)
    fig = axes[0].figure
    fig.savefig(output_prefix + ".pca.svg", **figkws)

    axes = sc.pl.umap(a, color=config["CLINVARS"], show=False)
    fig = axes[0].figure
    fig.savefig(output_prefix + ".umap.svg", **figkws)

    colors = {"phenotypes": np.asarray(sns.color_palette("tab10"))[[2, 4, 3]]}

    fig, stats = swarmboxenplot(
        data=dcs.join(y),
        x="phenotypes",
        y=dcs.columns,
        plot_kws=dict(palette=colors["phenotypes"]),
        test_kws=dict(parametric=False),
    )
    fig.savefig(
        output_prefix + f".correlation.swarmboxenplot.svg",
        **figkws,
    )
    stats.to_csv(output_prefix + ".correlation.csv", index=False)

    ### volcano plot
    fig = volcano_plot(stats)
    fig.savefig(output_prefix + ".volcano.svg", **figkws)


def series_matrix2csv(
    matrix_url: str, prefix: str = None
) -> Tuple[DataFrame, DataFrame]:
    """
    Get a GEO series matrix file describing an experiment and
    parse it into project level and sample level data.

    Parameters
    ----------
    matrix_url: str
        FTP URL of gziped txt file with GEO series matrix.
    prefix: str
        Prefix path to write files to.
    """
    import os
    from typing import Tuple, Union
    import tempfile
    from collections import Counter
    import urllib.request as request
    from contextlib import closing
    import gzip

    with closing(request.urlopen(matrix_url)) as r:
        content = gzip.decompress(r.fp.file.read())
    lines = content.decode("utf-8").strip().split("\n")

    # separate lines with only one field (project-related)
    # from lines with >2 fields (sample-related)

    # # if the same key appears more than once, keep all but rename
    # # them with a suffix
    prj_lines = dict()
    sample_lines = dict()
    idx_counts: Counter = Counter()
    col_counts: Counter = Counter()

    for line in lines:
        cols = line.strip().split("\t")
        key = cols[0].replace('"', "")
        if len(cols) == 2:
            if key in idx_counts:
                key = f"{key}_{idx_counts[key] + 1}"
            idx_counts[key] += 1
            prj_lines[key] = cols[1].replace('"', "")
        elif len(cols) > 2:
            if key in col_counts:
                key = f"{key}_{col_counts[key] + 1}"
            col_counts[key] += 1
            sample_lines[key] = [x.replace('"', "") for x in cols[1:]]

    prj = pd.Series(prj_lines)
    prj.index = prj.index.str.replace("!Series_", "")

    samples = pd.DataFrame(sample_lines)
    samples.columns = samples.columns.str.replace("!Sample_", "")

    if prefix is not None:
        prj.to_csv(os.path.join(prefix + ".project_annotation.csv"), index=True)
        samples.to_csv(
            os.path.join(prefix + ".sample_annotation.csv"), index=False
        )

    return prj, samples


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\t - Exiting due to user interruption.")
        sys.exit(1)
