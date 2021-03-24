# coding: utf-8

"""
This script analyses targeted spatial transcriptomics data (GeoMx)
for lung tissue of healthy donors and patients with lung infections.
"""

import sys
import io
from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
import imageio
import parmap
import PIL
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
from anndata import AnnData
import scanpy as sc
import pingouin as pg

from ngs_toolkit.general import enrichr
from imc.types import Path, DataFrame
from imc.graphics import get_grid_dims

from seaborn_extensions import clustermap, swarmboxenplot


# Some defaults
swarmboxenplot = partial(swarmboxenplot, test_kws=dict(parametric=False))
clustermap = partial(clustermap, metric="correlation", dendrogram_ratio=0.1)

figkws = dict(dpi=300, bbox_inches="tight")
data_dir = Path("data") / "geomx"
output_dir = Path("results") / "geomx"
gene_set_library_dir = Path("data") / "gene_set_libraries"
colors = {"phenotypes": np.asarray(sns.color_palette("tab10"))[[2, 5, 4, 3]]}
cells = [
    "Epithelial",
    "Mesenchym",
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
urls = {
    # "metadata": "https://wcm.box.com/shared/static/l8sxs6luu4fkpiabpjbiqytsb8dgkfxz.pq",
    "metadata": "https://zenodo.org/record/4635286/files/data/geomx/metadata_matrix.pq?download=1",
    # "expression": "https://wcm.box.com/shared/static/b7nuqfoiey5o5fwnp4z006rdhmm4ascv.pq",
    "expression": "https://zenodo.org/record/4635286/files/data/geomx/expression_matrix.pq?download=1",
}


def main() -> int:
    expr, meta = get_X_Y()

    # get metadata per sample too
    meta_sample = meta.drop_duplicates(subset="sample_id").set_index(
        ["sample_id", "location", "days_hospitalized"]
    )[["phenotypes"]]
    meta_sample = meta.drop_duplicates(subset="sample_id").set_index(
        ["sample_id"]
    )[["phenotypes", "location", "days_hospitalized"]]

    _expr = np.log1p(expr)
    exprnorm = (_expr / _expr.sum()) * 1e4

    # Convert gene expression to signature space
    # # both pathway and cell type level
    enr, ct, ct2 = get_ssGSEA_space(expr)
    enr = enr.reindex(expr.columns, axis=1)
    ct = ct.reindex(expr.columns, axis=1)
    ct2 = ct2.reindex(expr.columns, axis=1)

    # Plot pathway space across disease groups
    plot_ssGSEA_space((enr, ct, ct2), meta)

    # Try to use Enrichr signatures to do the same
    get_enrichr_gene_sets(enr, meta, meta_sample)

    # Plot cell type space across disease groups
    joint_ct = ct.append(ct2)
    plot_joint_cell_types(joint_ct, meta, meta_sample)

    # Compare effect size estimates between IMC and GeoMx
    compare_imc_and_geomx_cell_type_coefficients()

    # Deconvolve data on cell type pathways
    corr_d = soft_deconvolve(expr, joint_ct)
    plot_soft_deconvolution(corr_d, meta, enr)

    #
    # NOT USED in paper!
    unsupervised_analysis(expr, meta, enr, joint_ct, meta_sample)

    # Compare marker expression between IMC and GeoMx
    compare_dataset_markers_head_to_head(exprnorm, meta)

    # Deconvolve data on scRNA-seq from Krasnow et al.
    soft_deconvolution_with_scRNAseq_data(expr, meta)

    # Create visualization of RNAscope
    plot_rna_scope_viz(meta)

    return 0
    # Usused stuff
    # ssc = expr.corr()
    # clustermap(
    #     ssc,
    #     row_colors=meta[["phenotypes"]],
    #     row_colors_cmaps=[colors["phenotypes"]],
    # )
    # plt.show()

    # # Intra patient, inter location correlation
    # means = (
    #     expr.T.join(meta[["disease", "location"]])
    #     .groupby(["disease", "location"])
    #     .mean()
    #     .T
    # )

    # corrs = means.corr()

    # grid = clustermap(
    #     corrs, center=0, cmap="RdBu_r", row_colors=corrs.index.to_frame()
    # )


def get_X_Y() -> Tuple[DataFrame, DataFrame]:
    try:
        import urllib

        expr = pd.read_parquet(urls["expression"])
        meta = pd.read_parquet(urls["metadata"])
        return expr, meta
    except urllib.request.HTTPError:
        pass
    # metadata and data
    # # meta
    meta = pd.read_csv(data_dir / "annotations_20200923.csv", index_col=0)
    meta["sample_id"] = (
        meta["DSP_scan"]
        .str.replace("L18AUG2020", "L_081820")
        .str.replace(r"\)", "", regex=False)
    )
    meta_add = pd.read_csv(
        io.StringIO(
            """
sample_id,phenotypes,RNA_spike,days_hospitalized
Covid01_050520,Early,positive,0
Covid02_050720,Early,positive,5
Covid17_050820,Late,negative,9
Covid21_050820,Late,negative,18
Covid55L_081820,Early,positive,5
Covid59L_081820,Late,positive,21
Covid73L_081820,Late,positive,24
Covid86L_081820,Early,positive,1
ARDS02_050820,ARDS,,
"""
        )
    )

    matching_samples = [
        "Covid01_050520",
        "Covid21_050820",
        "Covid55L_081820",
    ]
    meta["matching_IMC"] = meta["sample_id"].isin(matching_samples)
    meta = (
        meta.reset_index()
        .merge(meta_add, on="sample_id", how="left")
        .set_index("sample_id")
    )
    meta["disease"] = meta["Disease"]
    meta["phenotypes"] = (
        meta["phenotypes"]
        .fillna(
            meta["Disease"]
            .reset_index()
            .drop_duplicates()
            .set_index("sample_id")
            .squeeze()
            .to_dict()
        )
        .replace("Non-viral", "Pneumonia")
    )
    meta["location"] = meta["Tissue"]
    meta = meta.reset_index().set_index("Sample_ID")

    meta["phenotypes"] = pd.Categorical(
        meta["phenotypes"],
        ordered=True,
        categories=["Normal", "Pneumonia", "Early", "Late"],
    )

    # gene expression data
    df = pd.read_csv(
        data_dir / "covid_five_classes_Q3Norm_TargetCountMatrix.txt",
        sep="\t",
        index_col=0,
    ).sort_index(axis=1)
    df.columns = df.columns.str.replace(".", "-", regex=False)

    # Expand/duplicate genes that are aggregated
    # this doesn't make much difference in the data but allows more genes to be
    # connected with databases and other data types.
    for idx in df.index[df.index.str.contains("/")]:
        n = len(idx.split("/")[-1])
        stem = idx.split("/")[0][:-n]
        for end in idx.replace(stem, "").split("/"):
            df.loc[stem + end] = df.loc[idx]
        df = df.drop(idx, axis=0)

    # align data and metadata
    df = df.reindex(meta.index, axis=1)
    df = df.loc[:, ~df.isnull().all()]
    meta = meta.reindex(df.columns, axis=0)

    return df, meta


def ssgsea(sample, database, x):
    res = gp.ssgsea(x.loc[:, sample], database)
    return res.resultsOnSamples["sample1"]


def get_ssGSEA_space(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    # Convert to signature space
    # # Hallmark pathways
    f = output_dir / "ssGSEA_enrichment.h.all.v7.2.new.csv"
    db = gene_set_library_dir / "h.all.v7.2.symbols.gmt"
    if not f.exists():
        res = parmap.map(ssgsea, df.columns, database=db.as_posix(), x=df)
        enr = pd.concat(res, axis=1)
        enr.columns = df.columns
        enr.to_csv(f)
    else:
        enr = pd.read_csv(f, index_col=0)

    # # Cell type signatures
    f = output_dir / "ssGSEA_enrichment.c8.all.v7.2.new.csv"
    db = gene_set_library_dir / "c8.all.v7.2.symbols.gmt"
    if not f.exists():
        res2 = parmap.map(ssgsea, df.columns, database=db.as_posix(), x=df)
        ct = pd.concat(res2, axis=1)
        ct.columns = df.columns
        ct.to_csv(f)
    else:
        ct = pd.read_csv(f, index_col=0)

    # f = output_dir / "ssGSEA_enrichment.scsig.all.v1.0.1.new.csv"
    f = output_dir / "ssGSEA_enrichment.scsig.all.v1.0.1.csv"
    db = gene_set_library_dir / "scsig.all.v1.0.1.symbols.gmt"
    if not f.exists():
        res3 = parmap.map(ssgsea, df.columns, database=db.as_posix(), x=df)
        ct2 = pd.concat(res3, axis=1)
        ct2.columns = df.columns
        ct2.to_csv(f)
    else:
        ct2 = pd.read_csv(f, index_col=0)
    return enr, ct, ct2


def plot_ssGSEA_space(
    dfs: Tuple[DataFrame, DataFrame, DataFrame],
    meta: DataFrame,
) -> None:
    enr, ct, ct2 = dfs

    for conf in ["abs", "z_score"]:
        grid = clustermap(
            enr.T,
            config=conf,
            row_colors=meta[["phenotypes", "location", "days_hospitalized"]],
            xticklabels=True,
            yticklabels=False,
        )
        grid.savefig(
            output_dir
            / f"ssGSEA_enrichment.clustermap.{conf}._normal_vs_covid.svg",
            **figkws,
        )

    # fig, stats = swarmboxenplot(
    #     data=enr.T.join(meta),
    #     x="phenotypes",
    #     y=["HALLMARK_IL6_JAK_STAT3_SIGNALING"],
    #     hue="location",
    # )

    fig, stats = swarmboxenplot(
        data=enr.T.join(meta),
        x="phenotypes",
        y=enr.index,
        hue="location",
    )
    fig.savefig(
        output_dir
        / "ssGSEA_enrichment.by_location.swarmboxenplot._normal_vs_covid.svg",
        **figkws,
    )
    stats.to_csv(output_dir / "ssGSEA_enrichment.by_location.stats.csv")

    fig, stats = swarmboxenplot(
        data=enr.T.join(meta),
        x="location",
        y=enr.index,
        hue="phenotypes",
        plot_kws=dict(palette=colors["phenotypes"]),
    )
    fig.savefig(
        output_dir
        / "ssGSEA_enrichment.by_phenotypes.swarmboxenplot._normal_vs_covid.svg",
        **figkws,
    )
    stats.to_csv(
        output_dir / "ssGSEA_enrichment.by_phenotypes.stats.csv", index=False
    )

    grid = clustermap(
        ct,
        center=0,
        cmap="RdBu_r",
        col_colors=meta[["phenotypes", "location", "days_hospitalized"]],
        cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
        rasterized=True,
    )
    grid.savefig(
        output_dir / "ssGSEA_enrichment.c8.all.v7.2.all_ROIs.clustermap.svg",
        **figkws,
    )

    grid = clustermap(
        ct2,
        center=0,
        cmap="RdBu_r",
        col_colors=meta[["phenotypes", "location", "days_hospitalized"]],
        cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
        rasterized=True,
    )
    grid.savefig(
        output_dir
        / "ssGSEA_enrichment.scsig.all.v1.0.1.all_ROIs.clustermap.svg",
        **figkws,
    )


def get_enrichr_gene_sets(
    df: DataFrame, meta: DataFrame, meta_sample: DataFrame
) -> None:
    out = output_dir / "enrichment"
    out.mkdir(exist_ok=True)
    gsllf = "metadata/enrichr.gene_set_libraries.txt"
    gsll = open(gsllf, "r").read().strip().split("\n")

    gsll = [
        "ARCHS4_Tissues",
        "WikiPathways_2019_Human",
        "COVID-19_Related_Gene_Sets",
        "Jensen_TISSUES",
        "Mouse_Gene_Atlas",
        "KEGG_2019_Human",
        "GO_Biological_Process_2018",
    ]
    for gsl in gsll:
        f = out / gsl + ".csv"
        if not f.exists():
            db = Path("gene_set_libraries") / gsl + ".gmt"
            r = parmap.map(ssgsea, df.columns, database=db.as_posix(), x=df)
            rr = pd.concat(r, axis=1)
            rr.columns = df.columns
            rr.to_csv(f)
        else:
            rr = pd.read_csv(f, index_col=0)

        grid = clustermap(
            rr,
            col_colors=meta[["phenotypes", "location", "days_hospitalized"]],
            cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
            rasterized=True,
        )
        grid.savefig(
            out / f"ssGSEA_enrichment.{gsl}.all_ROIs.clustermap.svg", **figkws
        )
        zkws = dict(
            z_score=0,
            center=0,
            cmap="RdBu_r",
            rasterized=True,
            cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
            metric="correlation",
        )
        grid = clustermap(
            rr,
            col_colors=meta[["phenotypes", "location", "days_hospitalized"]],
            **zkws,
        )
        grid.savefig(
            out / f"ssGSEA_enrichment.{gsl}.all_ROIs.clustermap.z_score.svg",
            **figkws,
        )
        q = rr.T.join(meta).query("location == 'Alveolar'")
        grid = clustermap(
            q.T.drop(meta.columns).astype(float),
            col_colors=q[["phenotypes", "location", "days_hospitalized"]],
            **zkws,
        )
        grid.savefig(
            out
            / f"ssGSEA_enrichment.{gsl}.all_ROIs.clustermap.z_score.alveolar.svg",
            **figkws,
        )
        qq = q.groupby("sample_id").mean()
        grid = clustermap(
            qq.T.drop(meta.columns, errors="ignore").astype(float),
            col_colors=meta_sample[["phenotypes", "days_hospitalized"]],
            **zkws,
        )
        grid.savefig(
            out
            / f"ssGSEA_enrichment.{gsl}.all_ROIs.clustermap.z_score.alveolar.reduced_sample.svg",
            **figkws,
        )

        gsl = "ARCHS4_Tissues"
        f = out / gsl + ".csv"
        rr = pd.read_csv(f, index_col=0)
        ctrr = rr.loc[
            [
                "LUNG EPITHELIAL CELL",
                "FIBROBLAST",
                "MYOFIBROBLAST",
                "ALVEOLAR MACROPHAGE",
                "MACROPHAGE",
                "NEUTROPHIL",
                "DENDRITIC CELL",
            ]
        ].T

        for location in meta["location"].unique():
            output_prefix = (
                output_dir
                / f"{gsl}.cell_type_aggregated.{location.replace(' ', '_')}.swarmboxenplot"
            )

            resloc = (
                ctrr.join(meta["location"])
                .query(f"location == '{location}'")
                .drop("location", axis=1)
            )

            n, m = get_grid_dims(resloc.shape[1])
            fig, axes = plt.subplots(n, m, figsize=(m * 3, n * 3), sharex=True)
            _stats = list()
            for i, c in enumerate(resloc.columns):
                ax = axes.flatten()[i]
                stats = swarmboxenplot(
                    data=resloc[[c]].join(meta),
                    x="phenotypes",
                    y=c,
                    ax=ax,
                    plot_kws=dict(palette=colors["phenotypes"]),
                    test_kws=dict(parametric=False),
                )
                ax.set(title=c, xlabel=None, ylabel=None)
                _stats.append(stats.assign(cell_type=c))
            for ax in axes.flatten()[i + 1 :]:
                ax.axis("off")
            fig.savefig(
                output_prefix + ".swarmboxenplot.svg",
                **figkws,
            )
            stats = pd.concat(_stats)
            stats.to_csv(output_prefix + ".csv", index=False)

            resloc_norm = (
                (resloc.T - resloc.min(1)) / (resloc.max(1) - resloc.min(1))
            ).T
            resloc_norm = (resloc_norm.T / resloc_norm.sum(1)).T
            # resloc_norm = (resloc_norm[cells[:5]].T - resloc_norm[cells[:5]].sum(1)).T

            n, m = get_grid_dims(resloc_norm.shape[1])
            fig, axes = plt.subplots(n, m, figsize=(m * 3, n * 3), sharex=True)
            for i, c in enumerate(resloc_norm.columns):
                ax = axes.flatten()[i]
                stats = swarmboxenplot(
                    data=resloc_norm[[c]].join(meta),
                    x="phenotypes",
                    y=c,
                    ax=ax,
                    plot_kws=dict(palette=colors["phenotypes"]),
                    test_kws=dict(parametric=False),
                )
                ax.set(title=c, xlabel=None, ylabel=None)
            for ax in axes.flatten()[i + 1 :]:
                ax.axis("off")
            fig.savefig(
                output_prefix + ".norm.swarmboxenplot.svg",
                **figkws,
            )
            stats = pd.concat(_stats)
            stats.to_csv(output_prefix + ".norm.csv", index=False)


def plot_joint_cell_types(
    joint_ct: DataFrame, meta: DataFrame, meta_sample: DataFrame
) -> None:
    accept = ["Normal", "Pneumonia", "Early", "Late"]

    joint_ct = joint_ct.T.join(meta["phenotypes"])
    joint_ct = (
        joint_ct.loc[joint_ct["phenotypes"].isin(accept)]
        .drop("phenotypes", 1)
        .T
    )

    meta = meta.loc[meta["phenotypes"].isin(accept)]
    meta["phenotypes"] = meta["phenotypes"].cat.remove_unused_categories()

    meta["location"] = pd.Categorical(
        meta["location"], ordered=True, categories=meta["location"].unique()
    )

    # # Z score
    joint_ct_z = ((joint_ct.T - joint_ct.mean(1)) / joint_ct.std(1)).T

    # Aggregate signatures by cell types
    _res = dict()
    for cell in cells:
        c = joint_ct_z.index[joint_ct_z.index.str.contains(cell, case=False)]
        # c = [x for x in c if ("NEURO" not in x)]  #  and ("PROGENITOR" not in x)
        if len(c) >= 3:
            print(cell, c)
            _res[cell] = joint_ct_z.loc[c].mean()
    res = pd.DataFrame(_res)

    # Standardize
    res_norm = ((res.T - res.min(1)) / (res.max(1) - res.min(1))).T
    res_norm = (res_norm.T / res_norm.sum(1)).T

    grid = clustermap(
        joint_ct,
        center=0,
        cmap="RdBu_r",
        col_colors=meta[["phenotypes", "location", "days_hospitalized"]],
        cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
        rasterized=True,
    )
    grid.savefig(
        output_dir / "ssGSEA_enrichment.cell_type.all_ROIs.clustermap.svg",
        **figkws,
    )

    grid = clustermap(
        joint_ct_z,
        center=0,
        cmap="RdBu_r",
        col_colors=meta[["phenotypes", "location", "days_hospitalized"]],
        cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
        rasterized=True,
    )
    grid.savefig(
        output_dir
        / "ssGSEA_enrichment.cell_type.all_ROIs.clustermap.z_score.svg",
        **figkws,
    )

    grid = clustermap(
        res.T.dropna(),
        metric="correlation",
        center=0,
        cmap="RdBu_r",
        col_colors=meta[["phenotypes", "location"]],
        cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
        figsize=(5, 4),
        xticklabels=False,
    )
    grid.savefig(
        output_dir
        / "ssGSEA_enrichment.cell_type_aggregated.all_ROIs.clustermap.svg",
        **figkws,
    )

    for location in ["All"] + meta["location"].unique().tolist():
        output_prefix = (
            output_dir
            / f"ssGSEA_enrichment.cell_type_aggregated.{location.replace(' ', '_')}.swarmboxenplot"
        )

        if location == "All":
            resloc = res
        else:
            resloc = (
                res.join(meta["location"])
                .query(f"location == '{location}'")
                .drop("location", 1)
            )
        fig, stats = swarmboxenplot(
            data=resloc.join(meta),
            x="phenotypes",
            y=resloc.columns.tolist(),
            plot_kws=dict(palette=colors["phenotypes"]),
            test_kws=dict(parametric=False),
        )
        fig.savefig(
            output_prefix + ".swarmboxenplot.svg",
            **figkws,
        )
        stats.rename(columns={"Variable": "cell_type"})
        stats.to_csv(output_prefix + ".csv", index=False)

        resloc_norm = (
            (resloc.T - resloc.min(1)) / (resloc.max(1) - resloc.min(1))
        ).T
        resloc_norm = (resloc_norm.T / resloc_norm.sum(1)).T
        # resloc_norm = (resloc_norm[cells[:5]].T - resloc_norm[cells[:5]].sum(1)).T

        fig, stats = swarmboxenplot(
            data=resloc_norm.join(meta),
            x="phenotypes",
            y=resloc_norm.columns.tolist(),
            plot_kws=dict(palette=colors["phenotypes"]),
            test_kws=dict(parametric=False),
        )
        fig.savefig(
            output_prefix + ".norm.swarmboxenplot.svg",
            **figkws,
        )
        stats = stats.rename(columns={"Variable": "cell_type"})
        stats.to_csv(output_prefix + ".norm.csv", index=False)

    for ph in meta["phenotypes"].unique():
        fig = swarmboxenplot(
            data=res_norm.join(meta).query(f"phenotypes == '{ph}'"),
            x="location",
            y=res_norm.columns.tolist(),
            test=False,
        )
        fig.suptitle(ph)

    res_red = (
        res_norm.join(meta[["sample_id", "location"]])
        .groupby(["sample_id", "location"])
        .mean()
        .dropna(subset=["Epithelial"])
    )
    res_red = res_red.loc[:, ~res_red.isnull().all()]

    idx = res_red.join(meta_sample)[
        ["phenotypes", "location", "days_hospitalized"]
    ]
    idx["location"] = idx.index.get_level_values("location")

    grid = clustermap(
        res_red.T,
        metric="correlation",
        center=0.1,
        cmap="RdBu_r",
        col_colors=idx,
        xticklabels=True,
        cbar_kws=dict(label="ssGSEA enrichment\nZ-score)"),
        figsize=(5, 4),
    )
    grid.savefig(
        output_dir
        / "ssGSEA_enrichment.cell_type_aggregated.sample_reduced.all_ROIs.clustermap.svg",
        **figkws,
    )

    idx = res_red.join(meta_sample)[
        ["phenotypes", "location", "days_hospitalized"]
    ]
    idx["location"] = idx.index.get_level_values("location")

    for loc in res_red.index.levels[1]:
        grid = clustermap(
            res_red.loc[:, loc, :].T,
            metric="correlation",
            center=0,
            cmap="RdBu_r",
            col_colors=idx.loc[:, loc, :][["phenotypes", "days_hospitalized"]],
            cbar_kws=dict(label="ssGSEA enrichment\nZ-score)"),
            figsize=(5, 4),
            xticklabels=True,
        )
        grid.savefig(
            output_dir
            / f"ssGSEA_enrichment.cell_type_aggregated.sample_reduced.all_ROIs.{loc}.clustermap.svg",
            **figkws,
        )


def compare_imc_and_geomx_cell_type_coefficients() -> None:
    # Compare "coefficients"
    to_repl = {
        "Mesenchym": "Mesenchymal",
    }

    location = "All"
    gmcoef = pd.read_csv(
        output_dir
        / f"ssGSEA_enrichment.cell_type_aggregated.{location}.swarmboxenplot.norm.csv",
    )

    accept = ["Normal", "Pneumonia", "Early", "Late"]
    gmcoef = gmcoef.loc[gmcoef["A"].isin(accept) & gmcoef["B"].isin(accept)]

    # gmcoef["median"] = gmcoef[["median_A", "median_B"]].mean(1)
    # m = gmcoef.groupby("cell_type")["median"].mean()
    # m = ((m - m.min()) / (m.max() - m.min())).sort_values()
    # gmcoef['hedges'] = np.log(gmcoef['median_A'] / gmcoef['median_B'])
    gmcoef["cell_type"] = (
        gmcoef["cell_type"]
        .str.replace("_", " ")
        .str.replace(" cell", "")
        .replace(to_repl)
        .str.replace(r"s$", "")
    )
    imcoef = pd.read_csv(
        Path("results")
        / "cell_type"
        / "clustering.roi_zscored.filtered.fraction.cluster_1.0.differences.csv"
    )
    imcoef = imcoef.loc[
        lambda x: (~x["cell_type"].str.contains(" - "))
        & (x["Contrast"] == "phenotypes")
        & (x["measure"] == "area")
        & (x["grouping"] == "roi")
    ]
    imcoef["cell_type"] = (
        imcoef["cell_type"]
        .str.replace(" cells", "")
        .str.replace("-cells", "")
        .str.replace(r"s$", "")
    )

    # plot
    for (a1, a2), (b1, b2), label in [
        (("Early", "Late"), ("COVID19_early", "COVID19_late"), "Late-vs-Early"),
        (("Normal", "Late"), ("Healthy", "COVID19_late"), "Late-vs-Healthy"),
        (("Normal", "Early"), ("Healthy", "COVID19_early"), "Early-vs-Healthy"),
        (
            ("Normal", "Pneumonia"),
            ("Healthy", "Pneumonia"),
            "Pneumonia-vs-Healthy",
        ),
        # (
        #     ("Pneumonia", "Early"),
        #     ("Pneumonia", "COVID19_early"),
        #     "Pneumonia-vs-COVID_early",
        # ),
        # (
        #     ("Pneumonia", "Late"),
        #     ("Pneumonia", "COVID19_late"),
        #     "Pneumonia-vs-COVID_late",
        # ),
    ]:
        a = (
            gmcoef.loc[(gmcoef["A"] == a1) & (gmcoef["B"] == a2)][
                ["cell_type", "hedges", "p-cor"]
            ]
            .set_index("cell_type")
            .sort_index()
        )
        a["mlogq"] = -np.log10(a["p-cor"])
        b = (
            imcoef.loc[(imcoef["A"] == b1) & (imcoef["B"] == b2)][
                ["cell_type", "hedges", "p-cor"]
            ]
            .set_index("cell_type")
            .sort_index()
        )
        b["mlogq"] = -np.log10(b["p-cor"])

        # sig = (a['p-cor'] < 0.05).to_frame('geomx').join((b['p-cor'] < 0.05).rename("imc"))
        # sklearn.metrics.accuracy_score(sig['imc'], sig['geomx'], sample_weight=None)
        # sklearn.metrics.hamming_loss(sig['imc'], sig['geomx'], sample_weight=None)
        # sklearn.metrics.jaccard_score(sig['imc'], sig['geomx'], sample_weight=None)
        # fp, tp, _ = sklearn.metrics.roc_curve(a['p-cor'] < 0.05, b['p-cor'].reindex(a.index), pos_label=True)
        # p, r, t = sklearn.metrics.precision_recall_curve(a['p-cor'] < 0.05, b['p-cor'].reindex(a.index), pos_label=True)
        # fig, axes = plt.subplots(1, 2)
        # axes[0].plot(fp, tp)
        # axes[1].plot(r, p)

        # fc = (a['hedges'] > 0).to_frame('geomx').join((b['hedges'] > 0).rename("imc"))
        # print(pg.chi2_independence(data=fc, x='imc', y='geomx'))

        # t = 0.2
        # a2 = (a['hedges'] > t).astype(int)
        # a3 = (a['hedges'] < -t).replace({True: -1}).astype(int)
        # b2 = (b['hedges'] > t).astype(int)
        # b3 = (b['hedges'] < -t).replace({True: -1}).astype(int)

        # fc = (a2 + a3).to_frame('geomx').join((b2 + b3).rename("imc"))
        # print(pg.chi2_independence(data=fc, x='imc', y='geomx'))

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

            sns.regplot(
                data=d,
                x="IMC",
                y="GeoMx",
                line_kws=dict(alpha=0.1, color="black"),
                color="black",
                scatter=True,
                ax=axes[i],
            )
            axes[i].scatter(
                data=d,
                x="IMC",
                y="GeoMx",
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
                xlim=(axes[i].get_xlim()[0] * 1.1, axes[i].get_xlim()[1] * 1.1),
                ylim=(axes[i].get_ylim()[0] * 1.1, axes[i].get_ylim()[1] * 1.1)
                # xlim=(vmin, vmax),
                # ylim=(vmin, vmax),
            )
            for t in d.index:
                axes[i].text(
                    d.loc[t, "IMC"],
                    d.loc[t, "GeoMx"],
                    s=t,
                    ha="left" if mm.loc[t] < 0 else "right",
                )
        fig.savefig(
            output_dir / f"coefficient_comparison.{label}.scatter.svg",
            **figkws,
        )

    # All together
    # x = gmcoef[['cell_type', 'A', 'B', 'hedges']].rename(columns={"hedges": "geomx"})
    # x['A'] = x['A'].replace({"Early": "COVID19_early", "Late": "COVID19_late", "Normal": "Healthy"})
    # x['B'] = x['B'].replace({"Early": "COVID19_early", "Late": "COVID19_late", "Normal": "Healthy"})
    # y = imcoef[['cell_type', 'A', 'B', 'hedges']].rename(columns={"hedges": "imc"})
    # p = x.merge(y)

    # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    # ax.scatter(p['imc'], p['geomx'])
    # s = pg.corr(d["IMC"], d["GeoMx"]).squeeze()
    # ax.set(title=f"{s['r']:.3f}; 95% CI: {s['CI95%']}\np = {s['p-val']:.3f}",)


def soft_deconvolve(res: DataFrame, enr: DataFrame) -> DataFrame:
    # Correlate signatures with cell types
    # r = joint_ct_z
    r = res.T.dropna()
    corr = r.T.join(enr.T).corr()
    corr_d = corr.loc[r.index, enr.index]
    return corr_d


def plot_soft_deconvolution(
    corr_d: DataFrame, meta: DataFrame, enr: DataFrame
) -> None:
    grid = clustermap(
        corr_d.T,
        metric="correlation",
        center=0,
        cmap="RdBu_r",
        cbar_kws=dict(label="Pearson correlation"),
        figsize=(6, 6),
        yticklabels=True,
    )
    grid.ax_heatmap.set(title="all")
    grid.savefig(
        output_dir
        / f"ssGSEA_enrichment.correlation_to_cell_type.aggregated.all_ROIs.clustermap.svg",
        **figkws,
    )

    norm = None

    corrs = dict()
    fig1, axes1 = plt.subplots(
        2, 3, figsize=(3 * 4, 2 * 6), sharex=True, sharey=True
    )
    fig2, axes2 = plt.subplots(
        2, 3, figsize=(3 * 4, 2 * 6), sharex=True, sharey=True
    )
    for i, pheno in enumerate(meta["phenotypes"].cat.categories):
        rr = (
            r.T.join(meta["phenotypes"])
            .query(f"phenotypes == '{pheno}'")
            .T.drop("phenotypes")
            .astype(float)
        )
        corr = rr.T.join(enr.T).corr()
        corr_d = corr.loc[rr.index, enr.index]
        p = corr_d.copy()
        p.index = pheno + " - " + p.index
        corrs[pheno] = p

        if pheno == "Normal":
            norm = corr_d

        ax1 = axes1.flatten()[i]
        sns.heatmap(
            corr_d.T.iloc[
                grid.dendrogram_row.reordered_ind,
                grid.dendrogram_col.reordered_ind,
            ],
            cmap="RdBu_r",
            center=0,
            yticklabels=True,
            ax=ax1,
            vmin=-1,
            vmax=1,
        )
        ax2 = axes2.flatten()[i]
        sns.heatmap(
            (corr_d - norm).T.iloc[
                grid.dendrogram_row.reordered_ind,
                grid.dendrogram_col.reordered_ind,
            ],
            cmap="RdBu_r",
            center=0,
            yticklabels=True,
            ax=ax2,
            vmin=-1,
            vmax=1,
        )
        ax1.set(title=pheno)
        ax2.set(title=pheno)
    fig1.savefig(
        output_dir
        / f"ssGSEA_enrichment.correlation_to_cell_type.aggregated.per_phenotype.clustermap.svg",
        **figkws,
    )
    fig2.savefig(
        output_dir
        / f"ssGSEA_enrichment.correlation_to_cell_type.aggregated.per_phenotype.over_normal.clustermap.svg",
        **figkws,
    )

    corrj = pd.concat(corrs.values())
    for z in [None, 0, 1]:
        grid = clustermap(
            corrj,
            z_score=z,
            metric="correlation",
            center=0,
            cmap="RdBu_r",
            cbar_kws=dict(label="Pearson correlation"),
            figsize=(10, 12),
            xticklabels=True,
            yticklabels=True,
        )
        grid.ax_heatmap.set(title="all")
        grid.savefig(
            output_dir
            / f"ssGSEA_enrichment.correlation_to_cell_type.aggregated.by_pheno_joint.z_score{z}.clustermap.svg",
            **figkws,
        )

    for ct in norm.index:
        p = corrj.loc[corrj.index.str.contains(ct)].T
        p.columns = p.columns.str.extract("(.*) - .*")[0]
        grid = clustermap(
            p,
            z_score=0,
            metric="correlation",
            center=0,
            col_cluster=False,
            cmap="RdBu_r",
            cbar_kws=dict(label="Pearson correlation (Z-score)"),
            figsize=(3, 8),
            yticklabels=True,
            vmin=-2,
            vmax=2,
        )
        grid.ax_heatmap.set(title=ct, xlabel=None, ylabel=None)
        grid.savefig(
            output_dir
            / f"ssGSEA_enrichment.correlation_to_cell_type.aggregated.by_cell-{ct}.clustermap.svg",
            **figkws,
        )


def unsupervised_analysis(
    df: DataFrame,
    meta: DataFrame,
    res: DataFrame,
    joint_ct: DataFrame,
    meta_sample: DataFrame,
) -> None:
    # Unsupervised dimres

    # # original gene space
    clinvars = ["sample_id", "phenotypes", "location", "days_hospitalized"]
    a = AnnData(df.T.dropna(), obs=df.T.join(meta)[clinvars])
    sc.pp.normalize_total(a)
    sc.pp.scale(a)
    sc.tl.pca(a)
    fig = sc.pl.pca(
        a,
        color=clinvars,
        components=["1,2", "2,3", "3,4", "4,5"],
        show=False,
    )[0].figure
    fig.savefig(output_dir / "gene_expression.pca.svg", **figkws)
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    fig = sc.pl.umap(a, color=clinvars, s=150, show=False)[0].figure
    fig.savefig(output_dir / "gene_expression.umap.svg", **figkws)

    # # # only alveoli
    df_alv = (
        df.T.join(meta[["location"]])
        .loc[lambda x: x["location"] == "Alveolar"]
        .drop("location", 1)
        .T
    )
    a = AnnData(
        df_alv.T.dropna(),
        obs=df_alv.T.join(meta)[clinvars],
    )
    sc.pp.normalize_total(a)
    sc.pp.scale(a)
    sc.tl.pca(a)
    fig = sc.pl.pca(
        a,
        color=clinvars,
        components=["1,2", "2,3", "3,4", "4,5"],
        show=False,
    )[0].figure
    fig.savefig(output_dir / "gene_expression.only_alveoli.pca.svg", **figkws)

    sc.pp.neighbors(a)
    sc.tl.umap(a)
    fig = sc.pl.umap(a, color=clinvars, s=150, show=False)[0].figure
    fig.savefig(output_dir / "gene_expression.only_alveoli.umap.svg", **figkws)

    # # signature space
    a = AnnData(
        res.T.dropna().T,
        obs=res.join(meta)[clinvars],
    )
    sc.tl.pca(a)
    fig = sc.pl.pca(
        a,
        color=clinvars,
        components=["1,2", "2,3", "3,4", "4,5"],
        show=False,
    )[0].figure
    fig.savefig(output_dir / "signature_space.pca.svg", **figkws)
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    fig = sc.pl.umap(a, color=clinvars, s=150, show=False)[0].figure
    fig.savefig(output_dir / "signature_space.umap.svg", **figkws)

    # # # only alveoli
    res_alv = (
        res.join(meta[["location"]])
        .loc[lambda x: x["location"] == "Alveolar"]
        .drop("location", 1)
    )

    a = AnnData(
        res_alv.T.dropna().T,
        obs=res_alv.join(meta)[clinvars],
    )
    sc.tl.pca(a)
    fig = sc.pl.pca(
        a,
        color=clinvars,
        components=["1,2", "2,3", "3,4", "4,5"],
        show=False,
    )[0].figure
    fig.savefig(output_dir / "signature_space.only_alveoli.pca.svg", **figkws)
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    fig = sc.pl.umap(a, color=clinvars, s=150, show=False)[0].figure
    fig.savefig(output_dir / "signature_space.only_alveoli.umap.svg", **figkws)

    # # cell type space
    # enrz = (enr.T - enr.mean(1)) / enr.std(1)
    joint_ct_z = ((joint_ct.T - joint_ct.mean(1)) / joint_ct.std(1)).T

    a = AnnData(joint_ct_z.T, obs=joint_ct_z.T.join(meta)[clinvars])
    sc.tl.pca(a)
    fig = sc.pl.pca(
        a,
        color=clinvars,
        components=["1,2", "2,3", "3,4", "4,5"],
        show=False,
    )[0].figure
    fig.savefig(output_dir / "cell_type_space.pca.svg", **figkws)
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    fig = sc.pl.umap(a, color=clinvars, s=150, show=False)[0].figure
    fig.savefig(output_dir / "cell_type_space.umap.svg", **figkws)

    # # # only alveoli
    joint_ct_z_alv = (
        joint_ct_z.T.join(meta[["location"]])
        .loc[lambda x: x["location"] == "Alveolar"]
        .drop("location", 1)
    )

    a = AnnData(joint_ct_z_alv, obs=joint_ct_z_alv.join(meta)[clinvars])
    sc.tl.pca(a)
    fig = sc.pl.pca(
        a,
        color=clinvars,
        components=["1,2", "2,3", "3,4", "4,5"],
        show=False,
    )[0].figure
    fig.savefig(output_dir / "cell_type_space.only_alveoli.pca.svg", **figkws)
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    fig = sc.pl.umap(a, color=clinvars, s=150, show=False)[0].figure
    fig.savefig(output_dir / "cell_type_space.only_alveoli.umap.svg", **figkws)

    # # # location averaged
    r = joint_ct_z.T.join(meta[["sample_id"]]).groupby("sample_id").mean()

    a = AnnData(r, obs=r.join(meta_sample)[["phenotypes"]])
    sc.tl.pca(a)
    fig = sc.pl.pca(
        a,
        color=["phenotypes"],
        components=["1,2", "2,3", "3,4", "4,5"],
        s=150,
        show=False,
    )[0].figure
    fig.savefig(
        output_dir / "cell_type_space.location_averaged.pca.svg", **figkws
    )
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    fig = sc.pl.umap(a, color=["phenotypes"], s=150, show=False).figure
    fig.savefig(
        output_dir / "cell_type_space.location_averaged.umap.svg", **figkws
    )


def plot_rna_scope_viz(meta: DataFrame) -> None:
    # RNAScope images
    from skimage.exposure import equalize_hist as eq
    from imc.utils import minmax_scale

    PIL.Image.MAX_IMAGE_PIXELS = 933120000

    img = imageio.imread(data_dir / "Covid21_050820 with ROIs.png")
    plt.imshow(img[2700:4800, 5600:8500])
    meta.query("sample_id == 'Covid21_050820' & ROI == 3").squeeze()  # Airway
    meta.query("sample_id == 'Covid21_050820' & ROI == 4").squeeze()  # Vascular
    meta.query(
        "sample_id == 'Covid21_050820' & ROI == 12"
    ).squeeze()  # Alveolar

    cyan = (0, 255, 255)
    r = imageio.imread(data_dir / "Covid21_TMPRSS2_Red.tiff")
    b = imageio.imread(data_dir / "Covid21_SARS_Green.tiff")
    g = imageio.imread(data_dir / "Covid21_ACE2_Cyan.tiff")
    tmpr = minmax_scale(eq(r[:, :, 0]))
    sars = minmax_scale(eq(b[:, :, 1]))
    ace2 = minmax_scale(eq(np.log1p(g[:, :, 1])))
    dna = np.asarray(
        [
            minmax_scale(eq(x))
            for x in np.stack([r[:, :, 2], b[:, :, 2], g[:, :, 2]])
        ]
    ).mean(0)
    m = np.moveaxis(np.stack([tmpr, sars, ace2]), 0, -1)

    #

    # plot
    y1 = slice(1450, 3050, 1)
    x1 = slice(4990, 7113, 1)
    y2 = slice(2773, 4457, 1)
    x2 = slice(5627, 7755, 1)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(6 * 2, 6 * 3),
        gridspec_kw=dict(wspace=0.01, hspace=0.0001),
    )
    axes = axes.flatten()
    axes[0].imshow(tmpr[1450:3050, 4990:7113], cmap="Reds")
    axes[1].imshow(sars[1450:3050, 4990:7113], cmap="Greens")
    axes[2].imshow(ace2[1450:3050, 4990:7113], cmap="Blues")
    axes[3].imshow(dna[1450:3050, 4990:7113], cmap="Purples")
    axes[4].imshow(m[1450:3050, 4990:7113])
    axes[5].imshow(img[2773:4457, 5627:7755])
    for ax, t in zip(
        axes,
        [
            "TMPRSS2",
            "SARS-CoV-2 Spike",
            "ACE2",
            "DNA",
            "Overlay (TMPRSS2, Spike, ACE2)",
            "Labeling image",
        ],
    ):
        ax.set(title=t)
        ax.axis("off")
    fig.savefig(
        output_dir / "RNAScope.COVID21.TMPRSS2_SARS_ACE2.svg",
        dpi=300,
        bbox_inches="tight",
    )


def compare_dataset_markers_head_to_head(
    df3: DataFrame, meta: DataFrame
) -> None:
    # Direct expression comparison between IMC and GeoMx
    matching_genes = {
        "iNOS": "NOS2",
        "cKIT": "KIT",
        "CD206": "MRC1",
        "CD16": "FCGR3A",
        "CD16": "FCGR3B",
        "CD163": "CD163",
        "CD14": "CD14",
        "CD11b": "ITGAM",
        "CD68": "CD68",
        "CD31": "PECAM1",
        "CD4": "CD4",
        "CD20": "MS4A1",
        "CD19": "CD19",
        "pSTAT3": "STAT3",
        "CD56": "NCAM1",
        "Keratin8": "KRT8",
        "Keratin18": "KRT18",
        "IL6": "IL6",
        "CD8": "CD8A",
        "CD15": "FUT4",
        "Arginase1": "ARG1",
        "IL1B": "IL1B",
        "Ki67": "MKI67",
        "CleavedCaspase3": "CASP3",
        "MasCellTryptase": "TPSAB1",
        "CD11c": "ITGAX",
        "MPO": "MPO",
        "TTF1": "NKX2-1",
        "CD45": "PTPRC",
        "CD57": "B3GAT1",
        "CD3": "CD3D",
        "CollagenTypeI": "COL1A1",
        "pCREB": "CREBBP",
    }

    mgm = {v: k for k, v in matching_genes.items()}

    dfp = df3.reindex(matching_genes.values()).dropna()
    dfp.index = dfp.index.to_series().replace(mgm)
    dfp = dfp.T.join(meta["phenotypes"])

    n, m = get_grid_dims(dfp.shape[1] - 1)

    fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 4), sharex=True)
    for i, g in enumerate(dfp.columns[:-1]):
        ax = axes.flatten()[i]
        stats = swarmboxenplot(
            data=dfp,
            x="phenotypes",
            y=g,
            ax=ax,
            plot_kws=dict(palette=colors["phenotypes"]),
        )
        ax.set(title=g, xlabel=None, ylabel=None)
    for ax in axes.flatten()[i + 1 :]:
        ax.axis("off")
    fig.savefig(
        output_dir
        / "IMC_marker_expression_in_GeoMx.by_phenotype.swarmboxenplot.svg",
        **figkws,
    )

    # explore coexpression
    corrs = dict()
    for pheno in dfp["phenotypes"].cat.categories:
        corrs[pheno] = dfp.query(f"phenotypes == '{pheno}'").corr()
    # get difference over healthy
    corrs_norm = {
        k: (c + 1) / 2 - (corrs["Normal"] + 1) / 2 for k, c in corrs.items()
    }

    # Plot correlation across all
    grid = clustermap(dfp.corr(), cmap="RdBu_r", center=0, metric="correlation")

    n, m = get_grid_dims(dfp["phenotypes"].nunique() - 1)
    fig, axes = plt.subplots(n, m, figsize=(n * 5, m * 5), sharex=True)
    for i, pheno in enumerate(dfp["phenotypes"].cat.categories):
        ax = axes.flatten()[i]
        p = corrs[pheno].iloc[
            grid.dendrogram_row.reordered_ind, grid.dendrogram_col.reordered_ind
        ]
        sns.heatmap(
            p,
            cmap="RdBu_r",
            center=0,
            cbar_kws=dict(label="Gene co-expression"),
            xticklabels=True,
            yticklabels=True,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
        )
        ax.set(title=pheno)
    fig.savefig(
        output_dir
        / "IMC_marker_expression_in_GeoMx.expression.by_phenotype.clustermap.svg",
        **figkws,
    )

    # Plot difference of co-expression over healthy

    n, m = get_grid_dims(dfp["phenotypes"].nunique() - 1)
    fig, axes = plt.subplots(n, m, figsize=(n * 5, m * 5), sharex=True)

    # # plot original correlation for healthy
    sns.heatmap(
        corrs["Normal"].iloc[
            grid.dendrogram_row.reordered_ind, grid.dendrogram_col.reordered_ind
        ],
        cmap="RdBu_r",
        center=0,
        cbar_kws=dict(label="Gene co-expression in Healthy lung"),
        xticklabels=True,
        yticklabels=True,
        vmin=-1,
        vmax=1,
        square=True,
        ax=axes[0, 0],
    )
    axes[0, 0].set(title="Normal")
    for i, pheno in enumerate(dfp["phenotypes"].cat.categories[1:], 1):
        ax = axes.flatten()[i]
        p = corrs_norm[pheno].iloc[
            grid.dendrogram_row.reordered_ind, grid.dendrogram_col.reordered_ind
        ]
        sns.heatmap(
            p,
            cmap="RdBu_r",
            center=0,
            cbar_kws=dict(label="Difference in co-expression over healthy"),
            xticklabels=True,
            yticklabels=True,
            vmin=-0.5,
            vmax=0.5,
            square=True,
            ax=ax,
        )
        ax.set(title=pheno)
    fig.savefig(
        output_dir
        / "IMC_marker_expression_in_GeoMx.expression.by_phenotype.over_normal.clustermap.svg",
        **figkws,
    )

    for gene in ["CD45", "CD31", "Keratin18", "CD3"]:
        cd45c = pd.DataFrame({k: c.loc[gene] for k, c in corrs.items()}).drop(
            gene
        )

        gkws = dict(
            cmap="RdBu_r",
            center=0,
            col_cluster=False,
            figsize=(3, 6),
            yticklabels=True,
        )
        grid = clustermap(
            cd45c,
            metric="correlation",
            cbar_kws=dict(label=f"{gene} co-expression"),
            **gkws,
        )
        grid.savefig(
            output_dir
            / f"IMC_marker_expression_in_GeoMx.co-expression_with_{gene}.by_phenotype.clustermap.svg",
            **figkws,
        )
        grid = clustermap(
            cd45c,
            metric="correlation",
            cbar_kws=dict(label=f"{gene} co-expression\n(row Z-score)"),
            **gkws,
            z_score=0,
        )
        grid.savefig(
            output_dir
            / f"IMC_marker_expression_in_GeoMx.co-expression_with_{gene}.by_phenotype.z_score.clustermap.svg",
            **figkws,
        )
        cd45cn = pd.DataFrame(
            {k: c.loc[gene] for k, c in corrs_norm.items()}
        ).drop("Normal", 1)
        grid = clustermap(
            cd45cn,
            cbar_kws=dict(label=f"{gene} co-expression (over Healthy)"),
            **gkws,
        )
        grid.savefig(
            output_dir
            / f"IMC_marker_expression_in_GeoMx.co-expression_with_{gene}.by_phenotype.over_normal.clustermap.svg",
            **figkws,
        )


def soft_deconvolution_with_scRNAseq_data(
    df: DataFrame, meta: DataFrame
) -> None:

    # Use scRNA-seq data
    input_dir = Path("data") / "krasnow_scrna-seq"
    input_dir.mkdir()
    mean_file = input_dir / "krasnow_hlca_10x.average.expression.csv"
    if not mean_file.exists():
        a = sc.read(
            input_dir / "droplet_normal_lung_blood_scanpy.20200205.RC4.h5ad"
        )
        obs = pd.read_csv(
            input_dir / "krasnow_hlca_10x_metadata.csv", index_col=0
        )
        mean = a.to_df().join(obs).groupby("free_annotation").mean().T
        mean.iloc[:-10, :].to_csv(mean_file)

    mean = pd.read_csv(mean_file, index_col=0)

    zm = ((mean.T - mean.mean(1)) / mean.std(1)).T.dropna()

    df.index[~df.index.isin(zm.index)]  # missing genes

    dg = zm.reindex(df.index).dropna()

    cell_groups = {
        "Basal": "Basal|Goblet",
        "Endothelial": "Artery|Vein|Vessel|Capilary",
        "Vascular": "Fibroblast|Pericyte|Smooth|Fibromyocyte|Myofibroblast",
        "Epithelial": "Epithelial",
        "Myeloid": "Monocyte|Macrophage|Dendritic",
        "Ciliated": "Ciliated|Ionocyte",
        "Lymphoid": "CD4|CD8|Plasma|B",
    }

    _super_means = dict()
    for group, groupstr in cell_groups.items():
        i = dg.columns.str.contains(groupstr)
        _super_means["SG_" + group] = dg.loc[:, i].mean(1)
    super_means = pd.DataFrame(_super_means)

    super_corrs = (
        dg.join(super_means).corr().loc[dg.columns, super_means.columns]
    )

    for ext, colcolors in [("", None), (".with_supergroups", super_corrs)]:
        grid = clustermap(
            dg.T,
            center=0,
            cmap="RdBu_r",
            yticklabels=True,
            xticklabels=False,
            robust=True,
            rasterized=True,
            dendrogram_ratio=0.1,
            cbar_kws=dict(label="Expression Z-score"),
            row_colors=colcolors,
        )
        grid.ax_heatmap.set(xlabel=f"GeoMx genes only (n = {df.shape[0]})")
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

    # soft deconvolve
    output_prefix = output_dir / "krasnow_scRNA_deconvolve.correlation"
    dc = df.join(dg).corr().loc[df.columns, dg.columns]

    sign = (dc > 0).astype(int).replace(0, -1)
    dcs = dc.abs() ** (1 / 3) * sign

    grid = clustermap(
        dc.T,
        center=0,
        cmap="RdBu_r",
        xticklabels=False,
        yticklabels=True,
        robust=True,
        col_colors=meta[["location", "phenotypes"]],
        metric="correlation",
        cbar_kws=dict(label="Pearson correlation"),
        dendrogram_ratio=0.1,
        rasterized=True,
    )
    grid.savefig(
        output_prefix + ".clustermap.svg",
        **figkws,
    )

    good = dc.index[dc.max(1) > 0.1]

    grid = clustermap(
        dc.loc[good].T,
        center=0,
        cmap="RdBu_r",
        xticklabels=False,
        yticklabels=True,
        robust=True,
        col_colors=meta[["location", "phenotypes"]],
        metric="correlation",
        cbar_kws=dict(label="Pearson correlation"),
        dendrogram_ratio=0.1,
        rasterized=True,
    )
    grid.savefig(
        output_prefix + ".clustermap.only_good_rois.svg",
        **figkws,
    )

    grid = clustermap(
        dc.T,
        center=0,
        cmap="RdBu_r",
        xticklabels=False,
        yticklabels=True,
        robust=True,
        col_colors=meta[["location", "phenotypes"]],
        metric="correlation",
        standard_scale=1,
        cbar_kws=dict(label="Pearson correlation"),
        dendrogram_ratio=0.1,
        rasterized=True,
    )
    grid.savefig(
        output_prefix + ".clustermap.std_scale.svg",
        **figkws,
    )

    p = (dcs.T + 1) / 2
    p -= p.min()
    normdcs = p / p.sum()

    grid = clustermap(
        normdcs * 100,
        xticklabels=False,
        yticklabels=True,
        robust=True,
        col_colors=meta[["location", "phenotypes"]],
        metric="correlation",
        cbar_kws=dict(label="Pearson correlation"),
        dendrogram_ratio=0.1,
        rasterized=True,
    )
    grid.savefig(
        output_prefix + ".clustermap.norm.svg",
        **figkws,
    )

    a = AnnData(dcs, obs=meta)
    sc.pp.pca(a)
    sc.pp.neighbors(a)
    sc.tl.umap(a)  # , gamma=0.0001)
    axes = sc.pl.pca(
        a, color=["sample_id", "location", "phenotypes"], show=False
    )
    fig = axes[0].figure
    fig.savefig(output_prefix + ".pca.svg", **figkws)

    axes = sc.pl.umap(
        a, color=["sample_id", "location", "phenotypes"], show=False
    )
    fig = axes[0].figure
    fig.savefig(output_prefix + ".umap.svg", **figkws)

    # s = dc.abs().mean(1).sort_values()
    # s = dc.max(1).sort_values()
    # plt.scatter(s.rank(), s)
    for matrix, label in [(dc, "correlation"), (normdcs.T, "norm_correlation")]:
        for loc in ["All"] + meta["location"].unique().tolist():

            if loc == "All":
                matrixloc = matrix
            else:
                matrixloc = (
                    matrix.join(meta["location"])
                    .query(f"location == '{loc}'")
                    .drop("location", 1)
                )

            n, m = get_grid_dims(matrixloc.shape[1])
            fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 4), sharex=True)
            _stats = list()
            for i, c in enumerate(matrixloc.columns):
                ax = axes.flatten()[i]
                stats = swarmboxenplot(
                    data=matrixloc[[c]].join(meta),
                    x="phenotypes",
                    y=c,
                    ax=ax,
                    plot_kws=dict(palette=colors["phenotypes"]),
                    test_kws=dict(parametric=False),
                )
                ax.set(title=c, xlabel=None, ylabel=True)
                means = (
                    matrixloc[[c]]
                    .join(meta["phenotypes"])
                    .groupby("phenotypes")
                    .mean()
                    .squeeze()
                )
                _stats.append(stats.assign(cell_type=c, **means.to_dict()))
            for ax in axes.flatten()[i + 1 :]:
                ax.axis("off")
            fig.savefig(
                output_prefix + f"{label}.only{loc}.swarmboxenplot.svg",
                **figkws,
            )
            stats = pd.concat(_stats).reset_index(drop=True)
            stats.to_csv(output_prefix + f".{label}.only{loc}.csv", index=False)

            ### volcano plot
            combs = stats[["A", "B"]].drop_duplicates().reset_index(drop=True)
            stats["hedges"] *= -1
            stats["logp-unc"] = -np.log10(stats["p-unc"].fillna(1))
            stats["logp-cor"] = -np.log10(stats["p-cor"].fillna(1))
            stats["p-cor-plot"] = (
                stats["logp-cor"] / stats["logp-cor"].max()
            ) * 5
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
            fig.savefig(
                output_prefix + f"{label}.only{loc}.volcano.svg",
                **figkws,
            )

    # Try to collapse the scRNA cell types to match the IMC ones
    not_used = [
        "Capillary Aerocyte",
        "Capillary Intermediate 1",
        "Capillary Intermediate 2",
        "Goblet",
        "Ionocyte",
        "Lymphatic",
        "Mucous",
        "Neuroendocrine",
        "Pericyte",
        "Platelet/Megakaryocyte",
        "Proliferating NK/T",
        "Serous",
    ]

    cell_type_mapping = {
        "B_cell": ["B", "Plasma"],
        "CD4_T": [
            "CD4+ Memory/Effector T",
            "CD4+ Naive T",
        ],
        "CD8_T": [
            "CD8+ Memory/Effector T",
            "CD8+ Naive T",
        ],
        "Club": ["Club"],
        "Dendritic": [
            "EREG+ Dendritic",
            "IGSF21+ Dendritic",
            "Myeloid Dendritic Type 1",
            "Myeloid Dendritic Type 2",
            # 'Plasmacytoid Dendritic',
            "TREM2+ Dendritic",
        ],
        "Epithelial": [
            "Alveolar Epithelial Type 1",
            "Alveolar Epithelial Type 2",
            "Signaling Alveolar Epithelial Type 2",
            "Ciliated",
            "Proximal Ciliated",
            "Basal",
            "Differentiating Basal",
            "Proliferating Basal",
            "Proximal Basal",
        ],
        "Endothelial": [
            "Artery",
            "Vein",
            # "Vessel",
            "Capillary",
            "Bronchial Vessel 1",
            "Bronchial Vessel 2",
        ],
        "Fibroblast": [
            "Adventitial Fibroblast",
            "Alveolar Fibroblast",
            "Fibromyocyte",
            "Lipofibroblast",
            "Myofibroblast",
        ],
        "Macrophage": [
            "Macrophage",
            "Proliferating Macrophage",
        ],
        "Mast": [
            "Basophil/Mast 1",
            "Basophil/Mast 2",
        ],
        "Mesenchymal": ["Mesothelial"],
        "Monocyte": [
            "Classical Monocyte",
            "Intermediate Monocyte",
            "Nonclassical Monocyte",
            "OLR1+ Classical Monocyte",
        ],
        "NK_cell": [
            "Natural Killer",
            "Natural Killer T",
        ],
        # "Neutrophil": [],
        "Smooth_muscle": [
            "Airway Smooth Muscle",
            "Vascular Smooth Muscle",
        ],
    }

    ctm = pd.DataFrame(
        {name: dc[group].mean(1) for name, group in cell_type_mapping.items()}
    )

    ctm = ctm.loc[ctm.sum(1) > 0.2]

    fig, stats = swarmboxenplot(
        data=ctm.join(meta),
        x="phenotypes",
        y=ctm.columns.tolist(),
        test_kws=dict(parametric=False),
        plot_kws=dict(palette=colors["phenotypes"]),
    )
    stats = stats.rename(columns={"Variable": "cell_type"})
    stats.to_csv(
        output_dir
        / f"krasnow_scRNA_deconvove.aggregated_matching_imc.All.swarmboxenplot.csv",
        index=False,
    )


def compare_imc_and_geomx_cell_type_coefficients_scrnaseq() -> None:
    # Compare "coefficients"

    gmcoef = pd.read_csv(
        output_dir
        / f"krasnow_scRNA_deconvove.aggregated_matching_imc.All.swarmboxenplot.csv",
    )
    gmcoef["cell_type"] = (
        gmcoef["cell_type"]
        .str.replace("_", " ")
        .str.replace(" cell", "")
        .str.replace(r"s$", "")
    )
    gmcoef["hedges"] *= -1
    gmcoef["A"] = (
        gmcoef["A"]
        .replace("Normal", "Healthy")
        .replace("Early", "COVID19_early")
        .replace("Late", "COVID19_late")
    )
    gmcoef["B"] = (
        gmcoef["B"]
        .replace("Normal", "Healthy")
        .replace("Early", "COVID19_early")
        .replace("Late", "COVID19_late")
    )
    imcoef = pd.read_csv(
        Path("results")
        / "cell_type"
        / "clustering.roi_zscored.filtered.fraction.cluster_1.0.differences.csv"
    )
    imcoef = imcoef.loc[
        lambda x: (~x["cell_type"].str.contains(" - "))
        & (x["Contrast"] == "phenotypes")
        & (x["measure"] == "area")
        & (x["grouping"] == "roi")
    ]
    imcoef["cell_type"] = (
        imcoef["cell_type"]
        .str.replace(" cells", "")
        .str.replace("-cells", "")
        .str.replace(r"s$", "")
    )
    imcoef["hedges"] *= -1

    gmcoef["mlogq"] = -np.log10(gmcoef["p-cor"])
    for metric in ["median_A", "median_B", "hedges", "p-cor", "mlogq"]:
        gmcoef[f"geo_{metric}"] = gmcoef[metric]
        gmcoef = gmcoef.drop(metric, 1)
    imcoef["mlogq"] = -np.log10(imcoef["p-cor"])
    for metric in ["median_A", "median_B", "hedges", "p-cor", "mlogq"]:
        imcoef[f"imc_{metric}"] = imcoef[metric]
        imcoef = imcoef.drop(metric, 1)

    p = gmcoef.merge(imcoef, on=["Contrast", "A", "B", "cell_type"])

    # plot
    for (x, y), label in [
        (("COVID19_early", "COVID19_late"), "Late-vs-Early"),
        (("Healthy", "COVID19_late"), "Late-vs-Healthy"),
        (("Healthy", "COVID19_early"), "Early-vs-Healthy"),
        (("Healthy", "Flu"), "Flu-vs-Healthy"),
        (("Healthy", "ARDS"), "ARDS-vs-Healthy"),
        (("Healthy", "Pneumonia"), "Pneumonia-vs-Healthy"),
    ]:
        p2 = p.loc[(p["A"] == x) & (p["B"] == y)].set_index("cell_type")

        fig, axes = plt.subplots(1, 3, figsize=(3 * 4.25, 1 * 3.75))
        for i, meas in enumerate(["hedges", "p-cor", "mlogq"]):
            xx = f"imc_{meas}"
            yy = f"geo_{meas}"
            mm = p2[[xx, yy]].mean(1)
            s = pg.corr(p2[xx], p2[yy]).squeeze()

            vmax = p2[[xx, yy]].abs().values.max()
            vmax += vmax * 0.1
            if meas == "hedges":
                vmin = -vmax
            else:
                vmin = -(vmax * 0.05)

            sns.regplot(
                data=p2,
                x=xx,
                y=yy,
                line_kws=dict(alpha=0.1, color="black"),
                color="black",
                scatter=True,
                ax=axes[i],
            )
            axes[i].scatter(
                data=p2,
                x=xx,
                y=yy,
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
            for t in p2.index:
                axes[i].text(
                    p2.loc[t, xx],
                    p2.loc[t, yy],
                    s=t,
                    ha="left" if mm.loc[t] < 0 else "right",
                )
        fig.savefig(
            output_dir
            / f"coefficient_comparison.scRNA-seq.{label}.scatter.svg",
            **figkws,
        )

    # # All together
    # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    # ax.scatter(p["imc_hedges"], p["geo_hedges"])
    # s = pg.corr(p["imc_hedges"], p["geo_hedges"]).squeeze()
    # ax.set(
    #     title=f"{s['r']:.3f}; 95% CI: {s['CI95%']}\np = {s['p-val']:.3f}",
    # )


def new_mac_signatures(df: DataFrame, meta: DataFrame) -> None:
    from typing import List
    import requests
    from bs4 import BeautifulSoup
    from tqdm import tqdm

    def get_enrichments(
        gssns: List[str], df: DataFrame, prev_enr: DataFrame = None
    ) -> DataFrame:
        if prev_enr is None:
            enr = pd.DataFrame(columns=df.columns)
        else:
            enr = prev_enr

        _enr = list()
        for gssn in tqdm(gssns):
            url = f"{base_url}/download_geneset.jsp?geneSetName={gssn}&fileType=gmt"
            req = requests.get(url)
            with open("geneset.gmt", "w") as handle:
                try:
                    handle.write(req.content.decode())
                except UnicodeDecodeError:
                    continue

            # n = len(open("geneset.gmt", "r").read().split("\t"))
            # tqdm.write(f"{gssn}: {n}")
            try:
                res = parmap.map(
                    ssgsea, df.columns, database="geneset.gmt", x=df
                )
            except Exception:
                continue
            enr = pd.concat(res, axis=1)
            enr.columns = df.columns
            _enr.append(enr)
        enr = pd.concat(_enr)
        return enr

    # Use a predetermined list of mac signatures
    gs = [
        "COATES_MACROPHAGE_M1_VS_M2_UP",
    ]
    enr = get_enrichments(gs, df)

    accept = ["Normal", "Pneumonia", "Early", "Late"]

    enr = enr.T.join(meta["phenotypes"])
    enr = enr.loc[enr["phenotypes"].isin(accept)].drop("phenotypes", 1).T

    meta = meta.loc[meta["phenotypes"].isin(accept)]
    meta["phenotypes"] = meta["phenotypes"].cat.remove_unused_categories()

    fig, stats = swarmboxenplot(
        data=enr.T.join(meta),
        x="phenotypes",
        y=enr.index.tolist(),  # sel
        plot_kws=dict(palette=colors["phenotypes"]),
    )
    fig.savefig(
        output_dir / "ssGSEA_enrichment.COATES_macrophage_signature.svg",
        **figkws,
    )

    fig, stats = swarmboxenplot(
        data=df.T.join(meta),
        x="phenotypes",
        y=["CD163", "CD14", "MRC1", "IL3RA"],
        plot_kws=dict(palette=colors["phenotypes"]),
    )
    fig.savefig(output_dir / "macrophage_marker_expression.svg", **figkws)

    # Or scrape GSEA
    base_url = "https://www.gsea-msigdb.org/gsea/msigdb"
    url = f"{base_url}/genesets.jsp?collection=C7"
    req = requests.get(url)

    soup = BeautifulSoup(req.content, "html.parser")
    table = soup.find("table", {"id": "geneSetTable"})
    t = pd.read_html(str(table))[0]
    _gssns = list()
    for col in t.columns:
        _gssns += t[col].str.split(" ").apply(pd.Series).stack().tolist()

    gssns = pd.Series(_gssns)
    macs = gssns[gssns.str.contains("MACROPHAGE")]

    try:  # in case it gets stuck in the loop, redo from here
        enr = pd.read_csv(
            output_dir / "ssGSEA_enrichment.C7.macrophage_signatures.all.csv",
            index_col=0,
        )
        gssn = enr.index[-1]
        macs = macs.loc[macs.index >= macs[macs == gssn].index[0]]
    except FileNotFoundError:
        enr = pd.DataFrame(columns=df.columns)

    enr = get_enrichments(macs, df, enr)
    enr.to_csv(
        output_dir / "ssGSEA_enrichment.C7.macrophage_signatures.all.csv"
    )

    # Plot all
    grid = clustermap(enr.T, config="z_score", row_colors=meta[["phenotypes"]])

    # Try to narrow it down
    enrm = enr.T.join(meta["phenotypes"]).groupby("phenotypes").mean().T

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(enrm.std(1), enrm.mean(1))

    sel = (enrm.std(1) / enrm.mean(1)).sort_values().tail(20).index.tolist()
    fc = np.log(enrm["Late"] / enrm["Early"])
    sel = fc.sort_values().dropna().tail(6).index.tolist()

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(enrm.mean(1), fc)

    fig, stats = swarmboxenplot(
        data=enr.T.join(meta),
        x="phenotypes",
        y=enr.index.tolist(),  # sel
        plot_kws=dict(palette=colors["phenotypes"]),
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\t - Exiting due to user interruption.")
        sys.exit(1)
