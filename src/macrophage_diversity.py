#!/usr/bin/env python

"""
Classification of cells into cell types/states.
"""

import re
from typing import List
from dataclasses import dataclass, field

from anndata import AnnData  # type: ignore[import]
import scanpy as sc  # type: ignore[import]

from imc.graphics import get_grid_dims, rasterize_scanpy
from imc.utils import align_channels_by_name

from src.config import *
from src.operations import (
    plot_umap,
    plot_umap_with_labeled_clusters,
    plot_cluster_heatmaps,
    plot_cluster_illustrations,
    plot_cluster_heatmaps_with_labeled_clusters,
)
from src.utils import z_score_by_column
from imc.graphics import rasterize_scanpy
from imc.types import Axis


@dataclass
class FakeFigure:
    axes: List[Axis] = field(default_factory=list)


# Let's declare some variables for later use
output_dir = results_dir / "cell_type" / "macrophage"
output_dir.mkdir()
ids = [
    "roi",
    "sample",
]
cell_filtering_string_query = (
    "area > -1.5 & solidity > -1 & eccentricity < 1 & total > 2 & total < 7"
)


# # Using original clusters
# prefix = "roi_zscored.filtered."
# ann = sc.read_h5ad(str(h5ad_file).replace("processed", f"{prefix}processed"))

# cluster_str = "cluster_1.0"
# new_labels = json.load(open("metadata/cluster_names.json"))[
#     f"{prefix};{cluster_str}"
# ]
# macrophage_ints = [
#     k
#     for k, v in new_labels.items()
#     if ("Macrophages" in v) or ("Monocytes" in v)
# ]

# ax = sc.pl.heatmap(
#     ann,
#     var_names=ann.var.index,
#     groupby="cluster_1.0",
#     use_raw=True,
#     show=False,
# )["heatmap_ax"]
# ax.figure.savefig(
#     output_dir / "original_clusters.all_clusters.heatmap.all_markers.svg",
#     **figkws,
# )

# macs2 = ann[ann.obs["cluster_1.0"].isin(macrophage_ints)]
# ax = sc.pl.heatmap(
#     macs2,
#     var_names=macs2.var.index,
#     groupby="cluster_1.0",
#     use_raw=True,
#     show=False,
# )["heatmap_ax"]
# ax.figure.savefig(
#     output_dir
#     / "original_clusters.macrophage_monocyte.heatmap.all_markers.svg",
#     **figkws,
# )

# # # get proportion per phenotype
# q = (
#     macs2.obs.groupby("cluster_1.0")["phenotypes"]
#     .value_counts()
#     .rename("percentage")
# )
# q /= macs2.obs.groupby("cluster_1.0").size()
# q *= 100
# ax = (
#     q.reset_index()
#     .pivot_table(index="cluster_1.0", columns="phenotypes", values="percentage")
#     .plot(
#         kind="bar",
#         stacked=True,
#         colormap=matplotlib.colors.ListedColormap(colors.get("phenotypes")),
#     )
# )
# ax.figure.savefig(
#     output_dir / "original_clusters.macrophage_monocyte.proportions.svg",
#     **figkws,
# )


# Re-cluster
prefix = "roi_zscored.filtered."
ann = sc.read_h5ad(str(h5ad_file).replace("processed", f"{prefix}processed"))

quant = pd.read_parquet(quantification_file).drop(
    channels_exclude, axis=1, errors="ignore"
)
# # Z-score per image
zquant = z_score_by_column(quant, "roi")

# # keep track of total per cell in original (log) units
zquant["total"] = np.log1p(
    quant.loc[:, quant.columns.str.contains(r"\(")].sum(1).values
)
zquant = zquant.query(cell_filtering_string_query).drop("total", 1)

quant = quant.loc[
    :, quant.columns[quant.columns.str.contains(r"\(")].tolist() + ids
].drop("SARSCoV2S1(Eu153)", 1)
zquant = zquant.loc[:, quant.columns]

# Plots
set_prj_clusters(aggregated=True)

c = ann.obs.merge(
    prj.clusters, left_on=["sample", "roi", "obj_id"], right_index=True
)
i = c[c["cluster"].str.contains("Macrophages|Monocytes")].index
macs = ann[i, :]

# macs_quant = (
#     quant.reset_index(drop=True)
#     .loc[zquant.reset_index().loc[i.astype(int)].index]
#     .drop(ids, 1)
#     .reset_index(drop=True)
# )

# macs = AnnData(macs_quant, obs=macs.obs.reset_index(drop=True))
# sc.pp.log1p(macs)
# macs.raw = macs
# sc.pp.scale(macs, max_value=2.5)
# sc.pp.pca(macs)
# sc.pp.neighbors(macs, n_neighbors=15)
# sc.tl.umap(macs)
# sc.tl.leiden(macs, resolution=0.2, key_added="macs_0.2")
# sc.tl.leiden(macs, resolution=0.5, key_added="macs_0.5")
# sc.tl.leiden(macs, resolution=1.0, key_added="macs_1.0")
# # sc.pl.pca(macs, color=["cluster_1.0", "macs_0.5", "macs_1.0"])
# sc.write(output_dir / "anndata.h5ad", macs)

# m = [
#     "CD68(Nd150)",
#     "CD206(Nd144)",
#     "CD14(Nd148)",
#     "CD16(Nd146)",
#     "CD123(Er167)",
#     "CD163(Sm147)",
# ]
# p = macs[macs.obs.index.to_series().sample(frac=1).values, :]
# p.uns["disease_colors"] = list(
#     map(matplotlib.colors.to_hex, colors.get("disease"))
# )
# p.uns["phenotypes_colors"] = list(
#     map(matplotlib.colors.to_hex, colors.get("phenotypes"))
# )

# sc.pl.umap(
#     p,
#     color=[
#         "disease",
#         "phenotypes",
#         "cluster_1.0",
#         "macs_0.2",
#         "macs_0.5",
#         "macs_1.0",
#     ]
#     + m,
#     use_raw=False,
#     vmax=3,
# )

# for cluster_str in ["macs_0.5", "macs_1.0"]:
#     ax = sc.pl.heatmap(
#         macs,
#         var_names=macs.var.index,
#         groupby=cluster_str,
#         use_raw=True,
#         vmax=2.5,
#         show=False,
#     )["heatmap_ax"]
#     ax.figure.savefig(
#         output_dir
#         / f"re-clustered.macrophage_monocyte.{cluster_str}.heatmap.all_markers.svg",
#         **figkws,
#     )

#     q = (
#         macs.obs.groupby(cluster_str)["phenotypes"]
#         .value_counts()
#         .rename("percentage")
#     )
#     q /= macs.obs.groupby(cluster_str).size()
#     q *= 100
#     ax = (
#         q.reset_index()
#         .pivot_table(
#             index=cluster_str, columns="phenotypes", values="percentage"
#         )
#         .plot(
#             kind="bar",
#             stacked=True,
#             colormap=matplotlib.colors.ListedColormap(colors.get("phenotypes")),
#         )
#     )
#     ax.figure.savefig(
#         output_dir
#         / f"re-clustered.macrophage_monocyte.{cluster_str}.proportions.svg",
#         **figkws,
#     )


zmacs_quant = (
    zquant.reset_index(drop=True)
    .loc[i.astype(int)]
    .drop(ids, 1)
    .reset_index(drop=True)
)
zmacs = AnnData(zmacs_quant, obs=macs.obs.reset_index(drop=True))
zmacs.raw = zmacs
sc.pp.pca(zmacs)
sc.pp.neighbors(zmacs, n_neighbors=15)
sc.tl.umap(zmacs)
sc.tl.leiden(zmacs, resolution=0.2, key_added="zmacs_0.2")
sc.tl.leiden(zmacs, resolution=0.5, key_added="zmacs_0.5")
sc.tl.leiden(zmacs, resolution=1.0, key_added="zmacs_1.0")
# sc.pl.pca(zmacs, color=["disease"])
sc.write(output_dir / "anndata.z-score.h5ad", zmacs)


m = [
    "CD68(Nd150)",
    "CD206(Nd144)",
    "CD14(Nd148)",
    "CD16(Nd146)",
    "CD123(Er167)",
    "CD163(Sm147)",
    "CD11c(Yb176)",
    "CD11b(Sm149)",
    "IL1beta(Er166)",
    "iNOS(Nd142)",
]
p = zmacs[zmacs.obs.index.to_series().sample(frac=1).values, :]
p.uns["disease_colors"] = list(
    map(matplotlib.colors.to_hex, colors.get("disease"))
)
p.uns["phenotypes_colors"] = list(
    map(matplotlib.colors.to_hex, colors.get("phenotypes"))
)

_fig = FakeFigure(
    sc.pl.umap(
        p,
        color=[
            "disease",
            "phenotypes",
            "cluster_1.0",
            "zmacs_0.2",
            "zmacs_0.5",
            "zmacs_1.0",
        ]
        + m,
        use_raw=False,
        vmax=3,
        show=False,
    )
)
rasterize_scanpy(_fig)
_fig.axes[0].figure.savefig(
    output_dir
    / f"re-clustered.macrophage_monocyte.z-score.umap.identity_markers.svgz",
    **figkws,
)


dists = pd.read_parquet(
    results_dir / "pathology" / "cell_distance_to_lacunae.pq"
)
dists.index.name = "obj_id"

lacs = ["pos_lacunae", "neg_lacunae"]  # 'parenchyma_lacunae',

for cluster_str in ["cluster_1.0", "zmacs_0.2", "zmacs_0.5", "zmacs_1.0"]:
    ax = sc.pl.heatmap(
        zmacs,
        var_names=zmacs.var.index,
        groupby=cluster_str,
        use_raw=True,
        vmax=2.5,
        show=False,
    )["heatmap_ax"]
    ax.figure.savefig(
        output_dir
        / f"re-clustered.macrophage_monocyte.z-score.{cluster_str}.heatmap.all_markers.svg",
        **figkws,
    )

    q = (
        zmacs.obs.groupby(cluster_str)["phenotypes"]
        .value_counts()
        .rename("percentage")
    )
    q /= zmacs.obs.groupby(cluster_str).size()
    q *= 100
    ax = (
        q.reset_index()
        .pivot_table(
            index=cluster_str, columns="phenotypes", values="percentage"
        )
        .plot(
            kind="bar",
            stacked=True,
            colormap=matplotlib.colors.ListedColormap(colors.get("phenotypes")),
        )
    )
    ax.figure.savefig(
        output_dir
        / f"re-clustered.macrophage_monocyte.z-score.{cluster_str}.proportions.svg",
        **figkws,
    )

    d = zmacs.obs.merge(dists.reset_index()).groupby(cluster_str)[lacs].mean()
    fig, ax = plt.subplots(1, 1, figsize=(2, 4))
    sns.heatmap((d - d.mean()) / d.std(), cmap="RdBu_r", center=0, ax=ax)
    fig.savefig(
        output_dir
        / f"re-clustered.macrophage_monocyte.z-score.{cluster_str}.distance_to_lacunae.svg",
        **figkws,
    )

#

#

#

#

#

#

#

#


# Using gating only

pos = pd.read_parquet(positive_file)
posc = pd.read_parquet(positive_count_file)


desc = {
    "Macrophages": [
        "CD68(Nd150)",
        "CD15(Dy163)",
        "CD3(Er170)",
        "CD206(Nd144)",
        "CD14(Nd148)",
        "CD16(Nd146)",
        "CD123(Er167)",
        "CD163(Sm147)",
        "CD11c(Yb176)",
        "CD11b(Sm149)",
        "CD4(Gd156)",
        "CD8a(Dy162)",
        "C5bC9(Gd155)",
        "CleavedCaspase3(Yb172)",
        "cKIT(Nd143)",
        "pSTAT3(Gd158)",
        "pCREB(Ho165)",
        "IL1beta(Er166)",
        "IL6(Gd160)",
        "MPO(Yb173)",
        "iNOS(Nd142)",
        "Arginase1(Dy164)",
        "Ki67(Er168)",
    ],
    "Neutrophils": [
        "CD68(Nd150)",
        "CD15(Dy163)",
        "CD3(Er170)",
        "CD206(Nd144)",
        "CD14(Nd148)",
        "CD16(Nd146)",
        "CD123(Er167)",
        "CD163(Sm147)",
        "CD11c(Yb176)",
        "CD11b(Sm149)",
        "CD4(Gd156)",
        "CD8a(Dy162)",
        "C5bC9(Gd155)",
        "CleavedCaspase3(Yb172)",
        "cKIT(Nd143)",
        "pSTAT3(Gd158)",
        "pCREB(Ho165)",
        "IL1beta(Er166)",
        "IL6(Gd160)",
        "MPO(Yb173)",
        "iNOS(Nd142)",
        "Arginase1(Dy164)",
        "Ki67(Er168)",
    ],
    "T-cells": [
        "CD68(Nd150)",
        "CD15(Dy163)",
        "CD3(Er170)",
        "CD206(Nd144)",
        "CD14(Nd148)",
        "CD16(Nd146)",
        "CD123(Er167)",
        "CD163(Sm147)",
        "CD11c(Yb176)",
        "CD11b(Sm149)",
        "CD4(Gd156)",
        "CD8a(Dy162)",
        "C5bC9(Gd155)",
        "CleavedCaspase3(Yb172)",
        "cKIT(Nd143)",
        "pSTAT3(Gd158)",
        "pCREB(Ho165)",
        "IL1beta(Er166)",
        "IL6(Gd160)",
        "MPO(Yb173)",
        "iNOS(Nd142)",
        "Arginase1(Dy164)",
        "Ki67(Er168)",
    ],
}

col = max([len(x) for x in desc.values()])

fig, axes = plt.subplots(len(desc), col, figsize=(col * 4, len(desc) * 4))
for i, (ct, markers) in enumerate(desc.items()):
    posm = pos.loc[pos["cluster"].str.contains(ct)]
    p = (
        (posm.groupby("roi")[markers].sum().T / posm.groupby("roi").size())
        * 100
    ).T.join(roi_attributes["phenotypes"])
    for j, marker in enumerate(markers):
        swarmboxenplot(
            data=p[["phenotypes", marker]],
            x="phenotypes",
            y=marker,
            ax=axes[i, j],
            plot_kws=dict(palette=colors.get("phenotypes")),
        )
fig.savefig(
    results_dir
    / "gating"
    / "macrophage_neutrophil.phenotypes.perc_positive.swarmboxenplots.svg",
    **figkws,
)
