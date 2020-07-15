#!/usr/bin/env python

"""
Spatial analysis of lung tissue
"""

from typing import Union

import parmap
import scipy.ndimage as ndi
import skimage as ski
import skimage.feature
from skimage.exposure import equalize_hist as eq
import tifffile
import pingouin as pg
import numpy_groupies as npg
from anndata import AnnData
import scanpy as sc

from imc.types import Path, Array

from src.config import *


output_dir = results_dir / "cell_type"
output_dir.mkdir()


# quantify single cells
quantification_file = output_dir / "quantification.pq"
if not quantification_file.exists():
    prj.quantify_cells()
    prj.quantification.to_parquet(quantification_file)

quant = pd.read_parquet(quantification_file).drop(channels_exclude, axis=1)

# read in distances to structures
dists = pd.read_parquet(results_dir / "spatial" / "cell_distance_to_lacunae.pq")

# join
matrix = pd.concat([quant, dists.drop(["sample", "roi"], 1)], axis=1)

# visualize
figure = output_dir / "matrix.clustermap.subsample_5000.svg"
if not figure.exists():
    grid = sns.clustermap(
        np.log1p(matrix.drop(["sample", "roi"], 1).sample(n=5000)),
        z_score=1,
        cmap="RdBu_r",
        center=0,
        robust=True,
        metric="correlation",
        xticklabels=True,
        rasterized=True,
    )
    grid.savefig(figure, **figkws)

# exclude small cells
matrix = matrix.loc[matrix["area"] > 6]

# exclude cells without DNA
matrix = matrix.loc[matrix["DNA2(Ir193)"] > 1]

# exclude small cells
matrix = matrix.loc[~(matrix.drop(channels_include, 1) == 0).any(1)]

figure = output_dir / "matrix.distplots.svg"
if not figure.exists():
    fig, axes = plt.subplots(9, 5, figsize=(5 * 3, 9 * 3))
    for ax, col in zip(axes.flat, matrix.columns.drop(["sample", "roi"])):
        sns.distplot(np.log1p(matrix[col]), kde=False, ax=ax)
        ax.set(title=col, yscale="log")
    fig.savefig(figure, **figkws)


# matrix2 = matrix.sample(n=150000).reset_index(drop=True)
matrix2 = matrix.reset_index(drop=True)
obs = (
    matrix2[["roi", "sample"]]
    .merge(roi_attributes, left_on="roi", right_index=True)
    .rename_axis(index="cell")
    .reset_index(drop=True)
).assign(obj_id=matrix.index.tolist())
# ann = AnnData(matrix2.drop(["roi", "sample"], axis=1), obs=obs)
cds = matrix2.columns[matrix2.columns.str.startswith("CD")].tolist()
mks = "Tryptase|Vimentin|Keratin|AlphaSMA|CREB|STAT|TTF1|KIT"
oth = matrix2.columns[matrix2.columns.str.contains(mks)].tolist()
matrix3 = matrix2[cds + oth]
ann = AnnData(matrix3, obs=obs)
ann.raw = ann
ann.raw._Xo = ann.raw._X
ann.raw._X = np.log1p(ann.raw.X)
sc.pp.log1p(ann)
# sc.pp.combat(ann, "sample")
# sc.pp.scale(ann)
sc.pp.pca(ann)

chs = ["DNA2(Ir193)", "AlphaSMA(Pr141)", "Keratin818(Yb174)", "CD206(Nd144)"]
# chs = matrix.columns.drop(["sample", "roi"]).tolist()
chs = channels_include.tolist()
cats = ["disease", "phenotypes"]
# sc.pl.pca(ann, color=ann.var_names, use_raw=True)


sc.pp.neighbors(ann, n_neighbors=3)
sc.tl.umap(ann)
sc.tl.leiden(ann, key_added="cluster", resolution=0.5)

ann.write_h5ad(output_dir / "anndata.all_cells.processed.h5ad")

# plot
ann = sc.read_h5ad(output_dir / "anndata.all_cells.processed.h5ad")

chs = ann.var_names.tolist()
fig = sc.pl.umap(
    ann,
    color=chs + cats,  #  + ["cluster"],
    use_raw=True,
    show=False,
    return_fig=True,
)
fig.savefig(output_dir / "clustering.umap.pdf")

# sc.tl.paga(ann, groups="cluster")


cluster_means = ann.to_df().groupby(ann.obs["cluster"]).mean()
cluster_counts = ann.obs["cluster"].value_counts().sort_values()
clusters = cluster_counts[cluster_counts >= 500].index


grid = sns.clustermap(
    cluster_means.loc[clusters],
    z_score=1,
    center=0,
    cmap="RdBu_r",
    robust=True,
    xticklabels=True,
    yticklabels=True,
    metric="correlation",
    # standard_scale=0,
)
grid.savefig(output_dir / "clustering.mean_per_cluster.svg", **figkws)


# from imc.operations import derive_reference_cell_type_labels

# new_labels = derive_reference_cell_type_labels(
#     mean_expr=cluster_means.T,
#     cluster_assignments=ann.obs["cluster"],
#     output_prefix=output_dir / "cell_type_labeling",
#     plot=True,
# )
# new_labels = new_labels.index.astype(str) + " - " + new_labels
# ann.obs["cluster"] = ann.obs["cluster"].replace(new_labels)


new_labels = {
    "0": "?",
    "1": "?",
    "2": "Neutrophil",
    "32": "Neutrophil",
    "13": "?",
    "8": "TTF1+",
    "15": "<EMPTY>",
    "3": "<EMPTY>",
    "4": "Neutrophil",
    "16": "Neutrophil",
    "6": "Macrophage",
    "6": "Macrophage",
    "30": "Macrophage",
    "22": "Macrophage",
    "26": "Macrophage",
    "23": "Macrophage",
    "5": "CD31+",
    "7": "CD31+",
    "11": "CD31+",
    "12": "CD31+",
    "24": "CD31+",
    "10": "<EMPTY>",  # Mast-cells?
    "17": "T-cells",
    "29": "NK-cells",
    "20": "Mast-cells",
    "9": "Fibroblast",
    "19": "Fibroblast",
    "34": "Fibroblast",
    "36": "Fibroblast",
    "35": "<EMPTY>",
    "21": "?",
    "25": "?",
    "18": "Keratin8/18+",
    "27": "Keratin8/18+",
    "28": "Keratin8/18+",
    "33": "Keratin8/18+",
    "14": "Keratin8/18+",
    "31": "Keratin8/18+",
    "37": "pDC",
}

ann.obs["cluster"] = ann.obs["cluster"].replace(new_labels)


clusters = ann.obs[["sample", "roi", "obj_id", "cluster"]].set_index(
    ["sample", "roi", "obj_id"]
)
prj.set_clusters(clusters, write_to_disk=True)

cluster_means = ann.to_df().groupby(ann.obs["cluster"]).mean()
cluster_counts = ann.obs["cluster"].value_counts().sort_values()
clusters = cluster_counts[cluster_counts >= 500].index


grid = sns.clustermap(
    cluster_means.loc[clusters].drop(["?", "<EMPTY>"], 0),
    z_score=1,
    center=0,
    cmap="RdBu_r",
    robust=True,
    xticklabels=True,
    yticklabels=True,
    metric="correlation",
    # standard_scale=0,
    figsize=(4, 4),
    cbar_kws=dict(label="Mean intensity\n(Z-score)"),
)
grid.savefig(
    output_dir / "clustering.mean_per_cluster.labels_reduced_simplified.svg",
    **figkws
)
