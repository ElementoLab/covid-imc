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
from tqdm import tqdm

from imc.types import Path, Array
from imc.operations import get_population
from imc.graphics import swarmboxenplot
from imc.utils import align_channels_by_name, z_score

from src.config import *


output_dir = results_dir / "cell_type"
output_dir.mkdir()


ids = [
    "roi",
    "sample",
]

# quantify single cells
quantification_file = output_dir / "quantification.pq"
quantification_file_sum = output_dir / "quantification.sum.pq"
if not quantification_file.exists():
    # WIth mean
    prj.quantify_cells()
    prj.quantification = align_channels_by_name(prj.quantification, 1)
    prj.quantification.to_parquet(quantification_file)

    # With sum
    quant_sum = prj.quantify_cell_intensity(red_func="sum")
    quant_sum = (
        quant_sum.reset_index()
        .merge(
            prj.quantification.iloc[:, -7:].reset_index(), on=["index"] + ids,
        )
        .set_index("index")
    )
    cols = quant_sum.columns.drop(ids).tolist() + ids
    quant_sum = quant_sum.loc[:, cols]
    assert quant_sum.shape == prj.quantification.shape
    assert all(quant_sum.columns == prj.quantification.columns)
    quant_sum = align_channels_by_name(quant_sum, 1)
    quant_sum.to_parquet(quantification_file_sum)


quant = pd.read_parquet(quantification_file).drop(
    channels_exclude, axis=1, errors="ignore"
)


# # Z-score per image
# ids = ["roi", "sample"]
# _zquant = list()
# for roi in quant["roi"].unique():
#     x = quant.query(f"roi == '{roi}'")
#     _zquant.append(z_score(x.drop(ids, 1)).join(x[ids]))
# zquant = pd.concat(_zquant)


# Exemplify with some manual gating
quant.loc[:, "DNA"] = quant[["DNA1(Ir191)", "DNA2(Ir193)"]].mean(1)

cd45 = "CD45(In115-Sm152)"
dna = "DNA"
sp = "SARSCoV2S1(Eu153)"

x = np.log1p(quant.loc[:, quant.dtypes == "float"])


immune = get_population(x[cd45], -1, plot=True)

x.loc[immune]


fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(x[dna], x[cd45], s=2, alpha=0.1)
ax.axhline(1.68, linestyle="--", color="grey")
ax.set(xlabel="DNA", ylabel="CD45")

"HistoneH3(In113)"


ax.scatter(quant[""], quant["CD45(In115-Sm152)"])


pop = get_population(x[cd45], -1, plot=True)
pop = get_population(x.sample(frac=0.1), -1, plot=True)


quant = quant.loc[~(quant.loc[:, quant.dtypes == "float"] > 1e4).any(1), :]


# read in distances to structures
dists = pd.read_parquet(results_dir / "spatial" / "cell_distance_to_lacunae.pq")

# join
matrix = pd.concat([quant, dists.drop(["sample", "roi"], 1)], axis=1)

# visualize
figure = output_dir / "matrix.clustermap.subsample_5000.svg"
if not figure.exists():
    grid = sns.clustermap(
        np.log1p(matrix.drop(["sample", "roi"], 1).sample(n=20000)),
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
matrix.loc[:, "DNA"] = matrix[["DNA1(Ir191)", "DNA2(Ir193)"]].mean(1)
matrix = matrix.loc[matrix["DNA"] > 1]

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
    matrix[["roi", "sample"]]
    .merge(roi_attributes, left_on="roi", right_index=True)
    .rename_axis(index="cell")
    .reset_index(drop=True)
).assign(obj_id=matrix.index.tolist())
# ann = AnnData(matrix2.drop(["roi", "sample"], axis=1), obs=obs)
cds = matrix2.columns[matrix2.columns.str.startswith("CD")].tolist()
mks = "Tryptase|Vimentin|Keratin|AlphaSMA|CREB|STAT|TTF1|KIT"
oth = matrix2.columns[matrix2.columns.str.contains(mks)].tolist()
matrix3 = matrix2[cds + oth]

h5ad_file = output_dir / "anndata.all_cells.processed.h5ad"

cats = ["disease", "phenotypes"]
if not h5ad_file.exists():
    ann = AnnData(matrix.reset_index(drop=True).drop(ids, 1), obs=obs)
    ann = AnnData(matrix3, obs=obs)
    ann.raw = ann
    ann.raw._Xo = ann.raw._X
    ann.raw._X = np.log1p(ann.raw.X)
    sc.pp.log1p(ann)
    # sc.pp.combat(ann, "sample")
    # sc.pp.scale(ann)
    sc.pp.pca(ann)
    # chs = matrix.columns.drop(["sample", "roi"]).tolist()
    chs = channels_include.tolist()
    # sc.pl.pca(ann, color=ann.var_names, use_raw=True)
    sc.pp.neighbors(ann, n_neighbors=3)
    sc.tl.umap(ann)
    sc.tl.leiden(ann, key_added="cluster", resolution=0.5)
    ann.write_h5ad(h5ad_file)


# plot
ann = sc.read_h5ad(h5ad_file)

ann = ann[ann.obs.index.to_series().sample(frac=1).values, :]

chs = ann.var_names.tolist()
from imc.graphics import rasterize_scanpy

for ch in chs + cats:
    fig = sc.pl.umap(ann, color=ch, use_raw=True, show=False, return_fig=True,)
    rasterize_scanpy(fig)
    fig.savefig(output_dir / f"clustering.umap.{ch}.pdf")

fig = sc.pl.umap(
    ann,
    color=chs + cats,  #  + ["cluster"],
    use_raw=True,
    show=False,
    return_fig=True,
)
rasterize_scanpy(fig)
fig.savefig(output_dir / f"clustering.umap.pdf")

# sc.tl.paga(ann, groups="cluster")


cluster_means = ann.to_df().groupby(ann.obs["cluster"]).mean()
cluster_counts = (
    ann.obs["cluster"].value_counts().sort_values().rename("Cells per cluster")
)
clusters = cluster_counts[cluster_counts >= 50].index.drop(["8", "24", "28"])


disease_counts = (
    pd.get_dummies(ann.obs["disease"]).groupby(ann.obs["cluster"]).sum()
)
phenotypes_counts = (
    pd.get_dummies(ann.obs["phenotypes"]).groupby(ann.obs["cluster"]).sum()
)


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
    row_colors=disease_counts.join(phenotypes_counts, rsuffix="_").join(
        cluster_counts
    ),
    colors_ratio=(0.01, 0.03),
    dendrogram_ratio=0.05,
    cbar_kws=dict(label="Z-score"),
)
grid.savefig(output_dir / "clustering.mean_per_cluster.svg", **figkws)


# Plot marker intensity per sample grouped by disease/phenotype
clinvars = ["disease", "phenotypes"]
for grouping in clinvars:
    fig, axes = plt.subplots(
        5,
        9,
        figsize=(9 * 3, 5 * 3),
        sharex=True,
        sharey=False,
        tight_layout=True,
    )
    axes = axes.flatten()
    for i, var in enumerate(ann.var.index):
        swarmboxenplot(
            data=x.groupby("sample").mean().join(sample_attributes[clinvars]),
            x=grouping,
            y=var,
            ax=axes[i],
        )
    for ax in axes[i + 1 :]:
        ax.axis("off")
    fig.savefig(output_dir / f"marker_intensity.by_{grouping}.svg", **figkws)


positives = pd.DataFrame(index=ann.obs.index, columns=ann.var.index, dtype=bool)
for i, var in tqdm(enumerate(ann.var.index)):
    positives[var] = get_population(ann[:, var].to_df().squeeze(), -1)

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
    row_colors=cluster_means.mean(1).rename("Cells per cluster"),
    col_colors=cluster_means.mean().rename("Channel mean"),
)
grid.ax_heatmap.set_xticklabels(grid.ax_heatmap.get_xticklabels(), fontsize=6)
grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), fontsize=6)
grid.savefig(
    output_dir / "clustering.mean_per_cluster.labels_reduced_simplified.svg",
    **figkws,
)
