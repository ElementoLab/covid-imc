#!/usr/bin/env python

"""
Classification of cells into cell types/states.
"""

import re

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


# Let's declare some variables for later use
output_dir = results_dir / "cell_type"
output_dir.mkdir()
ids = [
    "roi",
    "sample",
]
cell_filtering_string_query = (
    "area > -1.5 & solidity > -1 & eccentricity < 1 & total > 2 & total < 7"
)

# Quantify single cells
if not quantification_file.exists():
    # by mean in mask
    prj.quantify_cells()
    prj.quantification = align_channels_by_name(prj.quantification, 1)
    prj.quantification.to_parquet(quantification_file)
quant = pd.read_parquet(quantification_file).drop(
    channels_exclude, axis=1, errors="ignore"
)

prefix = "roi_zscored.filtered."
use_zscore = True

# Let's process these cells and get their cell type identities through clustering
# # This is roughly equivalent to what is in `src.operations.process_single_cell`,
# # but we'll use some of the intermediate steps later too.

# # Z-score per image
zquant = z_score_by_column(quant, "roi")

# # keep track of total per cell in original (log) units
zquant["total"] = np.log1p(
    quant.loc[:, quant.columns.str.contains(r"\(")].sum(1).values
)
# # Filter out cells based on the criteria on the query string.
# # These are mostly just capping the distribution tails
zquant = zquant.query(cell_filtering_string_query).drop("total", 1)

if not h5ad_file.exists():
    obs = (
        zquant[["roi", "sample"]]
        .merge(roi_attributes, left_on="roi", right_index=True)
        .rename_axis(index="cell")
        .reset_index(drop=True)
    ).assign(obj_id=zquant.index.tolist())

    ann = AnnData(zquant.reset_index(drop=True).drop(ids, 1), obs=obs)
    ann.raw = ann
    sc.pp.pca(ann)  # no need to scale/center as these are already Z-scores
    sc.pp.neighbors(ann, n_neighbors=3)  # higher values show little difference
    sc.tl.umap(ann)
    for res in [0.5, 1.0]:
        sc.tl.leiden(ann, key_added=f"cluster_{res}", resolution=res)
    for c in ann.obs.columns.to_series().filter(like="cluster"):
        ann.obs[c] = (ann.obs[c].astype(int) + 1).astype(str)
    ann.write_h5ad(str(h5ad_file).replace("processed", f"{prefix}processed"))


# Plots
# # randomize order of cells in order to better portray UMAP
ann = sc.read_h5ad(str(h5ad_file).replace("processed", f"{prefix}processed"))
ann = ann[ann.obs.index.to_series().sample(frac=1).values, :]

# # UMAP
plot_umap(ann)
plot_umap_with_labeled_clusters(ann)

# # Means per cluster
plot_cluster_heatmaps(ann)

# # Means per cluster labeled with biological names
plot_cluster_heatmaps_with_labeled_clusters(ann)

# Settle on the most resolved Leiden clustering
cluster_str = "cluster_1.0"

# Set `prj.clusters` to the discovered clusters.
# These cluster identities are numeric! (we want to keep them as such)
prj.set_clusters(
    ann.obs.set_index(["sample", "roi", "obj_id"])[cluster_str]
    .rename("cluster")
    .astype(int),
    write_to_disk=True,
)

# But for plotting purposes, we can replace the numeric clusters
# with strings that add to the biological identity of the clusters
cluster_names = json.load(open("metadata/cluster_names.json"))
new_labels = cluster_names[f"{prefix};{cluster_str}"]
new_labels = {int(k): v for k, v in new_labels.items()}
# If a numeric cluster is not in the mapping (lowly abundant clusters),
# it will be grouped in this 'other' group:
for k in prj.clusters.unique():
    if k not in new_labels:
        new_labels[k] = "999 - ?()"
# Set the clusters as the string labels but don't safe it to disk
prj.set_clusters(
    prj.clusters.replace(new_labels), write_to_disk=False,
)

# Let's generate some dataframes that will be useful later:
# # counts of cells per image, per cluster
roi_counts = (
    prj.clusters.reset_index()
    .assign(count=1)
    .pivot_table(
        index="roi",
        columns="cluster",
        values="count",
        aggfunc=sum,
        fill_value=0,
    )
)
roi_counts.to_parquet(counts_file)

# # counts of cells per image, per cluster, for meta-clusters
agg_counts = (
    roi_counts.T.groupby(
        roi_counts.columns.str.extract(r"\d+ - (.*) \(")[0].values
    )
    .sum()
    .T
)
agg_counts.to_parquet(counts_agg_file)

# # Area per image
roi_areas = pd.Series(
    {r.name: r.area for r in prj.rois}, name="area"
).rename_axis("roi")
roi_areas.to_csv(roi_areas_file)

# # counts of cells per sample, per cluster
sample_counts = (
    prj.clusters.reset_index()
    .assign(count=1)
    .pivot_table(
        index="sample",
        columns="cluster",
        values="count",
        aggfunc=sum,
        fill_value=0,
    )
)
# # Area per sample
sample_areas = pd.Series(
    {s.name: sum([r.area for r in s]) for s in prj}, name="area"
)
sample_areas.to_csv(sample_areas_file)


# Plot fraction of cells per ROI/Sample and grouped by disease/phenotype
# # Heatmaps
for grouping, df, area, attributes in [
    ("sample", sample_counts, sample_areas, sample_attributes),
    ("roi", roi_counts, roi_areas, roi_attributes),
]:
    attributes = sample_attributes if grouping == "sample" else roi_attributes
    kws = dict(
        figsize=(16, 8),
        rasterized=True,
        metric="correlation",
        col_colors=attributes,
        yticklabels=True,
    )
    grid = sns.clustermap((df.T / df.sum(1)), **kws)
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{cluster_str}.svg",
        **figkws,
    )
    grid = sns.clustermap((df.T / df.sum(1)), standard_scale=0, **kws)
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{cluster_str}.standard_scale.svg",
        **figkws,
    )

    df2 = df.copy()
    df2.columns = df2.columns.str.extract(r"\d+ - (.*) \(")[0]
    df2 = df2.T.groupby(level=0).sum().T.rename_axis(columns="Cluster")
    kws = dict(
        figsize=(10, 8),
        rasterized=True,
        metric="correlation",
        col_colors=attributes,
        yticklabels=True,
    )
    grid = sns.clustermap((df2.T / df2.sum(1)), **kws)
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{cluster_str}.reduced_clusters.svg",
        **figkws,
    )
    grid = sns.clustermap((df2.T / df2.sum(1)), standard_scale=0, **kws)
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{cluster_str}.reduced_clusters.standard_scale.svg",
        **figkws,
    )

    dfarea = (df.T / area).T * 1e6
    grid = sns.clustermap(
        dfarea, xticklabels=True, row_colors=attributes, figsize=(10, 12)
    )
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{cluster_str}.per_area.svg",
        **figkws,
    )

    dfarea_red = (
        dfarea.T.groupby(dfarea.columns.str.extract(r"\d+ - (.*)\(")[0].values)
        .sum()
        .T
    )
    grid = sns.clustermap(
        dfarea_red,
        metric="correlation",
        xticklabels=True,
        row_colors=attributes,
        figsize=(10, 6),
    )
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{cluster_str}.per_area.reduced_clusters.svg",
        **figkws,
    )
    grid = sns.clustermap(
        np.log1p(dfarea_red),
        metric="correlation",
        xticklabels=True,
        row_colors=attributes,
        figsize=(10, 6),
    )
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{cluster_str}.per_area.reduced_clusters.log1p.svg",
        **figkws,
    )
    grid = sns.clustermap(
        np.log1p(dfarea_red),
        z_score=1,
        center=0,
        cmap="RdBu_r",
        robust=True,
        metric="correlation",
        xticklabels=True,
        row_colors=attributes,
        figsize=(10, 6),
    )
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{cluster_str}.per_area.reduced_clusters.log1p.zscore.svg",
        **figkws,
    )

# # Swarmboxenplots
_stats = list()
clinvars = ["disease", "phenotypes"]
for grouping, df, area in [
    ("sample", sample_counts, sample_areas),
    ("roi", roi_counts, roi_areas),
]:
    attributes = sample_attributes if grouping == "sample" else roi_attributes

    for measure, df2 in [
        ("percentage", (df.T / df.sum(1)).T * 100),
        ("area", (df.T / area).T * 1e6),
    ]:
        df2.index.name = grouping
        df2 = (
            df2.join(attributes[clinvars])
            .reset_index()
            .melt(
                id_vars=[grouping] + clinvars,
                value_name=measure,
                var_name="cell_type",
            )
        )
        for var in clinvars:
            cts = df2["cell_type"].unique()
            n = len(cts)
            n, m = get_grid_dims(n)
            fig, axes = plt.subplots(
                n, m, figsize=(m * 3, n * 3), sharex=True, tight_layout=True
            )
            axes = axes.flatten()
            for i, ct in enumerate(cts):
                sts = swarmboxenplot(
                    data=df2.query(f"cell_type == '{ct}'"),
                    x=var,
                    y=measure,
                    test_kws=dict(parametric=False),
                    plot_kws=dict(palette=colors[var]),
                    ax=axes[i],
                )
                _stats.append(
                    sts.assign(
                        grouping=grouping,
                        variable="original",
                        cell_type=ct,
                        measure=measure,
                    )
                )
                axes[i].set(title="\n(".join(ct.split(" (")))
            for ax in axes[i + 1 :]:
                ax.axis("off")
            fig.savefig(
                output_dir
                / f"clustering.{prefix}fraction.{cluster_str}.cells_per_{grouping}.by_{var}.{measure}.svg",
                **figkws,
            )
            plt.close(fig)

            df3 = df2.copy()
            df3["cell_type"] = df3["cell_type"].str.extract(r"\d+ - (.*) \(")[0]
            df3 = (
                df3.groupby(clinvars + [grouping, "cell_type"])
                .sum()
                .reset_index()
                .dropna()
            )
            cts = df3["cell_type"].unique()
            n = len(cts)
            n, m = get_grid_dims(n)
            fig, axes = plt.subplots(
                n, m, figsize=(m * 3, n * 3), sharex=True, tight_layout=True
            )
            axes = axes.flatten()
            for i, ct in enumerate(cts):
                sts = swarmboxenplot(
                    data=df3.query(f"cell_type == '{ct}'"),
                    x=var,
                    y=measure,
                    test_kws=dict(parametric=False),
                    plot_kws=dict(palette=colors[var]),
                    ax=axes[i],
                )
                _stats.append(
                    sts.assign(
                        grouping=grouping,
                        variable="aggregated",
                        cell_type=ct,
                        measure=measure,
                    )
                )
                axes[i].set(title=ct)
            for ax in axes[i + 1 :]:
                ax.axis("off")
            fig.savefig(
                output_dir
                / f"clustering.{prefix}fraction.{cluster_str}.cells_per_{grouping}.reduced_clusters.by_{var}.{measure}.svg",
                **figkws,
            )
            plt.close(fig)

stats = pd.concat(_stats)
stats.to_csv(
    output_dir / f"clustering.{prefix}fraction.{cluster_str}.differences.csv",
    index=False,
)

# Illustrate clusters
plot_cluster_illustrations(ann)
