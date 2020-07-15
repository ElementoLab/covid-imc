#!/usr/bin/env python

"""
A module investigate relationships between Samples/ROIs and channels in an
unsupervised way.
"""

from src.config import *

summary, fig = prj.channel_summary(channel_exclude=channels_exclude)
summary.to_csv(qc_dir / prj.name + ".channel_summary.csv")
fig = prj.channel_correlation(channel_exclude=channels_exclude)

# # better plotting
cell_density = pd.Series(
    [r.cells_per_area_unit() for r in prj.rois],
    index=roi_names,
    name="cell_density",
).rename_axis("roi")


channel_means = summary.mean(1).rename("channel_mean").rename_axis("roi")

kwargs = dict(
    xticklabels=True,
    yticklabels=True,
    center=0,
    cmap="RdBu_r",
    z_score=0,
    cbar_kws=dict(label="Mean counts (Z-score)"),
    rasterized=True,
    metric="correlation",
    robust=True,
)

grid = sns.clustermap(
    summary,
    row_colors=channel_means,
    col_colors=roi_attributes.join(cell_density),
    **kwargs,
)
grid.ax_heatmap.set_xticklabels(grid.ax_heatmap.get_xticklabels(), fontsize=4)
grid.fig.savefig(
    qc_dir / prj.name + ".channel_summary.per_roi.clustermap.svg",
    dpi=300,
    bbox_inches="tight",
)


sample_summary = summary.T.groupby(
    summary.columns.str.split("-").to_series().apply(pd.Series)[0].values
).mean()
grid = sns.clustermap(
    sample_summary.T,
    row_colors=channel_means,
    col_colors=sample_attributes,
    figsize=(5, 10),
    **kwargs,
)
# grid.ax_heatmap.set_xticklabels(grid.ax_heatmap.get_xticklabels(), fontsize=4)
grid.fig.savefig(
    qc_dir / prj.name + ".channel_summary.per_sample.clustermap.svg",
    dpi=300,
    bbox_inches="tight",
)

# Sample correlation


# Sample dimres

# # PCA
# # UMAP
# # TSNE
# # MDS
# # NMF
