#!/usr/bin/env python

"""
A module investigate relationships between Samples/ROIs and channels in an
unsupervised way.
"""

from imc.graphics import swarmboxenplot

from src.config import *

output_dir = results_dir / "unsupervised"
output_dir.mkdir()

summary_file = qc_dir / prj.name + ".channel_summary.csv"
if not summary_file.exists():
    summary, fig = prj.channel_summary(channel_exclude=channels_exclude)
    summary.to_csv(summary_file)
summary = pd.read_csv(summary_file, index_col=0)

fig = prj.channel_correlation(channel_exclude=channels_exclude)
fig.savefig(
    output_dir / prj.name + ".channel_summary.per_roi.clustermap.svg", **figkws
)


# Cell density
density_file = results_dir / "cell_density_per_roi.csv"
if not density_file.exists():
    cell_density = pd.Series(
        [r.cells_per_area_unit() for r in prj.rois],
        index=roi_names,
        name="cell_density",
    ).rename_axis("roi")
    cell_density.to_csv(density_file)
cell_density = pd.read_csv(density_file, index_col=0, squeeze=True)
for grouping in ["disease", "phenotypes"]:
    fig = swarmboxenplot(
        data=roi_attributes.join(cell_density),
        x=grouping,
        y="cell_density",
        ax=None,
    )
    fig.savefig(
        output_dir / f"cell_density.per_roi.by_{grouping}.swarmboxenplot.svg",
        dpi=300,
        bbox_inches="tight",
    )


# TODO: recalculate with only (non-)immune cells


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
    output_dir / "channel_summary.per_roi.clustermap.svg", **figkws
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
    output_dir / "channel_summary.per_sample.clustermap.svg", **figkws,
)

# Sample correlation


# Sample dimres

# # PCA
# # UMAP
# # TSNE
# # MDS
# # NMF
