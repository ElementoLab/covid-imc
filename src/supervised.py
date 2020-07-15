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


output_dir = results_dir / "supervised"
output_dir.mkdir()


prj.sample_comparisons(
    sample_attributes=roi_attributes.columns[:2].tolist(),
    output_prefix=output_dir / "comparisons.",
    channel_exclude=channels_exclude,
)


df = prj.clusters.reset_index()
c = df["cluster"].value_counts()
c = c[c > 500].index.tolist()

df = df.loc[df["cluster"].isin(c)]
df = df.loc[~df["cluster"].isin(["?", "<EMPTY>"])]

df = df.merge(roi_attributes, left_on="roi", right_index=True)

perc = (
    df.groupby("roi")
    .apply(lambda x: (x["cluster"].value_counts() / x.shape[0]) * 100)
    .rename("percentage")
)
perc = roi_attributes.join(perc)
perc.index.names = ["roi", "cluster"]


# grid = sns.catplot(
#     data=perc.reset_index(),
#     x="cluster",
#     y="percentage",
#     hue=attr,
#     kind="boxen",
# )

_test_res = list()
for attr in roi_attributes.columns:
    # Test for differences
    aov = pd.concat(
        [
            pg.anova(
                data=perc.loc[perc.index.get_level_values(1) == val],
                dv="percentage",
                between=attr,
            ).assign(variable=val)
            for val in perc.index.levels[1]
        ]
    ).set_index("variable")
    _test_res.append(aov)

    kws = dict(
        data=perc.reset_index(), x=attr, y="percentage", palette="tab10",
    )
    grid = sns.FacetGrid(
        data=perc.reset_index(),
        col="cluster",
        height=3,
        col_wrap=4,
        sharey=False,
    )
    grid.map_dataframe(sns.boxenplot, saturation=0.5, dodge=False, **kws)
    for ax in grid.axes.flat:
        [
            x.set_alpha(0.25)
            for x in ax.get_children()
            if isinstance(
                x,
                (
                    matplotlib.collections.PatchCollection,
                    matplotlib.collections.PathCollection,
                ),
            )
        ]
    grid.map_dataframe(sns.swarmplot, **kws)
    for ax in grid.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # for ax in grid.axes.flat:
    #     if ax.get_title().endswith("_number"):
    #         ax.set_yscale("log")
    for ax in grid.axes.flat:
        var = ax.get_title().replace("cluster = ", "")
        f = aov.loc[var, "F"]
        p = aov.loc[var, "p-unc"]
        stats = f"\nF = {f:.3f}; p = {p:.3e}"
        ax.set_title(var + stats)

    grid.savefig(
        output_dir / f"differential_cell_types.{attr}.boxen_swarm_plot.svg",
        **figkws,
    )
    plt.close(grid.fig)


test_res = pd.concat(_test_res)
test_res.to_csv(output_dir / "differential_cell_types.anova_test_results.csv")
