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

from imc.types import Path, Array

from src.config import *


def get_lacunae(
    image: Union[Path, Array],
    selem_diam: int = 5,
    min_diam: int = 25,
    max_area_percentage: float = 50,
) -> Array:
    if isinstance(image, Path):
        image = tifffile.imread(image.as_posix())

    # threshold, close
    img = image > ski.filters.threshold_otsu(image)
    img = ski.morphology.binary_dilation(
        img, selem=ski.morphology.disk(selem_diam)
    )
    img = ski.morphology.closing(img, ski.morphology.disk(5))

    # clean up small objects inside
    img = ~ndi.binary_fill_holes(~img)
    img = ~ski.morphology.remove_small_objects(~img, min_size=min_diam ** 2)

    lac = ndi.label(~img)[0]

    # remove objects too large
    remove = [
        i
        for i in np.unique(lac)
        if ((lac == i).sum() / img.size) * 100 > max_area_percentage
    ]
    if remove:
        for i in remove:
            lac[lac == i] = 0
    return lac


def quantify_lacunae(lac: Array, stack: Array, selem_diam: int = 15):
    n_objs = lac.max()
    objs = range(1, n_objs + 1)
    if selem_diam > 0:
        selem = ski.morphology.disk(selem_diam)
        if np.asarray(stack.shape).argmin() == 0:
            stack = np.moveaxis(stack, 0, -1)
        res = list()
        for i in objs:  # skip background 0
            dil = ski.morphology.binary_dilation(lac == i, selem=selem)
            res.append(stack[dil].mean(0))
        return np.asarray(res)
    return np.asarray(
        [stack[j][lac == i].mean() for j in range(stack.shape[0]) for i in objs]
    ).reshape((n_objs, len(stack)))


def add_object_labels(label, ax=None):  # cmap=None,
    if ax is None:
        ax = plt.gca()
    index = np.unique(label)
    centroids = ndi.center_of_mass(
        np.ones_like(label), labels=label, index=index
    )
    # cmap = plt.get_cmap(cmap)
    for i, (y, x) in zip(index, centroids):
        ax.text(x, y, s=i)  # , color=cmap(i))


def get_lung_lacunae(
    roi,
    pos_ch="AlphaSMA(Pr141)",
    neg_ch="Keratin818(Yb174)",
    plot=True,
    output_prefix: Path = None,
    overwrite: bool = False,
):
    if output_prefix is None:
        output_prefix = roi.prj.results_dir / "qc" / "lacunae" / roi.name + "."
    output_figure = output_prefix + "lacunae.svg"
    output_figure.parent.mkdir()

    parenc_file = output_prefix + "parenchyma.tiff"
    pos_file = output_prefix + "pos_lacunae.tiff"
    neg_file = output_prefix + "neg_lacunae.tiff"

    if neg_file.exists() and not overwrite:
        return

    # Get lacunae based on channel mean
    # # one could use any single channel too, just make sure to max normalize
    stack = roi.stack[roi.channel_labels.isin(channels_include)]
    mean = eq(stack.mean(0))
    mean_lac = get_lacunae(mean, min_diam=25)
    mean_lac = np.ma.masked_array(mean_lac, mask=mean_lac == 0)

    # Quantify lacunae in relevant channels
    chs = [pos_ch, neg_ch]
    stack_mini = roi.stack[roi.channel_labels.isin(chs)]
    quant = pd.DataFrame(
        quantify_lacunae(mean_lac, stack_mini, selem_diam=10), columns=chs
    )
    quant = quant / quant.mean()
    quant.index += 1

    # Get ratio between AlphaSMA and Keratin to distinguish
    ratio = np.log(quant[pos_ch] / quant[neg_ch])
    # plt.scatter(ratio.rank(), ratio)

    # # select Alpha lacunae
    pos_lac = ratio[ratio > 0].index.values
    mean_pos_lac = mean_lac.copy()
    mean_pos_lac[~np.isin(mean_pos_lac, pos_lac)] = 0
    mean_pos_lac = np.ma.masked_array(mean_pos_lac, mask=mean_pos_lac == 0)

    # # select Keratin lacunae
    neg_lac = ratio[ratio < 0].index.values
    mean_neg_lac = mean_lac.copy()
    mean_neg_lac[~np.isin(mean_neg_lac, neg_lac)] = 0
    mean_neg_lac = np.ma.masked_array(mean_neg_lac, mask=mean_neg_lac == 0)

    # Get parenchyma as the remaining
    parenc = np.ma.masked_array(np.asarray(mean_lac), mask=~mean_lac.mask)

    tifffile.imwrite(parenc_file, parenc)
    tifffile.imwrite(pos_file, mean_pos_lac)
    tifffile.imwrite(neg_file, mean_neg_lac)
    ret = [parenc, mean_pos_lac, mean_neg_lac]
    if not plot:
        return ret

    # Plot
    # # Demo plot with channel and lacunae images
    vmax = mean_lac.max()
    fig, axes = plt.subplots(2, 4, figsize=(4 * 4, 4 * 2))
    axes = axes.flatten()
    axes[0].imshow(mean, rasterized=True)
    axes[0].set(title="Channel Mean")
    axes[0].axis("off")
    roi.plot_channel("DNA", ax=axes[1])
    roi.plot_channel(pos_ch, ax=axes[2])
    roi.plot_channel(neg_ch, ax=axes[3])
    axes[4].imshow(parenc, cmap="coolwarm", rasterized=True)
    axes[4].set(title="Parenchyma")
    axes[4].axis("off")
    axes[5].imshow(mean_lac, cmap="coolwarm", vmax=vmax, rasterized=True)
    axes[5].set(title="All lacunae")
    axes[5].axis("off")
    add_object_labels(mean_lac, ax=axes[5])
    axes[6].imshow(mean_pos_lac, cmap="coolwarm", vmax=vmax, rasterized=True)
    axes[6].set(title=f"{pos_ch} lacunae")
    axes[6].axis("off")
    add_object_labels(mean_pos_lac, ax=axes[6])
    axes[7].imshow(mean_neg_lac, cmap="coolwarm", vmax=vmax, rasterized=True)
    axes[7].set(title=f"{neg_ch} lacunae")
    axes[7].axis("off")
    add_object_labels(mean_neg_lac, ax=axes[7])
    fig.savefig(output_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ret

    # grid = sns.clustermap(np.log1p(quant), standard_scale=1)
    # grid = sns.clustermap(quant, z_score=1, cmap="RdBu_r", center=0)
    # grid = sns.clustermap(
    #     quant.loc[:, quant.columns.str.contains("|".join(chs))],
    #     metric="correlation",
    #     standard_scale=1,
    # )


def summarize_lung_lacunae(roi, prefix: Path = None):
    def summarize(lac):
        objs = np.unique(lac)[1:]
        n = len(objs)
        if n == 0:
            return [0.0] * 5
        areas = np.asarray([((lac == i).sum() / lac.size) * 100 for i in objs])
        return [
            n,
            areas.sum(),
            areas.mean(),
            areas.max(),
            # mean ratio of largest to other lacunae
            (areas.max() / np.delete(areas, areas.argmax())).mean()
            if n > 1
            else np.nan,
        ]

    if prefix is None:
        prefix = roi.prj.results_dir / "qc" / "lacunae" / roi.name + "."
    # parenc_file = prefix + "parenchyma.tiff"
    pos_file = prefix + "pos_lacunae.tiff"
    neg_file = prefix + "neg_lacunae.tiff"

    pos = tifffile.imread(pos_file)
    neg = tifffile.imread(neg_file)
    index = pd.Series(
        ["number", "area", "mean_area", "max_area", "mean_maxratio"]
    )
    return pd.Series(
        summarize(pos + neg) + summarize(pos) + summarize(neg),
        index=("lacunae_" + index).tolist()
        + ("pos_lacunae_" + index).tolist()
        + ("neg_lacunae_" + index).tolist(),
        name=roi.name,
    )


def get_cell_distance_to_lung_lacunae(
    roi, prefix: Path = None,  # plot: bool = False
):
    def get_cell_distance_to_mask(cells, mask):
        # fill in the gaps between the cells
        max_i = 2 ** 16 - 1
        cells[cells == 0] = max_i
        # set parenchyma to zero
        cells[mask > 0] = 0
        # get distance to parenchyma (zero)
        dist = ndi.distance_transform_edt(cells)
        # reduce per cell
        return (
            (
                pd.Series(
                    npg.aggregate(
                        cells.ravel(),
                        dist.ravel(),
                        func="mean",
                        fill_value=np.nan,
                    )
                )
                .dropna()
                .drop(max_i)
            )
            .rename(roi.name)
            .rename_axis("cell")
        )

    if prefix is None:
        prefix = roi.prj.results_dir / "qc" / "lacunae" / roi.name + "."
    parenc_file = prefix + "parenchyma.tiff"
    pos_file = prefix + "pos_lacunae.tiff"
    neg_file = prefix + "neg_lacunae.tiff"

    cells = roi.mask
    roi._cell_mask = None
    parenc = tifffile.imread(parenc_file)
    pos = tifffile.imread(pos_file)
    neg = tifffile.imread(neg_file)

    r = pd.concat(
        [
            get_cell_distance_to_mask(cells.copy(), x)
            for x in [parenc, pos, neg]
        ],
        axis=1,
    )
    r = r.drop(0)  # background
    r.columns = ["parenchyma_lacunae", "pos_lacunae", "neg_lacunae"]
    r["sample"] = roi.sample.name
    r["roi"] = roi.name

    # if not plot:
    #     return r
    # fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    # axes[0].imshow(cells, rasterized=True)
    # axes[1].imshow(parenc, rasterized=True)
    # axes[2].imshow(dist, rasterized=True)
    # for ax in axes:
    #     ax.axis("off")
    # fig.savefig(prefix + "cell_distance_to_lacunae.svg", **figkws)

    return r


output_dir = results_dir / "spatial"
output_dir.mkdir()

# Inspect
# sample = prj[6]
# roi = sample[0]
# roi.plot_channels(
#     [
#         "DNA",
#         "CD31",
#         "AlphaSMA",
#         "Collagen",
#         "Keratin",
#         "CD11c(Yb176)",
#         "CD68",
#         "MastCell",
#         "cKIT",
#     ]
# )
# roi.plot_channels(["CD68", "MastCell", "DNA", "cKIT"], merged=True, log=False)


# Get lacunae
for sample in prj.samples:
    parmap.map(get_lung_lacunae, sample.rois)


# measurements per image

# # number of lacunae, lacunae average area,
# # number of lacunae of a given type, area of a given type
# #

res = pd.concat(parmap.map(summarize_lung_lacunae, prj.rois), axis=1).T
res.index.name = "roi"
res.to_csv(output_dir / "lacunae.quantification_per_image.csv")

res = pd.read_csv(
    output_dir / "lacunae.quantification_per_image.csv", index_col=0
)

# scale
for col in res.columns[res.columns.str.contains("ratio")]:
    res[col] = np.log1p(res[col])
for col in res.columns[res.columns.str.endswith("area")]:
    res[col] = np.sqrt(res[col])

# add jitter
res2 = res.fillna(0) + np.random.random(res.shape) * 1e-25

# Visualize all data as heatmap
grid = sns.clustermap(
    res2, metric="euclidean", standard_scale=1, row_colors=roi_attributes,
)
grid.savefig(output_dir / "lacunae.clustermap.euclidean.svg", **figkws)
plt.close(grid.fig)

grid = sns.clustermap(
    res2, metric="correlation", standard_scale=1, row_colors=roi_attributes,
)
grid.savefig(output_dir / "lacunae.clustermap.correlation.svg", **figkws)
plt.close(grid.fig)

# Boxen-swarm plots
res2 = res.reset_index().melt(id_vars="roi").set_index("roi")
_test_res = list()
for col in roi_attributes.columns:

    data = res2.join(roi_attributes[col])

    # Test for differences
    aov = pd.concat(
        [
            pg.anova(
                data=data.query(f"variable == '{val}'"),
                dv="value",
                between=col,
            ).assign(variable=val)
            for val in data["variable"].unique()
        ]
    ).set_index("variable")
    _test_res.append(aov)

    kws = dict(data=data, x=col, y="value", palette="tab10",)
    grid = sns.FacetGrid(
        data=data, col="variable", height=3, col_wrap=4, sharey=False,
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
        var = ax.get_title().replace("variable = ", "")
        f = aov.loc[var, "F"]
        p = aov.loc[var, "p-unc"]
        stats = f"\nF = {f:.3f}; p = {p:.3e}"
        ax.set_title(var + stats)

    grid.savefig(output_dir / f"lacunae.{col}.boxen_swarm_plot.svg", **figkws)
    plt.close(grid.fig)

test_res = pd.concat(_test_res)
test_res.to_csv(output_dir / "lacunae.anova_test_results.csv")


# Get the distance of each cell to these structures

for roi in prj.rois:
    get_cell_distance_to_lung_lacunae(roi)

dists = pd.concat(
    parmap.map(get_cell_distance_to_lung_lacunae, prj.rois), axis=0
).fillna(
    0
)  #  these are inside
dists.to_parquet(output_dir / "cell_distance_to_lacunae.pq")

# dists = pd.read_csv(output_dir / "cell_distance_to_lacunae.csv", index_col=0)
